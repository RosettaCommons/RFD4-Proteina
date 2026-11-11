"""Serve the upstream generation entry point with one resident model."""

import copy
import os
from pathlib import Path

from .errors import ApiError, ErrorCode


class Runner:
    """Own the model; upstream owns request configuration, compilation, and sampling."""

    def __init__(self):
        # Deferred imports keep HTTP-only tests independent of model dependencies.
        import torch
        from hydra import compose, initialize_config_dir

        from rfproteina.generate import load_ckpt_n_configure_inference

        root = Path(os.environ["PROJECT_ROOT"])
        with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
            self.base = compose(
                config_name="inference",
                overrides=[
                    "~metrics",
                    "num_workers=0",
                    "skip_existing=false",
                    "inputs=null",
                    "out_dir=/workspace/startup",
                    "seed=42",
                ],
            )
        # Loading only resolves configuration; no example file is needed at startup.
        self.model = load_ckpt_n_configure_inference(self.base).eval().to("cuda")
        self.loads = 1
        self.device = torch.cuda.get_device_name(0)

    def generate(
        self,
        input_path: Path,
        output_dir: Path,
        settings: dict[str, int],
        max_tokens: int,
        max_samples: int,
    ) -> list[Path]:
        """Keep upstream bucketing/cache behavior and isolate each request's outputs."""
        from atomworks.ml.utils.token import get_token_count
        from torch import OutOfMemoryError
        from torch.utils.data import Dataset

        from rfproteina.generate import run_generation

        cfg = copy.deepcopy(self.base)
        cfg.inputs = str(input_path)
        # Upstream exits if a results CSV exists in out_dir.parent. Each request
        # owns that parent, so another request cannot trigger its skip/exit path.
        cfg.out_dir = str(output_dir)
        cfg.seed = settings["seed"]
        cfg.num_replicates = settings["num_replicates"]

        def prepare_dataset(dataset: Dataset) -> list[dict]:
            if len(dataset) > max_samples:
                raise ApiError(
                    ErrorCode.REQUEST_TOO_LARGE,
                    f"At most {max_samples} design samples per request.",
                    ["sampling.num_replicates"],
                )
            # Cache the exact transformed samples: a second pass could sample different lengths.
            samples = []
            for sample in dataset:
                if get_token_count(sample["atom_array"]) > max_tokens:
                    raise ApiError(ErrorCode.REQUEST_TOO_LARGE, f"At most {max_tokens} tokens per design.", ["input"])
                samples.append(sample)
            return samples

        try:
            run_generation(cfg, model=self.model, prepare_dataset=prepare_dataset)
        except OutOfMemoryError as exc:
            raise ApiError(
                ErrorCode.OUT_OF_MEMORY, "The design does not fit on the GPU; reduce the input size.", ["input"]
            ) from exc
        generated = sorted(output_dir.rglob("*_generated.cif"))
        if not generated:
            raise RuntimeError("Generation completed without generated CIF files")
        return generated
