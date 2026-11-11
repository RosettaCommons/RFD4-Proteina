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
        from torch import OutOfMemoryError

        from rfproteina.generate import SampleLimitError, TokenLimitError, run_generation

        cfg = copy.deepcopy(self.base)
        cfg.inputs = str(input_path)
        cfg.out_dir = str(output_dir)
        cfg.seed = settings["seed"]
        cfg.num_replicates = settings["num_replicates"]
        # Upstream exits if a results CSV exists in out_dir.parent. Each request
        # owns that parent, so another request cannot trigger its skip/exit path.
        try:
            run_generation(cfg, model=self.model, max_tokens=max_tokens, max_samples=max_samples)
        except TokenLimitError as exc:
            raise ApiError(
                ErrorCode.REQUEST_TOO_LARGE, f"This API accepts at most {exc.max_tokens} tokens per design", ["input"]
            ) from exc
        except SampleLimitError as exc:
            raise ApiError(
                ErrorCode.REQUEST_TOO_LARGE,
                f"This API accepts at most {exc.max_samples} design samples per request",
                ["sampling.num_replicates"],
            ) from exc
        except OutOfMemoryError as exc:
            raise ApiError(
                ErrorCode.OUT_OF_MEMORY, "The design does not fit on the GPU; reduce the input size.", ["input"]
            ) from exc
        generated = sorted(output_dir.rglob("*_generated.cif"))
        if not generated:
            raise RuntimeError("Generation completed without generated CIF files")
        return generated
