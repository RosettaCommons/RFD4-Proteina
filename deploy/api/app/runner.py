"""Serve the upstream generation entry point with one resident model."""

import os
from pathlib import Path


class Runner:
    """Own the model; upstream owns request configuration, compilation, and sampling."""

    def __init__(self):
        # Deferred imports keep HTTP-only tests independent of model dependencies.
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

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
        self.omega = OmegaConf

    def generate(self, input_path: Path, output_dir: Path, settings: dict[str, int]) -> list[Path]:
        """Keep upstream bucketing/cache behavior and isolate each request's outputs."""
        from rfproteina.generate import TokenLimitError, run_generation

        cfg = self.omega.create(self.omega.to_container(self.base, resolve=False))
        cfg.inputs = str(input_path)
        cfg.out_dir = str(output_dir)
        cfg.seed = settings["seed"]
        cfg.num_replicates = settings["num_replicates"]
        # Upstream exits if a results CSV exists in out_dir.parent. Each request
        # owns that parent, so another request cannot trigger its skip/exit path.
        try:
            run_generation(cfg, model=self.model, max_tokens=512, max_samples=4)
        except TokenLimitError as exc:
            raise ValueError(f"This API accepts at most {exc.max_tokens} tokens per design") from exc
        generated = sorted(output_dir.rglob("*_generated.cif"))
        if not generated:
            raise RuntimeError("Generation completed without generated CIF files")
        return generated
