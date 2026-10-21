"""Reuse upstream preprocessing, sampling, and export with one resident model."""

import os
from pathlib import Path
from typing import Any

from app.storage import Store


class Runner:
    """Own the resident model and reuse the repository inference pipeline."""

    def __init__(self, bucket: Store):
        # Model imports are deferred so API-only tests need no CUDA runtime.
        import lightning as L
        import torch
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from rfproteina.generate import load_ckpt_n_configure_inference

        self.loads = 0
        root = Path(os.environ["PROJECT_ROOT"])
        with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
            self.base = compose(
                config_name="inference",
                overrides=[
                    "~metrics",
                    "compile_lengths=null",
                    "num_workers=0",
                    "skip_existing=false",
                    "inputs=" + str(Path(__file__).resolve().parents[1] / "examples/monomer-short.cif"),
                    "out_dir=/workspace/startup",
                    "seed=42",
                ],
            )
        torch.set_float32_matmul_precision("medium")
        self.model = load_ckpt_n_configure_inference(self.base).eval().to("cuda")
        self.loads += 1
        self.trainer = L.Trainer(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            logger=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            precision=self.base.precision,
        )
        self.device = torch.cuda.get_device_name(0)
        self.omega = OmegaConf

    def generate(self, input_path: Path, output_dir: Path, settings: dict[str, int]) -> list[Path]:
        """Generate CIF artifacts serially, resetting configuration and RNG per request."""
        import lightning as L
        from atomworks.ml.utils.token import get_token_count
        from torch.utils.data import DataLoader

        from rfproteina.datapipes.collate import collate_batch
        from rfproteina.generate import (
            get_dataset_from_cfg,
            load_ckpt_n_configure_inference,
        )

        cfg = self.omega.create(self.omega.to_container(self.base, resolve=False))
        cfg.inputs = str(input_path)
        cfg.out_dir = str(output_dir)
        cfg.seed = settings["seed"]
        cfg.num_replicates = settings["num_replicates"]
        L.seed_everything(cfg.seed, verbose=False)
        load_ckpt_n_configure_inference(cfg, model=self.model)
        self.model.inference_seed = cfg.seed
        dataset = get_dataset_from_cfg(cfg, job_id=0)

        def collate_checked(samples: list[dict[str, Any]]) -> Any:
            if any(get_token_count(sample["atom_array"]) > 512 for sample in samples):
                raise ValueError("This API accepts at most 512 tokens per design")
            return collate_batch(
                samples,
                schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
            )

        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_checked,
        )
        self.trainer.validate(self.model, loader)
        # Lightning may move parameters to CPU during strategy teardown.
        self.model.to("cuda")
        generated = sorted(output_dir.rglob("*_generated.cif"))
        if not generated:
            raise RuntimeError("Generation completed without generated CIF files")
        return generated
