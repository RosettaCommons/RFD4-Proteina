"""Verification test suite for 3D point-group symmetry design in RFD4-Proteina."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

for _libstdcpp in ("/lib/x86_64-linux-gnu/libstdc++.so.6", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"):
    if os.path.exists(_libstdcpp):
        ctypes.CDLL(_libstdcpp, mode=ctypes.RTLD_GLOBAL)
        break

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
WORKTREE_SRC = WORKTREE_ROOT / "src"

os.environ.setdefault("PROJECT_ROOT", str(WORKTREE_ROOT))
os.environ.setdefault("USER", "pytest")
os.environ.setdefault("LOGNAME", "pytest")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor-pytest")

if str(WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKTREE_SRC))


def assert_worktree_imports() -> None:
    """Assert rfproteina and rfd4_proteina resolve inside the feature git worktree."""
    import rfd4_proteina
    import rfproteina

    rfp_path = Path(rfproteina.__file__).resolve()
    rfd4_path = Path(rfd4_proteina.__file__).resolve()
    if not rfp_path.is_relative_to(WORKTREE_SRC):
        raise AssertionError(f"rfproteina imported from {rfp_path}, expected under {WORKTREE_SRC}")
    if not rfd4_path.is_relative_to(WORKTREE_SRC):
        raise AssertionError(f"rfd4_proteina imported from {rfd4_path}, expected under {WORKTREE_SRC}")


def build_tiny_cpu_proteina(store_dir: str | Path) -> object:
    """Build a fast 1-layer CPU Proteina + AutoEncoder instance using the SOTA feature contract."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    from rfproteina.proteina import Proteina

    config_dir = str((WORKTREE_ROOT / "configs").resolve())

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg_ae = compose(
            config_name="training_ae",
            overrides=[
                "nn_ae.latent_z_dim=8",
                "nn_ae.use_torch_compile=false",
                "+nn_ae.attention_type=torch",
                "nn_ae.encoder.nlayers=1",
                "nn_ae.encoder.token_dim=32",
                "nn_ae.encoder.nheads=4",
                "nn_ae.encoder.pair_repr_dim=16",
                "nn_ae.decoder.nlayers=1",
                "nn_ae.decoder.token_dim=32",
                "nn_ae.decoder.nheads=4",
                "nn_ae.decoder.pair_repr_dim=16",
            ],
        )

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg_flow = compose(
            config_name="training_proteina",
            overrides=[
                "autoencoder_ckpt_path=null",
                "hardware.accelerator=cpu",
                "compress_outputs=false",
                "nn.nlayers=1",
                "nn.token_dim=32",
                "nn.nheads=4",
                "nn.pair_repr_dim=16",
                "nn.use_torch_compile=false",
                "nn.attention_type=torch",
                "nn.tri_mult_backend=torch",
                "generation.compile_lengths=null",
                "training.n_recycle_max=1",
                "training.n_recycle_val_loss=1",
            ],
        )

    OmegaConf.set_struct(cfg_flow, False)
    cfg_flow.metrics = None
    cfg_flow.autoencoder_config = cfg_ae
    cfg_flow.datasets = OmegaConf.create({"train": {}, "val": {}})
    if "generation" in cfg_flow and cfg_flow.generation is not None:
        cfg_flow.generation.n_recycle_schedule = None
        if "model" in cfg_flow.generation:
            for dm in ("bb_ca", "local_latents"):
                if dm in cfg_flow.generation.model and "schedule" in cfg_flow.generation.model[dm]:
                    cfg_flow.generation.model[dm].schedule.tail_steps = 0
    OmegaConf.set_struct(cfg_flow, True)

    model = Proteina(cfg_flow, store_dir=str(store_dir), load_training_metrics=False)
    return model.to("cpu").eval()
