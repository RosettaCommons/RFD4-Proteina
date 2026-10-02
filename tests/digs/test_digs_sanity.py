"""DIGS host-environment sanity checks.

Guards what a DIGS run depends on: the paths ``.ipd/.env.digs`` names all exist,
the JAX cache is writable, ``AF3Oracle`` builds a valid payload, and ``setup.sh``
installs from a fresh clone. Nothing folds -- that is the GPU lane's job, and the
AF3 image itself is smoke-tested host-side in ``.github/ci/run_digs_tests.sh``.

Only run on a DIGS node (see :mod:`tests.digs.conftest`), by the self-hosted
runner (``.github/workflows/digs_tests.yaml``).
"""

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.config import ParseConfig
from atomworks.ml.conditions import C_CRD
from biotite.structure.io.pdbx import CIFFile
from omegaconf import OmegaConf

from rfproteina.metrics.oracles.af3_oracle import AF3Oracle
from rfproteina.metrics.oracles.oracle_metric import OracleJobInput
from tests.digs.conftest import ENV_DIGS, env_digs, running_on_digs

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_CONFIG = REPO_ROOT / "configs" / "inference.yaml"
TRAINING_CONFIG = REPO_ROOT / "configs" / "training_proteina.yaml"


def _squash_sources() -> list[str]:
    """Host-side ``.sqfs`` paths from ``SQUASHFS_MOUNTS`` ("HOST:CONTAINER"), so they cannot drift.

    Empty off-DIGS: nothing to check there, and sourcing the env file would run on
    every lane. On DIGS an empty list is a broken env file, not "no datasets".
    """
    if not running_on_digs():
        return []
    mounts = env_digs('printf "%s\\n" "${SQUASHFS_MOUNTS[@]}"').splitlines()
    assert mounts, f"SQUASHFS_MOUNTS is empty in {ENV_DIGS}"
    return [m.rsplit(":", 1)[0] for m in mounts]


_SQUASH = _squash_sources()
_PATH_VARS = ("AF3_IMAGE", "AF3_MODEL_DIR", "AF3_DB_DIR", "PDB_MIRROR_PATH", "CCD_MIRROR_PATH", "LATENT_APPTAINER_SIF")


def env_digs_value(name: str) -> str:
    """Return ``name``'s expanded value from ``.ipd/.env.digs``."""
    return env_digs(f'printf "%s" "${{{name}?}}"')


@pytest.mark.digs
@pytest.mark.parametrize("var", _PATH_VARS)
def test_env_digs_path_exists(var: str) -> None:
    """Every path ``.ipd/.env.digs`` names exists on DIGS."""
    path = Path(env_digs_value(var))
    assert path.exists(), f"{var} from {ENV_DIGS} is missing on DIGS: {path}"


@pytest.mark.digs
@pytest.mark.parametrize("squashfs", _SQUASH)
def test_squashfs_dataset_exists(squashfs: str) -> None:
    """Every ``SQUASHFS_MOUNTS`` dataset exists, so training can mount it."""
    assert Path(squashfs).is_file(), f"squashfs from {ENV_DIGS} is missing on DIGS: {squashfs}"


@pytest.mark.digs
def test_autoencoder_checkpoint_exists() -> None:
    """The autoencoder checkpoint the training config interpolates exists."""
    os.environ.setdefault("DATA_PATH", env_digs_value("DATA_PATH"))
    ckpt = Path(str(OmegaConf.load(TRAINING_CONFIG).autoencoder_ckpt_path))
    assert ckpt.is_file(), f"autoencoder_ckpt_path from {TRAINING_CONFIG} is missing on DIGS: {ckpt}"


@pytest.mark.digs
def test_jax_cache_dir_is_writable() -> None:
    """AF3 binds this dir in and writes to it mid-fold; writing is the only check NFS ACLs cannot fool."""
    cache_dir = Path(env_digs_value("AF3_JAX_CACHE_DIR"))
    probe = cache_dir / f"digs_ci_write_probe_{os.getpid()}"
    try:
        probe.write_text("probe")
    except OSError as error:
        pytest.fail(
            f"AF3_JAX_CACHE_DIR unusable on DIGS: {cache_dir} ({error}). "
            f"Fix: mkdir -p {cache_dir} && chmod -R 777 {cache_dir}"
        )
    finally:
        probe.unlink(missing_ok=True)


@pytest.mark.digs
def test_af3_oracle_builds_manifest(tmp_path: Path) -> None:
    """``AF3Oracle`` builds a well-formed payload; ``_build_manifest`` is private but public ``fold_batch`` needs a GPU."""
    mirror = Path(env_digs_value("PDB_MIRROR_PATH"))
    config = ParseConfig.from_preset("rcsb", hydrogen_policy="remove", return_atom_array_plus=True)
    design = parse(mirror / "w3" / "2w3o.cif.gz", config=config)["assemblies"]["1"][0]
    design = design[~np.isnan(design.coord).any(axis=-1)]
    template_mask = design.chain_id == "A"
    C_CRD.set_annotation(
        design,
        np.where(template_mask[:, None], C_CRD.ground_truth_value(design), C_CRD.default_annotation(design)),
    )

    oracle = AF3Oracle(oracle_name="af3", template_query="sa_mask('coordinate')", seed=1)
    entries, states = oracle._build_manifest([OracleJobInput(design, "2w3o", tmp_path)], tmp_path)

    assert len(entries) == len(states) == 1
    assert entries[0]["name"] == "2w3o"
    assert entries[0]["template"], "chain A is coordinate-masked, so a template should be built"
    CIFFile.read(io.StringIO(entries[0]["cif"]))  # a payload AF3 cannot parse is the failure guarded against


@pytest.mark.digs
def test_default_inference_checkpoint_exists() -> None:
    """The default checkpoint ``configs/inference.yaml`` interpolates from DATA_PATH exists."""
    os.environ.setdefault("DATA_PATH", env_digs_value("DATA_PATH"))
    cfg = OmegaConf.load(INFERENCE_CONFIG)
    ckpt = Path(str(cfg.ckpt_path)) / str(cfg.ckpt_name)
    assert ckpt.is_file(), f"inference checkpoint from {INFERENCE_CONFIG} is missing on DIGS: {ckpt}"


@pytest.mark.digs
@pytest.mark.slow
def test_setup_sh_fresh_install(tmp_path: Path) -> None:
    """``setup.sh`` installs from a fresh offline clone, with an isolated HOME so the runner's rc is untouched."""
    clone = tmp_path / "repo"
    fake_home = tmp_path / "home"
    fake_home.mkdir()

    # Isolated git config: allow file-protocol submodules (blocked by default) and
    # supply an identity, without touching the runner's real ~/.gitconfig.
    (fake_home / ".gitconfig").write_text(
        '[protocol "file"]\n\tallow = always\n' "[user]\n\temail = ci@digs\n\tname = digs-ci\n"
    )

    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    # Isolate the fresh install: drop the lane's PYTHONPATH/VIRTUAL_ENV so setup.sh's
    # venv is what actually gets exercised, not /workspace's source tree.
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)

    subprocess.run(["git", "clone", "--quiet", f"file://{REPO_ROOT}", str(clone)], check=True, env=env)
    # Resolve the submodule from the local, already-initialized copy so the clone stays offline.
    local_atomworks = REPO_ROOT / "lib" / "atomworks"
    subprocess.run(
        ["git", "-C", str(clone), "config", "-f", ".gitmodules", "submodule.lib/atomworks.url", str(local_atomworks)],
        check=True,
        env=env,
    )
    subprocess.run(["git", "-C", str(clone), "submodule", "sync", "--quiet"], check=True, env=env)

    # `1\n` answers setup.sh's cluster prompt (1 = digs). 30 min covers the uv pip install.
    setup = subprocess.run(
        ["bash", "setup.sh"],
        cwd=str(clone),
        env=env,
        input="1\n",
        text=True,
        capture_output=True,
        timeout=1800,
    )
    assert setup.returncode == 0, (
        f"setup.sh failed (rc={setup.returncode}) in a fresh DIGS install.\n"
        f"--- stdout tail ---\n{setup.stdout[-3000:]}\n--- stderr tail ---\n{setup.stderr[-3000:]}"
    )

    venv_python = clone / ".venv" / "bin" / "python"
    assert venv_python.is_file(), f"setup.sh did not create a venv interpreter at {venv_python}"

    imported = subprocess.run(
        [str(venv_python), "-c", "import rfproteina"],
        cwd=str(clone),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert imported.returncode == 0, (
        f"`import rfproteina` failed in the freshly installed venv.\n" f"--- stderr tail ---\n{imported.stderr[-3000:]}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
