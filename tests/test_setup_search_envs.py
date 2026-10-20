"""Behaviour of scripts/setup_search_envs.sh, run against a scratch copy of the scripts."""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MARK_START = "# >>> rfd4 search environments >>>"


@pytest.fixture
def scratch(tmp_path):
    """A minimal checkout: the scripts, the inference config and an empty .env.local."""
    shutil.copytree(REPO / "scripts", tmp_path / "scripts")
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO / "configs" / "inference.yaml", tmp_path / "configs" / "inference.yaml")
    (tmp_path / ".env.local").write_text("DATA_PATH=${PROJECT_ROOT}/data_path\n")
    return tmp_path


def run(scratch, *args, stdin="", prompt=False):
    env = {"PATH": "/usr/bin:/bin", "HOME": str(scratch), "RFD4_SETUP_PROMPT": "1" if prompt else ""}
    return subprocess.run(
        ["bash", str(scratch / "scripts" / "setup_search_envs.sh"), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=env,
        cwd=scratch,
        check=False,
    )


def env_block(scratch):
    text = (scratch / ".env.local").read_text()
    return text.split(MARK_START)[1] if MARK_START in text else ""


def test_without_terminal_or_flags_nothing_happens(scratch):
    result = run(scratch)
    assert result.returncode == 0
    assert "nothing was requested" in result.stdout
    assert MARK_START not in (scratch / ".env.local").read_text()


def test_answers_select_environments_arch_and_cuda(scratch):
    # AF2 yes, AF3 yes, OpenFold3 weights no, AF2 parameters no, arm64, CUDA 12.
    result = run(scratch, stdin="y\ny\nn\nn\n2\n1\n", prompt=True)
    assert result.returncode == 0, result.stderr
    block = env_block(scratch)
    assert "arch=arm64, cuda=12" in block
    # The image reference is a docker tag where docker is installed and an Apptainer .sif path otherwise.
    assert any(
        ref in block
        for ref in (
            "AF2_IMAGE=${DATA_PATH}/af2/runtime/alphafold2-pallas-cuda12-arm64.sif",
            "AF2_IMAGE=alphafold2-pallas-cuda12-arm64:latest",
        )
    )
    assert "AF3_JAX_CACHE_DIR=${DATA_PATH}/openfold3/jax_cache" in block
    assert "AF3_MODEL_DIR" not in block  # weights declined
    assert "AF3_IMAGE" not in block  # no prebuilt arm64 image
    assert "No prebuilt AlphaFold 3 / OpenFold3 image is published for arm64" in result.stdout


def test_declining_both_environments_writes_nothing(scratch):
    result = run(scratch, stdin="n\nn\n", prompt=True)
    assert result.returncode == 0
    assert "No search environments requested" in result.stdout
    assert MARK_START not in (scratch / ".env.local").read_text()


def test_rerun_replaces_the_block_instead_of_appending(scratch):
    run(scratch, "--af3", "--no-of3-weights", "--arch", "amd64", "--cuda", "13", "--dry-run")
    for cuda in ("12", "13"):
        run(scratch, "--af2", "--no-af2-params", "--arch", "amd64", "--cuda", cuda)
    text = (scratch / ".env.local").read_text()
    assert text.count(MARK_START) == 1
    assert "alphafold2-pallas-cuda13-amd64" in text
    assert "alphafold2-pallas-cuda12-amd64" not in text
    assert text.startswith("DATA_PATH=")  # the rest of the file is untouched


def test_dry_run_changes_nothing(scratch):
    result = run(
        scratch, "--af2", "--af2-params", "--af3", "--of3-weights", "--arch", "amd64", "--cuda", "13", "--dry-run"
    )
    assert result.returncode == 0, result.stderr
    assert "[dry run]" in result.stdout
    assert MARK_START not in (scratch / ".env.local").read_text()
    assert not (scratch / "data_path").exists()


@pytest.mark.parametrize("args", [["--arch", "riscv"], ["--cuda", "11"], ["--bogus"]])
def test_bad_options_are_rejected(scratch, args):
    assert run(scratch, *args).returncode == 2


def test_setup_scripts_use_only_the_hub_and_deepmind():
    """Model, test data and OpenFold3 artifacts come from the Hugging Face repository, not an object store."""
    for name in ("download_assets.sh", "oracle_artifacts.sh", "setup_search_envs.sh"):
        text = (REPO / "scripts" / name).read_text()
        assert "cwobject" not in text and "RFPROTEINA_DATA_URL" not in text, name
    assert "OF3_RUNTIME_FILE=" in (REPO / "scripts" / "oracle_artifacts.sh").read_text()
