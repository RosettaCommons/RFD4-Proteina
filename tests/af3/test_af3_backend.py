"""Tests for the AF3 worker backend switch and its argv builders.

``af3_apptainer`` is loaded by path rather than imported as
``rfproteina.af3.af3_apptainer`` on purpose: the module exists to be importable
from the host's bare ``/usr/bin/python3`` in the slurm launcher, where package
init (``import torch``, the cuEquivariance kernels) is unavailable. Importing it
the heavy way here would test a path nothing in production takes.

These are argv-construction tests. They need no GPU, no weights and no image —
every environment variable below points at a tmp_path stand-in. That matters
because the repo's CPU lane deselects ``gpu``, so an end-to-end fold would not
run in CI at all; what can be checked here is the command, and the command is
what decides whether the two backends compute the same thing.

Deliberately narrow. Each test below guards either a bug that already shipped
or an invariant the two backends must share. Earlier revisions also covered
case normalisation, ``~`` expansion, dangling-symlink rejection and the
serve/manifest variants; those exercised defensive branches rather than
anything that had gone wrong, and were dropped as noise.
"""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "rfproteina" / "af3" / "af3_apptainer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("af3_apptainer_under_test", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def af3(monkeypatch):
    """The module with every AF3_* variable cleared, so defaults are real defaults."""
    for name in (
        "AF3_RUNTIME_BACKEND",
        "AF3_IMAGE",
        "AF3_PYTHON",
        "AF3_MODEL_DIR",
        "AF3_JAX_CACHE_DIR",
        "APPTAINER_BIND_PATHS",
    ):
        monkeypatch.delenv(name, raising=False)
    return _load_module()


@pytest.fixture
def venv_python(tmp_path):
    """A venv-shaped ``bin/python``: a symlink to an interpreter living elsewhere.

    This is the shape that matters. Real venvs link ``bin/python`` at the base
    interpreter and derive ``sys.prefix`` from the adjacent ``pyvenv.cfg``.
    """
    base = tmp_path / "base" / "bin"
    base.mkdir(parents=True)
    real = base / "python3.12"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)

    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    link = venv_bin / "python"
    link.symlink_to(real)
    (tmp_path / "venv" / "pyvenv.cfg").write_text(f"home = {base}\n")
    return link


@pytest.fixture
def common_env(monkeypatch, tmp_path):
    weights = tmp_path / "weights"
    cache = tmp_path / "jax_cache"
    weights.mkdir()
    cache.mkdir()
    monkeypatch.setenv("AF3_MODEL_DIR", str(weights))
    monkeypatch.setenv("AF3_JAX_CACHE_DIR", str(cache))
    return weights, cache


class TestResolveBackend:
    def test_defaults_to_apptainer(self, af3):
        """Every existing deployment sets no backend variable at all."""
        assert af3.resolve_backend() == "apptainer"

    def test_rejects_unknown_backend(self, af3, monkeypatch):
        """A typo must fail here, not silently run the other backend."""
        monkeypatch.setenv("AF3_RUNTIME_BACKEND", "docker")
        with pytest.raises(ValueError, match="docker"):
            af3.resolve_backend()


class TestRequiredEnvVars:
    """The daemon checks these before binding its socket, per backend."""

    def test_apptainer_wants_an_image_not_an_interpreter(self, af3):
        required = af3.required_env_vars("apptainer")
        assert "AF3_IMAGE" in required
        assert "AF3_PYTHON" not in required

    def test_native_wants_an_interpreter_not_an_image(self, af3):
        required = af3.required_env_vars("native")
        assert "AF3_PYTHON" in required
        assert "AF3_IMAGE" not in required


class TestNativeCommand:
    def test_does_not_resolve_the_venv_symlink(self, af3, monkeypatch, common_env, venv_python):
        """Regression: resolving ``bin/python`` silently deactivates the venv.

        A venv is active only when its own interpreter path is the one executed.
        ``.resolve()`` here launches the base interpreter instead, which has no
        ``alphafold3`` in site-packages, and the worker dies with
        ``ModuleNotFoundError`` far from the cause. This shipped once and was
        found only by running a fold on a cluster.
        """
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        cmd = af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert str(venv_python) in cmd
        assert str(venv_python.resolve()) not in cmd

    def test_requires_af3_python(self, af3, common_env):
        """Missing configuration should fail here, not on an allocated GPU."""
        with pytest.raises(OSError, match="AF3_PYTHON"):
            af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)


class TestXlaConfiguration:
    def test_container_sets_flags_after_the_image_environment(self, af3, monkeypatch, common_env, tmp_path):
        """Placement, not presence, is what this guards.

        The image sets ``XLA_FLAGS`` in its ``%environment``, which is sourced at
        container start and beats both a host export and ``--env``. The value has
        to be applied inside the invocation — after the image path — or the two
        backends silently run under different XLA configurations, which makes any
        comparison between them a comparison of two different computations.
        """
        sif = tmp_path / "af3.sif"
        sif.write_text("")
        monkeypatch.setenv("AF3_IMAGE", str(sif))
        cmd = af3.build_apptainer_fold_command(
            manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False
        )
        assert cmd[cmd.index(str(sif)) + 1] == "env"
        assert f"XLA_FLAGS={af3.AF3_XLA_FLAGS}" in cmd[cmd.index(str(sif)) :]


class TestDispatch:
    def test_both_backends_agree_on_the_worker_arguments(self, af3, monkeypatch, common_env, venv_python, tmp_path):
        """The container was only ever a dependency shipper.

        Whatever else differs, the two routes must hand ``run_alphafold_batch.py``
        the same flags — otherwise "native matches the container" is a claim about
        two different computations.
        """
        sif = tmp_path / "af3.sif"
        sif.write_text("")
        monkeypatch.setenv("AF3_IMAGE", str(sif))
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))

        kwargs = {"manifest_path": "/tmp/manifest.json", "gpu_device": 0, "save_full_outputs": False}
        container = af3.build_apptainer_fold_command(**kwargs)
        native = af3.build_native_fold_command(**kwargs)

        def worker_flags(cmd):
            # Everything before the batch script is the launcher's own argv
            # (apptainer's --nv/--bind, or env's VAR=VALUE prefix); only what
            # follows it reaches run_alphafold_batch.py.
            script = next(i for i, part in enumerate(cmd) if part.endswith("run_alphafold_batch.py"))
            return [part for part in cmd[script + 1 :] if part.startswith("--")]

        assert worker_flags(container) == worker_flags(native)
