"""Tests for the AF3 worker backend switch and its argv builders.

``af3_apptainer`` is loaded by path rather than imported as
``rfproteina.af3.af3_apptainer`` on purpose: the module exists to be importable
from the host's bare ``/usr/bin/python3`` in the slurm launcher, where package
init (``import torch``, the cuEquivariance kernels) is unavailable. Importing it
the heavy way here would test a path nothing in production takes.

These are argv-construction tests. They need no GPU, no weights and no image —
every environment variable below points at a tmp_path stand-in.
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
        assert af3.resolve_backend() == "apptainer"

    def test_reads_env(self, af3, monkeypatch):
        monkeypatch.setenv("AF3_RUNTIME_BACKEND", "native")
        assert af3.resolve_backend() == "native"

    def test_explicit_argument_beats_env(self, af3, monkeypatch):
        monkeypatch.setenv("AF3_RUNTIME_BACKEND", "native")
        assert af3.resolve_backend("apptainer") == "apptainer"

    @pytest.mark.parametrize("raw", ["NATIVE", " native ", "Native"])
    def test_normalises_case_and_whitespace(self, af3, monkeypatch, raw):
        monkeypatch.setenv("AF3_RUNTIME_BACKEND", raw)
        assert af3.resolve_backend() == "native"

    def test_rejects_unknown_backend(self, af3, monkeypatch):
        monkeypatch.setenv("AF3_RUNTIME_BACKEND", "docker")
        with pytest.raises(ValueError, match="docker"):
            af3.resolve_backend()

    def test_empty_env_falls_back_to_default(self, af3, monkeypatch):
        # An exported-but-empty variable is the usual shape of a half-written
        # .env file; it must not raise, and must not select native.
        monkeypatch.setenv("AF3_RUNTIME_BACKEND", "")
        assert af3.resolve_backend() == "apptainer"


class TestRequiredEnvVars:
    def test_apptainer_wants_an_image_not_an_interpreter(self, af3):
        required = af3.required_env_vars("apptainer")
        assert "AF3_IMAGE" in required
        assert "AF3_PYTHON" not in required

    def test_native_wants_an_interpreter_not_an_image(self, af3):
        required = af3.required_env_vars("native")
        assert "AF3_PYTHON" in required
        assert "AF3_IMAGE" not in required

    @pytest.mark.parametrize("backend", ["apptainer", "native"])
    def test_both_want_weights_and_cache(self, af3, backend):
        required = af3.required_env_vars(backend)
        assert "AF3_MODEL_DIR" in required
        assert "AF3_JAX_CACHE_DIR" in required


class TestNativeCommand:
    def test_does_not_resolve_the_venv_symlink(self, af3, monkeypatch, common_env, venv_python):
        """Regression: resolving ``bin/python`` silently deactivates the venv.

        A venv is active only when its own interpreter path is the one executed.
        ``.resolve()`` here launches the base interpreter instead, which has no
        ``alphafold3`` in site-packages, and the worker dies with
        ``ModuleNotFoundError`` far from the cause.
        """
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        cmd = af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert str(venv_python) in cmd
        assert str(venv_python.resolve()) not in cmd

    def test_normalises_without_following_links(self, af3, monkeypatch, common_env, venv_python):
        messy = venv_python.parent / ".." / "bin" / venv_python.name
        monkeypatch.setenv("AF3_PYTHON", str(messy))
        cmd = af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert str(venv_python) in cmd

    def test_expands_user(self, af3, monkeypatch, common_env, venv_python):
        monkeypatch.setenv("HOME", str(venv_python.parents[2]))
        monkeypatch.setenv("AF3_PYTHON", "~/venv/bin/python")
        cmd = af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert str(venv_python) in cmd

    def test_rejects_a_directory(self, af3, monkeypatch, common_env, tmp_path):
        monkeypatch.setenv("AF3_PYTHON", str(tmp_path))
        with pytest.raises(OSError, match="does not point at an interpreter"):
            af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)

    def test_rejects_a_dangling_symlink(self, af3, monkeypatch, common_env, tmp_path):
        link = tmp_path / "python"
        link.symlink_to(tmp_path / "nowhere")
        monkeypatch.setenv("AF3_PYTHON", str(link))
        with pytest.raises(OSError, match="does not point at an interpreter"):
            af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)

    def test_requires_af3_python(self, af3, common_env):
        with pytest.raises(OSError, match="AF3_PYTHON"):
            af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)

    def test_never_invokes_a_container(self, af3, monkeypatch, common_env, venv_python):
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        monkeypatch.setenv("AF3_IMAGE", "/should/not/be/read.sif")
        cmd = af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert "apptainer" not in cmd
        assert not any("read.sif" in part for part in cmd)

    def test_carries_env_through_the_env_prefix(self, af3, monkeypatch, common_env, venv_python):
        weights, cache = common_env
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        cmd = af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=3, save_full_outputs=False)
        assert cmd[0] == "env"
        assert "CUDA_VISIBLE_DEVICES=3" in cmd
        assert f"AF3_MODEL_DIR={weights}" in cmd
        assert f"--jax_compilation_cache_dir={cache}" in cmd

    def test_serve_mode_takes_no_manifest(self, af3, monkeypatch, common_env, venv_python):
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        cmd = af3.build_native_fold_command(manifest_path=None, gpu_device=0, save_full_outputs=False, serve=True)
        assert "--serve" in cmd
        assert not any(part.startswith("--manifest=") for part in cmd)

    def test_manifest_mode_requires_a_manifest(self, af3, monkeypatch, common_env, venv_python):
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        with pytest.raises(ValueError, match="manifest_path"):
            af3.build_native_fold_command(manifest_path=None, gpu_device=0, save_full_outputs=False, serve=False)


class TestXlaFlags:
    """Both routes must run the worker under one XLA configuration.

    The image sets ``XLA_FLAGS`` in its ``%environment``, which is sourced at
    container start and overrides the host, so the value has to be applied
    inside the invocation rather than passed in from outside.
    """

    def test_container_sets_flags_after_the_image_environment(self, af3, monkeypatch, common_env, tmp_path):
        sif = tmp_path / "af3.sif"
        sif.write_text("")
        monkeypatch.setenv("AF3_IMAGE", str(sif))
        cmd = af3.build_apptainer_fold_command(
            manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False
        )
        # ... which means: after the image path, not as an --env before it.
        assert cmd[cmd.index(str(sif)) + 1] == "env"
        assert f"XLA_FLAGS={af3.AF3_XLA_FLAGS}" in cmd[cmd.index(str(sif)) :]

    def test_native_sets_the_same_flags(self, af3, monkeypatch, common_env, venv_python):
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        cmd = af3.build_native_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert f"XLA_FLAGS={af3.AF3_XLA_FLAGS}" in cmd

    def test_both_routes_agree(self, af3, monkeypatch, common_env, venv_python, tmp_path):
        sif = tmp_path / "af3.sif"
        sif.write_text("")
        monkeypatch.setenv("AF3_IMAGE", str(sif))
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        kwargs = {"manifest_path": "/tmp/m.json", "gpu_device": 0, "save_full_outputs": False}

        def xla(cmd):
            return [part for part in cmd if part.startswith("XLA_FLAGS=")]

        assert xla(af3.build_apptainer_fold_command(**kwargs)) == xla(af3.build_native_fold_command(**kwargs))

    def test_default_disables_triton_gemm(self, af3):
        # AF3 upstream's own recommendation, and what the image has always set.
        assert "--xla_gpu_enable_triton_gemm=false" in af3.AF3_XLA_FLAGS

    def test_overridable(self, monkeypatch):
        monkeypatch.setenv("AF3_XLA_FLAGS", "--xla_gpu_autotune_level=0")
        # Read at import, so a reload is what picks the override up.
        assert _load_module().AF3_XLA_FLAGS == "--xla_gpu_autotune_level=0"


class TestDispatch:
    def test_native_backend_selects_the_native_builder(self, af3, monkeypatch, common_env, venv_python):
        monkeypatch.setenv("AF3_RUNTIME_BACKEND", "native")
        monkeypatch.setenv("AF3_PYTHON", str(venv_python))
        cmd = af3.build_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert cmd[0] == "env"

    def test_default_backend_selects_apptainer(self, af3, monkeypatch, common_env, tmp_path):
        sif = tmp_path / "af3.sif"
        sif.write_text("")
        monkeypatch.setenv("AF3_IMAGE", str(sif))
        cmd = af3.build_fold_command(manifest_path="/tmp/manifest.json", gpu_device=0, save_full_outputs=False)
        assert cmd[0] == "apptainer"

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
