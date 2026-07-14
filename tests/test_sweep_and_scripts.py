"""Tests for sweep.py utilities and launch script sanity checks."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # repo root
SWEEP_PY = PROJECT_ROOT / "src" / "script_utils" / "sweep.py"
SLURM_DIR = PROJECT_ROOT / "slurm_utils"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Add script_utils to path so we can import sweep
sys.path.insert(0, str(PROJECT_ROOT / "src" / "script_utils"))
import sweep  # noqa: E402

# ==========================================================================
# sweep.py unit tests
# ==========================================================================


class TestParseSweepArg:
    """Test CLI --sweep argument parsing."""

    def test_int_list(self):
        key, vals = sweep.parse_sweep_arg("nsteps=[100,200,400]")
        assert key == "nsteps"
        assert vals == [100, 200, 400]

    def test_float_list(self):
        key, vals = sweep.parse_sweep_arg("temperature=[0.5,1.0]")
        assert key == "temperature"
        assert vals == [0.5, 1.0]

    def test_string_list(self):
        key, vals = sweep.parse_sweep_arg('ckpt_name=["a.ckpt","b.ckpt"]')
        assert key == "ckpt_name"
        assert vals == ["a.ckpt", "b.ckpt"]

    def test_bool_list(self):
        key, vals = sweep.parse_sweep_arg("self_cond=[true,false]")
        assert key == "self_cond"
        assert vals == [True, False]

    def test_single_value(self):
        key, vals = sweep.parse_sweep_arg("nsteps=100")
        assert key == "nsteps"
        assert vals == [100]

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Invalid --sweep format"):
            sweep.parse_sweep_arg("no_equals_sign")


class TestLoadSweepConfig:
    """Test YAML sweep config loading."""

    def test_load_valid_config(self, tmp_path):
        cfg = {
            "base_config": "inference",
            "sweep": {"nsteps": [100, 200], "temp": [0.5, 1.0]},
            "output_dir": str(tmp_path / "out"),
        }
        cfg_path = tmp_path / "sweep.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        loaded = sweep.load_sweep_config(cfg_path)
        assert loaded["base_config"] == "inference"
        assert loaded["sweep"]["nsteps"] == [100, 200]

    def test_missing_sweep_key_raises(self, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump({"base_config": "inference"}, f)

        with pytest.raises(ValueError, match="must have a 'sweep' key"):
            sweep.load_sweep_config(cfg_path)


class TestGenerateSweepConfigs:
    """Test end-to-end config generation (requires Hydra)."""

    def test_generates_correct_number_of_configs(self, tmp_path):
        """Sweep over 2 num_replicates × 2 gen_njobs → 4 configs."""
        output_dir = str(tmp_path / "sweep_out")
        configs = sweep.generate_sweep_configs(
            base_config_name="inference",
            sweeper={"num_replicates": [10, 50], "gen_njobs": [1, 2]},
            output_dir=output_dir,
            config_path=str(CONFIGS_DIR),
        )
        assert len(configs) == 4
        assert all(p.exists() for p in configs)
        assert all(p.suffix == ".yaml" for p in configs)

    def test_empty_sweep_generates_one_config(self, tmp_path):
        output_dir = str(tmp_path / "sweep_out")
        configs = sweep.generate_sweep_configs(
            base_config_name="inference",
            sweeper={},
            output_dir=output_dir,
            config_path=str(CONFIGS_DIR),
        )
        assert len(configs) == 1

    def test_generated_configs_contain_sweep_values(self, tmp_path):
        """Verify swept values appear in generated configs."""
        output_dir = str(tmp_path / "sweep_out")
        configs = sweep.generate_sweep_configs(
            base_config_name="inference",
            sweeper={"num_replicates": [42]},
            output_dir=output_dir,
            config_path=str(CONFIGS_DIR),
        )
        assert len(configs) == 1
        with open(configs[0]) as f:
            cfg = yaml.safe_load(f)
        assert cfg["num_replicates"] == 42

    def test_run_name_prefix(self, tmp_path):
        output_dir = str(tmp_path / "sweep_out")
        configs = sweep.generate_sweep_configs(
            base_config_name="inference",
            sweeper={"num_replicates": [10, 20]},
            output_dir=output_dir,
            run_name_prefix="test_prefix",
            config_path=str(CONFIGS_DIR),
        )
        for i, cfg_path in enumerate(configs):
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            assert cfg["run_name_"] == f"test_prefix_{i}"


# ==========================================================================
# YAML sweep config file tests
# ==========================================================================


class TestSweepConfigFiles:
    """Test that the provided sweep config examples are valid."""

    @pytest.mark.parametrize(
        "config_name",
        [
            "ppi_benchmark.yaml",
            "enzyme_benchmark.yaml",
            "offline_val_ckpt_sweep.yaml",
            "search_strategy_sweep.yaml",
        ],
    )
    def test_sweep_config_loads(self, config_name):
        cfg_path = CONFIGS_DIR / "sweeps" / config_name
        if not cfg_path.exists():
            pytest.skip(f"Config not found: {cfg_path}")

        cfg = sweep.load_sweep_config(cfg_path)
        assert "base_config" in cfg
        assert "sweep" in cfg
        assert isinstance(cfg["sweep"], dict)
        assert len(cfg["sweep"]) > 0


# ==========================================================================
# Launch script sanity checks (syntax / --help parsing, no SLURM needed)
# ==========================================================================


class TestLaunchScripts:
    """Test that launch scripts are syntactically valid and source common.sh."""

    @pytest.mark.parametrize(
        "script",
        [
            "launch_infer.sh",
            "launch_sweep.sh",
        ],
    )
    def test_script_exists_and_executable(self, script):
        path = SLURM_DIR / script
        assert path.exists(), f"Script not found: {path}"
        assert os.access(path, os.X_OK), f"Script not executable: {path}"

    @pytest.mark.parametrize(
        "script",
        [
            "launch_infer.sh",
            "launch_sweep.sh",
        ],
    )
    def test_script_syntax_valid(self, script):
        """bash -n checks syntax without executing."""
        path = SLURM_DIR / script
        result = subprocess.run(
            ["bash", "-n", str(path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error in {script}: {result.stderr}"

    @pytest.mark.parametrize(
        "script",
        [
            "launch_infer.sh",
            "launch_sweep.sh",
        ],
    )
    def test_script_sources_common(self, script):
        """Each launch script should source common.sh."""
        path = SLURM_DIR / script
        content = path.read_text()
        assert 'source "$SCRIPT_DIR/common.sh"' in content, f"{script} does not source common.sh"

    def test_common_sh_exists_and_valid(self):
        common = SLURM_DIR / "common.sh"
        assert common.exists()
        result = subprocess.run(
            ["bash", "-n", str(common)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error in common.sh: {result.stderr}"

    def test_launch_infer_uses_resolve_config(self):
        """launch_infer.sh should use resolve_hydra_config, not grep."""
        content = (SLURM_DIR / "launch_infer.sh").read_text()
        assert "resolve_hydra_config" in content, "launch_infer.sh should use resolve_hydra_config"

    def test_launch_infer_has_scontrol_check(self):
        content = (SLURM_DIR / "launch_infer.sh").read_text()
        assert "scontrol" in content, "launch_infer.sh should check scontrol for auto-requeue"

    def test_launch_sweep_has_dry_run(self):
        content = (SLURM_DIR / "launch_sweep.sh").read_text()
        assert "--dry-run" in content


# ==========================================================================
# sweep.py CLI tests
# ==========================================================================


class TestSweepCLI:
    """Test sweep.py command-line interface."""

    def test_dry_run_no_error(self, tmp_path):
        """--dry-run should print info and exit 0 without creating files."""
        cfg = {
            "base_config": "inference",
            "sweep": {"num_replicates": [10, 50]},
            "output_dir": str(tmp_path / "should_not_exist"),
        }
        cfg_path = tmp_path / "sweep.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        existing_pythonpath = os.environ.get("PYTHONPATH", "")
        merged_pythonpath = os.pathsep.join(p for p in [str(PROJECT_ROOT / "script_utils"), existing_pythonpath] if p)
        result = subprocess.run(
            [sys.executable, str(SWEEP_PY), "--sweep-config", str(cfg_path), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": merged_pythonpath},
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert not (tmp_path / "should_not_exist").exists()

    def test_inline_sweep_generates_configs(self, tmp_path):
        output_dir = str(tmp_path / "inline_out")
        result = subprocess.run(
            [
                sys.executable,
                str(SWEEP_PY),
                "--base-config",
                "inference",
                "--sweep",
                "num_replicates=[10,20]",
                "--output-dir",
                output_dir,
                "--config-path",
                str(CONFIGS_DIR),
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        generated = list(Path(output_dir).glob("*.yaml"))
        assert len(generated) == 2
