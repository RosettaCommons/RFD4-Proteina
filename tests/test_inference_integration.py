"""Integration tests for proteinfoundation inference pipeline.

These tests exercise the full inference path end-to-end on a GPU,
from input parsing through dataset construction, collation,
model forward pass, and output saving.

Run with:
    pytest projects/proteinfoundation/tests/test_inference_integration.py -v
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.utils.condition import load_atom_array_with_conditions_from_cif
from biotite.structure import get_residue_starts

# Load .env BEFORE any atomworks imports (they read env vars at import time)
from dotenv import load_dotenv

load_dotenv(override=True)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from proteinfoundation.datapipes.pipeline import build_design_validation_pipeline
from proteinfoundation.generate import (
    get_contig_or_design_problem_dataset,
    run_generation,
)
from proteinfoundation.inference_utils.datasets import ContigJsonDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
_CONFIG_DIR = str((PROJECT_ROOT / "configs").resolve())

_TEST_LENGTH = "10-10"

# Columns that should always be present in the per-example output dataframe.
_EXPECTED_DF_COLUMNS = {
    "example_id",
    "designed_structure_path",
    "task",
    "dataset_name",
    "seed",
    "bb_ca_noise_scale",
    "latents_noise_scale",
    "nsteps",
    "chain_of_interest",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def inference_pipeline():
    """Build the standard inference transform pipeline (no metrics)."""
    return build_design_validation_pipeline(metrics_tags=None)


@pytest.fixture(scope="module")
def monomer_json_path():
    """Path to the monomer benchmark JSON (simple unconditional generation)."""
    return str(BENCHMARKS_DIR / "monomer.json")


@pytest.fixture
def tmp_output_dir():
    """Provide a fresh temporary directory for test outputs, cleaned up after."""
    d = tempfile.mkdtemp(prefix="inference_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def minimal_monomer_json(tmp_output_dir):
    """Write a single-example monomer JSON to disk."""
    data = {
        "tiny_monomer": {
            "length": _TEST_LENGTH,
            "dialect": 2,
        }
    }
    path = os.path.join(tmp_output_dir, "tiny_monomer.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def two_example_json(tmp_output_dir):
    """Write a two-example JSON to disk for subsetting and replicate tests."""
    data = {
        "example_a": {
            "length": _TEST_LENGTH,
            "dialect": 2,
        },
        "example_b": {
            "length": _TEST_LENGTH,
            "dialect": 2,
        },
    }
    path = os.path.join(tmp_output_dir, "two_examples.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_input_json(out_dir: str, data: dict, filename: str = "input.json") -> str:
    """Write an input JSON file and return its path."""
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def _build_test_cfg(overrides: list[str] | None = None) -> DictConfig:
    """Load the inference_like_rfd3 config with test-specific overrides.

    Composes the real Hydra config from disk and applies base overrides
    (checkpoint paths, reduced diffusion steps, etc.) plus any
    caller-supplied overrides.

    Args:
        overrides: Additional Hydra-style overrides (e.g. ``["num_replicates=2"]``).

    Returns:
        Fully composed DictConfig ready for :func:`run_generation`.
    """
    GlobalHydra.instance().clear()

    base_overrides = [
        "inference_experiment=inference_like_rfd3",
        f"ckpt_path={_CKPT_PATH}",
        f"ckpt_name={_CKPT_NAME}",
        f"autoencoder_ckpt_path={_AE_CKPT_PATH or 'null'}",
        "seed=42",
        "num_workers=0",
        "metrics_tags=[]",
        "generation.args.nsteps=10",
    ]

    all_overrides = base_overrides + (overrides or [])

    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="inference", overrides=all_overrides)

    return cfg


def _load_output_dataframe(
    out_dir: str,
    global_prefix: str | None = None,
    check_spec_columns: bool = False,
) -> pd.DataFrame:
    """Load the per-example metrics CSV and validate expected columns.

    Args:
        out_dir: Root output directory for the inference run.
        global_prefix: Prefix used during generation (affects CSV filename).
        check_spec_columns: If True, also verify ``input_spec.*`` columns exist.

    Returns:
        Loaded dataframe with validated columns.
    """
    metrics_dir = Path(out_dir) / "inference_metrics"
    prefix = f"{global_prefix}_" if global_prefix else ""
    csv_path = metrics_dir / f"{prefix}per_example_metrics.csv"

    assert csv_path.exists(), f"Expected metrics CSV at {csv_path}"
    df = pd.read_csv(csv_path)

    missing = _EXPECTED_DF_COLUMNS - set(df.columns)
    assert not missing, f"Missing expected columns in output dataframe: {missing}"

    if check_spec_columns:
        spec_cols = [c for c in df.columns if c.startswith("input_spec.")]
        assert len(spec_cols) > 0, f"Expected input_spec.* columns for JSON inputs, got columns: {list(df.columns)}"

    return df


# ---------------------------------------------------------------------------
# Dataset construction tests (CPU-only, no model needed)
# ---------------------------------------------------------------------------


class TestDatasetConstruction:
    """Tests for dataset creation, replication, subsetting, and prefixing.

    These verify the input-parsing side of inference without requiring a GPU.
    """

    def test_contig_json_dataset_basic(self, minimal_monomer_json, inference_pipeline):
        """Verify ContigJsonDataset loads a single example and applies transforms."""
        dataset = get_contig_or_design_problem_dataset(
            minimal_monomer_json,
            num_replicates=1,
            transform=inference_pipeline,
        )
        assert isinstance(dataset, ContigJsonDataset)
        assert len(dataset) == 1
        assert "tiny_monomer_0" in dataset.names

        example = dataset[0]
        assert "atom_array" in example
        assert "example_id" in example
        assert "feats" in example

    def test_num_replicates(self, minimal_monomer_json, inference_pipeline):
        """Verify num_replicates correctly multiplies dataset size."""
        dataset = get_contig_or_design_problem_dataset(
            minimal_monomer_json,
            num_replicates=3,
            transform=inference_pipeline,
        )
        assert len(dataset) == 3
        assert len(set(dataset.names)) == 3

    def test_global_prefix(self, minimal_monomer_json, inference_pipeline):
        """Verify global_prefix is prepended to example IDs."""
        prefix = "myrun"
        dataset = get_contig_or_design_problem_dataset(
            minimal_monomer_json,
            num_replicates=1,
            transform=inference_pipeline,
            global_prefix=prefix,
        )
        for name in dataset.names:
            assert name.startswith(f"{prefix}_"), f"Expected prefix '{prefix}_' in name '{name}'"

    def test_subset_to_examples(self, two_example_json, inference_pipeline):
        """Verify subset_to_examples filters the dataset to requested keys."""
        dataset = get_contig_or_design_problem_dataset(
            two_example_json,
            num_replicates=1,
            transform=inference_pipeline,
            subset_to_examples=["example_a"],
        )
        assert len(dataset) == 1
        assert any("example_a" in name for name in dataset.names)
        assert not any("example_b" in name for name in dataset.names)

    def test_subset_to_examples_comma_string(self, two_example_json, inference_pipeline):
        """Verify subset_to_examples works when passed as a comma-separated string."""
        dataset = get_contig_or_design_problem_dataset(
            two_example_json,
            num_replicates=1,
            transform=inference_pipeline,
            subset_to_examples="example_a,example_b",
        )
        assert len(dataset) == 2

    def test_global_prefix_with_replicates(self, minimal_monomer_json, inference_pipeline):
        """Verify global_prefix and num_replicates compose correctly."""
        dataset = get_contig_or_design_problem_dataset(
            minimal_monomer_json,
            num_replicates=2,
            transform=inference_pipeline,
            global_prefix="run1",
        )
        assert len(dataset) == 2
        for name in dataset.names:
            assert name.startswith("run1_")


# ---------------------------------------------------------------------------
# GPU integration tests (require checkpoint + GPU)
# ---------------------------------------------------------------------------

# TODO: This default path should be updated to a checkpoint trained on latest main!
_CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    "/path/to/shared/storage",
)
_CKPT_NAME = os.environ.get("CKPT_NAME", "last.ckpt")
_AE_CKPT_PATH = os.environ.get("AE_CKPT_PATH", None)

_has_gpu = torch.cuda.is_available()

skip_if_no_gpu = pytest.mark.skipif(not _has_gpu, reason="GPU not available")


@skip_if_no_gpu
class TestEndToEndInference:
    """End-to-end inference tests that load the model and generate structures.

    Set environment variables before running::

        export CKPT_PATH=/path/to/checkpoint/dir
        export CKPT_NAME=model.ckpt
        export AE_CKPT_PATH=/path/to/ae_checkpoint.ckpt
    """

    def test_generation_with_replicates_and_prefix_and_refolding(self, tmp_output_dir):
        """Verify num_replicates, global_prefix, and diffusion_batch_size alias."""
        prefix = "testprefix"
        num_replicates = 2
        bb_ca_noise_scale = 0.2
        latents_noise_scale = 0.3
        input_path = _write_input_json(
            tmp_output_dir,
            {"tiny": {"length": _TEST_LENGTH, "dialect": 2}},
        )

        cfg = _build_test_cfg(
            [
                f"inputs={input_path}",
                f"out_dir={tmp_output_dir}",
                f"num_replicates={num_replicates}",
                f"global_prefix={prefix}",
                "diffusion_batch_size=1",
                f"generation.model.bb_ca.simulation_step_params.sc_scale_noise={bb_ca_noise_scale}",
                f"generation.model.local_latents.simulation_step_params.sc_scale_noise={latents_noise_scale}",
                "metrics_tags=unconditional_refolding"
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_replicates_prefix")

        # Verify correct number of CIF files with prefix in names
        cif_files = list(Path(tmp_output_dir).glob("inference_samples/*.cif"))
        assert (
            len(cif_files) == num_replicates
        ), f"Expected {num_replicates} CIF files, got {len(cif_files)}: {cif_files}"
        for cif_file in cif_files:
            assert prefix in cif_file.name, f"Expected prefix '{prefix}' in filename '{cif_file.name}'"

        # Verify output dataframe has expected fields (including input_spec.* for JSON inputs)
        df = _load_output_dataframe(tmp_output_dir, global_prefix=prefix, check_spec_columns=True)
        assert len(df) == num_replicates
        assert all(df["seed"] == 42)
        assert all(df["bb_ca_noise_scale"] == bb_ca_noise_scale)
        assert all(df["latents_noise_scale"] == latents_noise_scale)
        assert all(df["chain_of_interest"].notna())
        assert any("rf3_metric_unconditional" in col for col in df.columns), "RF3Metric did not run."
        for eid in df["example_id"]:
            assert prefix in eid

    def test_generation_with_subsetting(self, tmp_output_dir):
        """Verify subset_to_examples and json_keys_subset (RFD3 alias)."""
        input_data = {
            "example_a": {"length": _TEST_LENGTH, "dialect": 2},
            "example_b": {"length": _TEST_LENGTH, "dialect": 2},
        }

        # Test subset_to_examples
        out_a = os.path.join(tmp_output_dir, "subset")
        os.makedirs(out_a)
        input_path_a = _write_input_json(out_a, input_data)

        cfg = _build_test_cfg(
            [
                f"inputs={input_path_a}",
                f"out_dir={out_a}",
                "subset_to_examples=example_a",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_subset")

        cif_files_a = list(Path(out_a).glob("inference_samples/*.cif"))
        assert len(cif_files_a) == 1
        assert "example_a" in cif_files_a[0].name

        df_a = _load_output_dataframe(out_a, check_spec_columns=True)
        assert len(df_a) == 1
        assert "example_a" in df_a["example_id"].iloc[0]

        # Test json_keys_subset alias
        out_b = os.path.join(tmp_output_dir, "alias")
        os.makedirs(out_b)
        input_path_b = _write_input_json(out_b, input_data)

        cfg = _build_test_cfg(
            [
                f"inputs={input_path_b}",
                f"out_dir={out_b}",
                "json_keys_subset=example_b",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_json_keys")

        cif_files_b = list(Path(out_b).glob("inference_samples/*.cif"))
        assert len(cif_files_b) == 1
        assert "example_b" in cif_files_b[0].name

        df_b = _load_output_dataframe(out_b, check_spec_columns=True)
        assert len(df_b) == 1
        assert "example_b" in df_b["example_id"].iloc[0]

    def test_skip_existing_prevents_regeneration(self, tmp_output_dir):
        """Verify skip_existing=True skips examples whose outputs already exist."""
        input_path = _write_input_json(
            tmp_output_dir,
            {"tiny": {"length": _TEST_LENGTH, "dialect": 2}},
        )

        # First run: generate
        cfg = _build_test_cfg(
            [
                f"inputs={input_path}",
                f"out_dir={tmp_output_dir}",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_skip")

        cif_files_before = sorted(Path(tmp_output_dir).glob("inference_samples/*.cif"))
        assert len(cif_files_before) >= 1
        mtimes_before = {f.name: f.stat().st_mtime for f in cif_files_before}

        # Verify first run produced a valid dataframe
        _load_output_dataframe(tmp_output_dir, check_spec_columns=True)

        # Second run: skip_existing should prevent regeneration
        cfg = _build_test_cfg(
            [
                f"inputs={input_path}",
                f"out_dir={tmp_output_dir}",
                "skip_existing=true",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_skip")

        cif_files_after = sorted(Path(tmp_output_dir).glob("inference_samples/*.cif"))
        mtimes_after = {f.name: f.stat().st_mtime for f in cif_files_after}

        assert set(mtimes_before.keys()) == set(mtimes_after.keys())
        for name in mtimes_before:
            assert mtimes_before[name] == mtimes_after[name], f"File {name} was regenerated despite skip_existing=True"

    def test_seeded_generation_is_deterministic(self, tmp_output_dir):
        """Verify that two runs with the same seed produce identical outputs."""
        input_path = _write_input_json(
            tmp_output_dir,
            {"tiny": {"length": _TEST_LENGTH, "dialect": 2}},
        )

        out_dirs = []
        for i in range(2):
            out_dir = os.path.join(tmp_output_dir, f"run_{i}")
            os.makedirs(out_dir)
            out_dirs.append(out_dir)

            cfg = _build_test_cfg(
                [
                    f"inputs={input_path}",
                    f"out_dir={out_dir}",
                    "seed=12345",
                ]
            )
            run_generation(cfg, job_id=0, config_name="test_seed")

        cifs_0 = sorted(Path(out_dirs[0]).glob("inference_samples/*.cif"))
        cifs_1 = sorted(Path(out_dirs[1]).glob("inference_samples/*.cif"))

        assert len(cifs_0) == len(cifs_1) >= 1

        for f0, f1 in zip(cifs_0, cifs_1, strict=True):
            aa0 = load_any(str(f0))
            aa1 = load_any(str(f1))
            np.testing.assert_allclose(
                aa0.coord,
                aa1.coord,
                atol=1e-3,
                err_msg="Seeded runs produced different coordinates",
            )

        # Verify both runs produced valid dataframes with matching seeds
        df_0 = _load_output_dataframe(out_dirs[0], check_spec_columns=True)
        df_1 = _load_output_dataframe(out_dirs[1], check_spec_columns=True)
        assert all(df_0["seed"] == 12345)
        assert all(df_1["seed"] == 12345)

    def test_output_cif_annotations(self, tmp_output_dir):
        """Verify that generated CIF files contain expected annotations."""
        input_data = {
            "example_a": {
                "contig": "5-5,A16-20",
                "dialect": 2,
                "input": str(BENCHMARKS_DIR / "motif_data_aa/1prw_aa.pdb"),
                "select_fixed_atoms": True
            },
        }

        # Test subset_to_examples
        out = os.path.join(tmp_output_dir, "subset")
        os.makedirs(out)
        input_path = _write_input_json(out, input_data)

        cfg = _build_test_cfg(
            [
                f"inputs={input_path}",
                f"out_dir={out}",
                "generation.args.nsteps=100",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_subset")

        cif_files = list(Path(out).glob("inference_samples/*.cif"))
        assert len(cif_files) == 1
        assert "example_a" in cif_files[0].name

        EXPECTED_EXTRA_FIELDS_AND_DEFAULTS = [
            ("mapped_to_coordinate_conditioned_region", False),
            ("mapped_to_coordinate_conditioned_region_token_lvl", False),
            ("coords_with_insertion", None), # The defaults will just be generated coords
            ("src_coordinate_conditioned_region", "."),
            ("src_component", "5"),
        ] 
        generated_path = str(cif_files[0])
        generated_aa = load_atom_array_with_conditions_from_cif(
            generated_path,
            extra_fields=[v[0] for v in EXPECTED_EXTRA_FIELDS_AND_DEFAULTS],
        )

        for field, default in EXPECTED_EXTRA_FIELDS_AND_DEFAULTS:
            assert field in generated_aa.get_annotation_categories(), f"Expected annotation '{field}' not found in generated CIF"
            if default is None:
                continue # No default to check, just presence of the field
            residue_starts_aa = generated_aa[get_residue_starts(generated_aa)]
            fraction_default = np.mean(residue_starts_aa.get_annotation(field) == default)
            assert fraction_default == 0.5, (
                f"Expected exactly half of residue starts to have default value '{default}' for annotation '{field}', " 
                f"but got {fraction_default:.2f} in generated CIF"
            )
            

        df = _load_output_dataframe(out, check_spec_columns=True)
        assert len(df) == 1
        assert "example_a" in df["example_id"].iloc[0]

    def test_partial_diffusion(self, tmp_output_dir):
        """Partial diffusion: fixed atoms are preserved, non-fixed atoms change.

        Runs partial diffusion on a stretch of 1PRW chain A, fixing a subset of
        residues with ``select_fixed_atoms``.  After aligning the generated structure
        onto the fixed region of the input, the fixed-region CA RMSD should be near
        zero while the diffused region should show meaningful coordinate change.
        """
        import biotite.structure as struc
        from atomworks.io.utils.io_utils import load_any
        from atomworks.io.parser import STANDARD_PARSER_ARGS, parse

        input_pdb = str(BENCHMARKS_DIR / "motif_data_aa/1ycr_aa.pdb")

        input_data = {
            "partial_diffusion_test_0": {
                "input": input_pdb,
                "contig": "B19-22",
                "select_fixed_atoms": "B19-22",
                "select_unfixed_sequence": "B23-26",
                "partial_t_backbone": 0.5,
                "partial_t_latent": 0.5,
                "dialect": 2,
            }
        }

        input_path = _write_input_json(tmp_output_dir, input_data)
        cfg = _build_test_cfg(
            [
                f"inputs={input_path}",
                f"out_dir={tmp_output_dir}",
                "generation.args.nsteps=400",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_partial_diffusion")

        cif_files = list(Path(tmp_output_dir).glob("inference_samples/*.cif"))
        assert len(cif_files) == 1, f"Expected 1 output CIF, got {len(cif_files)}"

        # Load input and subset to CA atoms
        input_aa = parse(input_pdb, **STANDARD_PARSER_ARGS)["asym_unit"][0]
        input_ca = input_aa[(input_aa.atom_name == "CA")]

        # Load generated CA atoms (output contains the same 10 residues in order)
        output_aa = load_any(str(cif_files[0]))[0]
        output_ca = output_aa[output_aa.atom_name == "CA"]

        # Superimpose output onto input using only the fixed region (first 5 CA atoms)
        fixed_mask =  (input_ca.res_id >= 19) & (input_ca.res_id <= 22)
        _, transform = struc.superimpose(input_ca[fixed_mask], output_ca[fixed_mask])

        # Apply the same rigid-body transform to the full output
        output_ca_aligned = transform.apply(output_ca)

        # Fixed region (B19-22): CA RMSD should be near zero, sequence should be identical
        fixed_rmsd = struc.rmsd(input_ca[fixed_mask], output_ca_aligned[fixed_mask])
        assert fixed_rmsd < 0.5, (
            f"Fixed-region CA RMSD {fixed_rmsd:.3f} Å should be < 0.5 Å"
            " — fixed atoms should retain input coordinates"
        )
        assert np.array_equal(input_ca.res_name[fixed_mask], output_ca_aligned.res_name[fixed_mask]), (
            "Fixed-region residue types should be unchanged in output"
        )

        # Non-fixed region: CA RMSD should be meaningfully > 0, sequences should be non-identical
        nonfixed_rmsd = struc.rmsd(input_ca[~fixed_mask], output_ca_aligned[~fixed_mask])
        assert nonfixed_rmsd > 1, (
            f"Non-fixed-region CA RMSD {nonfixed_rmsd:.3f} Å should be > 1 Å"
            " — diffused atoms should differ from input coordinates"
        )
        assert not np.array_equal(input_ca.res_name[~fixed_mask], output_ca_aligned.res_name[~fixed_mask]), (
            "Non-fixed-region residue types should be different in output"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])