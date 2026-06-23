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

import biotite.structure as struc
import numpy as np
import pandas as pd
import pytest
from atomworks.enums import ChainType
from atomworks.io.config import ParseConfig
from atomworks.io.parser import parse
from atomworks.io.utils.io_utils import load_any
from biotite.structure import get_residue_starts

# Load .env BEFORE any atomworks imports (they read env vars at import time)
from dotenv import load_dotenv

load_dotenv(override=True)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from pydantic_core._pydantic_core import ValidationError

from foundry.testing import gpu  # noqa: F401
from foundry.utils.standard_annotations import SA_CRDSRC, SA_CWI, SA_SRC
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


def _inference_generated_cifs(out_dir: str | Path) -> list[Path]:
    """Return generated structure CIFs (sharded under ``inference_samples/``)."""
    return list(Path(out_dir).glob("inference_samples/**/*_generated.cif"))


_TEST_LENGTH = "10-10"

# Columns that should always be present in the per-example output dataframe.
_EXPECTED_DF_COLUMNS = {
    "example_id",
    "designed_structure_path",
    "task",
    "dataset_name",
    "seed",
    "input_json_path",
    "chain_of_interest",
    "ckpt_path",
    "ckpt_name",
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
def diffused_na_json(tmp_output_dir):
    """Write a single-example JSON with a mix of RNA, DNA, and protein residues."""
    data = {
        "diffused_na": {
            "length": "15-15",
            "contig": "5D,/0,5R,/0,5P",
            "dialect": 2,
        }
    }
    path = os.path.join(tmp_output_dir, "diffused_na.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def bad_diffused_na_json(tmp_output_dir):
    """Write a single-example JSON with an invalid polymer type setup (varies within chain)."""
    data = {
        "bad_diffused_na": {
            "length": "15-15",
            "contig": "5D,/0,5R,5P",
            "dialect": 2,
        }
    }
    path = os.path.join(tmp_output_dir, "bad_diffused_na.json")
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
        "metrics/refolding_oracles=rf3",
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


def _assert_coords_identical(cif_path_a: str, cif_path_b: str) -> None:
    """Assert two CIF files contain identical all-atom coordinates after alignment.

    Superimposes structure B onto structure A, then verifies all-atom coordinates
    match within 1e-3 Å.

    Args:
        cif_path_a: Path to the reference CIF file.
        cif_path_b: Path to the CIF file to compare against the reference.
    """
    aa_a = load_any(cif_path_a)
    aa_b = load_any(cif_path_b)
    assert (
        aa_a.coord.shape == aa_b.coord.shape
    ), f"Structures have different shapes: {aa_a.coord.shape} vs {aa_b.coord.shape}"
    _, transform = struc.superimpose(aa_a, aa_b)
    aa_b_aligned = transform.apply(aa_b)
    np.testing.assert_allclose(
        aa_a.coord,
        aa_b_aligned.coord,
        atol=1e-3,
        err_msg="Structures produced different coordinates after alignment",
    )


def _load_output_dataframe(
    out_dir: str,
    global_prefix: str | None = None,
    check_input_json_path: bool = False,
) -> pd.DataFrame:
    """Load the per-example metrics CSV and validate expected columns.

    Args:
        out_dir: Root output directory for the inference run.
        global_prefix: Prefix used during generation (affects CSV filename).
        check_input_json_path: If True, verify ``input_json_path`` is set and exists.

    Returns:
        Loaded dataframe with validated columns.
    """
    metrics_dir = Path(out_dir) / "inference_metrics"
    prefix = f"{global_prefix}_" if global_prefix else ""
    csv_path = metrics_dir / f"{prefix}per_example_metrics.csv"

    assert csv_path.exists(), f"Expected metrics CSV at {csv_path}"
    df = pd.read_csv(csv_path)

    present = set(df.columns)
    missing = _EXPECTED_DF_COLUMNS - present
    assert not missing, (
        f"Missing expected columns in {csv_path}: {sorted(missing)}. "
        f"Present columns ({len(present)}): {sorted(present)}"
    )

    if check_input_json_path:
        assert df["input_json_path"].notna().all(), "Expected input_json_path for JSON inputs"
        for json_path in df["input_json_path"].dropna().unique():
            assert Path(json_path).is_file(), f"input_json_path does not exist: {json_path}"

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
        assert example["specification"]["input_json_path"] == str(Path(minimal_monomer_json).resolve())

    def test_diffused_na_contigs(self, diffused_na_json, bad_diffused_na_json, inference_pipeline):
        """Verify ContigJsonDataset loads a single example and applies transforms."""
        dataset = get_contig_or_design_problem_dataset(
            diffused_na_json,
            num_replicates=1,
            transform=inference_pipeline,
        )

        example = dataset[0]
        assert "atom_array" in example
        atom_array = example["atom_array"]
        annots = ["chain_type", "is_protein", "is_rna", "is_dna", "is_ligand"]
        for annot in annots:
            assert annot in atom_array.get_annotation_categories(), f"Expected '{annot}' annotation in atom array"

        residue_starts = get_residue_starts(atom_array)
        assert len(residue_starts) == 15, f"Expected 15 residues but got {len(residue_starts)}"
        rna_start, protein_start = residue_starts[5], residue_starts[10]

        aa_and_expectations = [
            (atom_array[:rna_start], ChainType.DNA, False, False, True, False),  # DNA residues
            (atom_array[rna_start:protein_start], ChainType.RNA, False, True, False, False),  # RNA residues
            (atom_array[protein_start:], ChainType.POLYPEPTIDE_L, True, False, False, False),  # Protein residues
        ]

        for chain_group in aa_and_expectations:
            aa = chain_group[0]
            for annot, expectation in zip(annots, chain_group[1:], strict=False):
                values = aa.get_annotation(annot)
                assert np.all(
                    values == expectation
                ), f"Expected annotation '{annot}' to be {expectation} but got values {values}."

        with pytest.raises(ValidationError, match="Chain types must be consistent"):
            _ = get_contig_or_design_problem_dataset(
                bad_diffused_na_json,
                num_replicates=1,
                transform=inference_pipeline,
            )

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

_CKPT_PATH = os.environ.get(
    "CKPT_PATH",
    "/path/to/data",
)

# Symlink to current sota checkpoint
_CKPT_NAME = os.environ.get("CKPT_NAME", "latest.ckpt")

# AE checkpoint lives under ``$DATA_PATH/latent_checkpoints/ae_checkpoints`` (per-cluster),
# matching the ``autoencoder_ckpt_path`` convention used across the configs.
_DATA_PATH = os.environ.get("DATA_PATH", "").strip()
_AE_CKPT_PATH = os.environ.get("AE_CKPT_PATH") or (
    os.path.join(_DATA_PATH, "latent_checkpoints", "ae_checkpoints", "AE_ucond_512.ckpt") if _DATA_PATH else None
)


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_CKPT_PATH, _CKPT_NAME)),
    reason=f"Checkpoint not available: {os.path.join(_CKPT_PATH, _CKPT_NAME)} (set CKPT_PATH/CKPT_NAME to override)",
)
@pytest.mark.skipif(
    _AE_CKPT_PATH is None or not os.path.isfile(_AE_CKPT_PATH),
    reason=f"Autoencoder checkpoint not available: {_AE_CKPT_PATH} (set DATA_PATH or AE_CKPT_PATH)",
)
class TestEndToEndInference:
    """End-to-end inference tests that load the model and generate structures.

    Set environment variables before running::

        export CKPT_PATH=/path/to/checkpoint/dir
        export CKPT_NAME=model.ckpt
        export AE_CKPT_PATH=/path/to/ae_checkpoint.ckpt
    """

    def test_generation_with_replicates_and_prefix_and_refolding(self, gpu, tmp_output_dir):  # noqa: F811
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
                "batch_size=null",
                "diffusion_batch_size=1",
                f"generation.model.bb_ca.simulation_step_params.sc_scale_noise={bb_ca_noise_scale}",
                f"generation.model.local_latents.simulation_step_params.sc_scale_noise={latents_noise_scale}",
                "metrics_tags=[unconditional_refolding]",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_replicates_prefix")

        # Verify correct number of CIF files with prefix in names
        cif_files = _inference_generated_cifs(tmp_output_dir)
        assert (
            len(cif_files) == num_replicates
        ), f"Expected {num_replicates} CIF files, got {len(cif_files)}: {cif_files}"
        for cif_file in cif_files:
            assert prefix in cif_file.name, f"Expected prefix '{prefix}' in filename '{cif_file.name}'"

        # Verify output dataframe has expected fields (including input_json_path for JSON inputs)
        df = _load_output_dataframe(tmp_output_dir, global_prefix=prefix, check_input_json_path=True)
        assert len(df) == num_replicates
        assert all(df["seed"] == 42)
        assert all(df["generation.model.bb_ca.simulation_step_params.sc_scale_noise"] == bb_ca_noise_scale)
        assert all(df["generation.model.local_latents.simulation_step_params.sc_scale_noise"] == latents_noise_scale)
        assert all(df["generation.args.nsteps"] == 10)
        assert all(df["chain_of_interest"].notna())
        assert any("oracle." in col for col in df.columns), "Refolding oracle did not run."
        for eid in df["example_id"]:
            assert prefix in eid

    def test_generation_with_subsetting(self, gpu, tmp_output_dir):  # noqa: F811
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

        cif_files_a = _inference_generated_cifs(out_a)
        assert len(cif_files_a) == 1
        assert "example_a" in cif_files_a[0].name

        df_a = _load_output_dataframe(out_a, check_input_json_path=True)
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

        cif_files_b = _inference_generated_cifs(out_b)
        assert len(cif_files_b) == 1
        assert "example_b" in cif_files_b[0].name

        df_b = _load_output_dataframe(out_b, check_input_json_path=True)
        assert len(df_b) == 1
        assert "example_b" in df_b["example_id"].iloc[0]

    def test_skip_existing_prevents_regeneration(self, gpu, tmp_output_dir):  # noqa: F811
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

        cif_files_before = sorted(_inference_generated_cifs(tmp_output_dir))
        assert len(cif_files_before) >= 1
        mtimes_before = {f.name: f.stat().st_mtime for f in cif_files_before}

        # Verify first run produced a valid dataframe
        _load_output_dataframe(tmp_output_dir, check_input_json_path=True)

        # Second run: skip_existing should prevent regeneration
        cfg = _build_test_cfg(
            [
                f"inputs={input_path}",
                f"out_dir={tmp_output_dir}",
                "skip_existing=true",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_skip")

        cif_files_after = sorted(_inference_generated_cifs(tmp_output_dir))
        mtimes_after = {f.name: f.stat().st_mtime for f in cif_files_after}

        assert set(mtimes_before.keys()) == set(mtimes_after.keys())
        for name in mtimes_before:
            assert mtimes_before[name] == mtimes_after[name], f"File {name} was regenerated despite skip_existing=True"

    def test_seeded_generation_is_deterministic(self, gpu, tmp_output_dir):  # noqa: F811
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

        cifs_0 = sorted(_inference_generated_cifs(out_dirs[0]))
        cifs_1 = sorted(_inference_generated_cifs(out_dirs[1]))

        assert len(cifs_0) == len(cifs_1) >= 1

        for f0, f1 in zip(cifs_0, cifs_1, strict=True):
            _assert_coords_identical(str(f0), str(f1))

        # Verify both runs produced valid dataframes with matching seeds
        df_0 = _load_output_dataframe(out_dirs[0], check_input_json_path=True)
        df_1 = _load_output_dataframe(out_dirs[1], check_input_json_path=True)
        assert all(df_0["seed"] == 12345)
        assert all(df_1["seed"] == 12345)

    def test_output_cif_annotations(self, gpu, tmp_output_dir):  # noqa: F811
        """Verify that generated CIF files contain expected annotations."""
        input_data = {
            "example_a": {
                "contig": "5-5,A16-20",
                "dialect": 2,
                "input": str(BENCHMARKS_DIR / "motif_data_aa/1prw_aa.cif"),
                "select_fixed_atoms": True,
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

        cif_files = _inference_generated_cifs(out)
        assert len(cif_files) == 1
        assert "example_a" in cif_files[0].name

        generated_path = str(cif_files[0])
        # StandardAnnotations require load_standard_annotations=True; add_missing_atoms=False for compatibility.
        generated_aa = parse(
            generated_path,
            config=ParseConfig.from_preset(
                "rcsb",
                load_standard_annotations=True,
                add_missing_atoms=False,
                convert_mse_to_met=False,
            ),
        )["asym_unit"][0]

        residue_starts_aa = generated_aa[get_residue_starts(generated_aa)]

        # Atom-level mapped mask (stored under "mapped_to_coordinate_conditioned_region" for backward compat)
        assert SA_CRDSRC.has_mask(generated_aa), f"Expected '{SA_CRDSRC.mask_name}' in generated CIF"
        assert np.mean(residue_starts_aa.get_annotation(SA_CRDSRC.mask_name) == False) == 0.5, (  # noqa: E712
            f"Expected exactly half of residue starts to have default value False for '{SA_CRDSRC.mask_name}'"
        )

        # Non-mask string annotations (stored under their full_name)
        assert SA_CRDSRC.has_annotation(generated_aa), f"Expected '{SA_CRDSRC.full_name}' in generated CIF"
        assert SA_SRC.has_annotation(generated_aa), f"Expected '{SA_SRC.full_name}' in generated CIF"
        assert (
            np.mean(residue_starts_aa.get_annotation(SA_CRDSRC.full_name) == ".") == 0.5
        ), f"Expected exactly half of residue starts to have default value '.' for '{SA_CRDSRC.full_name}'"
        assert (
            np.mean(residue_starts_aa.get_annotation(SA_SRC.full_name) == "5P") == 0.5
        ), f"Expected exactly half of residue starts to have value '5P' for '{SA_SRC.full_name}'"

        # coords_with_insertion is now a StandardAnnotation and is loaded automatically
        assert SA_CWI.has_annotation(generated_aa), f"Expected '{SA_CWI.full_name}' in generated CIF"

        df = _load_output_dataframe(out, check_input_json_path=True)
        assert len(df) == 1
        assert "example_a" in df["example_id"].iloc[0]

    def test_partial_diffusion(self, gpu, tmp_output_dir):  # noqa: F811
        """Partial diffusion: fixed atoms are preserved, non-fixed atoms change.

        Runs partial diffusion on a stretch of 1PRW chain A, fixing a subset of
        residues with ``select_fixed_atoms``.  After aligning the generated structure
        onto the fixed region of the input, the fixed-region CA RMSD should be near
        zero while the diffused region should show meaningful coordinate change.
        """
        import biotite.structure as struc
        from atomworks.io.parser import STANDARD_PARSER_ARGS
        from atomworks.io.utils.io_utils import load_any

        input_pdb = str(BENCHMARKS_DIR / "motif_data_aa/1ycr_aa.cif")

        input_data = {
            "partial_diffusion_test_0": {
                "input": input_pdb,
                "contig": "B19-22",
                "select_fixed_atoms": "B19-22",
                "select_unfixed_sequence": "B23-26",
                "partial_t_backbone": 0.4,
                "partial_t_latent": 0.4,
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

        cif_files = _inference_generated_cifs(tmp_output_dir)
        assert len(cif_files) == 1, f"Expected 1 output CIF, got {len(cif_files)}"

        # Load input and subset to CA atoms
        input_aa = parse(input_pdb, **STANDARD_PARSER_ARGS)["asym_unit"][0]
        input_ca = input_aa[(input_aa.atom_name == "CA")]

        # Load generated CA atoms (output contains the same 10 residues in order)
        output_aa = load_any(str(cif_files[0]))[0]
        output_ca = output_aa[output_aa.atom_name == "CA"]

        # Superimpose output onto input using only the fixed region (first 5 CA atoms)
        fixed_mask = (input_ca.res_id >= 19) & (input_ca.res_id <= 22)
        _, transform = struc.superimpose(input_ca[fixed_mask], output_ca[fixed_mask])

        # Apply the same rigid-body transform to the full output
        output_ca_aligned = transform.apply(output_ca)

        # Fixed region (B19-22): CA RMSD should be near zero, sequence should be identical
        fixed_rmsd = struc.rmsd(input_ca[fixed_mask], output_ca_aligned[fixed_mask])
        assert (
            fixed_rmsd < 0.5
        ), f"Fixed-region CA RMSD {fixed_rmsd:.3f} Å should be < 0.5 Å — fixed atoms should retain input coordinates"
        assert np.array_equal(
            input_ca.res_name[fixed_mask], output_ca_aligned.res_name[fixed_mask]
        ), "Fixed-region residue types should be unchanged in output"

        # Non-fixed region: CA RMSD should be meaningfully > 0, sequences should be non-identical
        nonfixed_rmsd = struc.rmsd(input_ca[~fixed_mask], output_ca_aligned[~fixed_mask])
        assert nonfixed_rmsd > 1, (
            f"Non-fixed-region CA RMSD {nonfixed_rmsd:.3f} Å should be > 1 Å"
            " — diffused atoms should differ from input coordinates"
        )
        assert not np.array_equal(
            input_ca.res_name[~fixed_mask], output_ca_aligned.res_name[~fixed_mask]
        ), "Non-fixed-region residue types should be different in output"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
