"""Integration tests for rfproteina inference pipeline.

These tests exercise the full inference path end-to-end on a GPU,
from annotated-CIF input parsing through dataset construction, collation,
model forward pass, and output saving.

Inputs are Condition-annotated CIFs (the legacy contig-JSON dialect has been
removed). CIF file / list / directory inputs route to a
:class:`~rfproteina.inference_utils.datasets.DesignProblemDataset`.

Run with:
    pytest projects/rfproteina/tests/test_inference_integration.py -v
"""

import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import biotite.structure as struc
import numpy as np
import pandas as pd
import pytest
import zstandard
from atomworks.io.parser import parse
from atomworks.io.utils.io_utils import load_any
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from biotite.structure import get_residue_starts

# Load .env BEFORE any atomworks imports (they read env vars at import time)
from dotenv import load_dotenv

load_dotenv(override=True)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from rfproteina.conditions import C_CRDGRP
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.generate import (
    _get_example_ids_from_dataset,
    get_contig_or_design_problem_dataset,
    run_generation,
)
from rfproteina.inference_utils.datasets import ConditionSpecDataset, DesignProblemDataset
from rfproteina.proteina import Proteina
from rfproteina.utils.standard_annotations import SA_CRDSRC, SA_CWI, SA_SRC, SA_UII
from rfproteina.utils.structure_files import iter_structure_files
from tests.conftest import TEST_DATA_DIR
from tests.e2e.helpers.regression_baselines import SOTA_AE_CKPT_PATH, SOTA_CKPT_PATH, load_design_cif
from tests.helpers import BENCHMARKS_DIR, PROJECT_ROOT, gpu  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_DIR = str((PROJECT_ROOT / "configs").resolve())

# Committed, regression-validated de-novo monomer CIFs (length-only, no input structure).
_MONOMER_CIF_DIR = BENCHMARKS_DIR / "monomer"
_MONOMER_CIF = _MONOMER_CIF_DIR / "monomer-short.cif"
_CATH_CIFS = [
    BENCHMARKS_DIR / "adherence" / "monomer_helical" / "monomer-short.cif",
    BENCHMARKS_DIR / "adherence" / "monomer_helical" / "monomer-medium.cif",
    BENCHMARKS_DIR / "adherence" / "monomer_helical" / "monomer-long.cif",
    BENCHMARKS_DIR / "adherence" / "monomer_ig" / "monomer-short.cif",
]
_PSEUDOKNOT_DIR = BENCHMARKS_DIR / "rna" / "pseudoknot"
_MIXED_CONDITION_CIF = BENCHMARKS_DIR / "mcsa" / "mcsa_37" / "M0664_2dhn.cif"

# Committed all-atom motif input structures. Structure-conditioned tests build their conditioning
# on the fly with a condition_spec (see :func:`_motif_condition_spec`) rather than committing a
# pre-built motif CIF fixture.
_1PRW_INPUT = BENCHMARKS_DIR / "motif_scaffolding" / "motif_data_aa" / "1prw_aa.cif"
_1YCR_INPUT = BENCHMARKS_DIR / "motif_scaffolding" / "motif_data_aa" / "1ycr_aa.cif"


def _inference_generated_cifs(out_dir: str | Path) -> list[Path]:
    """Return generated structure CIFs (one per-design directory under ``inference_samples/``)."""
    return iter_structure_files(Path(out_dir) / "inference_samples", stem_suffix="_generated")


_MOTIF_RMSD_MAX = 1.0

# Checkpoint-path -> loaded model, shared across TestEndToEndInference so the SOTA checkpoint
# is loaded once instead of per run_generation call (see the _cache_checkpoint_load fixture).
_SHARED_CHECKPOINT_CACHE: dict = {}


def _conditioned_res_names(aa) -> list[str]:
    """Sorted residue names at the sequence-conditioned residues."""
    starts = get_residue_starts(aa)
    return sorted(aa.res_name[starts][C_SEQ.mask(aa, default="generate")[starts]].tolist())


# Columns that should always be present in the per-example output dataframe, regardless of input
# kind. The source column differs by input kind (CIF inputs record ``input_cif_path``;
# condition_spec json/yaml inputs record ``input_json_path``) and is checked separately via the
# ``source_column`` argument to :func:`_load_output_dataframe`.
_EXPECTED_DF_COLUMNS = {
    "example_id",
    "designed_structure_path",
    "task",
    "dataset_name",
    "seed",
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


@pytest.fixture
def tmp_output_dir():
    """Provide a fresh temporary directory for test outputs, cleaned up after."""
    d = tempfile.mkdtemp(prefix="inference_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_test_cfg(overrides: list[str] | None = None) -> DictConfig:
    """Load the inference_on_user_inputs config with test-specific overrides.

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
        "inference_experiment=inference_on_user_inputs",
        "metrics/refolding_oracles=af3",
        f"autoencoder_ckpt_path={_AE_CKPT_PATH or 'null'}",
        "seed=42",
        "num_workers=0",
        "metrics_tags=[]",
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
    source_column: str | None = None,
) -> pd.DataFrame:
    """Load the per-example metrics CSV and validate expected columns.

    Args:
        out_dir: Root output directory for the inference run.
        global_prefix: Prefix used during generation (affects CSV filename).
        source_column: If given, the input-source column to verify (``"input_cif_path"``
            for CIF inputs, ``"input_json_path"`` for condition_spec inputs); the column
            must be present, fully populated, and point at existing files.

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

    if source_column is not None:
        assert source_column in present, f"Expected source column {source_column!r} in {csv_path}"
        assert df[source_column].notna().all(), f"Expected {source_column} for every row"
        for src_path in df[source_column].dropna().unique():
            assert Path(src_path).is_file(), f"{source_column} does not exist: {src_path}"

    return df


def _motif_condition_spec(input_cif: Path, contig: str, fixed_select: str) -> dict:
    """Build a condition_spec example that fully fixes a motif (coordinates + sequence).

    Coordinate-conditions (``C_CRD``) and sequence-conditions (``C_SEQ``) every atom the
    ``fixed_select`` query matches, resolved against the input's *original* residue numbering.
    Any contig residue outside the selection (a diffused expandable segment, or a preserved
    block left unselected) carries no conditioning, so it is generated / partially diffused.

    Args:
        input_cif: All-atom CIF providing the motif coordinates.
        contig: Contig string assembling the design (e.g. ``"5-5, A16-20"``).
        fixed_select: Selection query for the coordinate+sequence-fixed motif residues
            (e.g. ``"res_id >= 16 and res_id <= 20"``).

    Returns:
        A single condition_spec example dict (``{input, contig, conditions}``).
    """
    return {
        "input": str(input_cif),
        "contig": contig,
        "conditions": {
            "C_CRD": {True: [{"select": fixed_select}]},
            "C_SEQ": {True: [{"select": fixed_select}]},
        },
    }


# ---------------------------------------------------------------------------
# Dataset construction tests (CPU-only, no model needed)
# ---------------------------------------------------------------------------


class TestDatasetConstruction:
    """Tests for dataset creation, replication, subsetting, and prefixing.

    These verify the input-parsing side of inference (annotated-CIF ->
    :class:`DesignProblemDataset`) without requiring a GPU.
    """

    def test_design_problem_dataset_basic(self, inference_pipeline):
        """A single CIF loads as a DesignProblemDataset and applies transforms."""
        dataset = get_contig_or_design_problem_dataset(
            str(_MONOMER_CIF),
            num_replicates=1,
            transform=inference_pipeline,
        )
        assert isinstance(dataset, DesignProblemDataset)
        assert len(dataset) == 1
        assert _get_example_ids_from_dataset(dataset) == ["monomer-short"]

        example = dataset[0]
        assert "atom_array" in example
        assert "example_id" in example
        assert "feats" in example
        assert Path(example["specification"]["input_cif"]) == _MONOMER_CIF

    def test_duplicate_cif_stems_are_qualified_by_parent_directory(self, inference_pipeline):
        """Explicit CIF lists use parent names when filename stems collide."""
        dataset = DesignProblemDataset(_CATH_CIFS, transform=inference_pipeline)

        assert _get_example_ids_from_dataset(dataset) == [
            "monomer_helical-monomer-long",
            "monomer_helical-monomer-medium",
            "monomer_helical-monomer-short",
            "monomer_ig-monomer-short",
        ]

    def test_compressed_cif_ids_drop_all_extensions(self, inference_pipeline):
        """Compressed CIFs keep extension-free IDs for subsetting and outputs."""
        dataset = DesignProblemDataset(
            _PSEUDOKNOT_DIR,
            transform=inference_pipeline,
            subset_to_keys=["W05", "Q20"],
        )

        assert _get_example_ids_from_dataset(dataset) == ["Q20", "W05"]

    @pytest.mark.parametrize(
        "input_cif",
        [_MIXED_CONDITION_CIF, BENCHMARKS_DIR / "ppi" / "ppi_all_atom" / "12_Claudin1.cif.zst"],
        ids=["enzyme", "ppi"],
    )
    def test_molecule_entities_from_annotated_cif(self, inference_pipeline, input_cif):
        """Designed chains remain single entities, distinct from their fixed partners."""
        dataset = get_contig_or_design_problem_dataset(str(input_cif), num_replicates=1, transform=inference_pipeline)
        example = dataset[0]
        atom_array = example["atom_array"]
        _, token_starts = np.unique(atom_array.token_id, return_index=True)
        generated = ~C_SEQ.mask(atom_array, default="raise")[token_starts]
        chain_ids = atom_array.chain_iid[token_starts]
        entity_ids = example["feats"]["molecule_entity_id"].numpy()

        assert generated.any() and (~generated).any()
        for chain in np.unique(chain_ids[generated]):
            in_chain = chain_ids == chain
            entities = np.unique(entity_ids[in_chain])
            assert len(entities) == 1
            assert not np.isin(entity_ids[~in_chain], entities).any()

    @pytest.mark.parametrize(
        "num_replicates,global_prefix,expected_ids",
        [
            (3, None, ["monomer-short_rep0", "monomer-short_rep1", "monomer-short_rep2"]),
            (1, "myrun", ["myrun_monomer-short"]),
            (2, "run1", ["run1_monomer-short_rep0", "run1_monomer-short_rep1"]),
        ],
    )
    def test_example_id_scheme(self, inference_pipeline, num_replicates, global_prefix, expected_ids):
        """num_replicates (``_rep{i}`` suffix) and global_prefix compose into example ids."""
        dataset = get_contig_or_design_problem_dataset(
            str(_MONOMER_CIF),
            num_replicates=num_replicates,
            transform=inference_pipeline,
            global_prefix=global_prefix,
        )
        assert len(dataset) == num_replicates
        assert _get_example_ids_from_dataset(dataset) == expected_ids

    def test_subset_by_condition_spec_keys(self, inference_pipeline, tmp_path):
        """``subset_to_examples`` keeps only the requested keys of a multi-example condition_spec.

        This exercises the real ``subset_to_keys`` path of
        :class:`~rfproteina.inference_utils.datasets.ConditionSpecDataset` (each top-level
        key is one design), not the CIF file-path notion of subsetting.
        """
        spec = {
            "1ycr_motif": _motif_condition_spec(_1YCR_INPUT, "B19-27", "res_id >= 19 and res_id <= 22"),
            "1prw_motif": _motif_condition_spec(_1PRW_INPUT, "5-5, A16-20", "res_id >= 16 and res_id <= 20"),
        }
        spec_path = tmp_path / "two_examples.json"
        spec_path.write_text(json.dumps(spec))

        # No subset: both examples load (``num_replicates=1`` appends a ``_0`` replicate suffix).
        dataset = get_contig_or_design_problem_dataset(
            str(spec_path),
            num_replicates=1,
            transform=inference_pipeline,
        )
        assert isinstance(dataset, ConditionSpecDataset)
        assert set(_get_example_ids_from_dataset(dataset)) == {"1ycr_motif_0", "1prw_motif_0"}

        # Subsetting to one key loads only that example.
        subset = get_contig_or_design_problem_dataset(
            str(spec_path),
            num_replicates=1,
            transform=inference_pipeline,
            subset_to_examples="1ycr_motif",
        )
        assert _get_example_ids_from_dataset(subset) == ["1ycr_motif_0"]


# ---------------------------------------------------------------------------
# GPU integration tests (require checkpoint + GPU)
# ---------------------------------------------------------------------------

# Allow an explicit test override; otherwise use the AE from the composed inference config.
_AE_CKPT_PATH = os.environ.get("AE_CKPT_PATH") or SOTA_AE_CKPT_PATH


@pytest.mark.skipif(
    SOTA_CKPT_PATH is None or not SOTA_CKPT_PATH.is_file(),
    reason=f"SOTA checkpoint not available: {SOTA_CKPT_PATH} (set DATA_PATH)",
)
@pytest.mark.skipif(
    _AE_CKPT_PATH is None or not os.path.isfile(_AE_CKPT_PATH),
    reason=f"Autoencoder checkpoint not available: {_AE_CKPT_PATH} (set DATA_PATH or AE_CKPT_PATH)",
)
class TestEndToEndInference:
    """End-to-end inference tests that load the model and generate structures."""

    @pytest.fixture(autouse=True)
    def _cache_checkpoint_load(self, request):
        """Reuse one loaded model across the class so tests don't reload the checkpoint per run.

        The load dominates per-test cost; ``configure_inference`` re-runs cheaply each call. The
        seed-determinism test opts out via the ``no_checkpoint_cache`` marker and loads fresh.
        """
        if request.node.get_closest_marker("no_checkpoint_cache"):
            yield
            return
        original = Proteina.load_from_checkpoint.__func__

        def cached(cls, checkpoint_path, *args, **kwargs):
            if checkpoint_path not in _SHARED_CHECKPOINT_CACHE:
                _SHARED_CHECKPOINT_CACHE[checkpoint_path] = original(cls, checkpoint_path, *args, **kwargs)
            return _SHARED_CHECKPOINT_CACHE[checkpoint_path]

        Proteina.load_from_checkpoint = classmethod(cached)
        try:
            yield
        finally:
            Proteina.load_from_checkpoint = classmethod(original)
            # A failed setup/call may leave the shared module dirty; evict it so the next test reloads fresh.
            reports = (getattr(request.node, "rep_setup", None), getattr(request.node, "rep_call", None))
            if any(r is not None and r.failed for r in reports):
                _SHARED_CHECKPOINT_CACHE.clear()

    def test_generation_with_replicates_and_prefix_and_refolding(self, gpu, tmp_output_dir):  # noqa: F811
        """Verify num_replicates, global_prefix, and diffusion_batch_size alias."""
        prefix = "testprefix"
        num_replicates = 2
        bb_ca_noise_scale = 0.2
        latents_noise_scale = 0.3

        cfg = _build_test_cfg(
            [
                f"inputs={_MONOMER_CIF}",
                f"out_dir={tmp_output_dir}",
                f"num_replicates={num_replicates}",
                f"global_prefix={prefix}",
                "batch_size=null",
                "diffusion_batch_size=1",
                f"generation.model.bb_ca.simulation_step_params.sc_scale_noise={bb_ca_noise_scale}",
                f"generation.model.local_latents.simulation_step_params.sc_scale_noise={latents_noise_scale}",
                "generation.args.nsteps=20",
                "generation.n_recycle_schedule=null",
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

        # Verify output dataframe has expected fields (including input_cif_path for CIF inputs)
        df = _load_output_dataframe(tmp_output_dir, global_prefix=prefix, source_column="input_cif_path")
        assert len(df) == num_replicates
        assert all(df["seed"] == 42)
        assert all(df["generation.model.bb_ca.simulation_step_params.sc_scale_noise"] == bb_ca_noise_scale)
        assert all(df["generation.model.local_latents.simulation_step_params.sc_scale_noise"] == latents_noise_scale)
        assert all(df["generation.args.nsteps"] == 20)
        assert all(df["chain_of_interest"].notna())
        assert any("oracle." in col for col in df.columns), "Refolding oracle did not run."
        # Replicates share a base id and differ only by the ``_rep{i}`` suffix.
        assert set(df["example_id"]) == {f"{prefix}_monomer-short_rep0", f"{prefix}_monomer-short_rep1"}

    def test_generation_with_subsetting(self, gpu, tmp_output_dir):  # noqa: F811
        """``subset_to_examples`` generates only the requested key of a multi-example condition_spec.

        This exercises the real ``subset_to_keys`` path (each top-level key is one design), not
        the CIF file-path notion of subsetting.
        """
        out = os.path.join(tmp_output_dir, "subset")
        os.makedirs(out)

        spec = {
            "1ycr_motif": _motif_condition_spec(_1YCR_INPUT, "B19-27", "res_id >= 19 and res_id <= 22"),
            "1prw_motif": _motif_condition_spec(_1PRW_INPUT, "5-5, A16-20", "res_id >= 16 and res_id <= 20"),
        }
        spec_path = Path(out) / "spec.json"
        spec_path.write_text(json.dumps(spec))

        cfg = _build_test_cfg(
            [
                f"inputs={spec_path}",
                f"out_dir={out}",
                "subset_to_examples=1ycr_motif",
                "generation.args.nsteps=20",
                "generation.n_recycle_schedule=null",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_subset")

        cif_files = _inference_generated_cifs(out)
        assert len(cif_files) == 1
        assert "1ycr_motif" in cif_files[0].name
        assert "1prw_motif" not in cif_files[0].name

        df = _load_output_dataframe(out, source_column="input_json_path")
        assert len(df) == 1
        assert "1ycr_motif" in df["example_id"].iloc[0]

    def test_skip_existing_prevents_regeneration(self, gpu, tmp_output_dir):  # noqa: F811
        """Verify skip_existing=True skips examples whose outputs already exist."""
        # First run: generate
        cfg = _build_test_cfg(
            [
                f"inputs={_MONOMER_CIF}",
                f"out_dir={tmp_output_dir}",
                "generation.args.nsteps=20",
                "generation.n_recycle_schedule=null",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_skip")

        cif_files_before = sorted(_inference_generated_cifs(tmp_output_dir))
        assert len(cif_files_before) >= 1
        mtimes_before = {f.name: f.stat().st_mtime for f in cif_files_before}

        # Verify first run produced a valid dataframe
        _load_output_dataframe(tmp_output_dir, source_column="input_cif_path")

        # Second run: skip_existing should prevent regeneration
        cfg = _build_test_cfg(
            [
                f"inputs={_MONOMER_CIF}",
                f"out_dir={tmp_output_dir}",
                "skip_existing=true",
                "generation.args.nsteps=20",
                "generation.n_recycle_schedule=null",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_skip")

        cif_files_after = sorted(_inference_generated_cifs(tmp_output_dir))
        mtimes_after = {f.name: f.stat().st_mtime for f in cif_files_after}

        assert set(mtimes_before.keys()) == set(mtimes_after.keys())
        for name in mtimes_before:
            assert mtimes_before[name] == mtimes_after[name], f"File {name} was regenerated despite skip_existing=True"

    @pytest.mark.no_checkpoint_cache
    def test_seeded_generation_is_deterministic(self, gpu, tmp_output_dir):  # noqa: F811
        """Verify that two runs with the same seed produce identical outputs."""
        out_dirs = []
        for i in range(2):
            out_dir = os.path.join(tmp_output_dir, f"run_{i}")
            os.makedirs(out_dir)
            out_dirs.append(out_dir)

            cfg = _build_test_cfg(
                [
                    f"inputs={_MONOMER_CIF}",
                    f"out_dir={out_dir}",
                    "seed=12345",
                    "generation.args.nsteps=20",
                    "generation.n_recycle_schedule=null",
                ]
            )
            run_generation(cfg, job_id=0, config_name="test_seed")

        cifs_0 = sorted(_inference_generated_cifs(out_dirs[0]))
        cifs_1 = sorted(_inference_generated_cifs(out_dirs[1]))

        assert len(cifs_0) == len(cifs_1) >= 1

        for f0, f1 in zip(cifs_0, cifs_1, strict=True):
            _assert_coords_identical(str(f0), str(f1))

        # Verify both runs produced valid dataframes with matching seeds
        df_0 = _load_output_dataframe(out_dirs[0], source_column="input_cif_path")
        df_1 = _load_output_dataframe(out_dirs[1], source_column="input_cif_path")
        assert all(df_0["seed"] == 12345)
        assert all(df_1["seed"] == 12345)

    def test_output_cif_annotations(self, gpu, tmp_output_dir):  # noqa: F811
        """Verify that a motif-scaffold generated CIF carries the expected annotations.

        The condition_spec (contig ``"5-5, A16-20"`` on 1PRW) is a deterministic 10-residue
        design: a 5-residue diffused segment plus a 5-residue coordinate+sequence-conditioned
        motif (chain-A residues 16-20 of 1PRW), so exactly half the residues are coordinate-fixed.
        """
        out = os.path.join(tmp_output_dir, "motif_scaffold")
        os.makedirs(out)

        spec = {
            "1prw_motif_scaffold": _motif_condition_spec(_1PRW_INPUT, "5-5, A16-20", "res_id >= 16 and res_id <= 20")
        }
        spec_path = Path(out) / "spec.json"
        spec_path.write_text(json.dumps(spec))

        cfg = _build_test_cfg(
            [
                f"inputs={spec_path}",
                f"out_dir={out}",
                "generation.args.nsteps=20",
                "generation.n_recycle_schedule=null",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_motif_scaffold")

        cif_files = _inference_generated_cifs(out)
        assert len(cif_files) == 1
        assert "1prw_motif_scaffold" in cif_files[0].name

        generated_path = str(cif_files[0])
        generated_aa = load_design_cif(generated_path)

        residue_starts_aa = generated_aa[get_residue_starts(generated_aa)]

        # Atom-level coordinate-source annotation; its non-default values imply the mapped mask.
        assert SA_CRDSRC.has_annotation(generated_aa), f"Expected '{SA_CRDSRC.full_name}' in generated CIF"
        assert np.mean(SA_CRDSRC.mask(residue_starts_aa) == False) == 0.5, (  # noqa: E712
            f"Expected exactly half of residue starts to be outside '{SA_CRDSRC.full_name}'"
        )

        # Non-mask string annotations (stored under their full_name).
        assert SA_CRDSRC.has_annotation(generated_aa), f"Expected '{SA_CRDSRC.full_name}' in generated CIF"
        assert SA_SRC.has_annotation(generated_aa), f"Expected '{SA_SRC.full_name}' in generated CIF"
        assert (
            np.mean(residue_starts_aa.get_annotation(SA_CRDSRC.full_name) == ".") == 0.5
        ), f"Expected exactly half of residue starts to have default value '.' for '{SA_CRDSRC.full_name}'"

        # coords_with_insertion is a StandardAnnotation and is loaded automatically.
        assert SA_CWI.has_annotation(generated_aa), f"Expected '{SA_CWI.full_name}' in generated CIF"

        df = _load_output_dataframe(out, source_column="input_json_path")
        assert len(df) == 1
        assert "1prw_motif_scaffold" in df["example_id"].iloc[0]

    def test_partial_diffusion(self, gpu, tmp_output_dir):  # noqa: F811
        """Partial diffusion: fixed atoms are preserved, non-fixed atoms change.

        Runs partial diffusion on chain B (residues 19-27) of 1YCR via a condition_spec
        (contig ``"B19-27"``), with residues 19-22 coordinate+sequence-conditioned and
        residues 23-27 left unconditioned (so they diffuse and redesign). After aligning
        the generated structure onto the fixed region, the fixed-region CA RMSD should be
        near zero while the diffused region should show meaningful coordinate change.
        """
        from atomworks.io.parser import STANDARD_PARSER_ARGS

        # The condition_spec preserves this structure's chain B (19-27), so the original file
        # provides the reference CA coordinates in the same residue order.
        input_pdb = str(_1YCR_INPUT)

        spec = {"1ycr_partial": _motif_condition_spec(_1YCR_INPUT, "B19-27", "res_id >= 19 and res_id <= 22")}
        spec_path = Path(tmp_output_dir) / "spec.json"
        spec_path.write_text(json.dumps(spec))

        cfg = _build_test_cfg(
            [
                f"inputs={spec_path}",
                f"out_dir={tmp_output_dir}",
                "generation.args.partial_trajectory_fraction_steps=0.7",
            ]
        )
        run_generation(cfg, job_id=0, config_name="test_partial_diffusion")

        cif_files = _inference_generated_cifs(tmp_output_dir)
        assert len(cif_files) == 1, f"Expected 1 output CIF, got {len(cif_files)}"

        # Load input and subset to CA atoms
        input_aa = parse(input_pdb, **STANDARD_PARSER_ARGS)["asym_unit"][0]
        input_ca = input_aa[(input_aa.atom_name == "CA")]

        # Load generated CA atoms (output contains the same 9 residues in order)
        output_aa = load_any(str(cif_files[0]))[0]
        output_ca = output_aa[output_aa.atom_name == "CA"]

        # Superimpose output onto input using only the fixed region (residues 19-22)
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

        # Non-fixed region (B23-27): CA RMSD should be meaningfully > 0, sequences should be non-identical
        nonfixed_rmsd = struc.rmsd(input_ca[~fixed_mask], output_ca_aligned[~fixed_mask])
        assert nonfixed_rmsd > 1, (
            f"Non-fixed-region CA RMSD {nonfixed_rmsd:.3f} Å should be > 1 Å"
            " — diffused atoms should differ from input coordinates"
        )
        assert not np.array_equal(
            input_ca.res_name[~fixed_mask], output_ca_aligned.res_name[~fixed_mask]
        ), "Non-fixed-region residue types should be different in output"

    def test_motif_conditioning_persists_through_partial_diffusion(self, gpu, tmp_output_dir):  # noqa: F811
        """Motif conditioning survives a generation + a partial-diffusion re-feed.

        Requires that:
            - Sequence stays consistent across the input, initial generation (gen-1), and partial diffusion (gen-2)
            - Coordinate and internal-coordinate conditioning match the input again once the re-feed re-locks the motif
        """
        # Fixture: indexed absolute + indexed internal + unindexed absolute coordinate conditioning, plus
        # sequence conditioning. Decompress the zstd fixture to a temp CIF.
        source = Path(tmp_output_dir) / "mixed_unindexed_motif.cif"
        source.write_bytes(
            zstandard.ZstdDecompressor().decompress((TEST_DATA_DIR / "mixed_unindexed_motif.cif.zst").read_bytes())
        )

        # TODO: run this in-memory instead of through the CLI -- ideally pass inference an AtomArray
        # directly rather than round-tripping through a CIF on disk.
        def _generate(inputs: Path, sub: str, extra: list[str]) -> Path:
            out = Path(tmp_output_dir) / sub
            cfg = _build_test_cfg(
                [
                    f"inputs={inputs}",
                    f"out_dir={out}",
                    "~metrics/refolding_oracles",
                    *extra,
                ]
            )
            run_generation(cfg, job_id=0, config_name=sub)
            return _inference_generated_cifs(out)[0]

        # The tags activate the coordinate-conditioning inline metrics (design_motif_*) read below.
        motif_metric_tags = "metrics_tags=[3d_coordinate_conditioning,internal_coordinate_conditioning,unindexed]"
        # Step 1: generate from the source (unindexed motif)
        gen1 = _generate(source, "gen1", [motif_metric_tags])
        # Step 2: re-feed gen1 with partial diffusion to produce gen2
        gen2 = _generate(gen1, "gen2", ["generation.args.partial_trajectory_fraction_steps=0.8", motif_metric_tags])
        inp, g1, g2 = (load_design_cif(c) for c in (source, gen1, gen2))

        # Sequence: the conditioned residues keep their identity across input, gen-1, and gen-2.
        want = _conditioned_res_names(inp)
        for label, aa in (("gen-1", g1), ("gen-2", g2)):
            got = _conditioned_res_names(aa)
            assert got == want, (
                f"sequence conditioning drifted on {label}: extra={sorted((Counter(got) - Counter(want)).elements())} "
                f"missing={sorted((Counter(want) - Counter(got)).elements())} (got {got} vs input {want})"
            )

        # Motif RMSD via the design_motif metrics (per-coord-group aligned): the whole motif on gen-1 and
        # (re-locked) gen-2, plus the freshly inserted unindexed motif specifically on gen-1.
        def _motif_rmsd(out_sub: str, col: str) -> pd.Series:
            df = pd.read_csv(Path(tmp_output_dir) / out_sub / "inference_metrics" / "per_example_metrics.csv")
            assert col in df.columns, f"expected {col} in {out_sub} metrics; got {sorted(df.columns)}"
            return df[col]

        for sub, col in (
            ("gen1", "design_motif_allatom_rmsd.rmsd"),
            ("gen1", "design_unindexed_motif_allatom_rmsd.rmsd"),
            ("gen2", "design_motif_allatom_rmsd.rmsd"),
        ):
            r = _motif_rmsd(sub, col)
            assert (
                r.notna().all() and (r < _MOTIF_RMSD_MAX).all()
            ), f"{sub} {col} exceeds {_MOTIF_RMSD_MAX} A: {r.tolist()}"

        # The unindexed motif should become an indexed fixed motif (not re-unindexed) after the first generation
        assert SA_UII.mask(g1).any(), "unindexed motif was not flagged as an insertion on gen-1"
        motif = C_CRD.mask(g2, default="generate") | C_CRDGRP.is_internal(g2)
        assert C_IDX.mask(g2, default="generate")[motif].all(), "motif was re-unindexed on the re-feed"

    def test_inverse_folding(self, gpu, tmp_output_dir):  # noqa: F811
        """Resampled-forward-marginal inverse folding preserves the backbone and redesigns sequence.

        Feeding an existing structure via ``inputs=<cif>`` with
        ``generation=inverse_folding`` runs partial diffusion where every token's
        backbone is independently resampled from its forward marginal at each step
        (clean at t=1) while the sequence channel (``local_latents``) is regenerated. The
        generated backbone should match the input (CA RMSD near zero) and the sequence should
        differ from the input.
        """
        from atomworks.io.parser import STANDARD_PARSER_ARGS

        from rfproteina.flow_matching import product_space_flow_matcher

        input_pdb = str(BENCHMARKS_DIR / "motif_scaffolding/motif_data_aa/1ycr_aa.cif")

        cfg = _build_test_cfg(
            [
                "generation=inverse_folding",
                f"inputs={input_pdb}",
                f"out_dir={tmp_output_dir}",
                # Deliberately not the config default, so the assertion below would catch
                # run_generation dropping the override.
                "generation.args.nsteps=20",
                "generation.n_recycle_schedule=null",
            ]
        )
        args_before = OmegaConf.to_container(cfg.generation.args, resolve=True)
        with patch.object(
            product_space_flow_matcher,
            "sample_noised_target_backbone",
            wraps=product_space_flow_matcher.sample_noised_target_backbone,
        ) as sample_backbone:
            run_generation(cfg, job_id=0, config_name="test_inverse_folding")

        assert sample_backbone.call_count == cfg.generation.args.nsteps
        assert OmegaConf.to_container(cfg.generation.args, resolve=True) == args_before

        cif_files = _inference_generated_cifs(tmp_output_dir)
        assert len(cif_files) == 1, f"Expected 1 output CIF, got {len(cif_files)}"

        input_aa = parse(input_pdb, **STANDARD_PARSER_ARGS)["asym_unit"][0]
        input_ca = input_aa[input_aa.atom_name == "CA"]
        output_aa = load_any(str(cif_files[0]))[0]
        output_ca = output_aa[output_aa.atom_name == "CA"]
        assert len(output_ca) == len(input_ca), f"Output has {len(output_ca)} CA atoms, expected {len(input_ca)}"

        # Backbone is pinned to the input coordinates (same frame): CA RMSD should be near zero.
        bb_rmsd = float(np.sqrt(((output_ca.coord - input_ca.coord) ** 2).sum(axis=-1).mean()))
        assert bb_rmsd < 0.01, f"Inverse-folding CA RMSD {bb_rmsd:.4f} Å should be < 0.01 Å"

        # Sequence is redesigned: residue types should differ from the input.
        assert not np.array_equal(
            input_ca.res_name, output_ca.res_name
        ), "Inverse folding should redesign the sequence (output residues should differ from input)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
