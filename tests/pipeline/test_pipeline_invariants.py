"""Pipeline invariant tests.

These invariants are properties that must hold for ALL pipeline outputs across:
- All input structures (proteins, nucleic acids, ligands, complexes)
- All design tasks (unconditional, inverse_folding, motif_scaffold, etc.)
- All pipeline configurations (with/without cropping, different seeds, etc.)
"""

import rootutils

from rfproteina.utils.env import load_dotenv

# Setup the project root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Setup environment variables
load_dotenv(override=True)

import atexit
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from atomworks.io import parse
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.standard_annotations import S_ATM
from atomworks.ml.conditions import C_IDX
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.token import get_token_starts
from biotite.structure import AtomArray, BondList, BondType, get_residue_starts

from rfproteina.datapipes.pipeline import build_design_train_pipeline
from rfproteina.transforms.design_task import (
    BinderHotspotWith3DTarget,
    InverseFoldingTask,
    MotifScaffoldTask,
    StructurePredictionTask,
    TipAtomActiveSiteTask,
    TipAtomPartialLigandTask,
    UnconditionalTask,
)
from rfproteina.transforms.invariants import check_bond_length_violations, find_index_group_res_id_gaps
from rfproteina.utils.logging import print_df_as_table
from rfproteina.utils.parse_presets import RCSB_WITH_METALS

_TEST_DATA_DIR = Path(__file__).parent.parent / "data" / "pdb"


def _parse_test_structure(pdb_id: str) -> dict:
    """Parse a test structure from the bundled test CIF files."""
    path = _TEST_DATA_DIR / f"{pdb_id}.cif"
    data = parse(path, config=RCSB_WITH_METALS)
    data["atom_array"] = data["assemblies"]["1"][0]
    data["pdb_id"] = pdb_id
    return data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global timing statistics: {(pdb_id, design_task): {transform_name: time}}
_timing_stats = defaultdict(dict)

# =============================================================================
# Test Configuration
# =============================================================================

TEST_STRUCTURES = [
    "5ocm",  # Protein dimer + small molecule
    "1ivo",  # Protein complex + oligosaccharides
    "6wtf",  # Protein + ligand
    "3wbm",  # protein-RNA complex
    "4kpy",  # DNA binding protein
    "6mtg",  # Protein with non-canonical amino acid (NCAA)
]


# Design tasks to test
GLOBAL_DESIGN_TASK_KWARGS = {
    "rasa_kwargs": {"p_add_rasa_condition": 0.0},
    "hbond_kwargs": {"p_add_hbond_condition": 0.0},
    "secondary_structure_kwargs": {"p_add_secondary_structure_condition": 0.0},
    "cath_kwargs": {
        "p_add_cath_c_condition": 0.0,
        "p_add_cath_ca_condition": 0.0,
        "p_add_cath_cat_condition": 0.0,
    },
}


DESIGN_TASKS = {
    "unconditional": {
        "transform": UnconditionalTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS),
        "frequency": 1.0,
    },
    "inverse_folding": {
        "transform": InverseFoldingTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS),
        "frequency": 1.0,
    },
    "motif_scaffold": {
        "transform": MotifScaffoldTask(
            min_tokens_per_motif=5,
            max_tokens_per_motif=20,
            global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
        ),
        "frequency": 1.0,
    },
    "structure_prediction": {
        "transform": StructurePredictionTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS),
        "frequency": 1.0,
    },
    "tip_atom_partial_ligand": {
        "transform": TipAtomPartialLigandTask(
            n_max_residues=8,
            global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
        ),
        "frequency": 1.0,
    },
    "tip_atom_pocket": {
        "transform": TipAtomActiveSiteTask(global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS),
        "frequency": 1.0,
    },
    "ppi_hotspot": {
        "transform": BinderHotspotWith3DTarget(
            hotspot_count_distribution="geometric",
            hotspot_geometric_p=0.2,
            max_hotspots_frac_to_provide=0.1,
            p_no_hotspots=0.05,
            hotspot_max_distance=4.5,
            ppi_crop_binder_fraction=0.667,
            p_sample_ppi_crop=0.75,
            global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
        ),
        "frequency": 1.0,
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def parse_and_run_pipeline(
    pdb_id: str,
    design_tasks: dict,
    seed: int = 42,
    crop_size: int | None = None,
    loss_weight_cfg: dict[str, float] | None = None,
) -> dict:
    """Parse structure and run through design train pipeline."""

    input_data = _parse_test_structure(pdb_id)
    input_data["example_id"] = pdb_id

    # Inject query_pn_unit_iids so ppi_hotspot tasks can run
    ensure_annotations(input_data["atom_array"], "is_protein")
    protein_iids = list(np.unique(input_data["atom_array"].pn_unit_iid[input_data["atom_array"].is_protein]))[:2]
    input_data["query_pn_unit_iids"] = protein_iids

    # Build pipeline with specified design tasks
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        pipeline = build_design_train_pipeline(
            crop_size=crop_size,
            design_tasks=design_tasks,
            is_inference=False,
            p_random_rotation=0.0,  # Disable rotation for deterministic testing
            p_atomize_residues=0.1,  # Small chance to atomize for testing
            loss_weight_cfg=loss_weight_cfg,
        )

        # Run pipeline
        output = pipeline(input_data)

    # Extract timing data from transform history
    if hasattr(output, "__transform_history__"):
        design_task_name = list(design_tasks.keys())[0]
        key = (pdb_id, design_task_name)
        for record in output.__transform_history__:
            transform_name = record["name"]
            processing_time = record.get("processing_time")
            if processing_time is not None:
                _timing_stats[key][transform_name] = processing_time

    return output


# =============================================================================
# Tests
# =============================================================================


def test_pooled_masks_distinguish_atomized_and_in_context_tokens() -> None:
    """Atomized and in-context-conditioning tokens retain distinct mask primitives after pooling."""
    task = TipAtomActiveSiteTask(
        p_unindex_all=1.0,
        p_unindex_any=0.0,
        global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
    )
    output = parse_and_run_pipeline(
        pdb_id="6wtf",
        design_tasks={"tip_atom_pocket": {"transform": task, "frequency": 1.0}},
        seed=42,
        crop_size=256,
    )
    feats = output["feats"]
    token_type = np.asarray(feats["token_type"])
    is_in_context_conditioning = np.asarray(feats["is_in_context_conditioning"])
    supports_local_latent = np.asarray(feats["supports_local_latent"])
    is_resolved_mask = np.asarray(feats["is_resolved_mask"])

    assert (
        token_type.shape
        == is_in_context_conditioning.shape
        == supports_local_latent.shape
        == is_resolved_mask.shape
        == feats["coords"].shape[:1]
    )
    atomized = token_type == 3
    currently_unindexed = ~C_IDX.mask(output["atom_array"], default="generate")[get_token_starts(output["atom_array"])]
    assert np.array_equal(is_in_context_conditioning, currently_unindexed)
    assert atomized.any() and is_in_context_conditioning.any(), "fixture must exercise both token kinds"
    assert not is_in_context_conditioning[atomized].any() and not supports_local_latent[atomized].any()
    assert not supports_local_latent[is_in_context_conditioning].any()
    atom_array = output["atom_array"]
    token_starts = get_token_starts(atom_array, add_exclusive_stop=True)
    expected_is_resolved = np.array(
        [
            np.all((atom_array.occupancy[start:stop] > 0.0) & ~np.isnan(atom_array.coord[start:stop]).any(axis=-1))
            for start, stop in zip(token_starts[:-1], token_starts[1:], strict=True)
        ]
    )
    assert np.array_equal(is_resolved_mask, expected_is_resolved)


@pytest.mark.parametrize("pdb_id", TEST_STRUCTURES)
@pytest.mark.parametrize(
    "design_task_name",
    list(DESIGN_TASKS.keys()),
    ids=lambda x: x,
)
def test_pipeline_invariants(pdb_id: str, design_task_name: str) -> None:
    """Test all pipeline invariants for a structure and design task.

    The pipeline includes :py:class:`~rfproteina.transforms.invariants.ValidateBatchInvariants`
    which raises :py:exc:`ValueError` on any violation, so running the pipeline is sufficient.
    """
    selected_design_tasks = {design_task_name: DESIGN_TASKS[design_task_name]}
    try:
        parse_and_run_pipeline(
            pdb_id=pdb_id,
            design_tasks=selected_design_tasks,
            seed=42,
            crop_size=256,
        )
    except AssertionError as e:
        if "No eligible tasks found" in str(e):
            pytest.skip(f"{design_task_name} not applicable to {pdb_id}")
        raise


def log_timing_statistics():
    """Log timing statistics collected during test execution."""
    if not _timing_stats:
        return

    logger.info("\n" + "=" * 80)
    logger.info("TRANSFORM TIMING STATISTICS")
    logger.info("=" * 80 + "\n")

    # Aggregate statistics by transform across all examples
    transform_times = defaultdict(list)
    for (_pdb_id, _task), timings in _timing_stats.items():
        for transform_name, time in timings.items():
            transform_times[transform_name].append(time)

    # Build aggregated DataFrame (sorted by mean time, slowest first)
    agg_data = {
        "Transform": [],
        "Mean (s)": [],
        "Min (s)": [],
        "Max (s)": [],
        "Std (s)": [],
        "Count": [],
    }
    for transform_name, times in transform_times.items():
        agg_data["Transform"].append(transform_name)
        agg_data["Mean (s)"].append(np.mean(times))
        agg_data["Min (s)"].append(np.min(times))
        agg_data["Max (s)"].append(np.max(times))
        agg_data["Std (s)"].append(np.std(times))
        agg_data["Count"].append(len(times))

    agg_df = pd.DataFrame(agg_data).sort_values("Mean (s)", ascending=False)
    print_df_as_table(agg_df, title="Aggregated across all examples", console_width=120)

    # Per-example breakdown
    logger.info("\n" + "=" * 80)
    logger.info("PER-EXAMPLE BREAKDOWN")
    logger.info("=" * 80 + "\n")

    for (pdb_id, task), timings in sorted(_timing_stats.items()):
        total_time = sum(timings.values())
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)[:10]

        # Build DataFrame for top 10 slowest transforms
        per_example_data = {
            "Transform": [],
            "Time (s)": [],
            "% of Total": [],
        }
        for transform_name, time in sorted_timings:
            pct = (time / total_time) * 100 if total_time > 0 else 0
            per_example_data["Transform"].append(transform_name)
            per_example_data["Time (s)"].append(time)
            per_example_data["% of Total"].append(pct)

        # Add total row
        per_example_data["Transform"].append("TOTAL")
        per_example_data["Time (s)"].append(total_time)
        per_example_data["% of Total"].append(100.0)

        per_example_df = pd.DataFrame(per_example_data)
        print_df_as_table(per_example_df, title=f"{pdb_id} ({task})", console_width=120)

    logger.info("\n" + "=" * 80)


def test_atomized_residue_is_not_a_res_id_gap():
    """A PTM mid-chain must not read as a break in residue numbering.

    `find_index_group_res_id_gaps` skips atomized residues as gap anchors; it must also not count
    the res_id they occupy as missing, or every peptide with an internal PTM/NCAA fails validation.
    Fixture is chain B of an EGFR phosphopeptide design input: residues 1-13 with PTR at 7.
    """
    aa = parse(
        Path(__file__).parents[1] / "data" / "egfr2_phosphopeptide.cif",
        load_standard_annotations=True,
        return_atom_array_plus=True,
        add_missing_atoms=False,
    )["assemblies"]["1"][0]

    starts = get_residue_starts(aa)
    res_ids = [int(aa.res_id[i]) for i in starts]
    assert res_ids == list(range(1, 14)), "fixture must be contiguous, or it tests nothing"
    assert S_ATM.mask(aa, default="generate")[starts].sum() == 1, "fixture must have one atomized residue"

    assert find_index_group_res_id_gaps(aa) == []


def _bonded_pair(distance: float, occupancy: tuple[float, float] = (1.0, 1.0)) -> AtomArray:
    aa = AtomArray(2)
    aa.coord = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    aa.set_annotation("occupancy", np.array(occupancy))
    aa.bonds = BondList(2, np.array([(0, 1, BondType.SINGLE)]))
    return aa


def test_bond_length_invariant_rejects_long_bond():
    violations = check_bond_length_violations(_bonded_pair(4.1))

    assert len(violations) == 1
    assert violations[0].check == "bond_length"
    assert violations[0].count == 1


@pytest.mark.parametrize(
    "atom_array",
    [
        pytest.param(_bonded_pair(4.0), id="boundary"),
        pytest.param(_bonded_pair(20.0, occupancy=(0.0, 1.0)), id="unresolved"),
    ],
)
def test_bond_length_invariant_allows_reasonable_or_unmeasured_bonds(atom_array: AtomArray):
    assert check_bond_length_violations(atom_array) == []


if __name__ == "__main__":
    atexit.register(log_timing_statistics)
    pytest.main(["-v", __file__, "-m", "not slow"])
