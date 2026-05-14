"""Pipeline invariant tests.

These invariants are properties that must hold for ALL pipeline outputs across:
- All input structures (proteins, nucleic acids, ligands, complexes)
- All design tasks (unconditional, inverse_folding, motif_scaffold, etc.)
- All pipeline configurations (with/without cropping, different seeds, etc.)
"""

import rootutils

from foundry.utils.env import load_dotenv

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
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state

from foundry.utils.logging import print_df_as_table
from proteinfoundation.datapipes.pipeline import build_design_train_pipeline
from proteinfoundation.transforms.design_task import (
    InverseFoldingTask,
    MotifScaffoldTask,
    PPIHotspotWith3DTarget,
    StructurePredictionTask,
    TipAtomActiveSiteTask,
    TipAtomPartialLigandTask,
    UnconditionalTask,
)
from proteinfoundation.utils.parse_presets import RCSB_WITH_METALS

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
        "transform": PPIHotspotWith3DTarget(
            hotspot_count_distribution="geometric",
            hotspot_geometric_p=0.2,
            max_hotspots_frac_to_provide=0.1,
            p_no_hotspots=0.05,
            hotspot_max_distance=4.5,
            ppi_crop_binder_fraction=0.667,
            p_sample_ppi_crop=0.75,
            bb_only_coord_conditioning=False,
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


@pytest.mark.parametrize("pdb_id", TEST_STRUCTURES)
@pytest.mark.parametrize(
    "design_task_name",
    list(DESIGN_TASKS.keys()),
    ids=lambda x: x,
)
def test_pipeline_invariants(pdb_id: str, design_task_name: str) -> None:
    """Test all pipeline invariants for a structure and design task.

    The pipeline includes :py:class:`~proteinfoundation.transforms.invariants.ValidateBatchInvariants`
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


if __name__ == "__main__":
    atexit.register(log_timing_statistics)
    pytest.main(["-v", __file__, "-m", "not slow"])
