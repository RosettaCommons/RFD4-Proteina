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

import numpy as np
import pandas as pd
import pytest
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from atomworks.ml.utils.testing import cached_parse
from projects.proteinfoundation.tests.pipeline.invariant_checks import (
    check_af3_token_centers,
    check_can_be_encoded_requirements,
    check_centroid_feature_accuracy,
    check_conditioning_consistency,
    check_coordinates_centered_on_conditioned_atoms,
    check_coordinates_not_nan,
    check_elements,
    check_sequence_conditioning_with_res_types,
    check_token_level_features_consistent_within_tokens,
    check_unresolved_tokens_not_lossed,
)

from foundry.utils.logging import print_df_as_table
from proteinfoundation.datapipes.pipeline import build_design_train_pipeline
from proteinfoundation.transforms.design_task import (
    InverseFoldingTask,
    MotifScaffoldTask,
    UnconditionalTask,
)

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
            min_motif_size=5,
            max_motif_size=20,
            global_design_task_kwargs=GLOBAL_DESIGN_TASK_KWARGS,
        ),
        "frequency": 1.0,
    },
}


ALL_CHECKS = [
    # Coordinate Conditioning & Centering
    check_coordinates_centered_on_conditioned_atoms,
    check_coordinates_not_nan,
    check_centroid_feature_accuracy,
    # Sequence Conditioning & Element / Atom Name Leakage
    check_sequence_conditioning_with_res_types,
    check_elements,
    # Encoding & Loss Annotations
    check_can_be_encoded_requirements,
    check_unresolved_tokens_not_lossed,
    # Token-Level Features
    check_af3_token_centers,
    check_token_level_features_consistent_within_tokens,
    # Conditioning Consistency
    check_conditioning_consistency,
]

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

    # Parse structure
    input_data = cached_parse(pdb_id)
    input_data["example_id"] = pdb_id

    # Build pipeline with specified design tasks
    with rng_state(create_rng_state_from_seeds(np_seed=seed, torch_seed=seed, py_seed=seed)):
        pipeline = build_design_train_pipeline(
            crop_size=crop_size,
            design_tasks=design_tasks,
            is_inference=False,
            p_random_rotation=0.0,  # Disable rotation for deterministic testing
            # TODO: Re-enable small chance of atomization once invariants support it
            p_atomize_residues=0.0,  # Small chance to atomize for testing
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
    """Test all pipeline invariants for a structure and design task."""
    # Select only the specified design task
    selected_design_tasks = {design_task_name: DESIGN_TASKS[design_task_name]}

    # Run pipeline (with cropping enabled since some tasks require it)
    output = parse_and_run_pipeline(
        pdb_id=pdb_id,
        design_tasks=selected_design_tasks,
        seed=42,
        crop_size=256,  # Enable cropping for all tests
    )

    # Run all invariant checks
    for check_func in ALL_CHECKS:
        try:
            check_func(output)
        except AssertionError as e:
            raise AssertionError(
                f"Invariant check failed for {pdb_id} with {design_task_name} task\n"
                f"Check: {check_func.__name__}\n"
                f"Error: {e!s}"
            ) from e


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
    # pytest.main(["-v", __file__, "-m", "not slow"])
    test_pipeline_invariants("3wbm", "motif_scaffold")
