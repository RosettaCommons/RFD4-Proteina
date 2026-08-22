"""Utility functions for design task testing."""

from typing import Any

import pytest
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.design_task import DesignTaskABC
from biotite.structure import AtomArray

from rfproteina.conditions import C_HB, C_RAS, C_SEC
from rfproteina.datapipes.pipeline import _build_design_clean_and_sample_pipeline
from rfproteina.transforms.design_task import (
    BinderHotspotWith3DTarget,
    InverseFoldingTask,
    MotifScaffoldTask,
    ProteinFromDNASequenceContextTask,
    ProteinFromDNAStructureContextTask,
    StructurePredictionTask,
    TipAtomActiveSiteTask,
    TipAtomPartialLigandTask,
    TipAtomTask,
    UnconditionalTask,
)
from rfproteina.transforms.random_atomize_residues import RandomAtomizeResidues

DEFAULT_REQUIRED_ARGUMENTS: dict[type, dict] = {}


def apply_design_task(
    atom_array_data: dict | AtomArray,
    design_task: DesignTaskABC | type[DesignTaskABC],
    task_kwargs: dict[str, Any] | None = None,
    extra_data: dict[str, Any] | None = None,
    extra_atomize_p: float = 0.0,
) -> dict:
    """Apply a design task to an atom array with standard setup.

    This helper eliminates boilerplate by:
    1. Creating a properly formatted data dict
    2. Checking if the task can be applied
    3. Running the task's forward() method
    4. Returning the modified data

    Args:
      atom_array_data: Data dict from fixture (must contain 'atom_array' key).
      design_task: DesignTask class or instantiated DesignTask to apply.
      task_kwargs: Optional kwargs to pass to task constructor (if re-instantiating).
      extra_data: Optional additional data to add to the data dict before applying task.

    Returns:
      Modified data dict after applying the design task.

    Raises:
      ValueError: If the design task cannot be applied to this atom array.

    Examples:
      With a class:

      >>> data = apply_design_task(simple_protein, UnconditionalTask)

      With an instance:

      >>> data = apply_design_task(simple_protein, UnconditionalTask())
      >>> atom_array = data["atom_array"]
      >>> assert "task" in data

      With custom task parameters:

      >>> task = TipAtomPartialLigandTask(n_max_ligands=2)
      >>> data = apply_design_task(protein_with_ligand, task)

      Re-instantiate with new params:

      >>> data = apply_design_task(protein_with_ligand, task, task_kwargs={"n_max_ligands": 3})
    """
    # Determine if design_task is a class or instance
    is_class = isinstance(design_task, type)

    if is_class:
        # It's a class - instantiate it with defaults + any provided kwargs
        task_class = design_task
        kwargs = DEFAULT_REQUIRED_ARGUMENTS.get(task_class, {})
        if task_kwargs is not None:
            kwargs = kwargs | task_kwargs
        design_task = task_class(**kwargs)
    elif task_kwargs is not None:
        # It's an instance but we want to reinstantiate with new kwargs
        task_class = type(design_task)
        kwargs = DEFAULT_REQUIRED_ARGUMENTS.get(task_class, {}) | task_kwargs
        design_task = task_class(**kwargs)
    # else: it's an instance and task_kwargs is None, use as-is

    # Create data dict
    if not isinstance(atom_array_data, dict):
        data = {
            "atom_array": as_atom_array_plus(atom_array_data["atom_array"].copy()),
            "example_id": atom_array_data.get("pdb_id", "test_example"),
        }
    else:
        data = atom_array_data

    # Add any extra data
    if extra_data is not None:
        data.update(extra_data)

    # Add required features to the atom array using the validation pipeline (does everything up to DesignTask)
    data = Compose(_build_design_clean_and_sample_pipeline(is_inference=True))(data)

    # Optionally force-atomize a fraction of standard residues. Used to stress-test design tasks
    # against atom arrays where most tokens are atomized (rare in real data, but important for
    # invariants like "atomized tokens must remain indexed").
    if extra_atomize_p > 0.0:
        data = RandomAtomizeResidues(p_atomize=extra_atomize_p)(data)

    # edge case where we need to apply same transformation to atom array in crop info (always a copy)
    if "crop_info" in data:
        data["crop_info"]["atom_array"] = data["atom_array"].copy()

    # Annotate for task selection (ensures required annotations are present)
    data = design_task.annotate_for_task_selection(data)

    # Check if task can be applied
    if not design_task.can_apply(data):
        raise ValueError(
            f"DesignTask {design_task.__class__.__name__} cannot be applied to this atom array. "
            f"Check that the atom array meets the task's requirements."
        )

    # Apply the task
    data = design_task.forward(data)

    return data


def check_no_design_task_modifiers(data: dict):
    """
    Check that no design task modifiers were added to the data dict.
    Currently only checks for RASA, HBOND, and secondary structure conditions.
    """
    assert not data["task"].get("added_rasa_condition", False)
    assert not data["task"].get("added_hbond_condition", False)
    assert not data["task"].get("added_secondary_structure_condition", False)
    assert C_HB.mask(data["atom_array"]).sum() == 0
    assert C_RAS.mask(data["atom_array"]).sum() == 0
    assert C_SEC.mask(data["atom_array"]).sum() == 0


"""
A dictionary mapping each design task class to a fixture that contains a minimally eligible example.
I.e. if the design task needs a protein and a ligand, the example may be protein_with_ligand.
Check conftest.py for all the fixtures.
"""
MINIMAL_ELIGIBLE_EXAMPLE_MAP = {
    UnconditionalTask: "simple_protein",
    StructurePredictionTask: "simple_protein",
    InverseFoldingTask: "simple_protein",
    TipAtomPartialLigandTask: "protein_with_ligand",
    TipAtomTask: "simple_protein",
    MotifScaffoldTask: "simple_protein",
    BinderHotspotWith3DTarget: "protein_protein_interface_crop_info",
    TipAtomActiveSiteTask: "simple_protein",
}

COMPLEX_ELIGIBLE_EXAMPLE_MAP = {
    UnconditionalTask: "complex_protein",
    StructurePredictionTask: "complex_protein",
    InverseFoldingTask: "complex_protein",
    TipAtomPartialLigandTask: "complex_protein",
    TipAtomTask: "complex_protein",
    MotifScaffoldTask: "complex_protein",
    BinderHotspotWith3DTarget: "complex_protein_crop_info",
    TipAtomActiveSiteTask: "complex_protein",
    # DNA-binder tasks require a protein + DNA example; ``complex_protein`` carries
    # both (plus RNA/ligand). They are complex-only — no minimal protein+DNA fixture
    # exists, and ``simple_protein`` has no DNA so ``can_apply`` would reject it.
    ProteinFromDNAStructureContextTask: "complex_protein",
    ProteinFromDNASequenceContextTask: "complex_protein",
}


# ============================== Decorators ==============================


def parametrize_minimal_sweep():
    """Parametrize test with all task classes using minimal eligible examples.

    Example:
        @parametrize_minimal_sweep()
        def test_something(task_class, atom_array_fixture):
            data = apply_design_task(atom_array_fixture, task_class())
            # ... your test
    """
    return pytest.mark.parametrize(
        "task_class,atom_array_fixture",
        [
            pytest.param(task_class, fixture_name, id=task_class.__name__)
            for task_class, fixture_name in MINIMAL_ELIGIBLE_EXAMPLE_MAP.items()
        ],
        indirect=["atom_array_fixture"],
    )


def parametrize_complex_sweep():
    """Parametrize test with all task classes using complex eligible examples.

    Example:
        @parametrize_complex_sweep()
        def test_something(task_class, atom_array_fixture):
            data = apply_design_task(atom_array_fixture, task_class())
            # ... your test
    """
    return pytest.mark.parametrize(
        "task_class,atom_array_fixture",
        [
            pytest.param(task_class, fixture_name, id=task_class.__name__)
            for task_class, fixture_name in COMPLEX_ELIGIBLE_EXAMPLE_MAP.items()
        ],
        indirect=["atom_array_fixture"],
    )


def parametrize_all_sweep():
    """Parametrize test with all task classes using both minimal and complex examples.

    Example:
        @parametrize_all_sweep()
        def test_something(task_class, atom_array_fixture, example_type):
            data = apply_design_task(atom_array_fixture, task_class())
            # ... your test
    """
    return pytest.mark.parametrize(
        "task_class,atom_array_fixture,example_type",
        [
            pytest.param(task_class, fixture_name, "minimal", id=f"{task_class.__name__}_minimal")
            for task_class, fixture_name in MINIMAL_ELIGIBLE_EXAMPLE_MAP.items()
        ]
        + [
            pytest.param(task_class, fixture_name, "complex", id=f"{task_class.__name__}_complex")
            for task_class, fixture_name in COMPLEX_ELIGIBLE_EXAMPLE_MAP.items()
        ],
        indirect=["atom_array_fixture"],
    )
