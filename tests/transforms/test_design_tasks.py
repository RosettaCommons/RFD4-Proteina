"""Tests for design tasks.

Design Task Testing Framework
==============================

We include common fixtures for dummy atom arrays in conftest.py. Please extend these as needed so other
tests can use them.

apply_design_task is a helper function that applies a design task to an atom array and returns the modified atom array.

The majority of the code in your tests should be checking annotations. All of the boilerplate for building atom arrays,
design tasks, and applying design tasks should be handled by the test framework.

Writing Tests
=============

The framework makes it easy to write two types of tests:

**1. Broad Coverage Tests**
    Test a property across many/all design tasks using parameterization:

    .. code-block:: python

        @pytest.mark.parametrize(
            "task_class",
            [
                UnconditionalTask,
                StructurePredictionTask,
                InverseFoldingTask,
                # ... all tasks
            ],
        )
        def test_all_tasks_set_required_fields(simple_protein, task_class):
            data = apply_design_task(simple_protein, task_class())

            # All tasks must set these fields
            assert "task" in data
            assert "class" in data["task"]
            assert "mask" in data["task"]

**2. Specific Task Tests**
    Test detailed behavior of a specific task:

    .. code-block:: python

        def test_inverse_folding_hides_sidechains(simple_protein):
            data = apply_design_task(simple_protein, InverseFoldingTask())
            atom_array = data["atom_array"]

            # Backbone coords should be provided
            backbone_mask = atom_array.is_protein_backbone
            assert C_CRD.mask(atom_array)[backbone_mask].all()

            # Standard AA sidechains should NOT be coordinate-conditioned
            sidechain_mask = atom_array.is_standard_aa & ~backbone_mask
            assert not C_CRD.mask(atom_array)[sidechain_mask].any()

**3. Parameterized Specific Tests**
    Combine both approaches for testing variations:

    .. code-block:: python

        @pytest.mark.parametrize(
            "n_max_ligands,expected_min",
            [
                (1, 1),
                (2, 1),
                (3, 1),
            ],
        )
        def test_tip_atom_ligand_task_samples_ligands(
            protein_with_ligand, n_max_ligands, expected_min
        ):
            task = TipAtomPartialLigandTask(n_max_ligands=n_max_ligands)
            data = apply_design_task(protein_with_ligand, task)

            # Should sample at least expected_min ligands
            n_sampled = len(data["task"]["all_masks"])
            assert n_sampled >= expected_min

"""

import numpy as np
import pytest
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_DIS, C_IDX, C_SEQ
from atomworks.ml.utils.token import apply_and_spread_token_wise

from rfproteina.conditions import C_HB, C_RAS, C_SEC, RasaEnum, SSEnum
from rfproteina.transforms.design_task import (
    InverseFoldingTask,
    ProteinFromDNASequenceContextTask,
    ProteinFromDNAStructureContextTask,
    StructurePredictionTask,
    UnconditionalTask,
)
from tests.transforms.design_task_test_utils import (
    apply_design_task,
    check_no_design_task_modifiers,
    parametrize_complex_sweep,
)

# ============================== Basic Tests ==============================


def test_unconditional_task_basic_properties(simple_protein):
    """Test that UnconditionalTask sets basic required fields and annotations.

    This is a simple example showing the framework in action.
    """
    # Apply the task using the helper
    data = apply_design_task(simple_protein, UnconditionalTask())
    atom_array = data["atom_array"]

    # Check required task fields are present
    assert "task" in data
    assert "class" in data["task"]
    assert "mask" in data["task"]
    assert data["task"]["class"] == "UnconditionalTask"

    # Check task mask properties
    task_mask = data["task"]["mask"]
    assert len(task_mask) == atom_array.array_length()
    # Unconditional task should have all-zeros mask (nothing is conditioned as a motif)
    assert task_mask.sum() == 0

    # Check that all atoms are indexed (this is typically set for all tasks)
    assert C_IDX.mask(atom_array).all()


@pytest.mark.parametrize(
    "task_class",
    [
        UnconditionalTask,
        StructurePredictionTask,
        InverseFoldingTask,
    ],
)
def test_all_simple_tasks_set_required_fields(simple_protein, task_class):
    """Test that all basic tasks set required fields in data dict.

    This demonstrates parameterized testing across multiple task classes.
    """
    data = apply_design_task(simple_protein, task_class())

    # All tasks must set these fields
    assert "task" in data
    assert "class" in data["task"]
    assert "mask" in data["task"]
    assert "name" in data["task"]


@pytest.mark.parametrize("atom_array_fixture", ["simple_protein", "complex_protein"], indirect=True)
def test_inverse_folding_conditions(atom_array_fixture):
    """Test that InverseFoldingTask provides backbone coords and hides sidechains.

    This demonstrates a task-specific test checking detailed condition behavior.
    """
    data = apply_design_task(atom_array_fixture, InverseFoldingTask())
    atom_array = data["atom_array"]

    # Standard AA backbone atoms should have coordinate conditioning
    backbone_mask = atom_array.is_standard_aa & atom_array.is_protein_backbone
    assert C_CRD.mask(atom_array)[backbone_mask].all()

    # Standard AA sidechain atoms should NOT have coordinate conditioning
    sidechain_mask = atom_array.is_standard_aa & ~atom_array.is_protein_backbone
    assert not C_CRD.mask(atom_array)[sidechain_mask].any()

    # Non-standard residues (ligands, modified AAs) should have both coord and sequence
    non_standard_mask = ~atom_array.is_standard_aa
    if non_standard_mask.any():
        assert C_CRD.mask(atom_array)[non_standard_mask].all()
        assert C_SEQ.mask(atom_array)[non_standard_mask].all()

    # Standard AAs should NOT have sequence conditioning
    assert not C_SEQ.mask(atom_array)[atom_array.is_standard_aa].any()

    # Shouldn't have RASA, HBOND, or SS conditions
    check_no_design_task_modifiers(data)


@pytest.mark.parametrize("atom_array_fixture", ["simple_protein", "complex_protein"], indirect=True)
def test_structure_prediction_conditions(atom_array_fixture):
    """
    Test that StructurePredictionTask provides sequence information and doesn't add any other conditions.
    """

    data = apply_design_task(atom_array_fixture, StructurePredictionTask())
    atom_array = data["atom_array"]

    # Sequence information should be provided
    assert C_SEQ.mask(atom_array).all()

    # Should have no coordinate or distance conditions
    assert C_CRD.mask(atom_array).sum() == 0
    assert C_DIS.mask(atom_array).as_dense_array().sum() == 0

    # Shouldn't have RASA, HBOND, or SS conditions
    check_no_design_task_modifiers(data)


@pytest.mark.parametrize("atom_array_fixture", ["complex_protein"], indirect=True)
def test_dna_structure_context_conditions(atom_array_fixture):
    """ProteinFromDNAStructureContextTask provides the DNA structure + sequence as
    context while the protein is generated from noise.
    """
    data = apply_design_task(atom_array_fixture, ProteinFromDNAStructureContextTask())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_dna", "is_standard_aa")

    is_dna = atom_array.is_dna
    assert is_dna.any(), "fixture must contain DNA"

    # DNA sequence and (finite-coord) DNA structure are both provided as context.
    assert C_SEQ.mask(atom_array)[is_dna].all()
    finite_coord = ~np.isnan(atom_array.coord).any(axis=1)
    assert C_CRD.mask(atom_array)[is_dna & finite_coord].all()

    # The protein is generated: standard AAs get neither sequence nor coord context.
    is_std_aa = atom_array.is_standard_aa
    assert not C_SEQ.mask(atom_array)[is_std_aa].any()
    assert not C_CRD.mask(atom_array)[is_std_aa].any()


@pytest.mark.parametrize("atom_array_fixture", ["complex_protein"], indirect=True)
def test_dna_sequence_context_conditions(atom_array_fixture):
    """ProteinFromDNASequenceContextTask provides only the DNA sequence as context;
    the DNA structure is co-generated and the protein is generated from noise.
    """
    data = apply_design_task(atom_array_fixture, ProteinFromDNASequenceContextTask())
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_dna", "is_standard_aa")

    is_dna = atom_array.is_dna
    assert is_dna.any(), "fixture must contain DNA"

    # DNA sequence is provided, but no DNA coordinates (structure is co-generated).
    assert C_SEQ.mask(atom_array)[is_dna].all()
    assert not C_CRD.mask(atom_array)[is_dna].any()

    # The protein is generated: standard AAs get neither sequence nor coord context.
    is_std_aa = atom_array.is_standard_aa
    assert not C_SEQ.mask(atom_array)[is_std_aa].any()
    assert not C_CRD.mask(atom_array)[is_std_aa].any()


# ============================== Comprehensive Tests ==============================


@parametrize_complex_sweep()
def test_no_design_tasks_leave_atomized_things_without_sequence_conditioning(task_class, atom_array_fixture):
    """
    Test every design task and make sure if there is a ligand or non-canonical amino acid, the sequence should be conditioned.
    """
    data = apply_design_task(atom_array_fixture, task_class)
    atom_array = data["atom_array"]

    ensure_annotations(atom_array, "is_ligand")
    should_be_sequence_conditioned = atom_array.is_ligand | atom_array.atomize

    assert C_SEQ.mask(atom_array)[should_be_sequence_conditioned].all()


@pytest.mark.parametrize("extra_atomize_p", [0.0, 0.9])
@parametrize_complex_sweep()
def test_unindexed_tokens_are_whole_and_atomized_atoms_stay_indexed(task_class, atom_array_fixture, extra_atomize_p):
    """Test that C_IDX is consistent token-wise and atomized atoms are never unindexed.

    Two invariants are checked across all design tasks:

    1. Every token has C_IDX uniformly set across all of its atoms. If any atom in a
       token is unindexed, all atoms in that token are unindexed.
    2. No atomized atom is unindexed. Atomized things (ligands, non-canonical residues,
       etc.) must always remain indexed because tasks rely on their identity.

    The ``extra_atomize_p=0.9`` variant force-atomizes most standard residues to
    stress-test invariant (2) on arrays where atomized tokens dominate.
    """
    data = apply_design_task(atom_array_fixture, task_class, extra_atomize_p=extra_atomize_p)
    atom_array = data["atom_array"]

    is_indexed = C_IDX.mask(atom_array)

    # Invariant 1: token-wise consistency of C_IDX (all atoms in a token agree).
    spread_all = apply_and_spread_token_wise(atom_array, is_indexed, np.all)
    spread_any = apply_and_spread_token_wise(atom_array, is_indexed, np.any)
    assert np.array_equal(
        spread_all, spread_any
    ), "Found a token with mixed C_IDX values; unindexing must apply to whole tokens."

    # Invariant 2: atomized atoms must remain indexed.
    if "atomize" in atom_array.get_annotation_categories():
        atomized = atom_array.atomize
        assert is_indexed[atomized].all(), "Atomized atoms must never be unindexed."


@parametrize_complex_sweep()
def test_coords_and_distances_only_applied_to_sequence_conditioned_or_backbone_atoms(task_class, atom_array_fixture):
    """
    Test that coords and distances are only applied to sequence-conditioned or backbone atoms.
    """
    data = apply_design_task(atom_array_fixture, task_class)
    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_protein_backbone", "is_nucleic_acid_backbone")

    can_be_coord_or_distance_conditioned = (
        C_SEQ.mask(atom_array) | atom_array.is_protein_backbone | atom_array.is_nucleic_acid_backbone
    )

    assert not C_CRD.mask(atom_array)[~can_be_coord_or_distance_conditioned].any()
    assert not C_DIS.mask(atom_array).as_dense_array().any(axis=1)[~can_be_coord_or_distance_conditioned].any()


# ============================== Edge Case Tests ==============================


def test_design_task_modifiers(protein_with_ligand):
    """
    Uses protein + ligand example and runs UnconditionalTask with design task modifiers set to 100% probability.
    Then makes sure the RASA, hbond, and secondary structure conditions are present and not all 0s
    """

    design_task = UnconditionalTask(
        rasa_kwargs={"p_add_rasa_condition": 1.0},
        hbond_kwargs={"p_add_hbond_condition": 1.0},
        secondary_structure_kwargs={"p_add_secondary_structure_condition": 1.0},
    )

    data = apply_design_task(protein_with_ligand, design_task)
    atom_array = data["atom_array"]

    # check task dict
    assert data["task"]["added_rasa_condition"]
    assert data["task"]["added_hbond_condition"]
    assert data["task"]["added_secondary_structure_condition"]

    # check masks
    assert C_RAS.mask(atom_array).sum() > 0
    assert C_HB.mask(atom_array).sum() > 0
    assert C_SEC.mask(atom_array).sum() > 0

    # check annotations
    assert not np.all(C_RAS.annotation(atom_array, default="raise") == RasaEnum.UNSPECIFIED)
    assert not np.all(C_SEC.annotation(atom_array, default="raise") == SSEnum.NONE)
