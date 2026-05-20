"""
Tests for the new Condition classes and design task modifiers added to support
RASA, hydrogen bonds, and secondary structure conditioning.

These tests focus on:
1. Condition class behavior (default masks, default annotations, enums)
2. Design task modifier functions (add_rasa_condition, add_hbond_condition, add_secondary_structure_condition)
3. Integration with the RFD3DesignTask infrastructure
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from atomworks.ml.conditions import C_IDX, C_SEQ
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from atomworks.ml.utils.rng import create_rng_state_from_seeds, rng_state
from biotite.structure import AtomArray, BondList

from proteinfoundation.conditions import (
    C_ABH,
    C_ABL,
    C_CDR,
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_RAS,
    C_SEC,
    HBondEnum,
    RasaEnum,
    SSEnum,
)
from proteinfoundation.transforms.design_task_modifiers import (
    add_hbond_condition,
    add_rasa_condition,
    add_secondary_structure_condition,
)
from proteinfoundation.transforms.utils.conditioning_utils import (
    sample_unindexed_groups,
)


@pytest.fixture
def deterministic_rng():
    """Fixture for deterministic random number generation."""
    with rng_state(create_rng_state_from_seeds(np_seed=42, torch_seed=42, py_seed=42)):
        yield


# ============================================================================
# Tests for Condition Classes
# ============================================================================


@pytest.mark.fast
class TestRasaCondition:
    """Tests for the RASA condition class."""

    def test_default_annotation(self, simple_protein):
        """Test that default RASA annotation is UNSPECIFIED for all atoms."""
        simple_protein = simple_protein["atom_array"]
        annotation = C_RAS.default_annotation(simple_protein)
        assert annotation.shape == (len(simple_protein),)
        assert np.all(annotation == RasaEnum.UNSPECIFIED)
        assert annotation.dtype == int

    def test_default_mask_no_annotation(self, simple_protein):
        """Test default mask when no RASA annotation exists."""
        simple_protein = simple_protein["atom_array"]
        mask = C_RAS.default_mask(simple_protein)
        assert mask.shape == (len(simple_protein),)
        assert np.all(~mask)  # All False when no annotation exists

    def test_default_mask_with_annotation(self, simple_protein):
        """Test default mask when RASA annotation is present."""
        simple_protein = simple_protein["atom_array"]
        # Add RASA annotation
        rasa_annotation = np.full(len(simple_protein), RasaEnum.UNSPECIFIED, dtype=int)
        rasa_annotation[0:10] = RasaEnum.BURIED
        rasa_annotation[10:20] = RasaEnum.EXPOSED
        # Leave rest as UNSPECIFIED
        C_RAS.set_annotation(simple_protein, rasa_annotation)

        mask = C_RAS.default_mask(simple_protein)
        # Mask should be True only for non-UNSPECIFIED atoms
        assert np.sum(mask) == 20
        assert np.all(mask[0:20])
        assert np.all(~mask[20:])


@pytest.mark.fast
class TestSecondaryStructureCondition:
    """Tests for the secondary structure condition class."""

    def test_default_annotation(self, simple_protein):
        """Test that default secondary structure annotation is NONE for all atoms."""
        simple_protein = simple_protein["atom_array"]
        annotation = C_SEC.default_annotation(simple_protein)
        assert annotation.shape == (len(simple_protein),)
        assert np.all(annotation == SSEnum.NONE)
        assert annotation.dtype == int

    def test_default_mask_no_annotation(self, simple_protein):
        """Test default mask when no secondary structure annotation exists."""
        simple_protein = simple_protein["atom_array"]
        mask = C_SEC.default_mask(simple_protein)
        assert mask.shape == (len(simple_protein),)
        assert np.all(~mask)

    def test_default_mask_with_annotation(self, simple_protein):
        """Test default mask when secondary structure annotation is present."""
        simple_protein = simple_protein["atom_array"]
        ss_annotation = np.full(len(simple_protein), SSEnum.NONE, dtype=int)
        ss_annotation[0:10] = SSEnum.HELIX
        ss_annotation[10:20] = SSEnum.SHEET
        ss_annotation[20:30] = SSEnum.LOOP
        C_SEC.set_annotation(simple_protein, ss_annotation)

        mask = C_SEC.default_mask(simple_protein)
        # Mask should be True only for non-NONE atoms
        assert np.sum(mask) == 30
        assert np.all(mask[0:30])
        assert np.all(~mask[30:])


@pytest.mark.fast
class TestHydrogenBondConditions:
    """Tests for hydrogen bond donor and acceptor conditions."""

    def test_default_annotations(self, simple_protein):
        """Test default annotations for hydrogen bond conditions."""
        simple_protein = simple_protein["atom_array"]
        hbond_annotation = C_HB.default_annotation(simple_protein)

        assert hbond_annotation.shape == (len(simple_protein),)
        assert hbond_annotation.dtype == int
        assert np.all(hbond_annotation == HBondEnum.NONE)


@pytest.mark.fast
class TestHotspotCondition:
    """Tests for the hotspot condition class."""

    def test_default_annotation(self, simple_protein):
        """Test default hotspot annotation."""
        simple_protein = simple_protein["atom_array"]
        annotation = C_HOT.default_annotation(simple_protein)
        assert annotation.shape == (len(simple_protein),)
        assert annotation.dtype == bool
        assert np.all(~annotation)

    def test_default_mask_no_annotation(self, simple_protein):
        """Test default mask when no hotspot annotation exists."""
        simple_protein = simple_protein["atom_array"]
        mask = C_HOT.default_mask(simple_protein)
        assert mask.shape == (len(simple_protein),)
        assert np.all(~mask)

    def test_default_mask_with_annotation(self, simple_protein):
        """Test that hotspot mask spreads across entire chains containing hotspots."""
        simple_protein = simple_protein["atom_array"]
        # Set hotspot annotation for some atoms
        hotspot_annotation = np.zeros(len(simple_protein), dtype=bool)
        hotspot_annotation[5] = True  # One hotspot atom in chain A
        C_HOT.set_annotation(simple_protein, hotspot_annotation)

        mask = C_HOT.default_mask(simple_protein)
        assert np.all(mask == hotspot_annotation)  # All atoms are in chain A in this test fixture


@pytest.mark.fast
class TestAntibodyConditions:
    """Tests for antibody-specific condition classes."""

    def test_cdr_default_mask(self, simple_protein):
        """Test CDR condition default mask."""
        simple_protein = simple_protein["atom_array"]
        mask = C_CDR.default_mask(simple_protein)
        assert mask.shape == (len(simple_protein),)
        assert np.all(~mask)
        assert mask.dtype == bool

    def test_antibody_chain_default_masks(self, simple_protein):
        """Test antibody heavy and light chain default masks."""
        simple_protein = simple_protein["atom_array"]
        heavy_mask = C_ABH.default_mask(simple_protein)
        light_mask = C_ABL.default_mask(simple_protein)

        assert heavy_mask.shape == (len(simple_protein),)
        assert light_mask.shape == (len(simple_protein),)
        assert np.all(~heavy_mask)
        assert np.all(~light_mask)


# ============================================================================
# Tests for Condition Aliases
# ============================================================================


@pytest.mark.fast
def test_condition_aliases():
    """Test that convenience aliases for conditions are properly set up."""
    from atomworks.ml.conditions import CONDITIONS

    assert CONDITIONS.hotspot == C_HOT
    assert CONDITIONS.cdr == C_CDR
    assert CONDITIONS.ab_heavy == C_ABH
    assert CONDITIONS.ab_light == C_ABL
    assert CONDITIONS.rasa_binned == C_RAS
    assert CONDITIONS.secondary_structure == C_SEC
    assert CONDITIONS.hbond == C_HB


# ============================================================================
# Tests for Design Task Modifier Functions
# ============================================================================


@pytest.fixture
def annotated_atom_array():
    """
    Load an atom array with ground-truth annotations for RASA, hydrogen bonds,
    and secondary structure. This fixture contains ground-truth values
    that can be compared against the modifier outputs.
    NOTE: The secondary structure outputs were fully verified by hand.
    Rasa and hbonds were spot-checked and seemed reasonable.
    """
    test_data_path = Path(__file__).parent / ".." / "data" / "1a1e_subsetted_with_modifier_gt.pkl"

    with open(test_data_path, "rb") as f:
        atom_array = pickle.load(f)

    # Rename things
    atom_array.set_annotation("rasa_gt", C_RAS.annotation(atom_array))

    atom_array.set_annotation(
        "hbond_acceptor_gt",
        atom_array.motif_active_acceptor_gt_between_motif_and_diffused,
    )
    atom_array.set_annotation("hbond_donor_gt", atom_array.motif_active_donor_gt_between_motif_and_diffused)

    # Create a combined secondary structure annotation for easier comparison
    ss_gt = np.full(atom_array.array_length(), SSEnum.NONE, dtype=int)
    ss_gt[atom_array.is_helix_gt] = SSEnum.HELIX
    ss_gt[atom_array.is_sheet_gt] = SSEnum.SHEET
    ss_gt[atom_array.is_loop_gt] = SSEnum.LOOP
    atom_array.set_annotation("secondary_structure_gt", ss_gt)

    return atom_array


@pytest.fixture
def unannotated_atom_array(annotated_atom_array):
    """
    Create a copy of the annotated atom array without the ground-truth annotations.
    This is what the modifier functions will operate on.
    """
    atom_array = annotated_atom_array.copy()
    # Remove ground-truth annotations
    for annot in [
        "rasa",
        "rasa_gt",
        "is_helix_gt",
        "is_sheet_gt",
        "is_loop_gt",
        "secondary_structure_gt",
        "hbond_donor_gt",
        "hbond_acceptor_gt",
        "motif_active_donor_gt_between_motif_and_diffused",
        "motif_active_acceptor_gt_between_motif_and_diffused",
        "active_donor_gt_between_motif_and_diffused",
        "active_acceptor_gt_between_motif_and_diffused",
    ]:
        if annot in atom_array.get_annotation_categories():
            atom_array.del_annotation(annot)
    return atom_array


@pytest.mark.fast
class TestAddRasaCondition:
    """Tests for the add_rasa_condition modifier function."""

    def test_rasa_calculation_and_binning(self, deterministic_rng, unannotated_atom_array):
        """Test that RASA values are calculated, added, and binned correctly."""

        # add sequence conditioning to everything so rasa conditioning is added
        C_SEQ.set_mask(unannotated_atom_array, np.ones(len(unannotated_atom_array), dtype=bool))

        data = {"atom_array": unannotated_atom_array}
        rasa_max = 0.2

        # Run the modifier with p_include_non_ligand=1.0 to ensure all atoms are included
        result = add_rasa_condition(data, p_include_non_ligand=1.0, rasa_max=rasa_max)
        atom_array = result["atom_array"]

        # Check that binned RASA annotation was added
        assert C_RAS.full_name in atom_array.get_annotation_categories()
        rasa_binned = C_RAS.annotation(atom_array)
        assert rasa_binned.shape == (len(atom_array),)
        assert rasa_binned.dtype == int

        # Verify binning: bins should be 0 (buried), 1 (intermediate), 2 (exposed), or 3 (unspecified)
        rasa_values = calculate_atomwise_rasa(
            atom_array,
            probe_radius=1.4,
            atom_radii="ProtOr",
            point_number=100,
            atom_filter=np.ones(len(atom_array), dtype=bool),
        )
        assert np.all(
            np.isin(
                rasa_binned,
                [
                    RasaEnum.BURIED,
                    RasaEnum.INTERMEDIATE,
                    RasaEnum.EXPOSED,
                    RasaEnum.UNSPECIFIED,
                ],
            )
        )

        # Check that specified values are binned correctly
        rasa_specified_mask = rasa_binned != RasaEnum.UNSPECIFIED
        for i in np.where(rasa_specified_mask)[0]:
            rasa_val = rasa_values[i]
            bin_val = rasa_binned[i]

            if rasa_val < 0.1:
                assert bin_val == RasaEnum.BURIED
            elif rasa_val < 0.2:
                assert bin_val == RasaEnum.INTERMEDIATE
            else:
                assert bin_val == RasaEnum.EXPOSED

    def test_rasa_inclusion_mask_ligand_only(self, deterministic_rng, unannotated_atom_array):
        """Test that with p_include_non_ligand=0, only ligand atoms get RASA conditioning."""
        # Set some atoms as ligand
        unannotated_atom_array.is_ligand[:10] = True

        data = {"atom_array": unannotated_atom_array}

        # Use deterministic RNG to ensure p_include_non_ligand branch is not taken
        # We'll run this multiple times to verify behavior
        result = add_rasa_condition(data, p_include_non_ligand=0.0, rasa_max=0.2)
        atom_array = result["atom_array"]

        rasa_binned = C_RAS.annotation(atom_array)

        # Non-ligand atoms should be UNSPECIFIED
        non_ligand_mask = ~atom_array.is_ligand
        assert np.all(rasa_binned[non_ligand_mask] == RasaEnum.UNSPECIFIED)

    def test_rasa_no_negative_values(self, deterministic_rng, unannotated_atom_array):
        """Test that negative RASA values raise an error."""
        data = {"atom_array": unannotated_atom_array}

        # Artificially inject negative RASA (this shouldn't happen in practice)
        # We'll patch the calculate_atomwise_rasa function to return negative values
        import proteinfoundation.transforms.design_task_modifiers as mod

        original_calc = mod.calculate_atomwise_rasa

        def mock_calc_with_negative(*args, **kwargs):
            rasa = original_calc(*args, **kwargs)
            rasa[0] = -0.1  # Inject a negative value
            return rasa

        mod.calculate_atomwise_rasa = mock_calc_with_negative

        try:
            with pytest.raises(ValueError, match="Negative RASA value"):
                add_rasa_condition(data, p_include_non_ligand=1.0)
        finally:
            mod.calculate_atomwise_rasa = original_calc


@pytest.mark.fast
class TestAddSecondaryStructureCondition:
    """Tests for the add_secondary_structure_condition modifier function."""

    def test_ss_annotation_and_subset_validation(self, deterministic_rng, unannotated_atom_array):
        """Test that SS annotations are added and are a subset of ground-truth SS."""
        data = {"atom_array": unannotated_atom_array}

        result = add_secondary_structure_condition(data)
        atom_array = result["atom_array"]

        # Check that the SS condition annotation was added
        assert C_SEC.full_name in atom_array.get_annotation_categories()


@pytest.mark.slow
class TestAddHbondCondition:
    """Tests for the add_hbond_condition modifier function."""

    def test_hbond_annotations_and_subsampling(self, unannotated_atom_array):
        """Test that hydrogen bond annotations are added and subsampling works correctly."""
        data = {"atom_array": unannotated_atom_array.copy()}

        # First run with subsampling enabled
        result = add_hbond_condition(data, p_hbond_subsample=1.0)
        atom_array = result["atom_array"]

        # Check that hbond annotations were added
        assert C_HB.full_name in atom_array.get_annotation_categories()

        hbond_annot = C_HB.annotation(atom_array)

        assert hbond_annot.dtype == int

        # Check that subsampling occurred (should be in log_dict)
        assert "hbond_subsample_atoms" in result["log_dict"]
        assert "hbond_total_count" in result["log_dict"]

        # Subsampled count should be < total count
        assert result["log_dict"]["hbond_subsample_atoms"] < result["log_dict"]["hbond_total_count"]

        # make sure there are actually some hbonds
        assert hbond_annot.sum() > 0


@pytest.mark.fast
class TestIndexGroupCondition:
    """Tests for the C_IDXGRP (IndexGroup) condition class."""

    def test_default_behavior(self, simple_protein):
        """Test default mask and annotation behavior with rigorous value checks."""
        # Test mask: all atoms should have an index group
        simple_protein = simple_protein["atom_array"]
        mask = C_IDXGRP.default_mask(simple_protein)
        assert mask.shape == (len(simple_protein),)
        assert mask.dtype == bool
        assert mask.sum() == len(
            simple_protein
        ), f"Expected all {len(simple_protein)} atoms to be masked, got {mask.sum()}"
        assert np.all(mask), "All atoms should be included in default mask"

        # Test annotation without C_IDX: all atoms get group 0 for single chain
        annotation = C_IDXGRP.default_annotation(simple_protein)
        assert annotation.shape == (len(simple_protein),)
        assert np.issubdtype(annotation.dtype, np.integer)
        expected_group = 0
        assert np.all(
            annotation == expected_group
        ), f"Expected all annotations to be {expected_group}, got unique values: {np.unique(annotation)}"
        assert annotation.sum() == len(simple_protein) * expected_group

        # Test annotation with C_IDX: indexed atoms get group 0, unindexed get group 1
        idx_mask = np.array([True] * 20 + [False] * 20 + [False] * (len(simple_protein) - 40))
        C_IDX.set_mask(simple_protein, idx_mask)
        annotation = C_IDXGRP.default_annotation(simple_protein)

        # Check indexed region (first 20 atoms)
        assert np.all(annotation[:20] == 0), f"Indexed atoms should have group 0, got {np.unique(annotation[:20])}"

        # Check unindexed region (last 20 atoms) - should all be in group 1
        assert np.all(
            annotation[20:] == 1
        ), f"Unindexed atoms should be in group 1, got unique values: {np.unique(annotation[20:])}"


@pytest.mark.fast
class TestSampleUnindexedGroups:
    """Tests for the sample_unindexed_groups function."""

    def test_group_assignment_by_residue_contiguity(self):
        """Test groups are assigned based on residue contiguity and indexing with rigorous value checks."""
        # Create atom array with residue gap and mixed indexing
        atom_array = AtomArray(30)
        atom_array.chain_id = np.array(["A"] * 30)
        atom_array.res_name = np.array(["ALA"] * 30)
        atom_array.bonds = BondList(30, np.asarray([]))
        # Residues: 1, 2 (gap) 10, 11
        atom_array.res_id = np.array([1] * 5 + [2] * 5 + [10] * 10 + [11] * 10)
        atom_array.atom_name = np.array(["CA"] * 30)

        # Middle section is indexed
        idx_mask = np.array([False] * 10 + [True] * 10 + [False] * 10)
        C_IDX.set_mask(atom_array, idx_mask)

        groups = sample_unindexed_groups(atom_array, p_merge_groups=0.0, p_split_group=0.0)

        # Validate return type and shape
        assert isinstance(groups, np.ndarray)
        assert groups.shape == (30,), f"Expected shape (30,), got {groups.shape}"
        assert np.issubdtype(groups.dtype, np.integer)
        assert np.all(groups >= 0), f"All groups should be non-negative, got min={groups.min()}"

        # Groups are assigned sequentially:
        # - First unindexed region: group 0 or higher
        # - Indexed region: different group
        # - Second unindexed region: different group
        first_unindexed = groups[:10]
        indexed_groups = groups[10:20]
        second_unindexed = groups[20:]

        # Each contiguous region should have a single group ID
        assert (
            len(np.unique(first_unindexed)) == 1
        ), f"First region should be contiguous (1 group), got {len(np.unique(first_unindexed))} groups: {np.unique(first_unindexed)}"
        assert (
            len(np.unique(indexed_groups)) == 1
        ), f"Indexed region should be contiguous (1 group), got {len(np.unique(indexed_groups))} groups: {np.unique(indexed_groups)}"
        assert (
            len(np.unique(second_unindexed)) == 1
        ), f"Second region should be contiguous (1 group), got {len(np.unique(second_unindexed))} groups: {np.unique(second_unindexed)}"

        # All three regions should have different group IDs
        first_unindexed_group_id = first_unindexed[0]
        indexed_group_id = indexed_groups[0]
        second_unindexed_group_id = second_unindexed[0]
        assert (
            first_unindexed_group_id != indexed_group_id
        ), f"First unindexed and indexed should have different IDs, both got {first_unindexed_group_id}"
        assert (
            first_unindexed_group_id != second_unindexed_group_id
        ), f"First and second unindexed regions should have different IDs, both got {first_unindexed_group_id}"
        assert (
            indexed_group_id != second_unindexed_group_id
        ), f"Indexed and second unindexed should have different IDs, both got {indexed_group_id}"


if __name__ == "__main__":
    pytest.main(sys.argv)
