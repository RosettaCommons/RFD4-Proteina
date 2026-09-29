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

from rfproteina.conditions import (
    C_HB,
    C_HOT,
    C_HPHOB,
    C_IDXGRP,
    C_MCOORD,
    C_RAS,
    C_SALT,
    C_SEC,
    HBondEnum,
    MetalCoordEnum,
    RasaEnum,
    SaltBridgeEnum,
    SSEnum,
)
from rfproteina.transforms.design_task_modifiers import (
    add_hbond_condition,
    add_hydrophobic_condition,
    add_metal_condition,
    add_rasa_condition,
    add_salt_condition,
    add_secondary_structure_condition,
)
from rfproteina.transforms.utils.conditioning_utils import (
    sample_unindexed_groups,
)
from tests.transforms.synthetic_example_generator import concat_atom_arrays_without_conflict


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
        mask = C_RAS.mask(simple_protein, default="generate")
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

        mask = C_RAS.mask(simple_protein, default="generate")
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
        mask = C_SEC.mask(simple_protein, default="generate")
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

        mask = C_SEC.mask(simple_protein, default="generate")
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
        mask = C_HOT.mask(simple_protein, default="generate")
        assert mask.shape == (len(simple_protein),)
        assert np.all(~mask)

    def test_default_mask_with_annotation(self, simple_protein):
        """Test that hotspot mask spreads across entire chains containing hotspots."""
        simple_protein = simple_protein["atom_array"]
        # Set hotspot annotation for some atoms
        hotspot_annotation = np.zeros(len(simple_protein), dtype=bool)
        hotspot_annotation[5] = True  # One hotspot atom in chain A
        C_HOT.set_annotation(simple_protein, hotspot_annotation)

        mask = C_HOT.mask(simple_protein, default="generate")
        assert np.all(mask == hotspot_annotation)  # All atoms are in chain A in this test fixture


# ============================================================================
# Tests for Condition Aliases
# ============================================================================


@pytest.mark.fast
def test_condition_aliases():
    """Test that convenience aliases for conditions are properly set up."""
    from atomworks.ml.conditions import CONDITIONS

    assert CONDITIONS.hotspot == C_HOT
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
        C_SEQ.set_annotation(unannotated_atom_array, C_SEQ.ground_truth_value(unannotated_atom_array))

        data = {"atom_array": unannotated_atom_array}
        rasa_max = 0.2

        # Run the modifier with p_include_non_ligand=1.0 to ensure all atoms are included
        result = add_rasa_condition(data, p_add_rasa_condition=1.0, p_include_non_ligand=1.0, rasa_max=rasa_max)
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

    def test_rasa_no_negative_values(self, deterministic_rng, unannotated_atom_array):
        """Test that negative RASA values raise an error."""
        data = {"atom_array": unannotated_atom_array}

        # Artificially inject negative RASA (this shouldn't happen in practice)
        # We'll patch the calculate_atomwise_rasa function to return negative values
        import rfproteina.transforms.design_task_modifiers as mod

        original_calc = mod.calculate_atomwise_rasa

        def mock_calc_with_negative(*args, **kwargs):
            rasa = original_calc(*args, **kwargs)
            rasa[0] = -0.1  # Inject a negative value
            return rasa

        mod.calculate_atomwise_rasa = mock_calc_with_negative

        try:
            with pytest.raises(ValueError, match="Negative RASA value"):
                add_rasa_condition(data, p_add_rasa_condition=1.0, p_include_non_ligand=1.0)
        finally:
            mod.calculate_atomwise_rasa = original_calc


@pytest.mark.fast
class TestAddSecondaryStructureCondition:
    """Tests for the add_secondary_structure_condition modifier function."""

    def test_ss_annotation_and_subset_validation(self, deterministic_rng, unannotated_atom_array):
        """Test that SS annotations are added and are a subset of ground-truth SS."""
        data = {"atom_array": unannotated_atom_array}

        result = add_secondary_structure_condition(data, p_add_secondary_structure_condition=1.0)
        atom_array = result["atom_array"]

        # Check that the SS condition annotation was added
        assert C_SEC.full_name in atom_array.get_annotation_categories()


@pytest.fixture
def protein_with_contacting_partner_chain(simple_protein):
    """``simple_protein`` plus a copy of itself as a second chain, offset by 2.5 A.

    An interaction annotation is only revealed while some *other* chain carries the
    complementary value within the detector's cutoff -- 3.0 A at the tightest (metal
    coordination). Duplicating the chain 2.5 A away gives every atom a cross-chain neighbour,
    so which annotations have a partner is decided purely by the values a test writes onto
    chain B, not by the geometry of the original structure.
    """
    chain_a = simple_protein["atom_array"].copy()
    chain_b = chain_a.copy()
    chain_b.coord = chain_b.coord + np.array([2.5, 0.0, 0.0])
    return concat_atom_arrays_without_conflict(chain_a, chain_b)


@pytest.mark.fast
class TestSubsetGroundTruthInteractionConditions:
    """Tests for the interaction-condition modifiers (hbond/salt/hydrophobic/metal).

    These modifiers subset a ground-truth interaction Condition that is already present on the
    AtomArray -- they do not compute interactions themselves. The ring-level conditions
    (pistack/pication) need a real aromatic ring to say anything, so they live in
    ``test_pi_ring_conditioning.py`` instead.
    """

    @pytest.mark.parametrize(
        "add_condition_fn,condition,none_value,active_value,partner_value,p_add_kwarg,p_subsample_kwarg,subsample_frac_kwarg",
        [
            (
                add_hbond_condition,
                C_HB,
                HBondEnum.NONE,
                HBondEnum.DONOR,
                HBondEnum.ACCEPTOR,
                "p_add_hbond_condition",
                "p_hbond_subsample",
                "hbond_subsample_frac",
            ),
            (
                add_salt_condition,
                C_SALT,
                SaltBridgeEnum.NONE,
                SaltBridgeEnum.POSITIVE,
                SaltBridgeEnum.NEGATIVE,
                "p_add_salt_condition",
                "p_salt_subsample",
                "salt_subsample_frac",
            ),
            (
                add_hydrophobic_condition,
                C_HPHOB,
                False,
                True,
                True,
                "p_add_hydrophobic_condition",
                "p_hydrophobic_subsample",
                "hydrophobic_subsample_frac",
            ),
            (
                add_metal_condition,
                C_MCOORD,
                MetalCoordEnum.NONE,
                MetalCoordEnum.METAL,
                MetalCoordEnum.COORDINATING,
                "p_add_metal_condition",
                "p_metal_subsample",
                "metal_subsample_frac",
            ),
        ],
    )
    def test_gates_masks_and_clears_ground_truth(
        self,
        deterministic_rng,
        protein_with_contacting_partner_chain,
        add_condition_fn,
        condition,
        none_value,
        active_value,
        partner_value,
        p_add_kwarg,
        p_subsample_kwarg,
        subsample_frac_kwarg,
    ):
        base_atom_array = protein_with_contacting_partner_chain
        n = base_atom_array.array_length()
        n_chain_a = int(np.sum(base_atom_array.chain_id == "A"))

        # Synthetic ground truth: the first 20 atoms of chain A carry a real interaction, but only
        # the first 10 are "motif" (sequence-conditioned) -- i.e., eligible to be revealed. Chain B
        # carries the complementary value throughout, so every chain-A annotation keeps a partner.
        ground_truth = np.full(n, none_value)
        ground_truth[:20] = active_value
        ground_truth[n_chain_a:] = partner_value
        condition.set_annotation(base_atom_array, ground_truth)

        motif_mask = np.zeros(n, dtype=bool)
        motif_mask[:10] = True
        C_SEQ.set_annotation(
            base_atom_array,
            np.where(
                motif_mask,
                C_SEQ.ground_truth_value(base_atom_array),
                C_SEQ.default_annotation(base_atom_array),
            ),
        )

        # p_add=0.0 -> ground truth must be fully cleared, even though it was present on the array.
        data = {"atom_array": base_atom_array.copy(), "task": {}}
        result = add_condition_fn(data, **{p_add_kwarg: 0.0, p_subsample_kwarg: 0.0})
        assert np.all(condition.annotation(result["atom_array"]) == none_value)

        # p_add=1.0, no subsampling -> only the motif-masked subset of ground truth is revealed.
        data = {"atom_array": base_atom_array.copy(), "task": {}}
        result = add_condition_fn(data, **{p_add_kwarg: 1.0, p_subsample_kwarg: 0.0})
        revealed = condition.annotation(result["atom_array"])
        assert np.all(revealed[:10] == active_value)
        assert np.all(revealed[10:] == none_value)

        # p_add=1.0 with subsampling forced -> a strict subset of the 10 motif atoms survives.
        data = {"atom_array": base_atom_array.copy(), "task": {}}
        result = add_condition_fn(data, **{p_add_kwarg: 1.0, p_subsample_kwarg: 1.0, subsample_frac_kwarg: 0.5})
        subsampled = condition.annotation(result["atom_array"])
        # keep_count = int(10 * 0.5) = 5, and survivors must still be within the motif region.
        assert np.sum(subsampled != none_value) == 5
        assert np.all(subsampled[10:] == none_value)

    def test_clears_condition_when_no_partner_survives(self, deterministic_rng, simple_protein):
        """A crop that took the partner chain leaves nothing to reveal, so nothing is flagged."""
        atom_array = simple_protein["atom_array"].copy()
        n = atom_array.array_length()

        ground_truth = np.full(n, HBondEnum.NONE)
        ground_truth[:20] = HBondEnum.DONOR
        C_HB.set_annotation(atom_array, ground_truth)

        motif_mask = np.zeros(n, dtype=bool)
        motif_mask[:10] = True
        C_SEQ.set_annotation(
            atom_array,
            np.where(motif_mask, C_SEQ.ground_truth_value(atom_array), C_SEQ.default_annotation(atom_array)),
        )

        data = {"atom_array": atom_array, "task": {}}
        result = add_hbond_condition(data, p_add_hbond_condition=1.0, p_hbond_subsample=0.0)

        assert np.all(C_HB.annotation(result["atom_array"]) == HBondEnum.NONE)
        assert "added_hbond_condition" not in result["task"]

    def test_union_degrades_to_the_role_that_keeps_a_partner(
        self, deterministic_rng, protein_with_contacting_partner_chain
    ):
        """``BOTH`` stays a union only while each of its two roles keeps a partner."""
        atom_array = protein_with_contacting_partner_chain
        n = atom_array.array_length()
        n_chain_a = int(np.sum(atom_array.chain_id == "A"))

        ground_truth = np.full(n, HBondEnum.NONE)
        ground_truth[:10] = HBondEnum.BOTH
        ground_truth[n_chain_a:] = HBondEnum.ACCEPTOR  # the partner chain accepts, but never donates
        C_HB.set_annotation(atom_array, ground_truth)

        motif_mask = np.zeros(n, dtype=bool)
        motif_mask[:10] = True
        C_SEQ.set_annotation(
            atom_array,
            np.where(motif_mask, C_SEQ.ground_truth_value(atom_array), C_SEQ.default_annotation(atom_array)),
        )

        data = {"atom_array": atom_array, "task": {}}
        result = add_hbond_condition(data, p_add_hbond_condition=1.0, p_hbond_subsample=0.0)

        # The donor half of BOTH keeps its acceptor partner; the acceptor half has no donor left.
        revealed = C_HB.annotation(result["atom_array"])
        assert np.all(revealed[:10] == HBondEnum.DONOR)
        assert np.all(revealed[10:] == HBondEnum.NONE)


@pytest.mark.fast
class TestIndexGroupCondition:
    """Tests for the C_IDXGRP (IndexGroup) condition class."""

    def test_default_behavior(self, simple_protein):
        """Test default mask and annotation behavior with rigorous value checks."""
        # Test mask: all atoms should have an index group
        simple_protein = simple_protein["atom_array"]
        mask = C_IDXGRP.mask(simple_protein, default="generate")
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
        C_IDX.set_annotation(simple_protein, idx_mask)
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
        C_IDX.set_annotation(atom_array, idx_mask)

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
