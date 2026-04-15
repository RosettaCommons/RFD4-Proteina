"""Library of validation checks for train pipeline invariants.

An "invariant" in this context is a property that remains true across:
- All input structures (proteins, nucleic acids, ligands, complexes)
- All design tasks (unconditional, inverse_folding, motif_scaffold, etc.)
- All pipeline configurations (with/without cropping, different seeds, etc.)

Each check function takes pipeline output and validates a specific invariant.
Raises AssertionError with descriptive message if the check fails.
"""

import numpy as np
import torch
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_IDX
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
from atomworks.ml.transforms._checks import check_atom_array_annotation
from atomworks.ml.utils.token import get_token_starts

# =============================================================================
# Utility Functions
# =============================================================================


def _get_atom_array(output: dict) -> AtomArrayPlus:
    """Extract AtomArray from pipeline output."""
    return as_atom_array_plus(output["atom_array"])


def _get_feats(output: dict) -> dict:
    """Extract features dict from pipeline output."""
    return output["feats"]


def _get_torch_tensor(feats: dict, key: str) -> torch.Tensor | None:
    """Extract torch tensor from feats dict"""
    assert key in feats, f"Required feature '{key}' not found in feats"
    value = feats[key]
    assert torch.is_tensor(value), f"Feature '{key}' is not a torch tensor"
    return value


def _has_annotation(output: dict, annotation: str) -> bool:
    """Check if atom array has an annotation."""
    aa = _get_atom_array(output)
    return annotation in aa.get_annotation_categories()


# =============================================================================
# Coordinate Conditioning & Centering
# =============================================================================


def check_coordinates_centered_on_conditioned_atoms(output: dict) -> None:
    """Verify coordinates are centered on coordinate-conditioned atom centroid.

    Why: We make the choice to center coordinates on conditioned atoms to avoid
    leaking the centroid of the ground truth. We thus may have a drifting
    centroid throughout the denoising process, but this is the lesser evil.
    """
    # Check 1: Verify consistency on AtomArrayPlus annotations
    aa = _get_atom_array(output)
    check_atom_array_annotation(output, required=["coord_to_be_noised"])

    crd_mask = C_CRD.mask(aa, default="generate")
    if not crd_mask.any():
        # No conditioned atoms, skip this check
        return

    coords = aa.coord_to_be_noised
    centroid = np.mean(coords[crd_mask], axis=0)
    centroid_norm = np.linalg.norm(centroid)

    assert centroid_norm < 1e-3, (
        f"Coordinates not centered on conditioned atoms. " f"Centroid norm: {centroid_norm:.2e}"
    )

    # Check 2: Verify consistency on feats tensor (token-level atom37 format)
    feats = _get_feats(output)
    coords_token = _get_torch_tensor(feats, "coords")  # [n_tokens, 37, 3]
    crd_mask_token = _get_torch_tensor(feats, "mask_coordinate_1_atom")  # [n_tokens, 37]
    atom37_mask = _get_torch_tensor(feats, "atom37_mask")  # [n_tokens, 37]

    # Get valid coordinate-conditioned atoms
    valid_and_conditioned = atom37_mask & crd_mask_token  # [n_tokens, 37]

    if not valid_and_conditioned.any():
        return  # No conditioned atoms in feats

    # Extract all conditioned atom coordinates
    coords_flat = coords_token[valid_and_conditioned]  # [n_conditioned, 3]
    centroid_feats = coords_flat.mean(dim=0).cpu().numpy()  # [3]
    centroid_norm_feats = np.linalg.norm(centroid_feats)

    assert centroid_norm_feats < 1e-3, (
        f"Coordinates not centered on conditioned atoms (feats tensor). " f"Centroid norm: {centroid_norm_feats:.2e}"
    )


def check_coordinates_not_nan(output: dict) -> None:
    """Verify coordinates are valid in feats tensor (actual model input).

    Validates:
    1. Valid atoms (atom37_mask=True) have no NaN/inf values
    2. Invalid atoms (atom37_mask=False) have correct padding value (0.0)

    Why: Model requires valid coordinate inputs for training and inference.
    """
    feats = _get_feats(output)

    coords = _get_torch_tensor(feats, "coords")  # [n_tokens, 37, 3]
    atom37_mask = _get_torch_tensor(feats, "atom37_mask")  # [n_tokens, 37]

    # Check 1: Valid atoms have no NaN/inf
    if atom37_mask.any():
        coords_valid = coords[atom37_mask]  # [n_valid_atoms, 3]
        has_nan = torch.isnan(coords_valid).any()
        has_inf = torch.isinf(coords_valid).any()

        assert not has_nan, "Found NaN in valid atom coordinates (model input)"
        assert not has_inf, "Found infinite values in valid atom coordinates (model input)"

    # Check 2: Invalid atoms (padding) have correct default value
    # From ATOM37_FEATURES: ("coords", (3,), 0.0) - padding value is 0.0
    invalid_mask = ~atom37_mask
    if invalid_mask.any():
        coords_invalid = coords[invalid_mask]  # [n_invalid_atoms, 3]
        expected_padding = 0.0

        assert (coords_invalid == expected_padding).all(), (
            f"Found {(coords_invalid != expected_padding).any(dim=1).sum().item()} "
            f"invalid atom slots with non-zero coordinates. "
            f"Padding slots should have coords={expected_padding}"
        )


def check_centroid_feature_accuracy(output: dict) -> None:
    """Verify centroid feature matches actual centroid of non-coordinate-conditioned atoms."""
    aa = _get_atom_array(output)
    feats = _get_feats(output)
    check_atom_array_annotation(output, required=["coord_to_be_noised"])

    feature_name = "centroid_of_atoms_without_coordinate_conditioning"
    centroid_feature = _get_torch_tensor(feats, feature_name)

    # Compute expected centroid from atom array annotations (numpy)
    crd_mask = C_CRD.mask(aa, default="generate")
    not_crd_mask = ~crd_mask
    coords = aa.coord_to_be_noised

    if not_crd_mask.any():
        expected_centroid = np.nanmean(coords[not_crd_mask], axis=0)
    else:
        # All atoms are conditioned; centroid should be NaN or special value
        expected_centroid = np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    # Convert to numpy for comparison
    centroid_feature_np = centroid_feature.cpu().numpy()

    # Compare (handle NaN case)
    if np.isnan(expected_centroid).all() and np.isnan(centroid_feature_np).all():
        return  # Both NaN, OK

    diff = np.linalg.norm(centroid_feature_np - expected_centroid)
    assert diff < 1e-4, f"Centroid feature doesn't match computed centroid. " f"Difference: {diff:.2e}"


# =============================================================================
# Sequence Conditioning & Element / Atom Name Leakage
# =============================================================================


def check_sequence_conditioning_with_res_types(output: dict) -> None:
    """Verify residue_type encoding is correct for both conditioned and non-conditioned tokens.

    Validates:
    1. Ground truth residue_type always has valid encoding
       - Never contains <M> (mask) token
       - Can contain <A> (atomized) token, which must match atomize annotation
    2. Sequence-conditioned tokens: condition_sequence matches ground truth
    3. Non-sequence-conditioned tokens: condition_sequence has mask token (<M>)
    4. Atomized tokens:
       a. Atomize annotation matches token_type == TokenType.ATOMIZED
       b. Ground truth residue_type has <A> token (verified in check 1)
       c. condition_sequence has <A> token

    Why: Prevents information leakage while ensuring model has necessary info.
    """
    from atomworks.ml.conditions import C_SEQ

    aa = _get_atom_array(output)
    feats = _get_feats(output)

    # Get token-level features
    seq_mask = _get_torch_tensor(feats, "mask_sequence_1_residue")  # [n_tokens]
    res_type_ground_truth = _get_torch_tensor(feats, "residue_type")  # [n_tokens]
    res_type_conditioned = _get_torch_tensor(feats, C_SEQ.full_name)  # [n_tokens] - "condition_sequence_1_residue"
    token_type = _get_torch_tensor(feats, "token_type")  # [n_tokens]

    token_starts = get_token_starts(aa)

    # Get special token indices from UNIFIED_ATOM37_ENCODING
    mask_token_idx = UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]  # Mask token for non-seq-conditioned
    atomized_token_idx = UNIFIED_ATOM37_ENCODING.token_to_idx["<A>"]  # Atomized token
    max_valid_idx = UNIFIED_ATOM37_ENCODING.n_tokens

    # Import TokenType enum for validation
    from proteinfoundation.datapipes.pipeline import TokenType

    # Check 1: Ground truth residue_type always valid
    assert (res_type_ground_truth >= 0).all(), "Ground truth residue_type has negative values"
    assert (res_type_ground_truth < max_valid_idx).all(), f"Ground truth residue_type has values >= {max_valid_idx}"

    # Ground truth should never be mask token (mask is only for conditioning non-seq-conditioned tokens)
    assert (res_type_ground_truth != mask_token_idx).all(), (
        f"Ground truth residue_type contains <M> (mask) token at {(res_type_ground_truth == mask_token_idx).sum().item()} positions. "
        "Ground truth should never use mask token - this is only for conditioning."
    )

    # Ground truth CAN be atomized token (<A>) for atomized tokens - verify consistency
    has_atomized_in_gt = (res_type_ground_truth == atomized_token_idx).any()
    if has_atomized_in_gt and _has_annotation(output, "atomize"):
        # If ground truth has <A> tokens, they should match the atomize annotation
        is_atomized_token = aa.atomize[token_starts]
        is_atomized_torch = torch.from_numpy(is_atomized_token).to(res_type_ground_truth.device)
        atomized_mask_from_gt = res_type_ground_truth == atomized_token_idx

        assert torch.equal(is_atomized_torch, atomized_mask_from_gt), (
            f"Ground truth residue_type has <A> token but doesn't match atomize annotation. "
            f"GT has {atomized_mask_from_gt.sum().item()} <A> tokens, "
            f"atomize annotation has {is_atomized_torch.sum().item()} tokens."
        )

    # Check 2: Sequence-conditioned tokens should have conditioned = ground truth
    if seq_mask.any():
        res_gt_conditioned = res_type_ground_truth[seq_mask]
        res_cond_conditioned = res_type_conditioned[seq_mask]

        matches = res_gt_conditioned == res_cond_conditioned
        assert matches.all(), (
            f"Sequence-conditioned tokens should have condition_sequence == residue_type. "
            f"Found {(~matches).sum().item()} mismatches."
        )

    # Check 3: Non-sequence-conditioned tokens have <M> (mask) token
    # In UNIFIED encoding, all non-seq-conditioned tokens use <M> regardless of polymer type
    not_seq_mask = ~seq_mask
    if not_seq_mask.any():
        not_seq_cond_vals = res_type_conditioned[not_seq_mask]
        assert (not_seq_cond_vals == mask_token_idx).all(), (
            f"Non-seq-conditioned tokens should have condition_sequence={mask_token_idx} (<M>). "
            f"Found: {torch.unique(not_seq_cond_vals).tolist()}"
        )

    # Check 4: Atomized tokens consistency
    if _has_annotation(output, "atomize"):
        is_atomized_token = aa.atomize[token_starts]  # [n_tokens] from atom array
        is_atomized_from_token_type = (token_type == TokenType.ATOMIZED).cpu().numpy()  # [n_tokens] from feats

        # Check 4a: Atomization annotation matches token_type
        assert np.array_equal(is_atomized_token, is_atomized_from_token_type), (
            f"Mismatch between atomize annotation and token_type. "
            f"atomize annotation: {is_atomized_token.sum()} tokens, "
            f"token_type==ATOMIZED: {is_atomized_from_token_type.sum()} tokens"
        )

        # Check 4b: Atomized tokens have <A> token in condition_sequence
        if is_atomized_token.any():
            is_atomized_torch = torch.from_numpy(is_atomized_token).to(seq_mask.device)
            atomized_cond_vals = res_type_conditioned[is_atomized_torch]

            # Atomized tokens are always sequence-conditioned, so should have <A> token
            assert (atomized_cond_vals == atomized_token_idx).all(), (
                f"Atomized tokens should have condition_sequence={atomized_token_idx} (<A>). "
                f"Found: {torch.unique(atomized_cond_vals).tolist()}"
            )


def check_elements(output: dict) -> None:
    """Verify element encoding is correct based on conditioning in atom37 format.

    Validates:
    1. Non-seq-conditioned tokens: ALL atom slots have element=-1 (no leakage)
    2. Seq-conditioned tokens: valid atoms have element>=0, padding has element=-1

    Why: Element masking prevents feature leakage while providing necessary chemical info.
    """
    feats = _get_feats(output)

    element = _get_torch_tensor(feats, "element")  # [n_tokens, 37]
    atom37_mask = _get_torch_tensor(feats, "atom37_mask")  # [n_tokens, 37]
    seq_mask = _get_torch_tensor(feats, "mask_sequence_1_residue")  # [n_tokens]

    # Check 1: Non-sequence-conditioned tokens have NO element information (all -1)
    non_seq_mask = ~seq_mask
    if non_seq_mask.any():
        element_non_seq = element[non_seq_mask]  # [n_non_seq, 37]
        assert (element_non_seq == -1).all(), (
            f"Element leakage detected: {(element_non_seq != -1).sum().item()} "
            "non-seq-conditioned atom slots have element information. "
            "Should be -1 to prevent leakage."
        )

    # Check 2: Sequence-conditioned tokens have valid elements for VALID atoms only
    if seq_mask.any():
        element_seq = element[seq_mask]  # [n_seq, 37]
        atom37_mask_seq = atom37_mask[seq_mask]  # [n_seq, 37]

        # Valid atoms (atom37_mask=True) should have element >= 0
        valid_atoms = atom37_mask_seq
        if valid_atoms.any():
            element_valid = element_seq[valid_atoms]
            assert (element_valid >= 0).all(), (
                f"Found {(element_valid < 0).sum().item()} valid atoms in "
                "seq-conditioned tokens without element information (element = -1). "
                "All valid atoms should have elements."
            )

        # Invalid atoms (atom37_mask=False) should have element = -1 (padding)
        invalid_atoms = ~atom37_mask_seq
        if invalid_atoms.any():
            element_invalid = element_seq[invalid_atoms]
            assert (element_invalid == -1).all(), (
                f"Found {(element_invalid != -1).sum().item()} padded atom slots with "
                "element information. Padded slots should have element=-1."
            )


# =============================================================================
# Encoding Annotations (controls which tokens are passed through the autoencoder)
# =============================================================================


def check_can_be_encoded_requirements(output: dict) -> None:
    """Verify can_be_encoded matches RESIDUE_NAMES_TO_ENCODE from pipeline.

    Validates bidirectional correctness:
    1. Forward: If can_be_encoded=True → residue name must be in RESIDUE_NAMES_TO_ENCODE and not atomized
    2. Reverse: If residue name in RESIDUE_NAMES_TO_ENCODE and not atomized → must have can_be_encoded=True

    Why: Encoder is trained on residues in RESIDUE_NAMES_TO_ENCODE (AAs, and potentially RNA/DNA).
    """
    from proteinfoundation.datapipes.pipeline import RESIDUE_NAMES_TO_ENCODE

    aa = _get_atom_array(output)
    feats = _get_feats(output)

    # Get token-level feature
    can_be_encoded = _get_torch_tensor(feats, "can_be_encoded")  # [n_tokens]
    can_be_encoded_np = can_be_encoded.cpu().numpy()

    # Convert atom-level annotations to token-level
    token_starts = get_token_starts(aa)
    res_names = aa.res_name[token_starts]  # [n_tokens] - residue names

    # Determine atomization status (token-level)
    if _has_annotation(output, "atomize"):
        is_atomized_token = aa.atomize[token_starts]
    else:
        is_atomized_token = np.zeros(len(token_starts), dtype=bool)

    # Define eligible tokens: residue name in RESIDUE_NAMES_TO_ENCODE AND not atomized
    is_encodable_residue = np.array([name in RESIDUE_NAMES_TO_ENCODE for name in res_names])
    eligible = is_encodable_residue & ~is_atomized_token

    # Forward check: if encoded, must be eligible
    if can_be_encoded.any():
        invalid_encoded = can_be_encoded_np & ~eligible
        if invalid_encoded.any():
            invalid_names = res_names[invalid_encoded]
            raise AssertionError(
                f"Found {invalid_encoded.sum()} tokens with can_be_encoded=True that are not eligible. "
                f"Invalid residue names: {np.unique(invalid_names).tolist()}. "
                f"Only non-atomized residues in RESIDUE_NAMES_TO_ENCODE can be encoded."
            )

    # Reverse check: if eligible, must be encoded
    if eligible.any():
        not_encoded_but_eligible = eligible & ~can_be_encoded_np
        if not_encoded_but_eligible.any():
            missing_names = res_names[not_encoded_but_eligible]
            raise AssertionError(
                f"Found {not_encoded_but_eligible.sum()} eligible tokens with can_be_encoded=False. "
                f"Missing residue names: {np.unique(missing_names).tolist()}. "
                f"All non-atomized residues in RESIDUE_NAMES_TO_ENCODE must have can_be_encoded=True."
            )


def check_unresolved_tokens_not_lossed(output: dict) -> None:
    """Verify tokens with any unresolved atoms are not lossed.

    Why: Tokens with partial resolution lack ground truth for unresolved atoms.
    Expected: should_be_lossed should be False for any token with at least one
    unresolved atom (occupancy == 0).
    """
    aa = _get_atom_array(output)
    feats = _get_feats(output)

    # Get token-level feature (already token-level in atom37 format)
    should_be_lossed = _get_torch_tensor(feats, "should_be_lossed").cpu().numpy()  # [n_tokens]

    # Get atom-level data
    token_id_atom_lvl = aa.token_id  # [n_atoms]
    is_resolved = aa.occupancy > 0.0  # [n_atoms]

    # Get unique token IDs (sorted)
    unique_tokens = np.unique(token_id_atom_lvl)
    violations = []

    # Check each token
    for token_idx, token_id in enumerate(unique_tokens):
        token_mask = token_id_atom_lvl == token_id
        token_resolved = is_resolved[token_mask]

        # Check if token has any unresolved atoms
        has_any_unresolved = not token_resolved.all()

        # Check if token-level annotation says it should be lossed
        # Note: should_be_lossed is [n_tokens], indexed by position in sorted unique tokens
        token_should_be_lossed = should_be_lossed[token_idx]

        if has_any_unresolved and token_should_be_lossed:
            n_atoms = token_mask.sum()
            n_unresolved = (~token_resolved).sum()
            violations.append(
                {
                    "token_id": int(token_id),
                    "token_idx": int(token_idx),
                    "n_atoms": int(n_atoms),
                    "n_unresolved": int(n_unresolved),
                }
            )

    if violations:
        error_msg = (
            f"Found {len(violations)} tokens with unresolved atoms where should_be_lossed=True.\n"
            f"Tokens with ANY unresolved atoms (occupancy == 0) must have should_be_lossed=False.\n"
        )
        raise AssertionError(error_msg)


# =============================================================================
# Token Features
# =============================================================================


def check_af3_token_centers(output: dict) -> None:
    """Verify AF3 token centers are correctly annotated.

    Why: Model uses token centers for geometric computations.
    Expected: CA for std AAs (unless atomized), C1' for std NAs, single atom for atomized.
    """
    aa = _get_atom_array(output)
    check_atom_array_annotation(
        output,
        required=[
            "is_token_center_atom",
            "token_id",
            "atom_name",
            "is_standard_aa",
            "is_standard_rna",
            "is_standard_dna",
        ],
    )

    # Check if atomization is present
    has_atomize = _has_annotation(output, "atomize")

    # Check each token has exactly one center atom
    for token_id in np.unique(aa.token_id):
        token_mask = aa.token_id == token_id
        is_center = aa.is_token_center_atom[token_mask]

        n_centers = is_center.sum()
        assert n_centers == 1, f"Token {token_id} has {n_centers} center atoms, expected 1"

        # Get the center atom
        token_atoms = np.where(token_mask)[0]
        center_idx = token_atoms[is_center][0]

        # Check if this token is atomized (check if ANY atom in the token is atomized)
        is_atomized_token = has_atomize and aa.atomize[token_mask].any()

        # Check if this is a single-atom token (e.g., from cropping or atomization)
        is_single_atom_token = token_mask.sum() == 1

        # Check if this token is unindexed (motif/scaffold region with unknown positions)
        is_unindexed = not C_IDX.mask(aa, default="generate")[center_idx]

        # Verify center atom makes sense
        if is_atomized_token:
            # Atomized tokens: single atom per token, so any atom is fine
            pass
        elif is_single_atom_token:
            # Single-atom tokens (e.g., from cropping): any atom is fine
            # These can occur when cropping removes most atoms from a residue
            pass
        elif is_unindexed:
            # Unindexed tokens: partial residues are allowed for motif extraction
            # These represent regions being designed, so any atom can be the center
            pass
        elif aa.is_standard_aa[center_idx]:
            # Standard AA (indexed, not atomized, multi-atom): center should be CA
            assert aa.atom_name[center_idx] == "CA", (
                f"Token {token_id} (standard AA) has center atom " f"{aa.atom_name[center_idx]}, expected CA"
            )
        elif aa.is_standard_rna[center_idx] or aa.is_standard_dna[center_idx]:
            # Standard nucleotide: center should be C1'
            assert aa.atom_name[center_idx] == "C1'", (
                f"Token {token_id} (standard NA) has center atom " f"{aa.atom_name[center_idx]}, expected C1'"
            )
        # For other cases (ligands, etc.), the single atom should be the center


def check_token_level_features_consistent_within_tokens(output: dict) -> None:
    """Verify token-level features are consistent across all atoms within each token.

    Why: ConvertToAtom37Format takes the first atom's value for token-level features.
    If atoms within a token have different values, this silently loses information.
    """
    from proteinfoundation.datapipes.pipeline import ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL

    aa = _get_atom_array(output)

    for feat_name in ATOM_LEVEL_FEATURES_TO_CONVERT_TO_TOKEN_LEVEL:
        if feat_name not in aa.get_annotation_categories():
            continue

        feat_values = aa.get_annotation(feat_name)
        for token_id in np.unique(aa.token_id):
            token_mask = aa.token_id == token_id
            token_vals = feat_values[token_mask]

            assert (
                len(np.unique(token_vals)) == 1
            ), f"Feature '{feat_name}' has inconsistent values in token {token_id}: {np.unique(token_vals)}"


# =============================================================================
# Conditioning Consistency Checks
# =============================================================================


def check_conditioning_consistency(output: dict) -> None:
    """Verify conditioning masks are consistent at token level.

    Validates:
    1. Atomized tokens are sequence conditioned (element info must be provided)
    2. Fully coord-conditioned chains are also seq-conditioned

    Note:
        Atomized tokens may or may not be coordinate conditioned - design tasks control this.
        Non-standard AAs are by definition atomized, so no separate check is needed.
    """
    aa = _get_atom_array(output)
    feats = _get_feats(output)
    check_atom_array_annotation(output, required=["chain_iid"])

    # Get token-level conditioning masks
    seq_mask = _get_torch_tensor(feats, "mask_sequence_1_residue")  # [n_tokens]
    crd_mask = _get_torch_tensor(feats, "mask_coordinate_1_atom")  # [n_tokens, 37]
    atom37_mask = _get_torch_tensor(feats, "atom37_mask")  # [n_tokens, 37]
    chain_iid = _get_torch_tensor(feats, "chain_iid")  # [n_tokens]

    # Check if ALL valid atoms are conditioned per token
    is_fully_crd_conditioned = (~atom37_mask | crd_mask).all(dim=1)  # [n_tokens]

    # Convert atom-level annotations to token-level
    token_starts = get_token_starts(aa)

    # Check 1: Atomized tokens are sequence conditioned (element info must be provided)
    if _has_annotation(output, "atomize"):
        atomize_token = aa.atomize[token_starts]
        if atomize_token.any():
            atomize_torch = torch.from_numpy(atomize_token).to(seq_mask.device)
            atomized_seq = seq_mask[atomize_torch]

            assert atomized_seq.all(), f"Found {(~atomized_seq).sum().item()} atomized tokens not sequence conditioned"

    # Check 2: Fully coord-conditioned chains are also seq-conditioned
    chain_iid_np = chain_iid.cpu().numpy()
    seq_mask_np = seq_mask.cpu().numpy()
    crd_mask_np = is_fully_crd_conditioned.cpu().numpy()

    for chain_id in np.unique(chain_iid_np):
        chain_mask = chain_iid_np == chain_id

        # If entire chain is coordinate conditioned
        if crd_mask_np[chain_mask].all():
            # Then entire chain should also be sequence conditioned
            seq_in_chain = seq_mask_np[chain_mask]
            assert seq_in_chain.all(), (
                f"Chain {chain_id} is fully coordinate-conditioned but "
                f"has {(~seq_in_chain).sum()} tokens not sequence conditioned"
            )


# =============================================================================
# Small Molecule Featurization Checks
# =============================================================================
