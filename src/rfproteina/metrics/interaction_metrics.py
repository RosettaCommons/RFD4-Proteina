import logging
from collections import defaultdict
from typing import Any

import numpy as np
from atomworks.ml.transforms.interactions import AnnotateInteractions
from biotite.structure import AtomArray, AtomArrayStack

from rfproteina.conditions import (
    C_HB,
    C_HPHOB,
    C_MCOORD,
    C_PICATION,
    C_PISTACK,
    C_SALT,
    HBondEnum,
    PiCationEnum,
)
from rfproteina.metrics.metric import Metric
from rfproteina.utils.chain import get_chain_of_interest

logger = logging.getLogger(__name__)

_ADHERENCE_KEYS = [
    "hbond_donor_conditioning_adherence",
    "hbond_acceptor_conditioning_adherence",
    "hbond_simultaneous_donor_acceptor_conditioning_adherence",
    "pistack_conditioning_adherence",
    "pication_aromatic_conditioning_adherence",
    "pication_cation_conditioning_adherence",
    "saltbridge_conditioning_adherence",
    "hydrophobic_conditioning_adherence",
    "metal_coord_conditioning_adherence",
]
_ATOM_COUNT_KEYS = [
    "hbond_atom_count_coi",
    "pistack_residue_count_coi",
    "pication_residue_count_coi",
    "saltbridge_residue_count_coi",
    "hydrophobic_atom_count_coi",
    "metal_coord_atom_count_coi",
]
_PAIR_COUNT_KEYS = [
    "hbond_atom_pair_count_coi",
    "pistack_residue_pair_count_coi",
    "pication_residue_pair_count_coi",
    "saltbridge_residue_pair_count_coi",
    "hydrophobic_atom_pair_count_coi",
    "metal_coord_atom_pair_count_coi",
]
_ALL_KEYS = _ADHERENCE_KEYS + _ATOM_COUNT_KEYS + _PAIR_COUNT_KEYS


def _count_pairs_involving_coi(ann2d_pairs: np.ndarray, coi_indices: np.ndarray) -> int:
    """Count interaction pairs where at least one atom belongs to the chain of interest.

    Args:
        ann2d_pairs: ``(N, 2)`` array of atom index pairs from a 2D annotation.
        coi_indices: Atom indices belonging to the chain of interest.

    Returns:
        Number of pairs where at least one atom is in the chain of interest.
    """
    if len(ann2d_pairs) == 0:
        return 0
    a_in_coi = np.isin(ann2d_pairs[:, 0], coi_indices)
    b_in_coi = np.isin(ann2d_pairs[:, 1], coi_indices)
    return int((a_in_coi | b_in_coi).sum())


def _chain_identifier(annotated: AtomArray) -> np.ndarray:
    """Return ``chain_iid`` if present, else ``chain_id``."""
    if "chain_iid" in annotated.get_annotation_categories():
        return annotated.get_annotation("chain_iid")
    return annotated.chain_id


def _count_residue_pairs_involving_coi(
    ann2d_pairs: np.ndarray,
    coi_indices: np.ndarray,
    annotated: AtomArray,
) -> int:
    """Count unique residue-level pairs where at least one residue is in the chain of interest.

    Deduplicates atom-level pairs (e.g. all ring-atom x cation-atom cross-products) down to
    unique ``(chain_identifier, res_id)`` pairs, matching PLIP's group-level counting.

    Args:
        ann2d_pairs: ``(N, 2)`` array of atom index pairs from a 2D annotation.
        coi_indices: Atom indices belonging to the chain of interest.
        annotated: Annotated AtomArray, used to map atom indices to residue keys.

    Returns:
        Number of unique residue pairs where at least one residue is in the chain of interest.
    """
    if len(ann2d_pairs) == 0:
        return 0
    chain_ids = _chain_identifier(annotated)
    res_ids = annotated.res_id
    coi_residues = {(chain_ids[i], res_ids[i]) for i in coi_indices}
    residue_pairs: set[tuple[object, ...]] = set()
    for ai, bi in ann2d_pairs:
        key_a = (chain_ids[ai], res_ids[ai])
        key_b = (chain_ids[bi], res_ids[bi])
        if key_a in coi_residues or key_b in coi_residues:
            residue_pairs.add((min(key_a, key_b), max(key_a, key_b)))
    return len(residue_pairs)


def _get_residue_count(annotated: AtomArray, coi_indices: np.ndarray, annot_key: str) -> float:
    """Count unique residues in the COI with any atom carrying a nonzero annotation value.

    Args:
        annotated: Annotated AtomArray from AnnotateInteractions.
        coi_indices: Atom indices belonging to the chain of interest.
        annot_key: Name of the 1D annotation to query.

    Returns:
        Residue count as float, or ``nan`` if the annotation is unavailable.
    """
    if annot_key not in annotated.get_annotation_categories():
        return np.nan
    vals = annotated.get_annotation(annot_key)[coi_indices]
    active = coi_indices[vals != 0]
    chain_ids = _chain_identifier(annotated)
    res_ids = annotated.res_id
    return float(len({(chain_ids[i], res_ids[i]) for i in active}))


def _get_residue_pair_count(annotated: AtomArray, coi_indices: np.ndarray, annot_key: str) -> float:
    """Count unique residue-level interaction pairs involving the chain of interest.

    Args:
        annotated: Annotated AtomArray from AnnotateInteractions.
        coi_indices: Atom indices belonging to the chain of interest.
        annot_key: Name of the 2D annotation to query.

    Returns:
        Residue pair count as float, or ``nan`` if the annotation is unavailable.
    """
    if annot_key not in annotated.get_annotation_2d_categories():
        return np.nan
    ann2d = annotated.get_annotation_2d(annot_key)
    return float(_count_residue_pairs_involving_coi(ann2d.pairs, coi_indices, annotated))


def _get_atom_count(annotated: AtomArray, coi_indices: np.ndarray, annot_key: str) -> float:
    """Return atom count for chain-of-interest atoms with a nonzero annotation value.

    Args:
        annotated: Annotated AtomArray from AnnotateInteractions.
        coi_indices: Atom indices belonging to the chain of interest.
        annot_key: Name of the 1D annotation to query.

    Returns:
        Count as float, or ``nan`` if the annotation is unavailable.
    """
    if annot_key not in annotated.get_annotation_categories():
        return np.nan
    vals = annotated.get_annotation(annot_key)[coi_indices]
    return float((vals != 0).sum())


def _get_pair_count(annotated: AtomArray, coi_indices: np.ndarray, annot_key: str) -> float:
    """Return pair count for interchain interactions involving the chain of interest.

    Args:
        annotated: Annotated AtomArray from AnnotateInteractions.
        coi_indices: Atom indices belonging to the chain of interest.
        annot_key: Name of the 2D annotation to query.

    Returns:
        Count as float, or ``nan`` if the annotation is unavailable.
    """
    if annot_key not in annotated.get_annotation_2d_categories():
        return np.nan
    ann2d = annotated.get_annotation_2d(annot_key)
    return float(_count_pairs_involving_coi(ann2d.pairs, coi_indices))


def get_interaction_metrics(atom_array: AtomArray) -> dict[str, float]:
    """Compute interaction adherence, atom counts, and pair counts for each interaction type.

    Runs :class:`~atomworks.ml.transforms.interactions.AnnotateInteractions` on the
    predicted structure (interchain only), then for each interaction type computes:

    - Adherence: fraction of conditioned interactions reproduced in the output.
      Omitted for an interaction type when no conditioning of that type was provided.
    - Count: for hbond/hydrophobic/metal_coord, ``_atom_count_coi`` is the number
      of COI atoms involved; for pistack/pication/saltbridge, ``_residue_count_coi``
      is the number of unique COI residues involved (deduplicates atom-level
      cross-products to group-level, matching PLIP).
    - Pair count: for hbond/hydrophobic/metal_coord, ``_atom_pair_count_coi`` is the
      number of atom pairs; for pistack/pication/saltbridge, ``_residue_pair_count_coi``
      is the number of unique residue pairs where at least one residue is in the COI.

    Interaction types assessed: hydrogen bonds (``C_HB``), pi-stacking
    (``C_PISTACK``; from ``interaction_pistacking`` 1D annotation, excludes pi-cation),
    pi-cation (``C_PICATION``; aromatic/cation roles), salt bridges (``C_SALT``),
    hydrophobic contacts (``C_HPHOB``), metal coordination (``C_MCOORD``).

    Args:
        atom_array: Predicted AtomArray with interaction conditioning annotations.

    Returns:
        Dict with atom-count, pair-count, and (if applicable) adherence metrics for each interaction type.
        Returns all ``nan`` if interaction annotation fails.
    """
    annotator = AnnotateInteractions(inter_chain_only=True, silence_rdkit_warnings=True, hbond_model="rosetta")
    try:
        data = annotator.forward({"atom_array": atom_array})
        annotated = data["atom_array"]
    except Exception:
        logger.warning("AnnotateInteractions failed", exc_info=True)
        return {k: np.nan for k in _ALL_KEYS}

    annot_cats = set(annotated.get_annotation_categories())

    chain_of_interest, _ = get_chain_of_interest(atom_array)
    chain_ids = _chain_identifier(annotated)
    coi_mask = chain_ids == chain_of_interest
    coi_indices = np.where(coi_mask)[0]

    output_metrics: dict[str, float] = {}

    # --- HBond ---
    # Conditioning: HBondEnum (NONE=0, DONOR=1, ACCEPTOR=2, BOTH=3)
    # Generated 1D: interaction_hbond_role  (NONE=0, DONOR=1, ACCEPTOR=2, BOTH=3)
    # Generated 2D: interaction_hbond       pairs
    hb_cond_mask = C_HB.mask(atom_array, default="generate")
    if hb_cond_mask.any() and "interaction_hbond_role" in annot_cats:
        hb_cond = C_HB.annotation(atom_array, default="generate")
        hb_gen = annotated.get_annotation("interaction_hbond_role")
        cond_vals = hb_cond[hb_cond_mask]
        gen_vals = hb_gen[hb_cond_mask]
        # We count it as donor or acceptor adherence even if conditioned with BOTH
        cond_both = cond_vals == HBondEnum.BOTH
        gen_both = gen_vals == HBondEnum.BOTH
        cond_donor = (cond_vals == HBondEnum.DONOR) | cond_both
        cond_acceptor = (cond_vals == HBondEnum.ACCEPTOR) | cond_both
        gen_donor = (gen_vals == HBondEnum.DONOR) | gen_both
        gen_acceptor = (gen_vals == HBondEnum.ACCEPTOR) | gen_both
        correct_donor = (cond_donor & gen_donor).astype(float)
        correct_acceptor = (cond_acceptor & gen_acceptor).astype(float)
        if cond_donor.sum() > 0:
            output_metrics["hbond_donor_conditioning_adherence"] = float(correct_donor.sum() / cond_donor.sum())
        if cond_acceptor.sum() > 0:
            output_metrics["hbond_acceptor_conditioning_adherence"] = float(
                correct_acceptor.sum() / cond_acceptor.sum()
            )

        # Separately, record how well the model adheres to making simultaneous donor/acceptor interactions
        correct_both = cond_both & gen_both
        if cond_both.sum() > 0:
            output_metrics["hbond_simultaneous_donor_acceptor_conditioning_adherence"] = float(
                correct_both.sum() / cond_both.sum()
            )

    output_metrics["hbond_atom_count_coi"] = _get_atom_count(annotated, coi_indices, "interaction_hbond_role")
    output_metrics["hbond_atom_pair_count_coi"] = _get_pair_count(annotated, coi_indices, "interaction_hbond")

    # --- PiStacking ---
    # Conditioning: C_PISTACK mask (bool)
    # Generated 2D: interaction_pistacking pairs → derive per-atom bool (excludes pi-cation)
    pistack_cond_mask = C_PISTACK.mask(atom_array, default="generate")
    if "interaction_pistacking" in annot_cats:
        is_pistack = annotated.get_annotation("interaction_pistacking")
        if pistack_cond_mask.any():
            output_metrics["pistack_conditioning_adherence"] = float(is_pistack[pistack_cond_mask].mean())

    output_metrics["pistack_residue_count_coi"] = _get_residue_count(annotated, coi_indices, "interaction_pistacking")
    output_metrics["pistack_residue_pair_count_coi"] = _get_residue_pair_count(
        annotated, coi_indices, "interaction_pistacking"
    )

    # --- PiCation ---
    # Conditioning: PiCationEnum (NONE=0, AROMATIC=1, CATION=2, BOTH=3)
    # Generated 1D: interaction_pication_role (NONE=0, AROMATIC=1, CATION=2, BOTH=3)
    # Generated 2D: interaction_pication pairs
    pication_cond_mask = C_PICATION.mask(atom_array, default="generate")
    if pication_cond_mask.any() and "interaction_pication_role" in annot_cats:
        pication_cond = C_PICATION.annotation(atom_array, default="generate")
        pication_gen = annotated.get_annotation("interaction_pication_role")
        cond_vals = pication_cond[pication_cond_mask]
        gen_vals = pication_gen[pication_cond_mask]
        cond_both = cond_vals == PiCationEnum.BOTH
        gen_both = gen_vals == PiCationEnum.BOTH
        cond_aromatic = (cond_vals == PiCationEnum.AROMATIC) | cond_both
        cond_cation = (cond_vals == PiCationEnum.CATION) | cond_both
        gen_aromatic = (gen_vals == PiCationEnum.AROMATIC) | gen_both
        gen_cation = (gen_vals == PiCationEnum.CATION) | gen_both
        if cond_aromatic.sum() > 0:
            output_metrics["pication_aromatic_conditioning_adherence"] = float(
                (cond_aromatic & gen_aromatic).sum() / cond_aromatic.sum()
            )
        if cond_cation.sum() > 0:
            output_metrics["pication_cation_conditioning_adherence"] = float(
                (cond_cation & gen_cation).sum() / cond_cation.sum()
            )

    output_metrics["pication_residue_count_coi"] = _get_residue_count(
        annotated, coi_indices, "interaction_pication_role"
    )
    output_metrics["pication_residue_pair_count_coi"] = _get_residue_pair_count(
        annotated, coi_indices, "interaction_pication"
    )

    # --- SaltBridge ---
    # Conditioning: SaltBridgeEnum (NONE=0, POSITIVE=1, NEGATIVE=2)
    # Generated 1D: interaction_charged_role (NONE=0, POSITIVE=1, NEGATIVE=2)
    # Generated 2D: interaction_saltbridge   pairs
    salt_cond_mask = C_SALT.mask(atom_array, default="generate")
    if salt_cond_mask.any() and "interaction_charged_role" in annot_cats:
        salt_cond = C_SALT.annotation(atom_array, default="generate")
        salt_gen = annotated.get_annotation("interaction_charged_role")
        adherent = salt_gen[salt_cond_mask] == salt_cond[salt_cond_mask]
        output_metrics["saltbridge_conditioning_adherence"] = float(adherent.mean())

    output_metrics["saltbridge_residue_count_coi"] = _get_residue_count(
        annotated, coi_indices, "interaction_charged_role"
    )
    output_metrics["saltbridge_residue_pair_count_coi"] = _get_residue_pair_count(
        annotated, coi_indices, "interaction_saltbridge"
    )

    # --- Hydrophobic ---
    # Conditioning: C_HPHOB mask (bool)
    # Generated 1D: interaction_hydrophobic (bool)
    # Generated 2D: interaction_hydrophobic pairs
    hphob_cond_mask = C_HPHOB.mask(atom_array, default="generate")
    if hphob_cond_mask.any() and "interaction_hydrophobic" in annot_cats:
        hphob_gen = annotated.get_annotation("interaction_hydrophobic")
        output_metrics["hydrophobic_conditioning_adherence"] = float(hphob_gen[hphob_cond_mask].mean())

    output_metrics["hydrophobic_atom_count_coi"] = _get_atom_count(annotated, coi_indices, "interaction_hydrophobic")
    output_metrics["hydrophobic_atom_pair_count_coi"] = _get_pair_count(
        annotated, coi_indices, "interaction_hydrophobic"
    )

    # --- MetalCoordination ---
    # Conditioning: MetalCoordEnum (NONE=0, METAL=1, COORDINATING=2)
    # Generated 1D: interaction_metal_role (NONE=0, METAL=1, COORDINATING=2)
    # Generated 2D: interaction_metal      pairs
    mcoord_cond_mask = C_MCOORD.mask(atom_array, default="generate")
    if mcoord_cond_mask.any() and "interaction_metal_role" in annot_cats:
        mcoord_cond = C_MCOORD.annotation(atom_array, default="generate")
        mcoord_gen = annotated.get_annotation("interaction_metal_role")
        adherent = mcoord_gen[mcoord_cond_mask] == mcoord_cond[mcoord_cond_mask]
        output_metrics["metal_coord_conditioning_adherence"] = float(adherent.mean())

    output_metrics["metal_coord_atom_count_coi"] = _get_atom_count(annotated, coi_indices, "interaction_metal_role")
    output_metrics["metal_coord_atom_pair_count_coi"] = _get_pair_count(annotated, coi_indices, "interaction_metal")

    return output_metrics


class InterchainInteractionMetric(Metric):
    """Compute adherence to interaction conditioning and interchain interaction counts.

    For each interaction type (hydrogen bond, pi-stacking, pi-cation, salt bridge,
    hydrophobic contact, metal coordination), reports:

    - ``*_conditioning_adherence``: fraction of conditioned interactions reproduced
      in the generated output. Omitted from the output if no conditioning of that
      type was provided.
    - ``*_atom_count_coi`` (hbond, hydrophobic, metal_coord): atoms in the COI
      involved in interchain interactions of that type.
    - ``*_residue_count_coi`` (pistack, pication, saltbridge): unique residues in
      the COI involved in interchain interactions of that type (group-level, matches PLIP).
    - ``*_atom_pair_count_coi`` (hbond, hydrophobic, metal_coord): interchain atom
      pairs where at least one atom is in the COI.
    - ``*_residue_pair_count_coi`` (pistack, pication, saltbridge): unique interchain
      residue pairs where at least one residue is in the COI (group-level, matches PLIP).

    All interaction detection is restricted to interchain contacts only.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array_or_stack": ("predicted_atom_array",),
        }

    def compute(self, atom_array_or_stack: AtomArray | AtomArrayStack | list[AtomArray]) -> dict[str, Any]:
        out_dict: dict[str, list] = defaultdict(list)

        if isinstance(atom_array_or_stack, AtomArray):
            atom_array_list = [atom_array_or_stack]
        elif isinstance(atom_array_or_stack, AtomArrayStack):
            atom_array_list = list(atom_array_or_stack)
        elif isinstance(atom_array_or_stack, list):
            atom_array_list = atom_array_or_stack
        else:
            raise ValueError(
                f"InterchainInteractionMetric expects AtomArray, AtomArrayStack, or list of "
                f"AtomArrays, but got {type(atom_array_or_stack)}"
            )

        for atom_array in atom_array_list:
            metrics = get_interaction_metrics(atom_array)
            for k, v in metrics.items():
                out_dict[k].append(v)

        return {k: np.nanmean(v) for k, v in out_dict.items()}
