import ast
from typing import Any

import numpy as np
import pandas as pd
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import (
    C_CRD,
    C_DIS,
    C_SEQ,
)
from atomworks.ml.transforms.sasa import calculate_atomwise_rasa
from biotite.structure import get_residue_count, spread_residue_wise
from scipy.spatial import cKDTree

from rfproteina.conditions import (
    C_CATH,
    C_HB,
    C_HPHOB,
    C_INTCRD,
    C_MCOORD,
    C_NCYS,
    C_PICATION,
    C_PISTACK,
    C_RAS,
    C_SALT,
    C_SEC,
    ConditionBase,
    HBondEnum,
    MetalCoordEnum,
    PiCationEnum,
    RasaEnum,
    SaltBridgeEnum,
)
from rfproteina.metrics.sse_metrics import annotate_secondary_structure
from rfproteina.transforms.utils.conditioning_utils import sample_island_tokens
from rfproteina.utils.common import true_with_prob


def add_rasa_condition(
    data: dict,
    p_add_rasa_condition: float = 0.0,
    probe_radius: float = 1.4,
    atom_radii: str | np.ndarray = "ProtOr",
    point_number: int = 100,
    p_include_non_ligand: float = 0.1,
    rasa_max: float = 0.2,
) -> dict:
    """Add binned RASA conditioning to an example.

    Args:
        data: Example dictionary containing an atom array.
        p_add_rasa_condition: Probability of adding this condition at all. Defaults to 0.0.
        probe_radius: Van der Waals radius of the probe in Angstrom. Defaults to
            1.4 for water.
        atom_radii: Atom radii used for the calculation. ``"ProtOr"`` excludes
            hydrogens and some ions or charged atoms. Defaults to ``"ProtOr"``.
        point_number: Number of points sampled by the Shrake-Rupley algorithm.
            Defaults to 100.
        p_include_non_ligand: Probability of conditioning non-ligand atoms.
            Defaults to 0.1.
        rasa_max: Maximum RASA value used to construct bins. Defaults to 0.2.

    Returns:
        The example dictionary with RASA annotations added to its atom array.
    """
    if not true_with_prob(p_add_rasa_condition):
        return data

    atom_array = as_atom_array_plus(data["atom_array"])

    ensure_annotations(atom_array, "is_ligand", "atomize")

    # Determine valid residues for rasa conditioning
    if np.random.rand() < p_include_non_ligand:
        seq_mask = C_SEQ.mask(atom_array, default="generate")
        crd_mask = C_CRD.mask(atom_array, default="generate") | C_INTCRD.mask(atom_array, default="generate")
        dis_mask = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
        inclusion_mask = seq_mask | crd_mask | dis_mask
    else:
        inclusion_mask = atom_array.is_ligand | atom_array.atomize

    rasa = calculate_atomwise_rasa(atom_array, probe_radius, atom_radii, point_number, atom_filter=inclusion_mask)

    # NOTE: Bin count is hard-coded at 3 to match the RasaEnum
    bin_edges = np.linspace(0, rasa_max, 3)  # e.g., [0.0, 0.1, 0.2]
    bins = np.digitize(rasa, bin_edges, right=False)  # bins in [0, 3] -- however, bin 0 would mean negative rasa!
    if (bins == 0).any():
        raise ValueError("Negative RASA value was detected. This indicates an error.")

    # Convert to RasaEnum
    enum_rasa_values = C_RAS.default_annotation(atom_array)
    enum_rasa_values[bins == 1] = RasaEnum.BURIED
    enum_rasa_values[bins == 2] = RasaEnum.INTERMEDIATE
    enum_rasa_values[bins == 3] = RasaEnum.EXPOSED
    enum_rasa_values[~inclusion_mask] = RasaEnum.UNSPECIFIED

    C_RAS.set_annotation(atom_array, enum_rasa_values.astype(int))

    data["atom_array"] = atom_array
    data.setdefault("task", {})["added_rasa_condition"] = True
    return data


def subsample_condition_enum(condition_array: np.ndarray, fraction: float, none_value: Any = 0) -> np.ndarray:
    """Randomly clear a subset of active entries in a condition array, keeping ``fraction`` of them.

    Args:
      condition_array: Array of condition values (boolean mask or integer enum).
      fraction: Fraction of active (non-``none_value``) entries to keep. Must be in ``(0, 1]``.
      none_value: Value denoting "no condition" for this array. Defaults to ``0``.

    Returns:
      A copy of ``condition_array`` with ``1 - fraction`` of its active entries reset to
      ``none_value``. At least one active entry is always kept.
    """
    if not (0 < fraction <= 1):
        raise ValueError("Fraction must be in the range (0, 1].")

    condition_array = condition_array.copy()
    active_indices = np.where(condition_array != none_value)[0]
    keep_count = max(1, int(len(active_indices) * fraction))

    np.random.shuffle(active_indices)
    condition_array[active_indices[keep_count:]] = none_value

    return condition_array


def _drop_annotations_without_cross_chain_partner(
    atom_array: Any,
    ground_truth: np.ndarray,
    stored: np.ndarray,
    none_value: Any,
    partners: dict[Any, Any],
    unions: dict[Any, tuple[Any, ...]],
    partner_cutoff: float,
    residue_level_values: tuple[Any, ...] = (),
) -> np.ndarray:
    """Drop two-body annotations whose interaction partner did not survive cropping.

    Args:
      ground_truth: Motif-masked values being revealed.
      stored: Full stored annotation on the crop, searched for partners.
      partners: Maps each base value to the value a partner atom must carry.
      unions: Describes enum values that combine multiple base values, e.g. ``HBondEnum.BOTH``.
      partner_cutoff: Distance cutoff (A) of the detector that produced the annotation.
      residue_level_values: Base values kept or dropped per residue rather than per atom. Used for the
        pi side of pi-stacking/pi-cation, where the interaction belongs to the ring as a whole and
        conditioning on part of a ring would be out of distribution.
    """
    chains = np.asarray(atom_array.chain_iid)
    coords = atom_array.coord
    residue_starts = get_residue_starts(atom_array, add_exclusive_stop=True)

    def carries(values: np.ndarray, base_value: Any) -> np.ndarray:
        """Atoms whose value is ``base_value``, or a union that includes it."""
        return np.isin(values, [base_value, *(u for u, bases in unions.items() if base_value in bases)])

    # Which atoms have a cross-chain atom carrying the complementary value within the cutoff.
    supported = {}
    for base_value, partner_value in partners.items():
        query = np.where(carries(ground_truth, base_value))[0]
        candidates = carries(stored, partner_value)
        mask = np.zeros(len(ground_truth), dtype=bool)
        for chain in np.unique(chains[query]):
            rows = query[chains[query] == chain]
            cross_chain = candidates & (chains != chain)
            if not cross_chain.any():
                continue
            tree = cKDTree(coords[cross_chain])
            mask[rows] = tree.query(coords[rows], k=1)[0] <= partner_cutoff
        if base_value in residue_level_values:
            # One supported atom carries its whole residue, but only over atoms that were already
            # annotated -- this widens nothing beyond what the motif mask revealed.
            residue_supported = np.logical_or.reduceat(mask, residue_starts[:-1])
            mask = spread_residue_wise(atom_array, residue_supported) & carries(ground_truth, base_value)
        supported[base_value] = mask

    kept = np.full_like(ground_truth, none_value)
    for base_value, mask in supported.items():
        kept[mask] = base_value
    # A union only stays a union while every value it combines still has a partner; otherwise the
    # loop above has already degraded it to the single value that does.
    for union, bases in unions.items():
        kept[np.logical_and.reduce([supported[base] for base in bases])] = union
    return kept


def _subset_ground_truth_interaction_condition(
    data: dict,
    condition: type[ConditionBase],
    none_value: Any,
    task_key: str,
    p_add_condition: float,
    p_subsample: float,
    subsample_frac: float,
    partners: dict[Any, Any],
    unions: dict[Any, tuple[Any, ...]],
    partner_cutoff: float,
    residue_level_values: tuple[Any, ...] = (),
) -> dict:
    """Subset a ground-truth interaction Condition already present on the AtomArray.

    The AtomArray carries the full, un-gated ground-truth interaction annotation. This either
    reveals a motif-masked, optionally subsampled view of it, or clears it back to its default so
    that the ground truth does not leak through when this modifier is not "applied".

    NOTE: Only meaningful on datasets loaded with ``load_standard_annotations: true``; elsewhere the
    annotation is absent and the "revealed" condition is empty.

    Args:
      data: Data dict containing atom_array.
      condition: The interaction ``ConditionBase`` subclass to subset (e.g. ``C_HB``).
      none_value: The "no interaction" value for this condition's dtype.
      task_key: Key set to ``True`` in ``data["task"]`` when the condition is applied.
      p_add_condition: Probability of revealing this condition at all.
      p_subsample: Probability of additionally subsampling the revealed condition.
      subsample_frac: Fraction of active entries to keep when subsampling.
      partners: See :py:func:`_drop_annotations_without_cross_chain_partner`.
      unions: See :py:func:`_drop_annotations_without_cross_chain_partner`.
      partner_cutoff: Distance cutoff (A) of the detector that produced the annotation.
      residue_level_values: See :py:func:`_drop_annotations_without_cross_chain_partner`.

    Returns:
      Updated data dict with ``condition`` set on the AtomArray.
    """
    atom_array = as_atom_array_plus(data["atom_array"])

    ground_truth = None
    if np.random.rand() < p_add_condition:
        seq_mask = C_SEQ.mask(atom_array, default="generate")
        crd_mask = C_CRD.mask(atom_array, default="generate")
        intcrd_mask = C_INTCRD.mask(atom_array, default="generate")
        dis_mask = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
        motif_mask = seq_mask | crd_mask | intcrd_mask | dis_mask

        stored = condition.annotation(atom_array, default="generate")
        ground_truth = stored.copy()
        ground_truth[~motif_mask] = none_value

        # The annotation predates cropping, so drop entries whose partner the crop removed.
        if np.any(ground_truth != none_value):
            ground_truth = _drop_annotations_without_cross_chain_partner(
                atom_array, ground_truth, stored, none_value, partners, unions, partner_cutoff, residue_level_values
            )

        # Nothing left to reveal (single-chain input, or the crop took every partner): fall
        # through to the "not applied" path rather than flagging an empty condition.
        if not np.any(ground_truth != none_value):
            ground_truth = None

    if ground_truth is None:
        atom_array.del_annotation(condition.mask_name)
        atom_array.del_annotation(condition.full_name)
        condition.set_mask(atom_array, condition.default_mask(atom_array))
        condition.set_annotation(atom_array, condition.default_annotation(atom_array))
        data["atom_array"] = atom_array
        return data

    if np.random.rand() < p_subsample:
        ground_truth = subsample_condition_enum(ground_truth, subsample_frac, none_value=none_value)

    condition.set_annotation(atom_array, ground_truth)
    condition.set_mask(atom_array, ground_truth != none_value)
    data.setdefault("task", {})[task_key] = True
    data["atom_array"] = atom_array
    return data


def add_hbond_condition(
    data: dict,
    p_add_hbond_condition: float = 0.0,
    p_hbond_subsample: float = 0.5,
    hbond_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth hydrogen-bond conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_HB,
        HBondEnum.NONE,
        "added_hbond_condition",
        p_add_hbond_condition,
        p_hbond_subsample,
        hbond_subsample_frac,
        partners={HBondEnum.DONOR: HBondEnum.ACCEPTOR, HBondEnum.ACCEPTOR: HBondEnum.DONOR},
        unions={HBondEnum.BOTH: (HBondEnum.DONOR, HBondEnum.ACCEPTOR)},
        partner_cutoff=4.1,
    )


def add_pistack_condition(
    data: dict,
    p_add_pistack_condition: float = 0.0,
    p_pistack_subsample: float = 0.5,
    pistack_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth pi-stacking conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_PISTACK,
        False,
        "added_pistack_condition",
        p_add_pistack_condition,
        p_pistack_subsample,
        pistack_subsample_frac,
        partners={True: True},
        unions={},
        # 5.5 A ring centre-to-centre maximum, plus a ring radius at each end.
        partner_cutoff=8.5,
        # Both sides are rings: keep or drop each one whole.
        residue_level_values=(True,),
    )


def add_pication_condition(
    data: dict,
    p_add_pication_condition: float = 0.0,
    p_pication_subsample: float = 0.5,
    pication_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth pi-cation conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_PICATION,
        PiCationEnum.NONE,
        "added_pication_condition",
        p_add_pication_condition,
        p_pication_subsample,
        pication_subsample_frac,
        partners={PiCationEnum.AROMATIC: PiCationEnum.CATION, PiCationEnum.CATION: PiCationEnum.AROMATIC},
        unions={PiCationEnum.BOTH: (PiCationEnum.AROMATIC, PiCationEnum.CATION)},
        # 6.0 A cation-to-ring-centre maximum, plus a ring radius.
        partner_cutoff=7.5,
        # The aromatic side is a ring: keep or drop it whole. The cation stays per-atom.
        residue_level_values=(PiCationEnum.AROMATIC,),
    )


def add_salt_condition(
    data: dict,
    p_add_salt_condition: float = 0.0,
    p_salt_subsample: float = 0.5,
    salt_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth salt-bridge conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_SALT,
        SaltBridgeEnum.NONE,
        "added_salt_condition",
        p_add_salt_condition,
        p_salt_subsample,
        salt_subsample_frac,
        partners={SaltBridgeEnum.POSITIVE: SaltBridgeEnum.NEGATIVE, SaltBridgeEnum.NEGATIVE: SaltBridgeEnum.POSITIVE},
        unions={},
        partner_cutoff=5.5,
    )


def add_hydrophobic_condition(
    data: dict,
    p_add_hydrophobic_condition: float = 0.0,
    p_hydrophobic_subsample: float = 0.5,
    hydrophobic_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth hydrophobic-contact conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_HPHOB,
        False,
        "added_hydrophobic_condition",
        p_add_hydrophobic_condition,
        p_hydrophobic_subsample,
        hydrophobic_subsample_frac,
        partners={True: True},
        unions={},
        partner_cutoff=4.0,
    )


def add_metal_condition(
    data: dict,
    p_add_metal_condition: float = 0.0,
    p_metal_subsample: float = 0.5,
    metal_subsample_frac: float = 0.5,
) -> dict:
    """Subset the ground-truth metal-coordination conditioning already present on the AtomArray."""
    return _subset_ground_truth_interaction_condition(
        data,
        C_MCOORD,
        MetalCoordEnum.NONE,
        "added_metal_condition",
        p_add_metal_condition,
        p_metal_subsample,
        metal_subsample_frac,
        partners={MetalCoordEnum.METAL: MetalCoordEnum.COORDINATING, MetalCoordEnum.COORDINATING: MetalCoordEnum.METAL},
        unions={},
        partner_cutoff=3.0,
    )


def add_secondary_structure_condition(
    data: dict[str, Any],
    p_add_secondary_structure_condition: float = 0.0,
    max_secondary_structure_frac_to_provide: float = 0.4,
    min_ss_island_len: int = 1,
    max_ss_island_len: int = 10,
    n_islands_max: int = 3,
) -> dict[str, Any]:
    """Add secondary structure conditioning.

    Uses P-SEA to annotate secondary structure, samples islands from residues
    with valid SSE annotations, and sets C_SEC condition. Sampling is done at
    residue granularity (not token granularity) so that residues spanning
    multiple tokens — e.g., atomized PTM residues — receive a uniform value, satisfying the
    ``Level.RESIDUE`` invariant on the secondary-structure condition.

    Args:
      data: Data dict containing atom_array.
      p_add_secondary_structure_condition: Probability of adding this condition at all. Defaults to 0.0.
      max_secondary_structure_frac_to_provide: Max fraction of valid SS residues
        to condition. Defaults to 0.4.
      min_ss_island_len: Minimum island length (in residues). Defaults to 1.
      max_ss_island_len: Maximum island length (in residues). Defaults to 10.
      n_islands_max: Maximum number of islands. Defaults to 3.

    Returns:
      Updated data dict with C_SEC conditioning.
    """
    if not true_with_prob(p_add_secondary_structure_condition):
        return data

    atom_array = as_atom_array_plus(data["atom_array"])

    # Step 1: Annotate with P-SEA (residue-uniform sse / sse_is_valid)
    atom_array = annotate_secondary_structure(
        atom_array, annotation_name="sse", is_valid_annotation_name="sse_is_valid"
    )

    # Step 2: Sample islands from eligible residues
    residue_starts = get_residue_starts(atom_array)
    residue_level_array = atom_array[residue_starts]

    # Eligible: residues with valid SSE annotations
    eligible_for_ss_info_mask = residue_level_array.sse_is_valid

    n_residues_with_valid_ss = eligible_for_ss_info_mask.sum()
    max_residues_with_ss_conditioning = int(np.ceil(n_residues_with_valid_ss * max_secondary_structure_frac_to_provide))

    where_to_show_ss = sample_island_tokens(
        n_residues_with_valid_ss,
        island_len_min=min_ss_island_len,
        island_len_max=max_ss_island_len,
        n_islands_min=1,
        n_islands_max=n_islands_max,
        max_length=max_residues_with_ss_conditioning,
    )

    # Convert to residue-level mask and spread to atoms
    residue_level_ss_mask = np.zeros(len(residue_starts), dtype=bool)
    residue_level_ss_mask[eligible_for_ss_info_mask] = where_to_show_ss
    ss_mask = spread_residue_wise(atom_array, residue_level_ss_mask)

    # Step 3: Set C_SEC annotation (SSEnum.NONE for unspecified, sse values where sampled)
    ss_annotation = np.full(atom_array.array_length(), 0, dtype=int)  # 0 is SSEnum.NONE
    ss_annotation[ss_mask] = atom_array.sse[ss_mask]
    C_SEC.set_annotation(atom_array, ss_annotation)

    data["atom_array"] = atom_array
    data.setdefault("task", {})["added_secondary_structure_condition"] = True
    return data


def add_not_cysteine_condition(
    data: dict[str, Any],
    p_add_not_cysteine_condition: float = 0.0,
    max_subsample_frac: float = 0.25,
) -> dict[str, Any]:
    """Mark a random subset of eligible (standard, non-CYS, non-sequence-conditioned) residues as "not cysteine".

    Always subsamples: marking every eligible residue would leave cysteines as the only
    unmarked ones, leaking their positions.

    Args:
      p_add_not_cysteine_condition: Probability of adding this condition at all. Defaults to 0.0.
      max_subsample_frac: Upper bound on the per-residue keep-fraction, drawn from
        Uniform(0, max_subsample_frac).
    """
    if not true_with_prob(p_add_not_cysteine_condition):
        return data

    atom_array = as_atom_array_plus(data["atom_array"])
    ensure_annotations(atom_array, "is_standard_aa")

    # Exclude sequence-conditioned residues (identity already leaked); sidechain-conditioned ones stay.
    seq_mask = C_SEQ.mask(atom_array, default="generate")
    eligible_atoms = atom_array.is_standard_aa & (atom_array.res_name != "CYS") & ~seq_mask

    residue_starts = get_residue_starts(atom_array)
    residue_eligible = eligible_atoms[residue_starts]
    if not residue_eligible.any():
        return data

    keep_frac = np.random.uniform(0, max_subsample_frac)
    residue_mask = residue_eligible & (np.random.rand(len(residue_starts)) < keep_frac)

    C_NCYS.set_annotation(atom_array, spread_residue_wise(atom_array, residue_mask))

    data["atom_array"] = atom_array
    data.setdefault("task", {})["added_not_cysteine_condition"] = True
    return data


def _fraction_of_chain_retained(data: dict[str, Any], chain_iid: str) -> float:
    """Fraction of a chain's residues that survived cropping; 1.0 if no crop was applied."""
    pre_crop = data.get("crop_info", {}).get("atom_array", None)
    if pre_crop is None:
        return 1.0

    pre_crop_chain = pre_crop.chain_iid == chain_iid
    if not pre_crop_chain.any():
        return 1.0

    atom_array = data["atom_array"]
    chain = atom_array.chain_iid == chain_iid
    n_post = get_residue_count(atom_array[chain]) if chain.any() else 0
    return n_post / get_residue_count(pre_crop[pre_crop_chain])


def add_cath_condition(
    data: dict[str, Any],
    query_pn_unit_iid: str | None = None,
    p_add_cath_c_condition: float = 0.0,
    p_add_cath_ca_condition: float = 0.0,
    p_add_cath_cat_condition: float = 0.0,
    min_retained_chain_fraction: float = 0.75,
) -> dict[str, Any]:
    """Add CATH annotation to the AtomArray.

    Args:
      data: Data dict containing atom_array.
      query_pn_unit_iid: Chain ID to use for CATH annotation. Defaults to None, which will skip annotation
        unless this is a single-chain structure.
      p_add_cath_c_condition: Probability of adding CATH conditioning at C level only.
      p_add_cath_ca_condition: Probability of adding CATH conditioning at C and A levels only.
      p_add_cath_cat_condition: Probability of adding CATH conditioning at C, A, and T levels.
      min_retained_chain_fraction: Skip CATH conditioning when the crop kept a smaller fraction of
        the labelled chain's residues than this.
    """

    sampled_value = np.random.rand()

    p_skip = 1 - (p_add_cath_c_condition + p_add_cath_ca_condition + p_add_cath_cat_condition)
    if sampled_value < p_skip:
        return data

    if "extra_info" not in data:
        return data

    extra_info = data["extra_info"]
    atom_array = data["atom_array"]

    if query_pn_unit_iid is None:
        # Only monomers allowed in this case
        if len(np.unique(atom_array.chain_iid)) != 1:
            return data

        query_pn_unit_iid = atom_array.chain_iid[0]
        cath_code_key = "cath_code"
    else:
        if query_pn_unit_iid not in np.unique(atom_array.chain_iid):
            return data

        if extra_info.get("pn_unit_1_iid") == query_pn_unit_iid:
            cath_code_key = "pn_unit_1_cath_code"
        elif extra_info.get("pn_unit_2_iid") == query_pn_unit_iid:
            cath_code_key = "pn_unit_2_cath_code"
        else:
            return data

    if cath_code_key not in extra_info:
        return data

    if pd.isna(extra_info[cath_code_key]):
        return data

    # Too little of the chain survived for its fold label to describe what remains.
    if _fraction_of_chain_retained(data, query_pn_unit_iid) < min_retained_chain_fraction:
        return data

    cath_code = ast.literal_eval(extra_info[cath_code_key])

    # Sample only one cath annotation
    if isinstance(cath_code, list):
        cath_code = np.random.choice(cath_code)

    cath_code_split = cath_code.split(".")

    if len(cath_code_split) != 4:
        raise ValueError(f"Incomplete or invalid CATH code: {cath_code}")

    if sampled_value < p_skip + p_add_cath_c_condition:
        cath_code_split[1] = "x"
        cath_code_split[2] = "x"
        cath_code_split[3] = "x"
        data["task"]["added_cath_c_condition"] = True
    elif sampled_value < p_skip + p_add_cath_c_condition + p_add_cath_ca_condition:
        cath_code_split[2] = "x"
        cath_code_split[3] = "x"
        data["task"]["added_cath_ca_condition"] = True
    else:
        cath_code_split[3] = "x"
        data["task"]["added_cath_cat_condition"] = True

    cath_code_output = ".".join(cath_code_split)

    # Apply only to the specified chain
    cath_annot = C_CATH.default_annotation(atom_array)
    cath_annot[atom_array.chain_iid == query_pn_unit_iid] = cath_code_output
    C_CATH.set_annotation(atom_array, cath_annot)

    return data
