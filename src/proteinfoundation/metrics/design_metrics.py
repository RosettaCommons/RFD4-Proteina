"""Bond length and steric clash metrics for generated protein structures."""

import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.bonds import filter_bonds_by_distance
from atomworks.ml.conditions import C_CRD
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack, CellList

from foundry.metrics.metric import Metric
from proteinfoundation.metrics.metrics_utils import unwrap_atom_array


def _bond_spatial_masks(
    atom_array: AtomArray,
    a1: np.ndarray,
    a2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Categorize bonds by spatial relationship between their two atoms.

    Args:
        atom_array: Structure with ``is_polymer`` annotation.
        a1: Atom indices for bond endpoint 1, shape ``(n_bonds,)``.
        a2: Atom indices for bond endpoint 2, shape ``(n_bonds,)``.

    Returns:
        Dict mapping category name to boolean mask over bonds.
        Categories are mutually exclusive (within_residue < between_residue.within_chain < between_chain).
        ``nonpolymer_to_polymer`` is a sub-category of ``between_chain``.
    """
    chain_identifier = (
        atom_array.chain_iid if "chain_iid" in atom_array.get_annotation_categories() else atom_array.chain_id
    )

    same_chain = chain_identifier[a1] == chain_identifier[a2]
    same_res = same_chain & (atom_array.res_id[a1] == atom_array.res_id[a2])

    cross_chain = ~same_chain
    is_poly_a1 = atom_array.is_polymer[a1]
    is_poly_a2 = atom_array.is_polymer[a2]

    return {
        "within_residue": same_res,
        "between_residue.within_chain": ~same_res & same_chain,
        "between_chain.polymer_polymer": cross_chain & is_poly_a1 & is_poly_a2,
        "between_chain.nonpolymer_to_polymer": cross_chain & (is_poly_a1 ^ is_poly_a2),
    }


def compute_bond_length_stats(atom_array: AtomArray, verbose: bool = False) -> dict[str, Any]:
    """Compute bond length reasonableness statistics.

    Uses atomworks element-dependent distance thresholds to flag bonds that are
    too long to be physically reasonable.

    Args:
        atom_array: Structure with explicit bonds and coordinates.
        verbose: If True, report breakdown by bond category (within/between residue,
            within/between chain, polymer/non-polymer).

    Returns:
        Dict with bond length statistics. Empty if no bonds are present.
    """
    if atom_array.bonds is None:
        return {}

    bond_arr = atom_array.bonds.as_array()
    if len(bond_arr) == 0:
        return {}

    arr_copy = atom_array.copy()
    filter_bonds_by_distance(arr_copy, policy="filter")

    n_orig = len(bond_arr)
    kept_arr = arr_copy.bonds.as_array() if arr_copy.bonds is not None else np.empty((0, 3), dtype=int)
    n_unreasonable = n_orig - len(kept_arr)

    out: dict[str, Any] = {
        "bond_length.frac_unreasonable": float(n_unreasonable / n_orig) if n_orig > 0 else float("nan"),
    }

    if not verbose:
        return out

    # Reconstruct which original bonds were removed for per-category breakdown.
    if n_unreasonable == 0:
        is_unreasonable = np.zeros(n_orig, dtype=bool)
    elif len(kept_arr) == 0:
        is_unreasonable = np.ones(n_orig, dtype=bool)
    else:
        orig_pairs = np.sort(bond_arr[:, :2], axis=1)
        kept_pairs = np.sort(kept_arr[:, :2], axis=1)
        dt = np.dtype([("a", orig_pairs.dtype), ("b", orig_pairs.dtype)])
        is_unreasonable = ~np.isin(orig_pairs.view(dt).reshape(-1), kept_pairs.view(dt).reshape(-1))

    a1, a2 = bond_arr[:, 0], bond_arr[:, 1]
    ensure_annotations(atom_array, "is_polymer")
    categories = _bond_spatial_masks(atom_array, a1, a2)
    for cat_name, cat_mask in categories.items():
        n_cat = int(cat_mask.sum())
        n_unreas_cat = int((is_unreasonable & cat_mask).sum())
        out[f"bond_length.frac_unreasonable.{cat_name}"] = float(n_unreas_cat / n_cat) if n_cat > 0 else float("nan")

    return out


def compute_clash_stats(
    atom_array: AtomArray,
    thresholds: tuple[float, ...] = (1.0, 1.5, 2.0),
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute steric clash statistics at multiple distance thresholds.

    Counts atoms that have at least one non-bonded neighbor closer than each threshold.
    Directly bonded atom pairs are excluded from the clash check.
    Uses a cell-list neighbor search for O(n·k) memory where k is the max local neighbor count.

    Args:
        atom_array: Structure with coordinates.
        thresholds: Distance thresholds in Angstroms to check. Defaults to ``(1.0, 1.5, 2.0)``.
        verbose: If True, report breakdown by spatial category (within/between residue,
            within/between chain, polymer/non-polymer).

    Returns:
        Dict with clash statistics.
    """
    n_atoms = atom_array.array_length()
    coords = atom_array.coord
    max_thresh = max(thresholds)

    # Encode bonded pairs as i * n_atoms + j (i < j) for vectorised isin lookup
    bonded_keys = np.empty(0, dtype=np.intp)
    if atom_array.bonds is not None:
        bond_arr = atom_array.bonds.as_array()
        if len(bond_arr) > 0:
            b1 = np.minimum(bond_arr[:, 0], bond_arr[:, 1]).astype(np.intp)
            b2 = np.maximum(bond_arr[:, 0], bond_arr[:, 1]).astype(np.intp)
            bonded_keys = b1 * n_atoms + b2

    # Cell-list neighbor search: O(n * k) memory where k = max local neighbor count
    cell_list = CellList(atom_array, cell_size=max_thresh)
    neighbor_matrix = cell_list.get_atoms(coords, radius=max_thresh)  # (n, max_k), -1 padded

    # Flatten to (i, j) pairs with i < j — avoids duplicates and self-pairs
    n, max_k = neighbor_matrix.shape
    i_rep = np.repeat(np.arange(n), max_k)
    j_flat = neighbor_matrix.ravel()
    valid = (j_flat >= 0) & (i_rep < j_flat)
    i_idx = i_rep[valid]
    j_idx = j_flat[valid]

    # Remove bonded pairs
    pair_keys = i_idx * n_atoms + j_idx
    keep = ~np.isin(pair_keys, bonded_keys)
    i_idx, j_idx = i_idx[keep], j_idx[keep]

    # Per-pair distances, then scatter-min to get per-atom minimum
    pair_dists = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=-1) if len(i_idx) > 0 else np.empty(0)
    min_dists = np.full(n_atoms, np.inf)
    if len(i_idx) > 0:
        np.minimum.at(min_dists, i_idx, pair_dists)
        np.minimum.at(min_dists, j_idx, pair_dists)

    out: dict[str, Any] = {}
    for thresh in thresholds:
        key = _threshold_key(thresh)
        out[f"clashes.frac_clashing_{key}"] = float((min_dists < thresh).sum()) / n_atoms

    if not verbose:
        return out

    # Verbose: per-category breakdown
    ensure_annotations(atom_array, "is_polymer")
    category_masks = (
        _bond_spatial_masks(atom_array, i_idx, j_idx)
        if len(i_idx) > 0
        else {
            cat: np.zeros(0, dtype=bool)
            for cat in (
                "within_residue",
                "between_residue.within_chain",
                "between_chain.polymer_polymer",
                "between_chain.nonpolymer_to_polymer",
            )
        }
    )

    for category_name, pair_mask in category_masks.items():
        cat_min_dists = np.full(n_atoms, np.inf)
        if pair_mask.any():
            cat_i, cat_j, cat_d = i_idx[pair_mask], j_idx[pair_mask], pair_dists[pair_mask]
            np.minimum.at(cat_min_dists, cat_i, cat_d)
            np.minimum.at(cat_min_dists, cat_j, cat_d)
        for thresh in thresholds:
            key = _threshold_key(thresh)
            out[f"clashes.frac_clashing_{key}.{category_name}"] = float((cat_min_dists < thresh).sum()) / n_atoms

    return out


def _threshold_key(thresh: float) -> str:
    """Format threshold as a metric key component and preserve dot-dict format, e.g. 1.0 -> '1,0'."""
    return f"{thresh:.1f}".replace(".", ",")


class BondLengthMetric(Metric):
    """Report unreasonable bond lengths using atomworks element-dependent thresholds.

    In verbose mode, unreasonable bonds are broken down by spatial category:
    within-residue, between-residue within-chain, between-chain polymer-polymer,
    and the special non-polymer-to-polymer bin (often unreliable in predictions).
    """

    def __init__(self, verbose: bool = False, **kwargs: Any) -> None:
        """Initialize metric.

        Args:
            verbose: If True, emit per-category breakdown in addition to overall stats.
        """
        self.verbose = verbose
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"atom_array": ("predicted_atom_array",)}

    def compute(self, atom_array: AtomArray) -> dict[str, Any]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        return compute_bond_length_stats(atom_array, verbose=self.verbose)


class ClashMetric(Metric):
    """Report steric clashes at configurable distance thresholds (default 1.0, 1.5, 2.0 Å).

    Bonded atom pairs are excluded. In verbose mode, counts are broken down by
    spatial category: within-residue, between-residue within-chain, and
    between-chain (polymer-polymer or non-polymer-to-polymer).
    """

    def __init__(
        self,
        thresholds: tuple[float, ...] = (1.0, 1.5, 2.0),
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize metric.

        Args:
            thresholds: Distance thresholds in Angstroms. Defaults to ``(1.0, 1.5, 2.0)``.
            verbose: If True, emit per-category breakdown in addition to overall stats.
        """
        self.thresholds = thresholds
        self.verbose = verbose
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"atom_array": ("predicted_atom_array",)}

    def compute(self, atom_array: AtomArray) -> dict[str, Any]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        return compute_clash_stats(atom_array, thresholds=self.thresholds, verbose=self.verbose)


def get_ligand_clash_metrics(atom_array: AtomArray, ligand_clash_threshold: float = 1.5) -> dict[str, float | int]:
    """Compute steric clash counts between ligand and protein atoms.

    Pairs where both atoms are CRD-conditioned (fixed input coordinates) are excluded,
    since those coordinates were not generated and should not be penalised.

    Args:
        atom_array: Structure with ``is_ligand``, ``is_protein_backbone``, and
            ``is_protein`` annotations, plus CRD conditioning state.
        ligand_clash_threshold: Distance threshold in Angstroms. Defaults to ``1.5``.

    Returns:
        Dict with backbone_ligand and allatom_ligand clash counts and minimum distances.
        Empty if no ligand atoms are present.
    """
    ensure_annotations(atom_array, "is_ligand", "is_protein_backbone", "is_protein")

    if not atom_array.is_ligand.any():
        return {}

    is_crd = C_CRD.mask(atom_array, default="generate")
    ligand_mask = atom_array.is_ligand
    xyz_ligand = atom_array[ligand_mask].coord  # (n_lig, 3)
    ligand_crd = is_crd[ligand_mask]  # (n_lig,)

    out: dict[str, float | int] = {}
    for mask, prefix in [
        (atom_array.is_protein_backbone, "backbone"),
        (atom_array.is_protein, "allatom"),
    ]:
        if mask.sum() == 0:
            continue
        xyz = atom_array[mask].coord  # (n_prot, 3)
        both_crd = is_crd[mask][:, None] & ligand_crd[None, :]  # (n_prot, n_lig)
        dists = np.linalg.norm(xyz[:, None] - xyz_ligand[None], axis=-1)  # (n_prot, n_lig)
        dists[both_crd] = np.inf
        out[f"n_clashing.{prefix}_ligand_clashes"] = int(np.sum(dists < ligand_clash_threshold))
        out[f"n_clashing.{prefix}_ligand_min_distance"] = float(dists.min())

    return out


class LigandClashMetric(Metric):
    """Measure steric clashes between ligand atoms and protein backbone/sidechain atoms.

    Reports both backbone-ligand and all-atom-ligand clash counts and minimum distances.
    Atom pairs where both atoms are CRD-conditioned (fixed input coordinates) are excluded.
    """

    def __init__(self, ligand_clash_threshold: float = 1.5, **kwargs: Any) -> None:
        """Initialize metric.

        Args:
            ligand_clash_threshold: Distance threshold in Angstroms. Defaults to ``1.5``.
        """
        self.ligand_clash_threshold = ligand_clash_threshold
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"atom_array": ("predicted_atom_array",)}

    def compute(self, atom_array: AtomArray | AtomArrayStack) -> dict[str, Any]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        return get_ligand_clash_metrics(
            unwrap_atom_array(atom_array), ligand_clash_threshold=self.ligand_clash_threshold
        )
