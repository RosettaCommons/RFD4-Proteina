"""Bond length, steric clash, and sequence-similarity metrics for generated protein structures."""

import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.bonds import filter_bonds_by_distance
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import C_CRD
from beartype.typing import Any
from biotite.structure import AtomArray, CellList

from foundry.metrics.metric import Metric


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


def compute_clash_stats(atom_array: AtomArray, threshold: float = 1.5) -> dict[str, Any]:
    """Fraction of atoms in a steric clash, by clash type.

    A clash is a non-bonded pair of atoms closer than ``threshold`` A; pairs
    where *both* atoms are CRD-conditioned (fixed input) are excluded.
    See :py:func:`_clash_category_masks` for the 5-category breakdown.
    """
    n_atoms = atom_array.array_length()
    coords = atom_array.coord
    key = _threshold_key(threshold)

    # Encode bonded pairs as i * n_atoms + j (i < j) for vectorised isin lookup
    bonded_keys = np.empty(0, dtype=np.intp)
    if atom_array.bonds is not None:
        bond_arr = atom_array.bonds.as_array()
        if len(bond_arr) > 0:
            b1 = np.minimum(bond_arr[:, 0], bond_arr[:, 1]).astype(np.intp)
            b2 = np.maximum(bond_arr[:, 0], bond_arr[:, 1]).astype(np.intp)
            bonded_keys = b1 * n_atoms + b2

    # Cell-list neighbor search: O(n * k) memory where k = max local neighbor count
    cell_list = CellList(atom_array, cell_size=threshold)
    neighbor_matrix = cell_list.get_atoms(coords, radius=threshold)  # (n, max_k), -1 padded

    # Flatten to (i, j) pairs with i < j — avoids duplicates and self-pairs
    n, max_k = neighbor_matrix.shape
    i_rep = np.repeat(np.arange(n), max_k)
    j_flat = neighbor_matrix.ravel()
    valid = (j_flat >= 0) & (i_rep < j_flat)
    i_idx = i_rep[valid]
    j_idx = j_flat[valid]

    # Drop bonded pairs and pairs where both atoms are CRD-conditioned (not generated)
    is_crd = C_CRD.mask(atom_array, default="generate")
    pair_keys = i_idx * n_atoms + j_idx
    keep = ~np.isin(pair_keys, bonded_keys) & ~(is_crd[i_idx] & is_crd[j_idx])
    i_idx, j_idx = i_idx[keep], j_idx[keep]

    pair_dists = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=-1) if len(i_idx) > 0 else np.empty(0)

    def _frac_atoms_clashing(pair_mask: np.ndarray) -> float:
        """Fraction of all atoms that have a clashing pair within ``pair_mask``."""
        min_dists = np.full(n_atoms, np.inf)
        if pair_mask.any():
            np.minimum.at(min_dists, i_idx[pair_mask], pair_dists[pair_mask])
            np.minimum.at(min_dists, j_idx[pair_mask], pair_dists[pair_mask])
        return float((min_dists < threshold).sum()) / n_atoms

    out: dict[str, Any] = {f"frac_atoms_clashing_{key}": _frac_atoms_clashing(np.ones(len(i_idx), dtype=bool))}

    ensure_annotations(atom_array, "is_polymer")
    categories = (
        "intra_residue_polymer",
        "intra_residue_ligand",
        "inter_residue_polymer_polymer",
        "inter_residue_polymer_ligand",
        "inter_residue_ligand_ligand",
    )
    category_masks = (
        _clash_category_masks(atom_array, i_idx, j_idx)
        if len(i_idx) > 0
        else {cat: np.zeros(0, dtype=bool) for cat in categories}
    )
    for category_name in categories:
        out[f"frac_atoms_clashing_{category_name}_{key}"] = _frac_atoms_clashing(category_masks[category_name])

    return out


def _threshold_key(thresh: float) -> str:
    """Format a distance threshold for a metric key, e.g. ``1.5`` -> ``'1p5A'``.

    Avoids ``.`` (the metric-system namespace separator) and ``,`` (a CSV delimiter).
    """
    return f"{thresh:.1f}".replace(".", "p") + "A"


def _clash_category_masks(
    atom_array: AtomArray,
    a1: np.ndarray,
    a2: np.ndarray,
) -> dict[str, np.ndarray]:
    """Classify clashing atom pairs into 5 mutually exclusive clash-type categories.

    Two-axis taxonomy: same vs. different residue x the molecule types of the
    two atoms.

    - ``intra_residue_polymer``: same residue, both polymer.
    - ``intra_residue_ligand``: same residue, both non-polymer.
    - ``inter_residue_polymer_polymer``: different residue, both polymer.
    - ``inter_residue_polymer_ligand``: different residue, one polymer + one non-polymer.
    - ``inter_residue_ligand_ligand``: different residue, both non-polymer.
    """
    chain_identifier = (
        atom_array.chain_iid if "chain_iid" in atom_array.get_annotation_categories() else atom_array.chain_id
    )
    is_poly_a1 = atom_array.is_polymer[a1]
    is_poly_a2 = atom_array.is_polymer[a2]
    both_polymer = is_poly_a1 & is_poly_a2
    both_ligand = ~is_poly_a1 & ~is_poly_a2
    polymer_ligand = is_poly_a1 ^ is_poly_a2
    same_res = (chain_identifier[a1] == chain_identifier[a2]) & (atom_array.res_id[a1] == atom_array.res_id[a2])

    return {
        "intra_residue_polymer": same_res & both_polymer,
        "intra_residue_ligand": same_res & both_ligand,
        "inter_residue_polymer_polymer": ~same_res & both_polymer,
        "inter_residue_polymer_ligand": ~same_res & polymer_ligand,
        "inter_residue_ligand_ligand": ~same_res & both_ligand,
    }


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
    """Fraction of atoms in a steric clash, by clash type, at a configurable threshold.

    Emits ``frac_atoms_clashing`` plus a 5-category breakdown (see
    :py:func:`compute_clash_stats`). The threshold is encoded in the metric name
    (e.g. ``..._1p5A``); pairs where *both* atoms are CRD-conditioned are excluded.
    """

    def __init__(self, threshold: float = 1.5, **kwargs: Any) -> None:
        """Initialize metric.

        Args:
            threshold: Clash distance threshold in Angstroms. Defaults to ``1.5``.
        """
        self.threshold = threshold
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"atom_array": ("predicted_atom_array",)}

    def compute(self, atom_array: AtomArray) -> dict[str, Any]:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        return compute_clash_stats(atom_array, threshold=self.threshold)


def _protein_sequence(atom_array: AtomArray) -> np.ndarray | None:
    """Extract three-letter residue name sequence from an AtomArray using first atom per residue.

    Returns None if no amino acid residues are present.
    """
    ensure_annotations(atom_array, "is_protein")
    protein = atom_array[atom_array.is_protein]
    if protein.array_length() == 0:
        return None
    residue_names = protein.res_name[get_residue_starts(protein)]
    return residue_names


class SequenceSimilarityToInput(Metric):
    """Fraction of residues with identical amino acid type between output and input structure.

    Returns ``nan`` when the predicted and input protein sequences differ in length.
    Intended for use in partial diffusion.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": ("predicted_atom_array",),
            "ground_truth_atom_array": ("ground_truth_atom_array",),
        }

    def compute(  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        predicted_atom_array: AtomArray,
        ground_truth_atom_array: AtomArray,
    ) -> dict[str, Any]:
        """Compute sequence identity between generated and input structures.

        Args:
            predicted_atom_array: Generated protein structure.
            ground_truth_atom_array: Input structure used to seed partial diffusion.

        Returns:
            Dict with ``sequence_identity_to_input`` in [0, 1], or ``nan`` if
            sequence lengths differ.
        """
        pred_seq = _protein_sequence(predicted_atom_array)
        input_seq = _protein_sequence(ground_truth_atom_array)

        if pred_seq is None or input_seq is None or len(pred_seq) != len(input_seq):
            return {"sequence_identity_to_input": float("nan")}

        identity = (pred_seq == input_seq).mean()
        return {"sequence_identity_to_input": float(identity)}
