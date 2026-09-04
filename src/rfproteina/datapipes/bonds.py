"""Atom-level bond utilities (bond order, aromaticity, distances).

Extracted from :py:mod:`~rfproteina.datapipes.pipeline` to keep that
module focused on transform composition.
"""

import numpy as np
from atomworks.constants import BIOTITE_BOND_TYPE_TO_BOND_ORDER, UNKNOWN_AA, UNKNOWN_DNA, UNKNOWN_RNA
from atomworks.io.utils.ccd import get_polymerization_atoms
from atomworks.io.utils.standard_annotations import S_SEGMIN
from atomworks.ml.utils.annotator import ensure_annotations
from atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite.structure import AtomArray
from biotite.structure.bonds import BondType
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from rfproteina.conditions import C_IDXGRP
from rfproteina.datapipes.schema.proteina import TokenType

_UNKNOWN_RES_BY_TOKEN_TYPE = {TokenType.PROTEIN: UNKNOWN_AA, TokenType.RNA: UNKNOWN_RNA, TokenType.DNA: UNKNOWN_DNA}

_AROMATIC_BOND_TYPES = (
    BondType.AROMATIC_SINGLE,
    BondType.AROMATIC_DOUBLE,
    BondType.AROMATIC_TRIPLE,
    BondType.AROMATIC,
)


def _build_symmetric_bond_matrix(
    atom_array: AtomArray, per_bond_values: np.ndarray, dtype: np.dtype, fill: float | bool | int = 0
) -> np.ndarray:
    """Scatter ``per_bond_values`` into a symmetric ``(n_atoms, n_atoms)`` matrix."""
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    n_atoms = atom_array.array_length()
    matrix = np.full((n_atoms, n_atoms), fill, dtype=dtype)
    matrix[atom1_idxs, atom2_idxs] = per_bond_values
    matrix[atom2_idxs, atom1_idxs] = per_bond_values
    return matrix


def _token_type_at(atom_array: AtomArray, atom_idxs: np.ndarray) -> np.ndarray:
    """Return token type (0=protein, 1=RNA, 2=DNA, 3=atomized, 4=unknown) at given atom positions."""
    # Use token type annotation, if it exists
    if "token_type" in atom_array.get_annotation_categories():
        return np.asarray(atom_array.token_type)[atom_idxs].astype(np.int16)
    ensure_annotations(
        atom_array,
        "is_standard_unknown_or_masked_aa",
        "is_standard_unknown_or_masked_dna",
        "is_standard_unknown_or_masked_rna",
    )
    types = np.full(len(atom_idxs), TokenType.UNKNOWN, dtype=np.int16)
    types[atom_array.get_annotation("is_standard_unknown_or_masked_dna")[atom_idxs]] = TokenType.DNA
    types[atom_array.get_annotation("is_standard_unknown_or_masked_rna")[atom_idxs]] = TokenType.RNA
    types[atom_array.get_annotation("is_standard_unknown_or_masked_aa")[atom_idxs]] = TokenType.PROTEIN
    types[atom_array.atomize[atom_idxs]] = TokenType.ATOMIZED
    return types


def _polymerisation_atoms(res_name: str, token_type: int) -> tuple[str, str]:
    """(leaving, entering) backbone atoms; ``""`` for a side the residue does not carry."""
    leaving, entering = get_polymerization_atoms(res_name)
    if leaving is None and entering is None:
        unknown = _UNKNOWN_RES_BY_TOKEN_TYPE.get(token_type)
        if unknown is not None:
            leaving, entering = get_polymerization_atoms(unknown)
    return leaving or "", entering or ""


def _link_atom_names(atom_array: AtomArray, atom_idxs: np.ndarray, token_type: np.ndarray) -> np.ndarray:
    """Per-atom ``[leaving, entering]`` names expected of that atom's residue."""
    res, types = atom_array.res_name[atom_idxs], token_type[atom_idxs]
    pairs = {(r, t): _polymerisation_atoms(str(r), int(t)) for r, t in set(zip(res, types, strict=False))}
    return np.array([pairs[k] for k in zip(res, types, strict=False)]).reshape(-1, 2)


def _is_link_atom(
    atom_array: AtomArray, atom_idxs: np.ndarray, token_type: np.ndarray, stands_in: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Whether each atom is its residue's leaving / entering polymerisation atom."""
    link = _link_atom_names(atom_array, atom_idxs, token_type)
    names, stand_in = atom_array.atom_name[atom_idxs], stands_in[atom_idxs]
    return (names == link[:, 0]) | stand_in, (names == link[:, 1]) | stand_in


def _inter_token_bond_edges(
    atom_array: AtomArray,
    atom_to_token: np.ndarray,
    atom_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return inter-token bond edges as a dict with keys ``tok1``, ``tok2``, ``bond_types``, ``atom1``, ``atom2``.

    Args:
        atom_array: Atom array with bond information.
        atom_to_token: Per-atom token ID (shape ``[n_atoms]``).
        atom_mask: If provided, only bonds where both atoms are ``True`` are kept.

    Returns:
        Dict of five arrays, each of length n_inter_token_bonds.
    """
    a1, a2, btypes = atom_array.bonds.as_array().T
    if atom_mask is not None:
        keep = atom_mask[a1] & atom_mask[a2]
        a1, a2, btypes = a1[keep], a2[keep], btypes[keep]
    t1 = atom_to_token[a1]
    t2 = atom_to_token[a2]
    inter = t1 != t2
    return {
        "tok1": t1[inter].astype(np.int32),
        "tok2": t2[inter].astype(np.int32),
        "bond_types": btypes[inter],
        "atom1": a1[inter],
        "atom2": a2[inter],
    }


def _compute_token_path_lengths(
    tok1: np.ndarray,
    tok2: np.ndarray,
    source_indices: np.ndarray,
    path_length_cutoff: int | None,
) -> np.ndarray:
    """Compute inter-token shortest-path distances via BFS.

    Args:
        tok1: Source token indices for inter-token bond edges.
        tok2: Target token indices for inter-token bond edges.
        source_indices: Token positions to run BFS from.
        path_length_cutoff: BFS depth limit. Pairs beyond this are left as 0.
            ``None`` keeps all distances.

    Returns:
        int16 array, shape ``[len(source_indices), len(source_indices)]``.
        0 = unreached (disconnected or beyond cutoff).
    """
    n_sources = len(source_indices)
    dist = np.zeros((n_sources, n_sources), dtype=np.int16)
    if n_sources == 0 or len(tok1) == 0:
        return dist

    # Multi-source BFS via scipy csgraph (same hop counts as the old networkx loop, much faster).
    n_nodes = max(int(tok1.max()), int(tok2.max()), int(source_indices.max())) + 1
    rows = np.concatenate([tok1, tok2])
    cols = np.concatenate([tok2, tok1])
    adj = coo_matrix((np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    d = shortest_path(adj, method="D", unweighted=True, directed=False, indices=source_indices)  # [n_sources, n_nodes]
    sub = d[:, source_indices]  # [n_sources, n_sources]

    # Keep hop counts 1..cutoff; self/unreachable/beyond-cutoff stay 0.
    valid = np.isfinite(sub) & (sub > 0)
    if path_length_cutoff is not None:
        valid &= sub <= path_length_cutoff
    dist[valid] = sub[valid].astype(np.int16)
    return dist


def _closes_ring(edges: dict[str, np.ndarray], res_id_per_tok: np.ndarray, mol_per_tok: np.ndarray) -> np.ndarray:
    """Whether each bond closes a ring: joins two non-consecutive residues of the same molecule.

    NOTE: Fallible heuristic: only consulted during inference, when both endpoints are expandable-segment
    stand-ins (S_SEGMIN), so chemistry cannot orient the bond.
    """
    tok1, tok2 = edges["tok1"], edges["tok2"]
    same_mol = mol_per_tok[tok1] == mol_per_tok[tok2]
    return same_mol & (np.abs(res_id_per_tok[tok1] - res_id_per_tok[tok2]) > 1)


def _upstream_direction(
    atom_array: AtomArray, edges: dict[str, np.ndarray], idx_tok_starts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bond ``(fwd, rev)`` masks: which endpoint of each inter-token bond is upstream."""
    # Chemistry: a bond runs from a residue's leaving atom into the next residue's entering atom.
    atom_type = _token_type_at(atom_array, np.arange(atom_array.array_length()))

    # Expandable-segment sentinels carry no backbone atoms, so they stand in for either role.
    stands_in = S_SEGMIN.mask(atom_array, default="generate")
    leaves1, enters1 = _is_link_atom(atom_array, edges["atom1"], atom_type, stands_in)
    leaves2, enters2 = _is_link_atom(atom_array, edges["atom2"], atom_type, stands_in)
    fwd, rev = leaves1 & enters2, leaves2 & enters1
    if not stands_in.any():
        return fwd, rev  # no expandable-segment stand-ins (all training, most inference): chemistry decides.

    # No chemistry (both endpoints stand-ins): fall back to chain order (lower token upstream),
    # except a ring-closing bond runs high->low, so flip the tie-break there.
    res_id_per_tok, mol_per_tok = atom_array.res_id[idx_tok_starts], atom_array.molecule_iid[idx_tok_starts]
    ring = _closes_ring(edges, res_id_per_tok, mol_per_tok)
    blind = stands_in[edges["atom1"]] & stands_in[edges["atom2"]]
    tok1_upstream = (edges["tok1"] < edges["tok2"]) ^ ring

    return (fwd & ~blind) | (blind & tok1_upstream), (rev & ~blind) | (blind & ~tok1_upstream)


def compute_token_bond_features(
    atom_array: AtomArray,
    path_length_cutoff: int | None = None,
    source_atom_array: AtomArray | None = None,
    source_token_idxs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute token-level bond features for all tokens, including unindexed expansion copies.

    Features are computed in the indexed block [n_indexed x n_indexed] from the bond
    graph of indexed atoms only (unindexed copies have no valid inter-token bonds).
    The output [n_total x n_total] matrices are then populated per IndexGroup.

    Path lengths are special: when ``source_atom_array`` and ``source_token_idxs`` are
    provided (the training path), the BFS is run on a wider bond graph from earlier in the
    pipeline, so that shortest paths through tokens dropped since then -- by the crop, or by
    :py:class:`RemoveTokensWithAnyNaNAtom` -- are preserved. The result is then subsetted
    to ``source_token_idxs``. Bond mask, order, aromaticity, and canonical ordering are
    computed on the post-crop indexed block only (direct bonds cannot span a crop boundary).

    Args:
        atom_array: Post-expansion atom array with ``insertion_source_idx`` annotation.
        path_length_cutoff: Paths longer than this are stored as 0 (treated as
            disconnected). ``None`` keeps all distances.
        source_atom_array: A wider atom array from earlier in the pipeline. When provided
            together with ``source_token_idxs``, path lengths are computed here and
            subsetted, preserving paths through tokens dropped since.
        source_token_idxs: Token-level indices into ``source_atom_array`` of the tokens
            that survive in ``atom_array``, in order. Must be provided iff
            ``source_atom_array`` is provided.

    Returns:
        Dict with five ``[n_total, n_total]`` arrays:

        - ``"bond_mask"`` (bool): True where the input contains a direct bond.
        - ``"bond_order"`` (int16): Kekulé order (0=none, 1-3, 4=coord).
          Poly-poly pairs and self-diagonal are set to 0
        - ``"is_aromatic"`` (bool): True if any inter-token bond is aromatic.
          Poly-poly pairs are False.
        - ``"token_bond_path_length"`` (int16): Shortest bond-graph hop count within
          IndexGroup. 0 = disconnected / cross-group / self. Paths at or exceeding
          ``path_length_cutoff`` are stored as ``path_length_cutoff + 1``.
        - ``"canonical_polymer_ordering"`` (int8): ``+1`` at ``[downstream, upstream]`` of a
          backbone linkage (C-N / O3'-P) between same-type canonical polymer residues
          (protein/RNA/DNA), ``-1`` at the transpose, else 0.
    """
    assert (source_atom_array is None) == (
        source_token_idxs is None
    ), "source_atom_array and source_token_idxs must be provided together"
    assert (
        "insertion_source_idx" in atom_array.get_annotation_categories()
    ), "atom_array must have insertion_source_idx annotation (set by UnindexFlaggedTokens)"

    token_starts = get_token_starts(atom_array)
    n_total = len(token_starts)

    # ── Partition into indexed / unindexed tokens ─────────────────────────────
    src_per_tok = atom_array.insertion_source_idx[token_starts]  # [n_total]
    is_indexed_tok = src_per_tok == -1  # [n_total]
    n_indexed = int(is_indexed_tok.sum())

    # Positions (in 0..n_total-1) and sources (in 0..n_indexed-1) of unindexed tokens.
    unindexed_positions = np.where(~is_indexed_tok)[0]
    src_idxs = src_per_tok[unindexed_positions].astype(np.int32)

    # IndexGroup per token from the full (post-expansion) atom array.
    tok_groups = C_IDXGRP.annotation(atom_array, default="generate")[token_starts]

    # ── Inter-token bond edges filtered to indexed atoms only ─────────────────
    is_indexed_atom = spread_token_wise(atom_array, is_indexed_tok)
    atom_to_token = spread_token_wise(atom_array, np.arange(n_total, dtype=np.int32))
    edges = _inter_token_bond_edges(atom_array, atom_to_token, atom_mask=is_indexed_atom)

    # ── Bond mask [n_indexed x n_indexed] ───────────────────────────────────
    idx_bond_mask = np.zeros((n_indexed, n_indexed), dtype=bool)
    idx_bond_mask[edges["tok1"], edges["tok2"]] = True
    idx_bond_mask[edges["tok2"], edges["tok1"]] = True
    np.fill_diagonal(idx_bond_mask, False)

    # ── Bond orders [n_indexed x n_indexed] ──────────────────────────────────
    _lookup = np.ones(max(int(bt) for bt in BondType) + 1, dtype=np.int16)
    for bt, order in BIOTITE_BOND_TYPE_TO_BOND_ORDER.items():
        _lookup[int(bt)] = order
    _lookup[BondType.QUADRUPLE] = 1
    _lookup[BondType.COORDINATION] = 4
    _lookup[BondType.AROMATIC] = 1
    bond_orders = _lookup[edges["bond_types"]]

    _ORDER_SENTINEL = np.iinfo(np.int16).max
    _order_work = np.full((n_indexed, n_indexed), _ORDER_SENTINEL, dtype=np.int16)
    np.minimum.at(_order_work, (edges["tok1"], edges["tok2"]), bond_orders)
    np.minimum.at(_order_work, (edges["tok2"], edges["tok1"]), bond_orders)
    idx_bond_order = np.where(_order_work < _ORDER_SENTINEL, _order_work, np.int16(0))
    np.fill_diagonal(idx_bond_order, 0)

    # ── Aromaticity [n_indexed x n_indexed] ──────────────────────────────────
    _aromatic_ints = [int(bt) for bt in _AROMATIC_BOND_TYPES]
    is_aromatic_feat = np.isin(edges["bond_types"], _aromatic_ints).astype(np.int8)
    _arom_work = np.zeros((n_indexed, n_indexed), dtype=np.int8)
    np.maximum.at(_arom_work, (edges["tok1"], edges["tok2"]), is_aromatic_feat)
    np.maximum.at(_arom_work, (edges["tok2"], edges["tok1"]), is_aromatic_feat)
    idx_is_aromatic = _arom_work.astype(bool)
    np.fill_diagonal(idx_is_aromatic, False)

    # Polymer-polymer pair mask. Used both by canonical_polymer_ordering (which
    # wants the unmasked bond graph including peptide bonds) and by the
    # subsequent suppression of poly-poly bond features.
    idx_tok_starts = token_starts[is_indexed_tok]
    is_poly = ~atom_array.atomize[idx_tok_starts]
    is_poly_poly = np.outer(is_poly, is_poly)

    # ── Canonical polymer ordering [n_indexed x n_indexed] ───────────────────
    # Computed BEFORE the bond-feature suppression below so this feature can
    # use the unmasked bond graph (peptide bonds visible). Same logic as #205.
    token_type = _token_type_at(atom_array, idx_tok_starts)
    is_canonical = np.isin(token_type, (TokenType.PROTEIN, TokenType.RNA, TokenType.DNA))
    same_type = token_type[:, None] == token_type[None, :]
    eligible = is_poly_poly & np.outer(is_canonical, is_canonical) & same_type

    # Orient each inter-token bond: which endpoint is upstream (chemistry, chain-order tie-break).
    fwd, rev = _upstream_direction(atom_array, edges, idx_tok_starts)

    up_tok = np.concatenate([edges["tok1"][fwd], edges["tok2"][rev]])
    dn_tok = np.concatenate([edges["tok2"][fwd], edges["tok1"][rev]])
    idx_canonical_ordering = np.zeros((n_indexed, n_indexed), dtype=np.int8)
    idx_canonical_ordering[dn_tok, up_tok] = 1
    idx_canonical_ordering[up_tok, dn_tok] = -1
    idx_canonical_ordering[~(eligible & idx_bond_mask)] = 0

    # ── Suppress backbone-linkage bond-feature signal ────────────────────────
    # Standard residue connectivity is already carried by token_bond_path_length /
    # canonical_polymer_ordering, so bond features are reserved for bonds outside the chain.
    suppress = np.zeros((n_indexed, n_indexed), dtype=bool)
    suppress[up_tok, dn_tok] = suppress[dn_tok, up_tok] = True
    suppress &= is_poly_poly
    idx_bond_mask[suppress] = False
    idx_bond_order[suppress] = 0
    idx_is_aromatic[suppress] = False

    # ── Path lengths [n_indexed x n_indexed] ─────────────────────────────────
    if source_atom_array is not None:
        # BFS only from the surviving tokens through the full source graph, so we never
        # allocate an [n_source x n_source] dense matrix.
        src_token_starts = get_token_starts(source_atom_array)
        n_source = len(src_token_starts)
        src_atom_to_token = spread_token_wise(source_atom_array, np.arange(n_source, dtype=np.int32))
        src_edges = _inter_token_bond_edges(source_atom_array, src_atom_to_token)
        idx_path_length = _compute_token_path_lengths(
            src_edges["tok1"],
            src_edges["tok2"],
            source_indices=source_token_idxs,
            path_length_cutoff=path_length_cutoff,
        )  # [n_indexed, n_indexed]
        source_mol_ids = source_atom_array.molecule_iid[src_token_starts[source_token_idxs]]
    else:
        # Inference path: no wider array; compute on the indexed block directly.
        idx_path_length = _compute_token_path_lengths(
            edges["tok1"],
            edges["tok2"],
            source_indices=np.arange(n_indexed),
            path_length_cutoff=path_length_cutoff,
        )
        source_mol_ids = atom_array.molecule_iid[idx_tok_starts]

    # Same-molecule pairs left at 0 by BFS cutoff: fill with path_length_cutoff
    # This differentiates connected components from max-bin distances
    if path_length_cutoff is not None:
        same_mol = source_mol_ids[:, None] == source_mol_ids[None, :]
        np.fill_diagonal(same_mol, False)
        idx_path_length[same_mol & (idx_path_length == 0)] = path_length_cutoff + 1

    # Mask out inter-group paths and self-paths
    indexed_tok_groups = tok_groups[is_indexed_tok]
    idx_path_length[indexed_tok_groups[:, None] != indexed_tok_groups[None, :]] = 0
    np.fill_diagonal(idx_path_length, 0)

    # ── Assemble [n_total x n_total] output per IndexGroup ───────────────────
    out_bond_mask = np.zeros((n_total, n_total), dtype=bool)
    out_bond_order = np.zeros((n_total, n_total), dtype=np.int16)
    out_is_aromatic = np.zeros((n_total, n_total), dtype=bool)
    out_path_length = np.zeros((n_total, n_total), dtype=np.int16)
    out_canonical = np.zeros((n_total, n_total), dtype=np.int8)

    idx_feats = (idx_bond_mask, idx_bond_order, idx_is_aromatic, idx_path_length, idx_canonical_ordering)
    out_feats = (out_bond_mask, out_bond_order, out_is_aromatic, out_path_length, out_canonical)

    for idx_f, out_f in zip(idx_feats, out_feats, strict=False):
        out_f[:n_indexed, :n_indexed] = idx_f

    unindexed_tok_groups = tok_groups[unindexed_positions]
    for grp in np.unique(unindexed_tok_groups):
        grp_mask = unindexed_tok_groups == grp
        grp_positions = unindexed_positions[grp_mask]
        grp_sources = src_idxs[grp_mask]
        for idx_f, out_f in zip(idx_feats, out_feats, strict=False):
            out_f[np.ix_(grp_positions, grp_positions)] = idx_f[np.ix_(grp_sources, grp_sources)]

    return {
        "bond_mask": out_bond_mask,
        "bond_order": out_bond_order,
        "is_aromatic": out_is_aromatic,
        "token_bond_path_length": out_path_length,
        "canonical_polymer_ordering": out_canonical,
    }


def get_bond_order_matrix(atom_array: AtomArray) -> np.ndarray:
    """Get the bond order matrix using Kekulé assignments for aromatic bonds.

    Values: 0=no bond, 1=single, 2=double, 3=triple, 4=coordination.
    Aromatic bonds get their Kekulé order (AROMATIC_SINGLE->1, AROMATIC_DOUBLE->2, AROMATIC_TRIPLE->3).
    QUADRUPLE and generic AROMATIC (no Kekulé info) are mapped to single (1).
    """
    _, _, bond_types = atom_array.bonds.as_array().T

    # Build lookup from BIOTITE_BOND_TYPE_TO_BOND_ORDER constant (sized to the biotite enum)
    lookup = np.ones(max(int(bt) for bt in BondType) + 1, dtype=np.int16)  # default: unknown -> single
    for bt, order in BIOTITE_BOND_TYPE_TO_BOND_ORDER.items():
        lookup[int(bt)] = order
    lookup[BondType.QUADRUPLE] = 1  # QUADRUPLE -> single
    lookup[BondType.COORDINATION] = 4  # COORDINATION -> own class
    lookup[BondType.AROMATIC] = 1  # generic AROMATIC (no Kekulé info) -> single

    bond_orders = lookup[bond_types]
    assert np.all(bond_orders >= 0) and np.all(bond_orders <= 4), "Invalid bond order detected"

    return _build_symmetric_bond_matrix(atom_array, bond_orders, dtype=np.int16, fill=0)


def get_is_aromatic_matrix(atom_array: AtomArray) -> np.ndarray:
    """Get boolean aromaticity matrix for an atom array.

    A bond is aromatic if its biotite BondType is any of AROMATIC_{SINGLE,DOUBLE,TRIPLE}
    or the generic AROMATIC.
    """
    _, _, bond_types = atom_array.bonds.as_array().T
    is_aromatic = np.isin(bond_types, [int(bt) for bt in _AROMATIC_BOND_TYPES])
    return _build_symmetric_bond_matrix(atom_array, is_aromatic, dtype=bool, fill=False)


def get_bond_distances(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond distance (adjacency) list as a 1D array."""
    coords = atom_array.coord
    atom1_idxs, atom2_idxs, _ = atom_array.bonds.as_array().T
    return np.linalg.norm(coords[atom1_idxs] - coords[atom2_idxs], axis=1)


def get_bond_distance_matrix(atom_array: AtomArray) -> np.ndarray:
    """Returns the bond adjacency matrix with bond distances as values."""
    return _build_symmetric_bond_matrix(atom_array, get_bond_distances(atom_array), dtype=float, fill=np.inf)
