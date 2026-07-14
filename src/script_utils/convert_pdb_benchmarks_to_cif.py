#!/usr/bin/env python3
"""Convert benchmark structure files to CIF using RCSB reference structures.

For each benchmark .pdb or .cif file the script:
  1. Detects a PDB ID: exactly one underscore-delimited 4-char alphanumeric group
     present in the PDB mirror. Fails if none or more than one match.
  2. Parses the benchmark file and its RCSB reference via atomworks.
  3. Matches every benchmark residue to the RCSB structure by residue name and
     coordinates (``subset_with_reference``). Benchmark chain_id/res_id are written
     back onto the matched RCSB atoms so original labelling is preserved.
  4. Optionally (``--reparsed_pdb_dir``): looks up the matched atoms in a
     pre-processed PDB mirror by RCSB-native identifiers, substitutes those
     coordinates, then restores benchmark chain/residue labels. A ≤1 Å RMSD
     sanity check guards against mismatches.
  5. Centers coordinates at the geometric mean and writes a CIF file.

Special cases:
  - Files in MODEL_PREDICTED_DIRS with no PDB ID are written directly (no RCSB lookup).
  - Files with "relaxed" in the stem are skipped (coordinates likely differ from RCSB).
  - Files in EXCLUDE_PROTEIN_IN_DIRS have protein chains stripped before matching
    (the PDB ID identifies the small-molecule target, not the protein source).

Side effects (batch mode only):
  - Deletes DUPLICATE_DIRS and any .cif co-located with a .pdb.
  - Deletes the original .pdb after successful conversion.
  - Rewrites benchmark JSON/YAML configs for PDB→CIF renames and cross-dir renames.
  - Writes a Markdown failure report to benchmarks/conversion_failures.md.

Usage:
    source ~/protocols/atomworks/.venv/bin/activate

    # Full batch:
    python projects/rfproteina/script_utils/convert_pdb_benchmarks_to_cif.py

    # With re-parsed PDB coordinates:
    python ... --reparsed_pdb_dir /pdb_no_missing_atoms

    # Debug a single file (full traceback on failure):
    python ... --file projects/.../file.pdb [--reparsed_pdb_dir ...]
"""

import argparse
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
from atomworks.io import parse, prepare_atom_array
from atomworks.io.config import ParseConfig
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import load_any, to_cif_file
from atomworks.ml.conditions.annotator import ensure_annotations
from atomworks.ml.utils.testing import get_pdb_mirror_path
from biotite.structure import AtomArray, get_residue_starts, superimpose

# ---------------------------------------------------------------------------
# Constants and configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # projects/rfproteina
BENCHMARKS_ROOT = PROJECT_ROOT / "data" / "benchmarks"

# Canonical benchmark directories to process.
BENCHMARK_DIRS = [
    BENCHMARKS_ROOT / "mcsa_40",  # canonical (renamed from mcsa_41/); mcsa_41_cifs/ and rfbench/mcsa_41/ are duplicates
    BENCHMARKS_ROOT / "dna_pdbs",  # canonical; dna_cifs/ is the duplicate
    BENCHMARKS_ROOT / "motif_data_aa",
    BENCHMARKS_ROOT / "complexa_targets",
    BENCHMARKS_ROOT / "rfbench",
    BENCHMARKS_ROOT / "paper_model_rasa_inference",
    BENCHMARKS_ROOT / "rfd3_ppi_paper_benchmark",
    BENCHMARKS_ROOT / "ss_conditioned_proteina",
    BENCHMARKS_ROOT / "tip_atom_no_ligand_training_examples",
    BENCHMARKS_ROOT / "tip_atom_plinder_training_examples",
]

# Protein chains are stripped before RCSB matching in these dirs.
# The PDB ID identifies the small-molecule target, not the protein source.
EXCLUDE_PROTEIN_IN_DIRS: set[Path] = {
    BENCHMARKS_ROOT / "rfbench" / "sm_binder",
    BENCHMARKS_ROOT / "paper_model_rasa_inference",
}

# Model-predicted structures in these dirs skip RCSB lookup when no PDB ID is found.
MODEL_PREDICTED_DIRS: set[Path] = {
    BENCHMARKS_ROOT / "complexa_targets" / "bindcraft_targets",
}

# Stale duplicate directories deleted before processing.
DUPLICATE_DIRS = [
    BENCHMARKS_ROOT / "mcsa_41_cifs",
    BENCHMARKS_ROOT / "rfbench" / "mcsa_41",  # legacy names
    BENCHMARKS_ROOT / "dna_cifs",
]

# Path prefix substitutions for benchmark configs that reference deleted duplicate dirs.
CROSS_DIR_RENAMES = [
    ("mcsa_41/", "mcsa_40/"),
    ("mcsa_41_cifs/", "mcsa_40/"),
    ("dna_cifs/", "dna_pdbs/"),
]

BENCHMARK_CONFIGS = sorted(list(BENCHMARKS_ROOT.rglob("*.json")) + list(BENCHMARKS_ROOT.rglob("*.yaml")))

FAILURE_REPORT_PATH = BENCHMARKS_ROOT / "conversion_failures.md"

# Used for both benchmark files and the RCSB reference mirror.
PARSE_CONFIG = ParseConfig(
    remove_waters=False,
    convert_mse_to_met=True,
    remove_ccds=(),
    build_assembly=None,
    hydrogen_policy="remove",
)

# Used for the re-parsed PDB mirror; retains hydrogens, skips missing-atom inference.
REPARSED_PARSE_CONFIG = ParseConfig(
    add_missing_atoms=False,
    remove_ccds=(),
    remove_waters=True,
    build_assembly=None,
    hydrogen_policy="keep",
    add_id_and_entity_annotations=True,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Failure:
    path: Path
    category: str  # no_pdb_id | ambiguous_pdb_id | parse_error | subset_failure | reparsed_failure
    details: str = ""


@dataclass
class Skipped:
    path: Path
    reason: str  # relaxed


class ConversionError(Exception):
    """Raised by convert_one with a machine-readable failure category."""

    def __init__(self, category: str, message: str = ""):
        super().__init__(message or category)
        self.category = category


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _strip_single_atom_ori(path: Path) -> Path:
    """Return a temp PDB with single-atom ORI residues removed, or the original path unchanged.

    Single-atom ORI residues are legacy non-structural origin tokens; ORI is also a
    real multi-atom CCD ligand, so only the single-atom form is removed. CIF files
    are returned unchanged.

    Args:
      path: Path to the input structure file (.pdb or .cif).

    Returns:
      Path to a temp PDB file with single-atom ORI records removed, or the original
      path if no removal was needed (or the file is a CIF).
    """
    if path.suffix != ".pdb":
        return path
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    ori_groups: dict[tuple[str, str], list[int]] = {}
    for i, line in enumerate(lines):
        if (line.startswith("HETATM") or line.startswith("ATOM  ")) and line[17:20].strip() == "ORI":
            key = (line[21:22], line[22:26].strip())
            ori_groups.setdefault(key, []).append(i)
    to_remove = {idx for indices in ori_groups.values() if len(indices) == 1 for idx in indices}
    if not to_remove:
        return path
    filtered = [line for i, line in enumerate(lines) if i not in to_remove]
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.writelines(filtered)
    return Path(tmp.name)


def _split_nonpolymer_chains(aa: AtomArray) -> AtomArray:
    """Ensure each bond-disconnected non-polymer molecule has its own chain_id.

    Some benchmark files place disjoint non-polymer molecules on the same chain.
    Polymers may legitimately be disjoint on a single chain (e.g. a cropped structure
    with a gap), so only non-polymers are split.

    Args:
      aa: Input atom array, possibly with multiple non-polymer molecules sharing a chain.

    Returns:
      Atom array with disjoint non-polymer molecules assigned to distinct chain IDs,
      Returns ``aa`` unchanged if no splitting needed.
    """
    ensure_annotations(aa, "is_polymer")
    if not (~aa.is_polymer).any():
        return aa

    res_starts = get_residue_starts(aa, add_exclusive_stop=True)
    # Map each atom index to its residue index (0-based into res_starts).
    atom_res_idx = np.searchsorted(res_starts[:-1], np.arange(aa.array_length()), side="right") - 1

    nonpolymer_res_set = set(np.where(~aa.is_polymer[res_starts[:-1]])[0].tolist())
    bond_graph = nx.Graph()
    bond_graph.add_nodes_from(nonpolymer_res_set)
    if aa.bonds is not None:
        for bond in aa.bonds.as_array():
            r1, r2 = int(atom_res_idx[bond[0]]), int(atom_res_idx[bond[1]])
            if r1 != r2 and r1 in nonpolymer_res_set and r2 in nonpolymer_res_set:
                bond_graph.add_edge(r1, r2)

    connected_components = list(nx.connected_components(bond_graph))

    # Collect chains that contain more than one disconnected component.
    # A component may legitimately span multiple chains; track it under all of them.
    chain_component_map: dict[str, list[int]] = {}
    for comp_idx, comp in enumerate(connected_components):
        comp_chains = {aa.chain_id[res_starts[r]] for r in comp}
        for chain in comp_chains:
            chain_component_map.setdefault(chain, []).append(comp_idx)
    chains_needing_split = {chain: comps for chain, comps in chain_component_map.items() if len(comps) > 1}
    if not chains_needing_split:
        return aa

    aa = aa.copy()
    chain_id_gen = create_chain_id_generator(unavailable_chain_ids=np.unique(aa.chain_id))
    n_chains_split = 0
    for chain, component_indices in chains_needing_split.items():
        # Keep the first component's residues on this chain; for each other component,
        # reassign only the residues currently on this chain to a new chain ID.
        # This correctly handles multi-chain components: only the portion on the
        # problematic chain moves; the rest stays on its original chain(s).
        for comp_idx in component_indices[1:]:
            new_chain = next(chain_id_gen)
            for r in connected_components[comp_idx]:
                if aa.chain_id[res_starts[r]] == chain:
                    aa.chain_id[res_starts[r] : res_starts[r + 1]] = new_chain
            n_chains_split += 1

    print(f"    INFO: Split {n_chains_split} non-polymer molecule(s) onto separate chains")

    return aa


def _parse_structure(path: Path, split_nonpolymer_chains: bool = False) -> AtomArray:
    """Parse a structure from a CIF or PDB file using the asymmetric unit.

    Falls back to load_any() if parse() raises (e.g. malformed charge fields).

    Args:
      path: Path to the structure file (.cif or .pdb).
      split_nonpolymer_chains: If True, run :func:`_split_nonpolymer_chains` to
        ensure that no chain_id contains part of two distinct non-polymer molecules.

    Returns:
      Parsed :class:`~biotite.structure.AtomArray`.
    """

    def _unwrap(result: AtomArray | tuple) -> AtomArray:
        return result if isinstance(result, AtomArray) else result[0]

    try:
        parsed = parse(path, config=PARSE_CONFIG)
        aa = _unwrap(parsed["asym_unit"])
    except Exception as err:
        print(f"    WARNING: parse() failed ({err}); retrying with load_any()")
        aa = prepare_atom_array(_unwrap(load_any(path)))

    if split_nonpolymer_chains:
        aa = _split_nonpolymer_chains(aa)
    return aa


# ---------------------------------------------------------------------------
# RCSB matching
# ---------------------------------------------------------------------------


def _evaluate_candidate_matches(
    candidate_matches: np.ndarray,
    rcsb_aa: AtomArray,
    rcsb_res_starts: np.ndarray,
    benchmark_res: AtomArray,
    atol: float,
) -> tuple[list, list, list]:
    """Test each candidate RCSB residue against benchmark_res by coordinate alignment.

    Args:
      candidate_matches: RCSB residue indices (into rcsb_res_starts) to evaluate.
      rcsb_aa: Full RCSB atom array.
      rcsb_res_starts: Residue start indices for rcsb_aa (with exclusive stop).
      benchmark_res: Benchmark atoms for a single residue to match.
      atol: Absolute coordinate tolerance for np.allclose.

    Returns:
      Three parallel lists for passing candidates: RCSB residue indices,
      inverse permutations (benchmark masked → RCSB atom order), and benchmark
      atom masks (atoms used in matching).
    """
    match_rcsb_res_indices, match_inv_perms, match_bench_masks = [], [], []
    for candidate_match in candidate_matches:
        candidate_res = rcsb_aa[rcsb_res_starts[candidate_match] : rcsb_res_starts[candidate_match + 1]]
        candidate_res = candidate_res[np.isin(candidate_res.atom_name, benchmark_res.atom_name)]

        benchmark_mask = np.ones(benchmark_res.array_length(), dtype=bool)
        if candidate_res.array_length() != benchmark_res.array_length():
            # For matching purposes, ignore certain benchmark atoms:
            #   OXT: appears on artificially terminal residues in benchmark crops but is
            #        absent in the RCSB asym_unit where the residue is internal.
            #   all-NaN coords: leaving-group atoms added by add_missing_atoms when the
            #        bond is absent; they carry no coordinate information.
            benchmark_extra = set(benchmark_res.atom_name) - set(candidate_res.atom_name)
            for name in benchmark_extra:
                if name == "OXT" or np.all(np.isnan(benchmark_res.coord[benchmark_res.atom_name == name])):
                    benchmark_mask &= benchmark_res.atom_name != name
            # If the lengths still differ, this candidate can't be a match.
            if candidate_res.array_length() != benchmark_res[benchmark_mask].array_length():
                continue

        benchmark_res_matched = benchmark_res[benchmark_mask]

        # Build permutation: permutation_indices[i] = RCSB index for benchmark atom i.
        permutation_indices = np.argsort(candidate_res.atom_name)[
            np.argsort(np.argsort(benchmark_res_matched.atom_name))
        ]
        assert np.all(candidate_res[permutation_indices].atom_name == benchmark_res_matched.atom_name)
        inv_perm = np.argsort(permutation_indices)  # benchmark (masked) → RCSB order
        benchmark_res_reordered = benchmark_res_matched[inv_perm]
        assert np.all(benchmark_res_reordered.atom_name == candidate_res.atom_name)

        # Superimpose only on non-NaN atoms; still enforce atom-name matching on all atoms.
        non_nan_rcsb = ~np.isnan(candidate_res.coord).any(axis=1)
        non_nan_bench = ~np.isnan(benchmark_res_reordered.coord).any(axis=1)
        if np.any(~non_nan_rcsb & non_nan_bench):
            continue  # benchmark has extra atoms with coordinates that RCSB lacks
        non_nan_mask = non_nan_rcsb & non_nan_bench

        aligned_nonnan, _ = superimpose(benchmark_res_reordered[non_nan_mask], candidate_res[non_nan_mask])

        if np.allclose(aligned_nonnan.coord, benchmark_res_reordered[non_nan_mask].coord, atol=atol):
            match_rcsb_res_indices.append(candidate_match)
            match_inv_perms.append(inv_perm)
            match_bench_masks.append(benchmark_mask)

    return match_rcsb_res_indices, match_inv_perms, match_bench_masks


def _inter_residue_bonds(aa: AtomArray, res_starts: np.ndarray) -> frozenset[tuple[int, int]]:
    """Return unordered (res_i, res_j) pairs connected by a covalent bond.

    Args:
      aa: Atom array with bond information.
      res_starts: Residue start indices (with exclusive stop) from get_residue_starts.

    Returns:
      Frozenset of (min_res_idx, max_res_idx) pairs for each inter-residue bond.
    """
    if aa.bonds is None:
        return frozenset()
    bond_array = aa.bonds.as_array()
    if len(bond_array) == 0:
        return frozenset()
    atom_res_idx = np.searchsorted(res_starts, np.arange(aa.array_length()), side="right") - 1
    r1, r2 = atom_res_idx[bond_array[:, 0]], atom_res_idx[bond_array[:, 1]]
    inter = r1 != r2
    if not inter.any():
        return frozenset()
    return frozenset(
        zip(
            np.minimum(r1[inter], r2[inter]).tolist(),
            np.maximum(r1[inter], r2[inter]).tolist(),
            strict=False,
        )
    )


def _match_mol_to_rcsb_mol(
    rcsb_aa: AtomArray,
    rcsb_res_starts: np.ndarray,
    rcsb_mol_res_indices: np.ndarray,
    benchmark_aa: AtomArray,
    bench_res_starts: np.ndarray,
    bench_res_in_mol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Match all residues of one benchmark molecule to one RCSB molecule.

    Matching uses two passes:
      1. Coordinate alignment (atol 1e-3, then 1e-4 for tiebreaking).
      2. Bond-based disambiguation: residues bonded to an already-confirmed match
         are further filtered to RCSB candidates bonded to that same match.

    Args:
      rcsb_aa: Full RCSB atom array.
      rcsb_res_starts: Residue start indices for rcsb_aa (with exclusive stop).
      rcsb_mol_res_indices: Residue indices (into rcsb_res_starts) for the RCSB molecule.
      benchmark_aa: Full benchmark atom array.
      bench_res_starts: Residue start indices for benchmark_aa (with exclusive stop).
      bench_res_in_mol: Residue indices (into bench_res_starts) for the benchmark molecule.

    Returns:
      Tuple of (rcsb_atom_indices, bench_atom_indices) where bench_atom_indices are
      permuted to match RCSB atom-name order within each residue.
      Raises ValueError if any residue cannot be matched uniquely.
    """
    rcsb_res_names = rcsb_aa.res_name[rcsb_res_starts[rcsb_mol_res_indices]]

    # confirmed: bench_res_idx → (rcsb_res_idx, inv_perm, benchmark_mask)
    confirmed: dict[int, tuple] = {}
    # pending:   bench_res_idx → (match_idxs, match_perms, match_bench_masks)
    pending: dict[int, tuple] = {}

    # Pass 1: coordinate-based matching.
    for i in bench_res_in_mol:
        bench_res = benchmark_aa[bench_res_starts[i] : bench_res_starts[i + 1]]
        candidates = rcsb_mol_res_indices[rcsb_res_names == bench_res.res_name[0]]

        m_idx, m_perm, m_masks = _evaluate_candidate_matches(candidates, rcsb_aa, rcsb_res_starts, bench_res, atol=1e-3)
        if len(m_idx) > 1:
            m_idx, m_perm, m_masks = _evaluate_candidate_matches(
                candidates, rcsb_aa, rcsb_res_starts, bench_res, atol=1e-4
            )

        if len(m_idx) == 0:
            raise ValueError(
                f"No match for residue at index {bench_res_starts[i]} ({bench_res.res_name[0]}) within RCSB molecule"
            )
        elif len(m_idx) == 1:
            confirmed[i] = (m_idx[0], m_perm[0], m_masks[0])
        else:
            pending[i] = (m_idx, m_perm, m_masks)

    # Pass 2: bond-based disambiguation, iterated until stable.
    if pending:
        bench_bonds = _inter_residue_bonds(benchmark_aa, bench_res_starts)
        rcsb_bonds = _inter_residue_bonds(rcsb_aa, rcsb_res_starts)
        bench_mol_set = set(bench_res_in_mol.tolist())

        changed = True
        while changed and pending:
            changed = False
            for i in list(pending.keys()):
                bonded_confirmed = {
                    j: confirmed[j][0]
                    for j in bench_mol_set
                    if j in confirmed and (min(i, j), max(i, j)) in bench_bonds
                }
                if not bonded_confirmed:
                    continue
                m_idx, m_perm, m_masks = pending[i]
                keep = [
                    k
                    for k, c in enumerate(m_idx)
                    if all((min(c, rj), max(c, rj)) in rcsb_bonds for rj in bonded_confirmed.values())
                ]
                if len(keep) == 0:
                    raise ValueError(
                        f"Bond constraints eliminated all candidates for residue at index "
                        f"{bench_res_starts[i]} ({benchmark_aa.res_name[bench_res_starts[i]]})"
                    )
                if len(keep) == 1:
                    k = keep[0]
                    confirmed[i] = (m_idx[k], m_perm[k], m_masks[k])
                    del pending[i]
                    changed = True
                else:
                    pending[i] = ([m_idx[k] for k in keep], [m_perm[k] for k in keep], [m_masks[k] for k in keep])

        # Any remaining pending residues are single-atom (e.g. metals) that can't be
        # further disambiguated; pick the first candidate arbitrarily.
        for i, (m_idx, m_perm, m_masks) in pending.items():
            bench_res = benchmark_aa[bench_res_starts[i] : bench_res_starts[i + 1]]
            if len(bench_res) != 1:
                raise ValueError(
                    f"Expected exactly one match for residue at index {bench_res_starts[i]} "
                    f"({bench_res.res_name[0]}), found {len(m_idx)}"
                )
            confirmed[i] = (m_idx[0], m_perm[0], m_masks[0])

    rcsb_atom_indices: list[np.ndarray] = []
    bench_atom_indices: list[np.ndarray] = []

    for i in bench_res_in_mol:
        rcsb_res_idx, inv_perm, benchmark_mask = confirmed[i]
        bench_res = benchmark_aa[bench_res_starts[i] : bench_res_starts[i + 1]]
        residue_global = np.arange(rcsb_res_starts[rcsb_res_idx], rcsb_res_starts[rcsb_res_idx + 1])
        rcsb_atom_indices.append(residue_global)
        bench_global = np.arange(bench_res_starts[i], bench_res_starts[i + 1])
        bench_atom_indices.append(bench_global[benchmark_mask][inv_perm])

    return np.concatenate(rcsb_atom_indices), np.concatenate(bench_atom_indices)


def subset_with_reference(
    rcsb_pdb_atom_array: AtomArray,
    benchmark_atom_array: AtomArray,
) -> AtomArray:
    """Match benchmark atoms to RCSB by residue name and coordinates.

    For each benchmark molecule, finds the matching RCSB molecule and aligns all
    residues by coordinate superimposition (with bond-based tiebreaking).

    Returns an AtomArray built from the matched benchmark atoms (preserving benchmark
    chain_id/res_id/coords), reordered to match RCSB atom-name order within each
    residue.  RCSB-native identifiers are stored as extra annotations (``rcsb_chain_id``,
    ``rcsb_res_id``, ``rcsb_ins_code``) so that :func:`extract_from_reparsed` can look
    up the same atoms in a re-parsed PDB without any separate return value.
    """
    assert isinstance(rcsb_pdb_atom_array, AtomArray) and isinstance(benchmark_atom_array, AtomArray)
    assert "chain_type" in rcsb_pdb_atom_array.get_annotation_categories()

    bench_res_starts = get_residue_starts(benchmark_atom_array, add_exclusive_stop=True)
    rcsb_res_starts = get_residue_starts(rcsb_pdb_atom_array, add_exclusive_stop=True)

    # Per-residue chain_id — groups residues belonging to the same chain (molecule instance).
    rcsb_res_chain = rcsb_pdb_atom_array.chain_id[rcsb_res_starts[:-1]]
    bench_res_chain = benchmark_atom_array.chain_id[bench_res_starts[:-1]]

    # Unique benchmark chains in order of first occurrence.
    _, first_occ = np.unique(bench_res_chain, return_index=True)
    bench_chains = bench_res_chain[np.sort(first_occ)]

    all_rcsb_atom_indices: list[np.ndarray] = []
    all_bench_atom_indices: list[np.ndarray] = []

    for bench_chain in bench_chains:
        bench_res_in_mol = np.where(bench_res_chain == bench_chain)[0]
        first_res_name = benchmark_atom_array.res_name[bench_res_starts[bench_res_in_mol[0]]]
        rcsb_res_names = rcsb_pdb_atom_array.res_name[rcsb_res_starts[:-1]]
        # Candidate RCSB chains that contain the first residue name of this benchmark chain.
        candidate_rcsb_chains = np.unique(rcsb_res_chain[rcsb_res_names == first_res_name])

        rcsb_atom_indices = bench_atom_indices = None
        last_err: Exception | None = None

        for rcsb_chain in candidate_rcsb_chains:
            rcsb_mol_res_indices = np.where(rcsb_res_chain == rcsb_chain)[0]
            try:
                rcsb_atom_indices, bench_atom_indices = _match_mol_to_rcsb_mol(
                    rcsb_pdb_atom_array,
                    rcsb_res_starts,
                    rcsb_mol_res_indices,
                    benchmark_atom_array,
                    bench_res_starts,
                    bench_res_in_mol,
                )
                break
            except ValueError as e:
                last_err = e

        if rcsb_atom_indices is None:
            raise ValueError(f"No consistent RCSB chain found for benchmark chain {bench_chain!r}: {last_err}")

        all_rcsb_atom_indices.append(rcsb_atom_indices)
        all_bench_atom_indices.append(bench_atom_indices)

    rcsb_atom_indices_flat = np.concatenate(all_rcsb_atom_indices)
    bench_atom_indices_flat = np.concatenate(all_bench_atom_indices)

    # Sort by RCSB index to preserve the original RCSB atom ordering in the output.
    rcsb_sort_order = np.argsort(rcsb_atom_indices_flat, kind="stable")
    rcsb_atom_indices_flat = rcsb_atom_indices_flat[rcsb_sort_order]
    bench_atom_indices_flat = bench_atom_indices_flat[rcsb_sort_order]

    # Build output from benchmark atoms (correct labels and coords already in place).
    # RCSB is used only for ordering and to supply native identifiers as annotations.
    output_aa = benchmark_atom_array[bench_atom_indices_flat]
    rcsb_matched_aa = rcsb_pdb_atom_array[rcsb_atom_indices_flat]

    # Sanity check: atom-name ordering must agree between benchmark (permuted) and RCSB.
    assert np.all(output_aa.atom_name == rcsb_matched_aa.atom_name), (
        "Atom name mismatch between benchmark and RCSB after permutation — "
        "check inv_perm logic in _evaluate_candidate_matches."
    )

    # Sanity check: benchmark coordinates should superimpose closely onto RCSB coordinates.
    non_nan = (~np.isnan(output_aa.coord).any(axis=1)) & (~np.isnan(rcsb_matched_aa.coord).any(axis=1))
    if non_nan.sum() >= 3:
        aligned, _ = superimpose(output_aa[non_nan], rcsb_matched_aa[non_nan])
        diff = aligned.coord - output_aa[non_nan].coord
        rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        assert rmsd < 1, (
            f"Coordinate RMSD between benchmark and RCSB is {rmsd:.4f} Å — "
            "possible mislabelling or coordinate mismatch after matching."
        )

    output_aa.set_annotation("rcsb_chain_id", rcsb_matched_aa.chain_id.copy())
    output_aa.set_annotation("rcsb_res_id", rcsb_matched_aa.res_id.copy())
    output_aa.set_annotation("rcsb_ins_code", rcsb_matched_aa.ins_code.copy())

    # Ensure polymers and non-polymers never share a chain_id (a common benchmark artifact).
    ensure_annotations(output_aa, "is_ligand")
    ligand_mask = output_aa.is_ligand
    duplicate_chain_ids = set(output_aa[ligand_mask].chain_id) & set(output_aa[~ligand_mask].chain_id)
    if duplicate_chain_ids:
        max_ord = max(ord(c) for c in np.unique(output_aa.chain_id))
        for idx, dup_chain in enumerate(duplicate_chain_ids):
            new_chain = chr(max_ord + idx + 1)
            lig_mask = output_aa.is_ligand & (output_aa.chain_id == dup_chain)
            output_aa.chain_id[lig_mask] = new_chain
            # rcsb_chain_id is intentionally NOT updated — it must keep the original
            # RCSB chain_id so that extract_from_reparsed can locate these atoms.
            print(
                f"    WARNING: Chain {dup_chain} has both polymer and non-polymer residues; "
                f"non-polymers reassigned to chain {new_chain}"
            )

    return output_aa


# ---------------------------------------------------------------------------
# Re-parsed PDB extraction
# ---------------------------------------------------------------------------


def get_reparsed_pdb_path(pdb_id: str, reparsed_pdb_dir: str, altloc_id: int | str = 0) -> str:
    """Return path to the pre-processed CIF for a given PDB entry.

    Tries ``{dir}/{mid2}/{pdb_id}/{pdb_id}_{altloc_id}.cif`` first, then
    the variant without the altloc suffix.
    """
    pdb_id = pdb_id.lower()
    primary = os.path.join(reparsed_pdb_dir, pdb_id[1:3], pdb_id, f"{pdb_id}_{altloc_id}.cif")
    fallback = os.path.join(reparsed_pdb_dir, pdb_id[1:3], pdb_id, f"{pdb_id}.cif")
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        return fallback
    raise ValueError(f"Reparsed CIF not found: tried {primary} and {fallback}")


def _enumerate_reparsed_paths(pdb_id: str, reparsed_pdb_dir: str) -> list[str]:
    """Return all available altloc CIF paths for pdb_id in ascending altloc order.

    Scans the PDB entry directory for files matching ``{pdb_id}_<int>.cif`` and
    returns them sorted by altloc index.  If no altloc-specific files exist,
    falls back to the non-suffixed ``{pdb_id}.cif``.

    Args:
      pdb_id: Four-character PDB identifier.
      reparsed_pdb_dir: Root of the re-parsed PDB mirror.

    Returns:
      Ordered list of existing CIF paths to try, or an empty list if none found.
    """
    pdb_id = pdb_id.lower()
    pdb_dir = Path(reparsed_pdb_dir) / pdb_id[1:3] / pdb_id
    if not pdb_dir.is_dir():
        return []

    prefix = f"{pdb_id}_"
    altloc_paths: list[tuple[int, Path]] = []
    for p in pdb_dir.iterdir():
        if p.suffix == ".cif" and p.stem.startswith(prefix):
            altloc = p.stem[len(prefix) :]
            if altloc.isdigit():
                altloc_paths.append((int(altloc), p))

    if altloc_paths:
        return [str(p) for _, p in sorted(altloc_paths)]

    fallback = pdb_dir / f"{pdb_id}.cif"
    return [str(fallback)] if fallback.exists() else []


def extract_from_reparsed(
    matched_aa: AtomArray,
    reparsed_aa: AtomArray,
) -> AtomArray:
    """Extract atoms from reparsed_aa that correspond to matched_aa.

    Uses the ``rcsb_chain_id``, ``rcsb_res_id``, and ``rcsb_ins_code`` annotations on
    ``matched_aa`` (set by :func:`subset_with_reference`) to look up each atom in the
    re-parsed PDB by its original RCSB identity.  Lookup is vectorised via sorted string keys and
    ``np.searchsorted``.

    The mapping must be unique — raises ValueError if any atom matches zero or more than
    one atom in the re-parsed PDB.  The returned array carries benchmark chain_id/res_id
    labels from ``matched_aa`` so it is labelled identically to the RCSB-match output.
    """

    def _make_keys(chain_id: np.ndarray, res_id: np.ndarray, ins_code: np.ndarray, atom_name: np.ndarray) -> np.ndarray:
        """Build a composite sort key per atom as a single string."""
        return np.array(
            [f"{c}\x00{r}\x00{i}\x00{a}" for c, r, i, a in zip(chain_id, res_id, ins_code, atom_name, strict=False)]
        )

    reparsed_keys = _make_keys(
        reparsed_aa.chain_id,
        reparsed_aa.res_id,
        reparsed_aa.ins_code,
        reparsed_aa.atom_name,
    )
    query_keys = _make_keys(
        matched_aa.rcsb_chain_id,
        matched_aa.rcsb_res_id,
        matched_aa.rcsb_ins_code,
        matched_aa.atom_name,
    )

    # Sort reparsed keys once; use searchsorted for O(n log n) lookup.
    sort_order = np.argsort(reparsed_keys)
    reparsed_keys_sorted = reparsed_keys[sort_order]
    search_positions = np.searchsorted(reparsed_keys_sorted, query_keys)

    in_bounds = search_positions < len(reparsed_keys_sorted)
    exact_match = in_bounds & (
        reparsed_keys_sorted[np.minimum(search_positions, len(reparsed_keys_sorted) - 1)] == query_keys
    )

    if not exact_match.all():
        missing = query_keys[~exact_match][0]
        parts = missing.split("\x00")
        raise ValueError(
            f"No atom in reparsed PDB matches RCSB key "
            f"chain={parts[0]!r} res_id={parts[1]} ins_code={parts[2]!r} atom={parts[3]!r}"
        )

    # Check uniqueness: a key appears more than once if the next sorted entry is identical.
    next_pos = search_positions + 1
    has_next = next_pos < len(reparsed_keys_sorted)
    is_duplicate = has_next & (reparsed_keys_sorted[np.minimum(next_pos, len(reparsed_keys_sorted) - 1)] == query_keys)
    if is_duplicate.any():
        dup = query_keys[is_duplicate][0]
        parts = dup.split("\x00")
        raise ValueError(
            f"Ambiguous match in reparsed PDB for RCSB key "
            f"chain={parts[0]!r} res_id={parts[1]} ins_code={parts[2]!r} atom={parts[3]!r}"
        )

    extracted_aa = reparsed_aa[sort_order[search_positions]]

    # Save original RCSB identifiers for reference
    extracted_aa.set_annotation("rcsb_chain_id", extracted_aa.chain_id.copy())
    extracted_aa.set_annotation("rcsb_res_id", extracted_aa.res_id.copy())

    # Overwrite with benchmark identifiers so benchmark jsons still work
    extracted_aa.res_id[:] = matched_aa.res_id
    extracted_aa.chain_id[:] = matched_aa.chain_id
    return extracted_aa


def _reparsed_rmsd(reference: AtomArray, reparsed: AtomArray) -> float:
    """RMSD between reference and re-parsed coords after superimposition (non-NaN only).

    Args:
      reference: RCSB-matched atom array used as the alignment target.
      reparsed: Reparsed atom array to superimpose onto reference.

    Returns:
      RMSD in Ångströms, or 0.0 if fewer than 3 valid (non-NaN) atoms are present.
    """
    non_nan_mask = ~np.isnan(reference.coord).any(axis=1)
    if non_nan_mask.sum() < 3:
        return 0.0
    aligned_reparsed, _ = superimpose(reference[non_nan_mask], reparsed[non_nan_mask])
    diff = aligned_reparsed.coord - reference[non_nan_mask].coord
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


# ---------------------------------------------------------------------------
# Per-file conversion
# ---------------------------------------------------------------------------


def _is_in_dir_set(path: Path, dirs: set[Path]) -> bool:
    """Return True if path is inside any directory in dirs.

    Args:
      path: File path to test.
      dirs: Set of directory paths to check against.

    Returns:
      True if path is a descendant of at least one directory in dirs.
    """
    for d in dirs:
        try:
            path.relative_to(d)
            return True
        except ValueError:
            pass
    return False


def find_pdb_id(path: Path, rcsb_pdb_dir: str) -> tuple[str | None, list[str]]:
    """Return (pdb_id, all_candidates) from underscore-delimited stem groups.

    Args:
      path: Structure file path whose stem is parsed for PDB ID candidates.
      rcsb_pdb_dir: Root directory of the RCSB PDB mirror.

    Returns:
      Tuple of (pdb_id, candidates) where pdb_id is None if zero or multiple
      candidates matched the mirror, and candidates is the full list of matches.
    """
    candidates = []
    for part in path.stem.split("_"):
        if len(part) == 4 and part.isalnum():
            try:
                get_pdb_mirror_path(part.lower(), base_dir=rcsb_pdb_dir)
                candidates.append(part.lower())
            except Exception:
                pass
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def center_structure(aa: AtomArray) -> AtomArray:
    """Translate coordinates so the geometric mean of all atoms is at the origin."""
    aa = aa.copy()
    aa.coord = aa.coord - np.nanmean(aa.coord, axis=-2)
    return aa


def convert_one(
    benchmark_path: Path,
    rcsb_pdb_dir: str,
    reparsed_pdb_dir: str | None = None,
) -> Path:
    """Convert one benchmark file to CIF and return the output path.

    Args:
      benchmark_path: Path to the input .pdb or .cif benchmark file.
      rcsb_pdb_dir: Root directory of the original RCSB PDB mirror.
      reparsed_pdb_dir: Optional root of a re-parsed PDB mirror; if given, output
        uses reparsed coordinates instead of RCSB-native coordinates.

    Returns:
      Path to the written CIF file (same stem as input, .cif suffix).
      Raises :class:`ConversionError` on expected failures (no PDB ID, subset mismatch,
      reparsed lookup failure) and may propagate other exceptions on parse errors.
    """
    pdb_id, candidates = find_pdb_id(benchmark_path, rcsb_pdb_dir)

    if pdb_id is None:
        # Model-predicted structures with no PDB ID: write parsed structure directly.
        if not candidates and _is_in_dir_set(benchmark_path, MODEL_PREDICTED_DIRS):
            clean_path = _strip_single_atom_ori(benchmark_path)
            try:
                benchmark_aa = _parse_structure(clean_path, split_nonpolymer_chains=True)
            finally:
                if clean_path != benchmark_path:
                    clean_path.unlink()
            output_path = benchmark_path.with_suffix(".cif")
            to_cif_file(center_structure(benchmark_aa), output_path)
            return output_path
        if not candidates:
            raise ConversionError(
                "no_pdb_id",
                "no 4-char alphanumeric underscore-delimited token found in PDB mirror",
            )
        raise ConversionError(
            "ambiguous_pdb_id",
            f"multiple valid PDB IDs: {', '.join(candidates)}",
        )

    clean_path = _strip_single_atom_ori(benchmark_path)
    try:
        benchmark_aa = _parse_structure(clean_path, split_nonpolymer_chains=True)
    finally:
        if clean_path != benchmark_path:
            clean_path.unlink()

    if _is_in_dir_set(benchmark_path, EXCLUDE_PROTEIN_IN_DIRS):
        ensure_annotations(benchmark_aa, "is_protein")
        benchmark_aa = benchmark_aa[~benchmark_aa.is_protein]
        if benchmark_aa.array_length() == 0:
            raise ConversionError(
                "no_pdb_id",
                "benchmark contains only protein atoms after protein exclusion",
            )

    rcsb_aa = _parse_structure(
        get_pdb_mirror_path(pdb_id, base_dir=rcsb_pdb_dir),
        split_nonpolymer_chains=False,
    )

    try:
        output_aa = subset_with_reference(rcsb_aa, benchmark_aa)
    except Exception as err:
        raise ConversionError("subset_failure", str(err)) from err

    if reparsed_pdb_dir is not None:
        reparsed_paths = _enumerate_reparsed_paths(pdb_id, reparsed_pdb_dir)
        if not reparsed_paths:
            raise ConversionError(
                "reparsed_failure",
                f"No reparsed CIF found for {pdb_id} in {reparsed_pdb_dir}",
            )

        def _unwrap(result: AtomArray | tuple) -> AtomArray:
            return result if isinstance(result, AtomArray) else result[0]

        best: tuple[float, AtomArray, str] | None = None  # (rmsd, array, path_name)
        last_err = ""
        for reparsed_path in reparsed_paths:
            reparsed_aa = _unwrap(parse(reparsed_path, config=REPARSED_PARSE_CONFIG)["asym_unit"])
            try:
                candidate = extract_from_reparsed(output_aa, reparsed_aa)
            except ValueError as e:
                last_err = str(e)
                continue
            rmsd = _reparsed_rmsd(output_aa, candidate)
            if best is None or rmsd < best[0]:
                best = (rmsd, candidate, Path(reparsed_path).name)

        if best is None or best[0] >= 1.0:
            if best is not None:
                last_err = f"RMSD {best[0]:.3f} Å ≥ 1.0 Å threshold"
            raise ConversionError("reparsed_failure", last_err)

        best_rmsd, best_aa, best_name = best
        if best_name != Path(reparsed_paths[0]).name:
            print(f"    INFO: using altloc {best_name} (lowest RMSD {best_rmsd:.3f} Å)")
        output_aa = best_aa

    output_path = benchmark_path.with_suffix(".cif")
    to_cif_file(
        center_structure(output_aa),
        output_path,
        extra_fields=["rcsb_res_id", "rcsb_chain_id"],
    )
    return output_path


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def collect_files(dirs: list[Path]) -> tuple[list[Path], list[Skipped]]:
    """Collect all .cif/.pdb files under the given directories, skipping relaxed variants.

    Args:
      dirs: Benchmark directories to recurse into (missing dirs are silently skipped).

    Returns:
      Tuple of (files_to_process, skipped) where files_to_process is deduplicated and
      sorted, and skipped contains :class:`Skipped` entries for relaxed variants.
    """
    files: list[Path] = []
    skipped: list[Skipped] = []
    seen: set[Path] = set()
    for d in dirs:
        if not d.exists():
            continue
        for ext in ("*.cif", "*.pdb"):
            for f in sorted(d.rglob(ext)):
                if f in seen:
                    continue
                seen.add(f)
                if "relaxed" in f.stem.lower():
                    skipped.append(Skipped(f, "relaxed"))
                else:
                    files.append(f)
    return files, skipped


def update_configs(replacements: dict[str, str]) -> int:
    """Replace old path strings with new ones in all benchmark JSON/YAML configs.

    Args:
      replacements: Mapping of old path string → new path string.

    Returns:
      Number of config files modified.
    """
    n_updated = 0
    for cfg_path in BENCHMARK_CONFIGS:
        text = cfg_path.read_text()
        new_text = text
        for old, new in replacements.items():
            new_text = new_text.replace(old, new)
        if new_text != text:
            cfg_path.write_text(new_text)
            n_updated += 1
            print(f"  Updated config: {cfg_path.relative_to(PROJECT_ROOT)}")
    return n_updated


def _find_referenced_files(config_files: list[Path]) -> dict[str, list[str]]:
    """Map each structure filename/path snippet to the configs that reference it.

    Args:
      config_files: List of JSON/YAML config paths to scan.

    Returns:
      Dict mapping each matched path snippet (e.g. ``"foo/bar.pdb"``) to a list of
      config labels (relative paths) that contain it.
    """
    refs: dict[str, list[str]] = {}
    pattern = re.compile(r"[\w./\-]+\.(?:pdb|cif)")
    for cfg in config_files:
        try:
            text = cfg.read_text()
        except Exception:
            continue
        label = cfg.relative_to(BENCHMARKS_ROOT).as_posix()
        for match in pattern.findall(text):
            refs.setdefault(match, []).append(label)
    return refs


def _referencing_configs(path: Path, refs: dict[str, list[str]]) -> list[str]:
    """Return config files that reference this path, matched by stem with either .pdb/.cif suffix.

    Args:
      path: Structure file path whose stem is used for matching.
      refs: Output of :func:`_find_referenced_files`.

    Returns:
      Sorted list of config labels that reference this file's stem.
    """
    stem = path.stem
    hits: set[str] = set()
    for key, cfgs in refs.items():
        if stem in key and (key.endswith(".pdb") or key.endswith(".cif")):
            hits.update(cfgs)
    return sorted(hits)


CATEGORY_META: dict[str, tuple[str, str]] = {
    "no_pdb_id": (
        "No PDB ID found",
        "No underscore-delimited 4-character alphanumeric group matched the PDB mirror. "
        "Includes de novo designs, AlphaFold models, and structures without an RCSB counterpart.",
    ),
    "ambiguous_pdb_id": (
        "Ambiguous PDB ID",
        "Multiple underscore-delimited 4-character groups each matched the PDB mirror; "
        "the correct reference cannot be determined unambiguously.",
    ),
    "parse_error": (
        "Parse error",
        "The benchmark file or its RCSB reference could not be parsed. "
        "Common causes: malformed charge fields, empty files, non-standard format.",
    ),
    "subset_failure": (
        "Subset/alignment failure",
        "subset_with_reference could not align all benchmark residues to the RCSB structure. "
        "Common causes: modified/mutated/de novo residues, heavily cropped structures, "
        "residue name mismatches.",
    ),
    "reparsed_failure": (
        "Re-parsed PDB failure",
        "RCSB matching succeeded but the re-parsed PDB lookup failed. "
        "Common causes: entry absent from the re-parsed mirror, non-unique atom identifier, "
        "or RMSD sanity check failure (coords differ by > 1 Å after superimposition).",
    ),
}

SKIP_META: dict[str, str] = {
    "relaxed": "Files with 'relaxed' in the name — coordinates likely differ from RCSB source.",
}


def write_failure_report(
    failures: list[Failure],
    skipped: list[Skipped],
    referenced: dict[str, list[str]],
    deleted_dirs: list[Path],
    missing_dirs: list[Path],
) -> None:
    """Write a Markdown conversion report to FAILURE_REPORT_PATH.

    Args:
      failures: List of :class:`Failure` records from the batch run.
      skipped: List of :class:`Skipped` records (e.g. relaxed variants).
      referenced: Output of :func:`_find_referenced_files` for ⚠️ annotations.
      deleted_dirs: Duplicate directories that were deleted before processing.
      missing_dirs: Benchmark directories that did not exist on disk.
    """
    lines = [
        "# Benchmark conversion report",
        "",
        f"**Failures:** {len(failures)}  |  **Skipped:** {len(skipped)}",
        "",
        "Files marked ⚠️ are referenced in at least one benchmark config (JSON/YAML) and may need manual attention.",
        "",
        "---",
        "",
    ]

    if deleted_dirs:
        lines += ["## Deleted duplicate directories", ""]
        for d in deleted_dirs:
            lines.append(f"- `{d.relative_to(BENCHMARKS_ROOT)}/`")
        lines.append("")

    if missing_dirs:
        lines += ["## Benchmark directories not found (skipped)", ""]
        for d in missing_dirs:
            lines.append(f"- `{d.relative_to(BENCHMARKS_ROOT)}/`")
        lines.append("")

    failures_by_category: dict[str, list[Failure]] = {}
    for f in failures:
        failures_by_category.setdefault(f.category, []).append(f)

    for cat, (title, description) in CATEGORY_META.items():
        items = failures_by_category.get(cat, [])
        if not items:
            continue
        lines += [f"## {title} ({len(items)})", "", description, ""]
        for f in sorted(items, key=lambda x: x.path):
            rel = f.path.relative_to(BENCHMARKS_ROOT).as_posix()
            refs = _referencing_configs(f.path, referenced)
            flag = " ⚠️" if refs else ""
            ref_note = f" — referenced in: {', '.join(refs)}" if refs else ""
            lines.append(f"- `{rel}`{flag}{ref_note}")
            if f.details:
                lines.append(f"  `{f.details}`")
        lines.append("")

    skipped_by_reason: dict[str, list[Skipped]] = {}
    for s in skipped:
        skipped_by_reason.setdefault(s.reason, []).append(s)

    if skipped_by_reason:
        lines += ["## Skipped files", ""]
        for reason, items in sorted(skipped_by_reason.items()):
            lines += [
                f"### {reason.replace('_', ' ').title()} ({len(items)})",
                "",
                SKIP_META.get(reason, ""),
                "",
            ]
            for s in sorted(items, key=lambda x: x.path):
                rel = s.path.relative_to(BENCHMARKS_ROOT).as_posix()
                refs = _referencing_configs(s.path, referenced)
                flag = " ⚠️" if refs else ""
                ref_note = f" — referenced in: {', '.join(refs)}" if refs else ""
                lines.append(f"- `{rel}`{flag}{ref_note}")
            lines.append("")

    FAILURE_REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nWrote report to {FAILURE_REPORT_PATH.relative_to(PROJECT_ROOT)}")


def debug_one(benchmark_path: Path, rcsb_pdb_dir: str, reparsed_pdb_dir: str | None) -> None:
    """Convert a single file, letting all exceptions propagate for full tracebacks."""
    rel = benchmark_path.relative_to(PROJECT_ROOT)
    print(f"Debug converting: {rel}")
    output_path = convert_one(benchmark_path, rcsb_pdb_dir, reparsed_pdb_dir)
    print(f"  OK → {output_path.relative_to(PROJECT_ROOT)}")


def main(rcsb_pdb_dir: str, reparsed_pdb_dir: str | None) -> None:
    """Run the full batch conversion pipeline over all BENCHMARK_DIRS.

    Args:
      rcsb_pdb_dir: Root of the RCSB PDB mirror used for reference matching.
      reparsed_pdb_dir: Optional re-parsed PDB mirror; if given, output uses its coords.
    """
    if not os.path.isdir(rcsb_pdb_dir):
        raise SystemExit(f"ERROR: RCSB PDB directory not found: {rcsb_pdb_dir}")
    if reparsed_pdb_dir is not None and not os.path.isdir(reparsed_pdb_dir):
        raise SystemExit(f"ERROR: Re-parsed PDB directory not found: {reparsed_pdb_dir}")

    # Delete stale duplicate directories before collecting files.
    print("Cleaning up duplicate directories...")
    deleted_dirs: list[Path] = []
    for duplicate_dir in DUPLICATE_DIRS:
        if duplicate_dir.exists():
            shutil.rmtree(duplicate_dir)
            deleted_dirs.append(duplicate_dir)
            print(f"  Deleted {duplicate_dir.relative_to(PROJECT_ROOT)}/")

    missing_dirs = [d for d in BENCHMARK_DIRS if not d.exists()]
    for d in missing_dirs:
        print(f"  WARNING: benchmark dir not found: {d.relative_to(PROJECT_ROOT)}/")

    # Remove any .cif that has a co-located .pdb with the same stem; the PDB is the original
    # After this workflow, only the converted CIFs should remain
    print("\nPre-processing: removing CIFs superseded by a co-located PDB...")
    for d in BENCHMARK_DIRS:
        if not d.exists():
            continue
        for pdb_file in sorted(d.rglob("*.pdb")):
            cif_file = pdb_file.with_suffix(".cif")
            if cif_file.exists():
                cif_file.unlink()
                print(f"  Deleted {cif_file.relative_to(PROJECT_ROOT)}")

    all_files, skipped = collect_files(BENCHMARK_DIRS)
    print(f"\nFound {len(all_files)} files to process ({len(skipped)} skipped before processing)")

    failures: list[Failure] = []
    pdb_to_cif: dict[Path, Path] = {}

    for benchmark_path in all_files:
        rel = benchmark_path.relative_to(PROJECT_ROOT)
        try:
            output_path = convert_one(benchmark_path, rcsb_pdb_dir, reparsed_pdb_dir)
            print(f"  OK   {rel}")
            if benchmark_path.suffix == ".pdb":
                pdb_to_cif[benchmark_path] = output_path
                benchmark_path.unlink()
        except ConversionError as e:
            failures.append(Failure(benchmark_path, e.category, str(e)))
            print(f"  FAIL [{e.category}]  {rel}")
        except Exception as e:
            failures.append(Failure(benchmark_path, "parse_error", str(e)))
            print(f"  FAIL [parse_error]  {rel}: {e}")

    # Update configs for PDB→CIF renames within each benchmark directory.
    print("\nUpdating config files for PDB→CIF renames...")
    replacements: dict[str, str] = {}
    for pdb_path, cif_path in pdb_to_cif.items():
        try:
            rel_pdb = str(pdb_path.relative_to(BENCHMARKS_ROOT))
            rel_cif = str(cif_path.relative_to(BENCHMARKS_ROOT))
        except ValueError:
            continue
        replacements[rel_pdb] = rel_cif
        replacements[f"../{rel_pdb}"] = f"../{rel_cif}"

    # Also update cross-directory references for the deleted duplicate dirs.
    for old_prefix, new_prefix in CROSS_DIR_RENAMES:
        replacements[old_prefix] = new_prefix
        replacements[f"../{old_prefix}"] = f"../{new_prefix}"

    n_updated_configs = update_configs(replacements)
    print(f"  {n_updated_configs} config file(s) updated.")

    referenced = _find_referenced_files(BENCHMARK_CONFIGS)
    write_failure_report(failures, skipped, referenced, deleted_dirs, missing_dirs)

    n_success = len(all_files) - len(failures)
    print(f"\n{n_success}/{len(all_files)} succeeded, {len(failures)} failed, {len(skipped)} skipped.")
    print(f"Report: {FAILURE_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=Path,
        metavar="PATH",
        help="Convert a single file in debug mode (exceptions propagate with full traceback)",
    )
    parser.add_argument(
        "--rcsb_pdb_dir",
        type=str,
        metavar="DIR",
        default="/path/to/data",
        help="Root of the original RCSB PDB mirror (default: %(default)s)",
    )
    parser.add_argument(
        "--reparsed_pdb_dir",
        type=str,
        metavar="DIR",
        default=None,
        help="Root of the re-parsed PDB mirror; if given, output uses reparsed coordinates",
    )
    args = parser.parse_args()

    if args.file:
        debug_one(args.file.resolve(), args.rcsb_pdb_dir, args.reparsed_pdb_dir)
    else:
        main(args.rcsb_pdb_dir, args.reparsed_pdb_dir)
