"""ANARCII-based antibody numbering and lazy CDR annotations for AtomWorks.

After sequence or chain edits, remove ``ANTIBODY_ANNOTATIONS`` before reusing masks.
"""

from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import numpy as np
from atomworks.io.utils.annotator import _register_lazy_annotator
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray, AtomArrayStack, filter_amino_acids, get_residue_starts

CHOTHIA_CDR_RANGES = {
    "H1": (26, 32),
    "H2": (52, 56),
    "H3": (95, 102),
    "L1": (24, 34),
    "L2": (50, 56),
    "L3": (89, 97),
}

# North et al. (2011) CDR boundaries in Chothia residue numbering, inclusive.
NORTH_CDR_RANGES = {
    "H1": (23, 35),
    "H2": (50, 58),
    "H3": (93, 102),
    "L1": (24, 34),
    "L2": (49, 56),
    "L3": (89, 97),
}


ANTIBODY_ANNOTATIONS = ("is_north_cdr", "is_north_cdrh3", "is_chothia_cdr")


@lru_cache(maxsize=1)
def _antibody_numberer() -> Any:
    """Create a cached CPU numberer without changing the caller's Torch thread count."""
    import torch
    from anarcii import Anarcii

    threads = torch.get_num_threads()
    try:
        return Anarcii(cpu=True, ncpu=1, verbose=False)
    finally:
        torch.set_num_threads(threads)


@lru_cache(maxsize=256)
def _number_antibody_sequence(sequence: str) -> tuple[dict[str, Any], ...]:
    """Cache Chothia domain records without changing the caller's Torch thread count."""
    import torch

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        numberer = _antibody_numberer()
        numberer.number({"query": sequence})
        return tuple(numberer.to_scheme("chothia").values())
    finally:
        torch.set_num_threads(threads)


def antibody_domains(
    atom_array: AtomArray | AtomArrayStack,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Yield chain atom indices, residue starts (with stop), domain numbering, and H/L type.

    Numbering is residue-level, with -1 outside the current domain. Insertion
    codes share their integer position. No numbering annotations are attached.
    """
    protein = filter_amino_acids(atom_array)
    chain_attr = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    chain_ids = atom_array.get_annotation(chain_attr)
    for chain_id in np.unique(chain_ids[protein]):
        indices = np.flatnonzero(protein & (chain_ids == chain_id))
        chain = atom_array[:, indices] if isinstance(atom_array, AtomArrayStack) else atom_array[indices]
        starts = get_residue_starts(chain, add_exclusive_stop=True)
        letters = []
        for name in chain.res_name[starts[:-1]]:
            try:
                letters.append(ProteinSequence.convert_letter_3to1(name))
            except KeyError:
                letters.append("X")
        sequence = "".join(letters)
        assigned = np.zeros(len(letters), dtype=bool)
        for record in _number_antibody_sequence(sequence):
            chain_type = record.get("chain_type")
            if record.get("error") or record.get("scheme") != "chothia" or chain_type not in {"H", "K", "L"}:
                continue
            entries = [(number, aa) for number, aa in (record.get("numbering") or []) if aa != "-"]
            start = record.get("query_start")
            if (
                not entries
                or start is None
                or start < 0
                or sequence[start : start + len(entries)] != "".join(aa for _, aa in entries)
            ):
                continue
            region = slice(start, start + len(entries))
            if assigned[region].any():
                raise ValueError("ANARCII returned overlapping antibody variable domains.")
            assigned[region] = True
            numbers = np.full(len(letters), -1, dtype=int)
            numbers[region] = [number for (number, _), _ in entries]
            yield indices, starts, numbers, "H" if chain_type == "H" else "L"


def _cdr_mask(atom_array: AtomArray | AtomArrayStack, ranges: dict[str, tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(atom_array.array_length(), dtype=bool)
    for indices, starts, numbers, prefix in antibody_domains(atom_array):
        residues = np.zeros(len(numbers), dtype=bool)
        for cdr, (lo, hi) in ranges.items():
            if cdr.startswith(prefix):
                residues |= (numbers >= lo) & (numbers <= hi)
        mask[indices] |= np.repeat(residues, np.diff(starts))
    return mask


@_register_lazy_annotator("is_north_cdr")
def is_north_cdr(atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
    """Select North CDR1-3; insertion-coded residues inherit their numbered region."""
    return _cdr_mask(atom_array, NORTH_CDR_RANGES)


@_register_lazy_annotator("is_north_cdrh3")
def is_north_cdrh3(atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
    """Select North heavy-chain CDR3, including its stem (positions 93-102)."""
    return _cdr_mask(atom_array, {"H3": NORTH_CDR_RANGES["H3"]})


@_register_lazy_annotator("is_chothia_cdr")
def is_chothia_cdr(atom_array: AtomArray | AtomArrayStack) -> np.ndarray:
    """Select basic Chothia CDR1-3 for the torsion reference distributions."""
    return _cdr_mask(atom_array, CHOTHIA_CDR_RANGES)
