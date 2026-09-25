"""Sequence-based antibody numbering and Chothia CDR interface metrics."""

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from anarcii import Anarcii
from biotite.sequence import ProteinSequence
from biotite.structure import AtomArray, filter_amino_acids, get_residue_starts
from scipy.spatial.distance import cdist

from rfproteina.metrics.metric import Metric

CDR_RANGES = {
    "H1": (26, 32),
    "H2": (52, 56),
    "H3": (95, 102),
    "L1": (24, 34),
    "L2": (50, 56),
    "L3": (89, 97),
}


@lru_cache(maxsize=1)
def _antibody_numberer() -> Anarcii:
    """Create a cached CPU numberer without changing the caller's Torch thread count."""
    threads = torch.get_num_threads()
    try:
        return Anarcii(cpu=True, ncpu=1, verbose=False)
    finally:
        torch.set_num_threads(threads)


@lru_cache(maxsize=256)
def _number_antibody_sequence(sequence: str) -> tuple[dict[str, Any], ...]:
    """Cache Chothia domain records without changing the caller's Torch thread count."""
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        numberer = _antibody_numberer()
        numberer.number({"query": sequence})
        return tuple(numberer.to_scheme("chothia").values())
    finally:
        torch.set_num_threads(threads)


def annotate_antibody_numbering(atom_array: AtomArray, overwrite: bool = False) -> AtomArray:
    """Return an array with atom-level Chothia numbering of antibody variable domains.

    All protein chains are numbered with ANARCII, using ``chain_iid`` when present.
    ``is_cdr`` uses the inclusive basic Chothia H1/H2/H3 and L1/L2/L3 ranges.
    ``chothia_numbering`` stores integer positions; insertion letters are retained in
    ``chothia_insertion_code``. ``chothia_chain_type`` is H, K, or L, and
    ``chothia_domain_id`` distinguishes domains, including those on the same chain.
    Unrecognized domains, nonprotein atoms, and residues outside numbered domains
    have false CDR masks, -1 numbers/domain IDs, and empty string annotations.

    The input is not mutated. An already fully annotated input is returned unchanged
    unless ``overwrite=True``; otherwise a copy is annotated. After changing a
    sequence, request overwrite to recompute annotations. Sequence results are cached.
    """
    annotation_names = {
        "is_cdr",
        "chothia_numbering",
        "chothia_insertion_code",
        "chothia_chain_type",
        "chothia_domain_id",
    }
    if not overwrite and annotation_names <= set(atom_array.get_annotation_categories()):
        return atom_array
    annotations = {
        "is_cdr": np.zeros(len(atom_array), dtype=bool),
        "chothia_numbering": np.full(len(atom_array), -1, dtype=int),
        "chothia_insertion_code": np.full(len(atom_array), "", dtype="U4"),
        "chothia_chain_type": np.full(len(atom_array), "", dtype="U1"),
        "chothia_domain_id": np.full(len(atom_array), -1, dtype=int),
    }
    protein = filter_amino_acids(atom_array)
    chain_attr = "chain_iid" if "chain_iid" in atom_array.get_annotation_categories() else "chain_id"
    chain_ids = atom_array.get_annotation(chain_attr)
    domain_id = 0
    for chain_id in np.unique(chain_ids[protein]):
        atom_indices = np.flatnonzero(protein & (chain_ids == chain_id))
        chain = atom_array[atom_indices]
        starts = get_residue_starts(chain, add_exclusive_stop=True)
        letters = []
        for name in chain.res_name[starts[:-1]]:
            try:
                letters.append(ProteinSequence.convert_letter_3to1(name))
            except KeyError:
                letters.append("X")
        sequence = "".join(letters)
        for record in _number_antibody_sequence(sequence):
            chain_type = record.get("chain_type")
            if (
                record.get("error")
                or not record.get("numbering")
                or record.get("scheme") != "chothia"
                or chain_type not in {"H", "K", "L"}
            ):
                continue
            entries = [(number, aa) for number, aa in record["numbering"] if aa != "-"]
            start = record.get("query_start")
            if start is None or start < 0 or sequence[start : start + len(entries)] != "".join(aa for _, aa in entries):
                continue
            prefix = "H" if chain_type == "H" else "L"
            for offset, ((number, insertion), _) in enumerate(entries, start=start):
                indices = atom_indices[starts[offset] : starts[offset + 1]]
                if np.any(annotations["chothia_domain_id"][indices] != -1):
                    raise ValueError("ANARCII returned overlapping antibody variable domains.")
                annotations["chothia_numbering"][indices] = number
                annotations["chothia_insertion_code"][indices] = insertion.strip()
                annotations["chothia_chain_type"][indices] = chain_type
                annotations["chothia_domain_id"][indices] = domain_id
                annotations["is_cdr"][indices] = any(
                    lo <= number <= hi for cdr, (lo, hi) in CDR_RANGES.items() if cdr.startswith(prefix)
                )
            domain_id += 1
    result = atom_array.copy()
    for name, values in annotations.items():
        result.set_annotation(name, values)
    return result


class AntibodyContacts(Metric):
    """Count antibody residues contacting a protein target within a CA distance cutoff.

    A contacting residue is counted once, regardless of its number of target neighbors.
    ANARCII identifies antibody variable domains across all chains. Basic Chothia
    ranges distinguish CDRs from framework. Targets are protein chains containing
    no recognized antibody domain. The fraction is CDR / (CDR + framework), and
    is NaN without contacts.
    """

    def __init__(self, contact_threshold: float = 7.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not np.isfinite(contact_threshold) or contact_threshold <= 0:
            raise ValueError("contact_threshold must be a positive finite distance in Angstroms.")
        self.contact_threshold = contact_threshold

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": ("predicted_atom_array",)}

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Score contacts using sequence-based Chothia annotations."""
        array = annotate_antibody_numbering(predicted_atom_array)
        empty = dict.fromkeys(("cdr_contact_count", "fw_contact_count", "cdr_contact_fraction"), float("nan"))
        antibody = array.chothia_numbering >= 0
        chain_attr = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
        chain_ids = array.get_annotation(chain_attr)
        target = ~np.isin(chain_ids, chain_ids[antibody])
        ca = filter_amino_acids(array) & (array.atom_name == "CA")
        cdr = array.is_cdr[ca & antibody]
        ab_coords = array.coord[ca & antibody]
        target_coords = array.coord[ca & target]
        if not len(target_coords) or not len(ab_coords):
            return empty
        if not np.isfinite(ab_coords).all() or not np.isfinite(target_coords).all():
            return empty
        contacts = np.any(cdist(ab_coords, target_coords) <= self.contact_threshold, axis=1)
        cdr_count = int(np.count_nonzero(contacts & cdr))
        fw_count = int(np.count_nonzero(contacts & ~cdr))
        total = cdr_count + fw_count
        return {
            "cdr_contact_count": float(cdr_count),
            "fw_contact_count": float(fw_count),
            "cdr_contact_fraction": cdr_count / total if total else float("nan"),
        }
