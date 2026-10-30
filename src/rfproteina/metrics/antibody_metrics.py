"""Antibody interface, confidence, and developability metrics using North CDRs."""

from typing import Any

import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from biotite.structure import AtomArray, filter_amino_acids
from scipy.spatial.distance import cdist

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.physics.developability_metrics import protein_one_letter_sequence, subset_to_designed_protein
from rfproteina.metrics.physics.sap import calculate_sap
from rfproteina.utils.antibody import antibody_domains


class AntibodyContacts(Metric):
    """Count antibody residues contacting a protein target within a CA distance cutoff.

    A contacting residue is counted once, regardless of its number of target neighbors.
    ANARCII identifies antibody variable domains across all chains. North CDR
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
        """Score contacts using sequence-derived North CDR annotations."""
        array = predicted_atom_array
        ensure_annotations(array, "is_north_cdr")
        empty = dict.fromkeys(("cdr_contact_count", "fw_contact_count", "cdr_contact_fraction"), float("nan"))
        antibody = np.zeros(len(array), dtype=bool)
        for indices, starts, numbers, _ in antibody_domains(array):
            antibody[indices] |= np.repeat(numbers >= 0, np.diff(starts))
        chain_attr = "chain_iid" if "chain_iid" in array.get_annotation_categories() else "chain_id"
        chain_ids = array.get_annotation(chain_attr)
        target = ~np.isin(chain_ids, chain_ids[antibody])
        ca = filter_amino_acids(array) & (array.atom_name == "CA")
        cdr = array.is_north_cdr[ca & antibody]
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


class AntibodyDevelopability(Metric):
    """Native SAP and sequence checks on the designed VHH, using North H3.

    Net charge is K + R - D - E (histidine and termini are neutral). Cysteine
    parity is a sequence-only screen; it does not establish disulfide bonding.
    H3 SAP retains the entire VHH as solvent and hydrophobic-neighborhood context.
    """

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array": "predicted_atom_array",
            "chain_of_interest": ("extra_info", "chain_of_interest"),
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset(["chain_of_interest"])

    def compute(self, predicted_atom_array: AtomArray, chain_of_interest: str | None = None) -> dict[str, float]:
        """Score one VHH; return NaNs when its heavy-chain domain/H3 is unrecognized."""
        array = subset_to_designed_protein(predicted_atom_array, chain_of_interest)
        ensure_annotations(array, "is_north_cdrh3")
        keys = ("sap_score", "cdrh3_sap_score", "net_charge", "cdrh3_net_charge", "unpaired_cysteines")
        h3 = array.is_north_cdrh3
        if not h3.any() or sum(prefix == "H" for _, _, _, prefix in antibody_domains(array)) != 1:
            return dict.fromkeys(keys, float("nan"))
        sequence = protein_one_letter_sequence(array)
        h3_sequence = protein_one_letter_sequence(array[h3])
        return {
            "sap_score": calculate_sap(array),
            "cdrh3_sap_score": calculate_sap(array, score_mask="is_north_cdrh3"),
            "net_charge": float(sum(sequence.count(aa) for aa in "KR") - sum(sequence.count(aa) for aa in "DE")),
            "cdrh3_net_charge": float(
                sum(h3_sequence.count(aa) for aa in "KR") - sum(h3_sequence.count(aa) for aa in "DE")
            ),
            "unpaired_cysteines": float(sequence.count("C") % 2),
        }


class AntibodyCDRConfidence(Metric):
    """Mean North CDR C-alpha pLDDT (0-100) from an oracle structure."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": "predicted_atom_array"}

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        """Use one CA per CDR residue, preserving low confidence on missing residues."""
        array = subset_to_designed_protein(predicted_atom_array)
        ensure_annotations(array, "is_north_cdr")
        cdr_ca = array.is_north_cdr & (array.atom_name == "CA")
        if not cdr_ca.any() or "b_factor" not in array.get_annotation_categories():
            return {"plddt": float("nan")}
        values = array.b_factor[cdr_ca]
        valid = np.isfinite(values).all() and ((values >= 0) & (values <= 100)).all()
        return {"plddt": float(np.mean(values)) if valid else float("nan")}
