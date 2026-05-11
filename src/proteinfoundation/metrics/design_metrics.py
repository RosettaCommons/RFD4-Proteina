import numpy as np
import torch
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.ml.conditions import C_CRD, C_DIS, C_IDX
from atomworks.ml.utils.token import (
    get_token_count,
    get_token_starts,
)
from beartype.typing import Any
from biotite.structure import AtomArray, AtomArrayStack

from foundry.common import exists
from foundry.metrics.metric import Metric
from proteinfoundation.conditions import C_INTCRD
from proteinfoundation.metrics.metrics_utils import (
    get_ss_metrics_and_rg,
    unwrap_atom_array,
)

STANDARD_CACA_DIST = 3.8


def get_chainbreaks(atom_array: AtomArray, chainbreak_threshold: float = 0.75) -> dict[str, float | int]:
    ca_atoms = atom_array[atom_array.atom_name == "CA"]
    xyz = ca_atoms.coord
    xyz = torch.from_numpy(xyz)
    ca_dists = torch.norm(xyz[1:] - xyz[:-1], dim=-1)
    deviation = torch.abs(ca_dists - STANDARD_CACA_DIST)

    # Allow leniency for expected chain breaks (e.g. PPI)
    expected_chain_breaks = ca_atoms.chain_iid[1:] != ca_atoms.chain_iid[:-1]
    deviation[expected_chain_breaks] = 0

    is_chainbreak = deviation > chainbreak_threshold
    return {
        "max_ca_deviation": float(deviation.max(-1).values.mean()),
        "n_chainbreaks": int(is_chainbreak.sum()),
    }


def get_interresidue_clashes(atom_array: AtomArray, clash_threshold: float = 1.5, backbone_only: bool = False) -> int:
    ensure_annotations(atom_array, "is_protein", "is_protein_backbone")
    protein_array = atom_array[atom_array.is_protein]
    resid = protein_array.res_id - protein_array.res_id.min()
    xyz = protein_array.coord
    dists = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)  # N_atoms x N_atoms

    # Block out intra-residue distances
    mask = np.triu(np.ones_like(dists), k=1).astype(bool)
    block_mask = np.abs(resid[:, None] - resid[None, :]) <= 1  # NOTE: Why is this <= instead of < ??
    mask[block_mask] = False
    dists[~mask] = 999

    if backbone_only:
        # Block out non-backbone atoms
        backbone_mask = protein_array.is_protein_backbone
        mask = backbone_mask[:, None] & backbone_mask[None, :]
        dists[~mask] = 999

    num_clashes_L = dists.min(axis=-1) < clash_threshold
    return int(num_clashes_L.sum())


def get_ligand_clash_metrics(atom_array: AtomArray, ligand_clash_threshold: float = 1.5) -> dict[str, float | int]:
    ensure_annotations(atom_array, "is_ligand", "is_protein_backbone")

    if not atom_array.is_ligand.any():
        return {}

    is_crd_conditioned = C_CRD.mask(atom_array, default="generate")
    backbone_mask = atom_array.is_protein_backbone
    protein_mask = atom_array.is_protein
    ligand_mask = atom_array.is_ligand

    xyz_ligand = atom_array[ligand_mask].coord
    ligand_is_crd = is_crd_conditioned[ligand_mask]

    output_dict = {}

    xyz_backbone = atom_array[backbone_mask].coord
    if xyz_backbone.shape[0] != 0:
        backbone_is_crd = is_crd_conditioned[backbone_mask]
        both_crd = backbone_is_crd[:, None] & ligand_is_crd[None, :]  # (n_backbone, n_ligand)
        dists = np.linalg.norm(xyz_backbone[:, None] - xyz_ligand[None], axis=-1)
        dists[both_crd] = 999
        output_dict["n_clashing.backbone_ligand_clashes"] = int(np.sum(dists < ligand_clash_threshold))
        output_dict["n_clashing.backbone_ligand_min_distance"] = float(dists.min())

    xyz_allatom = atom_array[protein_mask].coord
    if xyz_allatom.shape[0] != 0:
        allatom_is_crd = is_crd_conditioned[protein_mask]
        both_crd = allatom_is_crd[:, None] & ligand_is_crd[None, :]  # (n_protein, n_ligand)
        dists = np.linalg.norm(xyz_allatom[:, None] - xyz_ligand[None], axis=-1)
        dists[both_crd] = 999
        output_dict["n_clashing.allatom_ligand_clashes"] = int(np.sum(dists < ligand_clash_threshold))
        output_dict["n_clashing.allatom_ligand_min_distance"] = float(dists.min())

    return output_dict


def get_clash_metrics(
    atom_array: AtomArray,
    clash_threshold: float = 1.5,
    ligand_clash_threshold: float = 1.5,
    chainbreak_threshold: float = 0.75,
) -> dict[str, float | int]:
    # Accumulate metrics
    o = {}
    o = o | get_chainbreaks(atom_array, chainbreak_threshold)
    o["n_clashing.interresidue_clashes_w_sidechain"] = get_interresidue_clashes(
        atom_array, clash_threshold, backbone_only=False
    )
    o["n_clashing.interresidue_clashes_w_backbone"] = get_interresidue_clashes(
        atom_array, clash_threshold, backbone_only=True
    )
    o |= get_ligand_clash_metrics(atom_array, ligand_clash_threshold)
    return {k: v for k, v in o.items() if exists(v)}


def convert_to_float_or_str(o: dict[str, Any]) -> dict[str, float | str | list | int]:
    """
    Converts elements of a dictionary to ensure all components are saveable with JSONs
    """
    for k, v in o.items():
        if not isinstance(v, int | float | str | list):
            try:
                o[k] = float(v)
            except Exception as e:
                raise ValueError(f"Unsupported type for key {k}: {type(v)}. Error: {e}") from e
    return o


class AtomArrayMetrics(Metric):
    """General metrics for the predicted atom array."""

    def __init__(
        self,
        compute_for_diffused_region_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clash_threshold = 1.2
        self.float_threshold = 3.0  # maximum closest-neighbour distance before considered a floating atom
        self.standard_ca_dist = 3.8
        self.compute_for_diffused_region_only = compute_for_diffused_region_only

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array": ("predicted_atom_array",),
        }

    def compute(self, atom_array: AtomArray | AtomArrayStack) -> dict[str, Any]:
        atom_array = unwrap_atom_array(atom_array)
        n_tokens = get_token_count(atom_array)

        is_motif_atom_unindexed = ~C_IDX.mask(atom_array, default="generate")  # TODO: Does this work?
        is_motif_atom_with_fixed_coord = C_CRD.mask(atom_array, default="generate")
        has_dist = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
        is_motif_atom = is_motif_atom_with_fixed_coord | has_dist | C_INTCRD.mask(atom_array, default="generate")

        # Subset to indexed tokens, and optionally consider only the diffused region
        valid_mask = ~(is_motif_atom_unindexed)
        if self.compute_for_diffused_region_only:
            valid_mask = valid_mask & ~(is_motif_atom)
        valid_atom_array = atom_array[valid_mask]

        out = get_ss_metrics_and_rg(valid_atom_array, ss_conditioning=None)
        out["n_tokens"] = n_tokens

        # Subset to token level array for consistency
        token_array = valid_atom_array[get_token_starts(valid_atom_array)]

        # Basic compositional statistics
        out["alanine_content"] = float(np.mean(token_array.res_name == "ALA"))
        out["glycine_content"] = float(np.mean(token_array.res_name == "GLY"))

        return out


class BackboneMetrics(Metric):
    def __init__(self, compute_for_diffused_region_only: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.clash_threshold = 1.2
        self.float_threshold = 3.0  # maximum closest-neighbour distance before considered a floating atom
        self.standard_ca_dist = 3.8
        self.compute_for_diffused_region_only = compute_for_diffused_region_only

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "atom_array": ("predicted_atom_array",),
        }

    def compute(self, atom_array: AtomArray | AtomArrayStack) -> dict[str, float]:
        atom_array = unwrap_atom_array(atom_array)
        ensure_annotations(atom_array, "is_protein", "is_protein_backbone", "token_id")

        xyz = atom_array.coord
        dists = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)  # N_atoms x N_atoms

        is_protein = atom_array.is_protein
        is_backbone = atom_array.is_protein_backbone
        token_id = atom_array.token_id

        is_unindexed = ~C_IDX.mask(atom_array, default="generate")
        is_motif_atom_with_fixed_coord = C_CRD.mask(atom_array, default="generate")
        has_dist = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
        has_coord_or_dist_conditioning = (
            is_motif_atom_with_fixed_coord | has_dist | C_INTCRD.mask(atom_array, default="generate")
        )

        mask = np.zeros_like(dists, dtype=bool)
        mask = mask | (np.eye(dists.shape[-1], dtype=bool))
        mask = mask | (token_id[:, None] == token_id[None, :])
        mask = mask | ~(is_protein[:, None] & is_protein[None, :])
        mask = mask | (is_unindexed[:, None] | is_unindexed[None, :])
        dists[mask] = 999

        num_clashes_L = (dists.min(axis=-1) < self.clash_threshold).astype(float)
        out = {
            "frac_clashing": float(num_clashes_L.mean()),
            "n_clashing": float(num_clashes_L.sum()),
        }

        is_backbone_pair = is_backbone[:, None] & is_backbone[None, :]
        dists[~is_backbone_pair] = 999
        out["frac_backbone_clashing"] = float((dists.min(axis=-1) < self.clash_threshold).astype(float).mean())
        out["n_backbone_clashing"] = float((dists.min(axis=-1) < self.clash_threshold).astype(float).sum())

        # We do this after clash detection, since that should consider both chains
        if self.compute_for_diffused_region_only:
            diffused_region = ~has_coord_or_dist_conditioning
            xyz = xyz[diffused_region]

        # Num floating
        dists = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)  # N_atoms x N_atoms
        dists[np.eye(dists.shape[-1], dtype=bool)] = 999

        is_floating = dists.min(axis=-1) > self.float_threshold
        out["frac_floating"] = float(is_floating.mean())

        # Calculate CA
        is_ca = atom_array.atom_name == "CA"
        if self.compute_for_diffused_region_only:
            is_ca = is_ca[diffused_region]
            is_protein = is_protein[diffused_region]
        idx_mask = is_ca & is_protein
        xyz = xyz[idx_mask]

        ca_dists = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
        deviation = np.abs(ca_dists - self.standard_ca_dist)
        is_chainbreak = deviation > 0.75

        if len(ca_dists) > 0:
            out["max_ca_deviation"] = float(deviation.max())
            out["fraction_chainbreaks"] = float(is_chainbreak.mean())
            out["n_chainbreaks"] = float(is_chainbreak.sum())

        return out


class LigandClashMetric(Metric):
    def __init__(self, ligand_clash_threshold: float = 1.5, **kwargs):
        self.ligand_clash_threshold = ligand_clash_threshold
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": "predicted_atom_array"}

    def compute(self, predicted_atom_array: AtomArray | AtomArrayStack) -> dict[str, float]:
        atom_array = unwrap_atom_array(predicted_atom_array)
        return get_ligand_clash_metrics(atom_array, ligand_clash_threshold=self.ligand_clash_threshold)
