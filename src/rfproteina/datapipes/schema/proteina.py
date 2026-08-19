"""Schema for Proteina with explicit specs: (dtype, default, dims)."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import torch
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from biotite.structure import AtomArray
from torch.nn import functional as F

from rfproteina.conditions import (
    C_CATH,
    C_CENTROID,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_HPHOB,
    C_IDXGRP,
    C_INTCRD,
    C_MCOORD,
    C_NCYS,
    C_PICATION,
    C_PISTACK,
    C_RAS,
    C_SALT,
    C_SEC,
    HBondEnum,
    MetalCoordEnum,
    PiCationEnum,
    RasaEnum,
    SaltBridgeEnum,
    SSEnum,
)
from rfproteina.datapipes.schema.base import DataSample

# Specs: name -> (dtype, default_value, dim_string)
# dim_string uses: I=tokens, S=MSA rows, numbers=fixed dims
SPECS: dict[str, tuple] = {
    # Token-level [I]
    "residue_type": (torch.int64, -1, "I"),
    C_SEQ.full_name: (torch.int64, -1, "I"),
    "token_type": (torch.int32, 4, "I"),
    "tok_idx": (torch.int16, -1, "I"),
    "chain_iid": (torch.int16, -1, "I"),
    "molecule_iid": (torch.int16, -1, "I"),
    "within_index_group_res_idx": (torch.int16, -1, "I"),
    "can_be_encoded": (torch.bool, False, "I"),
    "should_be_lossed": (torch.bool, False, "I"),
    "insertion_source_idx": (torch.int32, -1, "I"),
    "loss_weight": (torch.float32, 1.0, "I"),
    "coord_cond_mask": (torch.bool, False, "I"),
    C_SEQ.mask_name: (torch.bool, False, "I"),
    C_SEC.full_name: (torch.int16, SSEnum.NONE, "I"),
    C_NCYS.full_name: (torch.bool, False, "I"),
    C_IDXGRP.full_name: (torch.int16, -1, "I"),
    "na_ss_state": (torch.int16, 0, "I"),
    "na_ss_state_mask": (torch.bool, False, "I"),
    "chain_breaks_per_residue": (torch.float32, 0.0, "I"),
    "missing_atom_count": (torch.float32, 0.0, "I"),
    # Token-level with extra dims [I, ...]
    C_CATH.full_name: (torch.int32, -1, "I 3"),
    C_CENTROID.full_name: (torch.float32, 0.0, "I 3"),
    "chirality": (torch.float32, 0.0, "I 18"),
    # Atom37-level [I, 37, ...]
    "coords": (torch.float32, 0.0, "I 37 3"),
    "element": (torch.int16, -1, "I 37"),
    "charge": (torch.int16, 0, "I 37"),
    "is_resolved_mask": (torch.bool, False, "I 37"),
    "atom37_mask": (torch.bool, False, "I 37"),
    C_CRD.mask_name: (torch.bool, False, "I 37"),
    C_INTCRD.mask_name: (torch.bool, False, "I 37"),
    C_IDX.mask_name: (torch.bool, True, "I 37"),
    C_CRDGRP.full_name: (torch.int16, C_CRDGRP.UNKNOWN_COORDS, "I 37"),
    C_INTCRD.full_name: (torch.float32, 0.0, "I 37 3"),
    C_HOT.full_name: (torch.bool, False, "I 37"),
    C_RAS.full_name: (torch.int16, RasaEnum.UNSPECIFIED, "I 37"),
    C_HB.full_name: (torch.int16, HBondEnum.NONE, "I 37"),
    C_PISTACK.full_name: (torch.bool, False, "I 37"),
    C_PICATION.full_name: (torch.int16, PiCationEnum.NONE, "I 37"),
    C_SALT.full_name: (torch.int16, SaltBridgeEnum.NONE, "I 37"),
    C_HPHOB.full_name: (torch.bool, False, "I 37"),
    C_MCOORD.full_name: (torch.int16, MetalCoordEnum.NONE, "I 37"),
    C_CRD.full_name: (torch.float32, 0.0, "I 37 3"),
    # Pair-level [I, I]
    "bond_mask": (torch.bool, False, "I I"),
    "bond_order": (torch.int16, 0, "I I"),
    "is_aromatic": (torch.bool, False, "I I"),
    "na_ss_pair": (torch.int16, 0, "I I"),
    "na_ss_pair_mask": (torch.bool, False, "I I"),
    "token_bond_path_length": (torch.int16, 0, "I I"),
    "canonical_polymer_ordering": (torch.int8, 0, "I I"),
    # MSA-level [S, I, ...]
    "msa_final": (torch.float32, 0.0, "S I 71"),
}


@dataclass
class ProteinaDataSample(DataSample):
    """Flat data sample - tensors dict with specs-based padding."""

    example_id: str
    tensors: dict[str, torch.Tensor]
    atom_array: AtomArray | None = None
    task: str = ""
    metrics_tags: set = field(default_factory=set)
    extra_info: dict[str, Any] = field(default_factory=dict)
    # Input design job metadata (JSON spec from ContigJsonDataset / CIF datasets),
    # distinct from tensor ``SPECS`` above; collated as ``specification_batch``.
    specification: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProteinaDataSample":
        feats = data["feats"]
        tensors = {}
        for k, v in feats.items():
            if not (isinstance(v, torch.Tensor) or hasattr(v, "__array__")):
                continue
            t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
            # Apply dtype from specs if defined
            if k in SPECS:
                dtype = SPECS[k][0]
                if t.dtype != dtype:
                    t = t.to(dtype)
            tensors[k] = t

        return cls(
            example_id=data["example_id"],
            tensors=tensors,
            atom_array=data.get("atom_array"),
            task=data.get("task", ""),
            metrics_tags=data.get("metrics_tags", set()),
            extra_info=data.get("extra_info", {}),
            specification=data.get("specification"),
        )

    @cached_property
    def max_lengths(self) -> dict[str, int]:
        lengths = {}
        for name, t in self.tensors.items():
            if t.ndim == 0:
                continue
            dims = SPECS[name][2].split() if name in SPECS else ["I"]
            for i, d in enumerate(dims):
                if not d.isdigit() and i < t.ndim:
                    lengths[d] = max(lengths.get(d, 0), t.shape[i])
        return lengths

    def pad_to_max_lengths(self, max_lengths: dict[str, int]) -> "ProteinaDataSample":
        padded = {}
        for name, t in self.tensors.items():
            if t.ndim == 0:
                padded[name] = t
                continue

            spec = SPECS.get(name)
            dims = spec[2].split() if spec else ["I"] + ["?"] * (t.ndim - 1)
            pv = spec[1] if spec else (False if t.dtype == torch.bool else 0)

            # Build padding (F.pad: last dim first)
            pad = []
            for i in range(t.ndim - 1, -1, -1):
                if i < len(dims) and not dims[i].isdigit() and dims[i] in max_lengths:
                    pad.extend([0, max_lengths[dims[i]] - t.shape[i]])
                else:
                    pad.extend([0, 0])

            padded[name] = F.pad(t, pad, value=pv) if any(p > 0 for p in pad) else t

        # Zero-fill missing SPECS keys so every batch has the same dict schema
        # (one torch.compile graph instead of N).
        for name in sorted(SPECS.keys()):
            if name in padded:
                continue
            dtype, default_val, dim_str = SPECS[name]
            shape = tuple(int(d) if d.isdigit() else max_lengths.get(d, 0) for d in dim_str.split())
            padded[name] = torch.full(shape, default_val, dtype=dtype)

        return ProteinaDataSample(
            example_id=self.example_id,
            tensors=padded,
            atom_array=self.atom_array,
            task=self.task,
            metrics_tags=self.metrics_tags,
            extra_info=self.extra_info,
            specification=self.specification,
        )
