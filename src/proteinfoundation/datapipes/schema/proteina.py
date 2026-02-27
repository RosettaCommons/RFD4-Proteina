"""Schema for Proteina with explicit specs: (dtype, default, dims)."""

import torch
from biotite.structure import AtomArray
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from functools import cached_property
from torch.nn import functional as F

from atomworks.ml.conditions import C_CRD, C_CTR, C_IDX, C_NTR, C_SEQ
from proteinfoundation.conditions import (
    C_HB, C_HOT, C_IDXGRP, C_CRDGRP, C_INTCRD, C_RAS, C_SEC, C_CATH,
    HBondEnum, RasaEnum, SSEnum,
)
from foundry.data.schema.base import DataSample


# Specs: name -> (dtype, default_value, dim_string)
# dim_string uses: I=tokens, S=MSA rows, numbers=fixed dims
SPECS: Dict[str, tuple] = {
    # Token-level [I]
    "residue_type": (torch.int64, -1, "I"),
    C_SEQ.full_name: (torch.int64, -1, "I"),
    "token_type": (torch.int32, 4, "I"),
    "tok_idx": (torch.int16, -1, "I"),
    "chain_iid": (torch.int16, -1, "I"),
    "within_index_group_res_idx": (torch.int16, -1, "I"),
    "can_be_encoded": (torch.bool, False, "I"),
    "should_be_lossed": (torch.bool, False, "I"),
    "coord_cond_mask": (torch.bool, False, "I"),
    C_SEQ.mask_name: (torch.bool, False, "I"),
    C_CTR.full_name: (torch.bool, False, "I"),
    C_NTR.full_name: (torch.bool, False, "I"),
    C_SEC.full_name: (torch.int16, SSEnum.NONE, "I"),
    C_IDXGRP.full_name: (torch.int16, -1, "I"),
    # Token-level with extra dims [I, ...]
    "graph_laplacian_pe_token_level": (torch.float32, 0.0, "I 32"),
    C_CATH.full_name: (torch.int32, -1, "I 3"),
    # Atom37-level [I, 37, ...]
    "coords": (torch.float32, 0.0, "I 37 3"),
    "element": (torch.int16, -1, "I 37"),
    "charge": (torch.int16, 0, "I 37"),
    "nhyd": (torch.int16, -1, "I 37"),
    "is_resolved_mask": (torch.bool, False, "I 37"),
    "is_center_atom_mask": (torch.bool, False, "I 37"),
    "atom37_mask": (torch.bool, False, "I 37"),
    C_CRD.mask_name: (torch.bool, False, "I 37"),
    C_INTCRD.mask_name: (torch.bool, False, "I 37"),
    C_IDX.mask_name: (torch.bool, True, "I 37"),
    C_CRDGRP.full_name: (torch.int16, C_CRDGRP.UNKNOWN_COORDS, "I 37"),
    C_INTCRD.full_name: (torch.float32, 0.0, "I 37 3"),
    C_HOT.full_name: (torch.bool, False, "I 37"),
    C_RAS.full_name: (torch.int16, RasaEnum.UNSPECIFIED, "I 37"),
    C_HB.full_name: (torch.int16, HBondEnum.NONE, "I 37"),
    C_CRD.full_name: (torch.float32, 0.0, "I 37 3"),
    # Pair-level [I, I]
    "bond_mask": (torch.bool, False, "I I"),
    "bond_order": (torch.int16, 0, "I I"),
    # MSA-level [S, I, ...]
    "msa_final": (torch.float32, 0.0, "S I 71"),
}


@dataclass
class ProteinaDataSample(DataSample):
    """Flat data sample - tensors dict with specs-based padding."""
    
    example_id: str
    tensors: Dict[str, torch.Tensor]
    atom_array: Optional[AtomArray] = None
    task: str = ""
    metrics_tags: set = field(default_factory=set)
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProteinaDataSample":
        feats = data["feats"]
        tensors = {}
        for k, v in feats.items():
            if not (isinstance(v, torch.Tensor) or hasattr(v, '__array__')):
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
        )

    @cached_property
    def max_lengths(self) -> Dict[str, int]:
        lengths = {}
        for name, t in self.tensors.items():
            if t.ndim == 0:
                continue
            dims = SPECS[name][2].split() if name in SPECS else ["I"]
            for i, d in enumerate(dims):
                if not d.isdigit() and i < t.ndim:
                    lengths[d] = max(lengths.get(d, 0), t.shape[i])
        return lengths

    def pad_to_max_lengths(self, max_lengths: Dict[str, int]) -> "ProteinaDataSample":
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
        
        return ProteinaDataSample(
            example_id=self.example_id,
            tensors=padded,
            atom_array=self.atom_array,
            task=self.task,
            metrics_tags=self.metrics_tags,
            extra_info=self.extra_info,
        )
