"""Collate function for batching variable-length proteins."""

import torch
import numpy as np
from atomworks.ml.conditions import C_CRD, C_CTR, C_IDX, C_NTR, C_SEQ
from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING

from proteinfoundation.conditions import (
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_CRDGRP,
    C_INTCRD,
    C_RAS,
    C_SEC,
    C_CATH,
    HBondEnum,
    RasaEnum,
    SSEnum,
)

# Token-level collation specs (shape: [batch, n_tokens])
# Token-level features for the model
#
# CRITICAL: We maintain BOTH ground truth and conditioned versions:
# - residue_type: Ground truth sequences (for autoencoder decoder)
# - C_SEQ.full_name: Conditioned sequences (for diffusion model)
#   * For unconditional: Contains <M> mask tokens (prevents sequence leakage)
#   * For folding: Contains ground truth (provides sequence conditioning)
TOKEN_LEVEL_SPECS = {
    "residue_type": (torch.int64, -1),  # Ground truth residue type (for autoencoder)
    C_SEQ.full_name: (torch.int64, -1),  # Conditioned residue type (for diffusion, with <M> masks)
    "token_type": (
        torch.int32,
        4,
    ),  # Token modality type (int32 for embedding): 0=protein, 1=RNA, 2=DNA, 3=atomized, 4=unknown
    "tok_idx": (torch.int16, -1),
    "chain_iid": (torch.int16, -1),
    "within_index_group_res_idx": (torch.int16, -1),
    "can_be_encoded": (torch.bool, False),  # Token-level: whether token can be encoded by autoencoder
    "should_be_lossed": (torch.bool, False),  # Token-level: whether token should be included in loss
    "coord_cond_mask": (
        torch.bool,
        False,
    ),  # Token-level: whether token has coordinate conditioning (for diagnostics)
    C_SEQ.mask_name: (torch.bool, False),
    C_CTR.full_name: (torch.bool, False),
    C_NTR.full_name: (torch.bool, False),
    C_SEC.full_name: (torch.int16, SSEnum.NONE),
    C_IDXGRP.full_name: (torch.int16, -1),
}

# Token-level with extra dimensions (shape: [batch, n_tokens, ...])
TOKEN_LEVEL_WITH_DIMS_SPECS = {
    "graph_laplacian_pe_token_level": (torch.float32, 0.0, (32,)),
    C_CATH.full_name: (torch.int32, -1, (3,)),
}

# Atom37-level collation specs (shape: [batch, n_tokens, 37, ...])
ATOM37_LEVEL_SPECS = {
    "coords": (torch.float32, 0.0, (3,)),  # Coordinates [B, num_tokens, 37, 3]
    "element": (torch.int16, -1, ()),
    "charge": (torch.int16, 0, ()),  # Atomic charge [B, num_tokens, 37]
    "nhyd": (torch.int16, -1, ()),  # Number of hydrogens [B, num_tokens, 37]
    "is_resolved_mask": (torch.bool, False, ()),  # Atom37-level: which atoms are resolved
    "is_center_atom_mask": (torch.bool, False, ()),  # Atom37-level: which atoms are token centers
    C_CRD.mask_name: (torch.bool, False, ()),
    C_INTCRD.mask_name: (torch.bool, False, ()),
    C_IDX.mask_name: (torch.bool, True, ()),
    C_CRDGRP.full_name: (torch.int16, C_CRDGRP.UNKNOWN_COORDS, ()),
    C_INTCRD.full_name: (torch.float32, 0.0, (3,)),
    C_HOT.full_name: (torch.bool, False, ()),
    C_RAS.full_name: (torch.int16, RasaEnum.UNSPECIFIED, ()),
    C_HB.full_name: (torch.int16, HBondEnum.NONE, ()),
    C_CRD.full_name: (
        torch.float32,
        0.0,
        (3,),
    ),  # Conditioned coordinates (for diffusion, with zeros for unconditioned atoms)
    "atom37_mask": (torch.bool, False, ()),  # Always present
}

# Pair-level collation specs (shape: [batch, n_tokens, n_tokens])
PAIR_LEVEL_SPECS = {
    "bond_mask": (torch.bool, False),
    "bond_order": (torch.int16, 0),
}

# MSA-level collation specs (shape: [batch, max_rows, max_tokens])
# MSA features are 2D per sample: [n_rows, n_tokens]
# NOTE: Only `msa` is strictly needed for MSA Pairformer - all masks are derived internally
MSA_LEVEL_SPECS = {
    "msa_final": (torch.float32, 0, 71), # pad with 0s, these values will be masked in the transformer
}


def atom37_collate_fn(batch: list[dict], fixed_max_length: int | None = None) -> dict:
    """Collate function for atom37 batches.

    Assumes input examples are already in token-level atom37 format from
    :py:class:`~proteinfoundation.datapipes.pipeline.ConvertToAtom37Format`.

    Args:
      batch: List of token-level examples with atom37 features.
      fixed_max_length: If provided, always pad to this many tokens instead of the
        batch's dynamic max. If None (default), pads to longest sample in batch.

    Returns:
        Dictionary with:
        - Token-level features [B, num_tokens]: residue_type, tok_idx, chain_iid, etc.
        - Atom37-level features [B, num_tokens, 37]: coords, element, masks, etc.
        - atom37_mask: [B, num_tokens, 37] - Valid atom slots
        - padding_mask: [B, num_tokens] - True for real tokens, False for padding
        - MSA features (if present):
            - msa: [B, max_rows, max_tokens] - Encoded MSA sequences
            - msa_row_mask: [B, max_rows] - True for real MSA rows
        - atom_array_batch, task_batch, etc. (kept unbatched)
    """
    # Infer max_tokens from coords shape (coords is [n_tokens, 37, 3])
    batch_max_tokens = max(ex["feats"]["coords"].shape[0] for ex in batch)
    max_tokens = fixed_max_length if fixed_max_length else batch_max_tokens

    if fixed_max_length and batch_max_tokens > fixed_max_length:
        raise ValueError(
            f"Sample in batch has {batch_max_tokens} tokens which exceeds " f"fixed_max_length={fixed_max_length}."
        )

    batch_size = len(batch)
    collated = {}

    # 1. Token-level features [batch, n_tokens]
    for output_name, (dtype, pad_value) in TOKEN_LEVEL_SPECS.items():
        # Map output name to input name
        input_name = output_name
        if input_name not in batch[0]["feats"]:
            continue

        if isinstance(dtype, torch.dtype):
            padded = torch.full((batch_size, max_tokens), pad_value, dtype=dtype)
        else:
            padded = np.full((batch_size, max_tokens), pad_value, dtype=dtype)
        for i, ex in enumerate(batch):
            feat = ex["feats"][input_name]
            num_tokens = feat.shape[0]
            padded[i, :num_tokens] = feat
        collated[output_name] = padded

    # 2. Token-level with extra dims [batch, n_tokens, ...]
    for feat_name, (dtype, pad_value, extra_dims) in TOKEN_LEVEL_WITH_DIMS_SPECS.items():
        if feat_name not in batch[0]["feats"]:
            continue

        shape = (batch_size, max_tokens, *extra_dims)
        padded = torch.full(shape, pad_value, dtype=dtype)
        for i, ex in enumerate(batch):
            feat = ex["feats"][feat_name]
            num_tokens = feat.shape[0]
            padded[i, :num_tokens] = feat
        collated[feat_name] = padded

    # 3. Atom37-level features [batch, n_tokens, 37, ...]
    for feat_name, (dtype, pad_value, extra_dims) in ATOM37_LEVEL_SPECS.items():
        if feat_name not in batch[0]["feats"]:
            continue

        shape = (batch_size, max_tokens, 37, *extra_dims) if extra_dims else (batch_size, max_tokens, 37)
        padded = torch.full(shape, pad_value, dtype=dtype)
        for i, ex in enumerate(batch):
            feat = ex["feats"][feat_name]
            num_tokens = feat.shape[0]
            padded[i, :num_tokens] = feat
        collated[feat_name] = padded

    # 4. Pair-level features [batch, n_tokens, n_tokens]
    # Bond features for atomized molecules/ligands
    for feat_name, (dtype, pad_value) in PAIR_LEVEL_SPECS.items():
        if feat_name not in batch[0]["feats"]:
            continue

        shape = (batch_size, max_tokens, max_tokens)
        padded = torch.full(shape, pad_value, dtype=dtype)
        for i, ex in enumerate(batch):
            feat = ex["feats"][feat_name]
            num_tokens = feat.shape[0]
            padded[i, :num_tokens, :num_tokens] = feat
        collated[feat_name] = padded

    # 5. MSA-level features [batch, max_rows, max_tokens]
    # Determine max MSA rows across batch (only if MSA features exist)
    if "msa_final" in batch[0]["feats"]:
        max_msa_rows = max(ex["feats"]["msa_final"].shape[0] for ex in batch)
        for feat_name, (dtype, pad_value, num_features) in MSA_LEVEL_SPECS.items():
            shape = (batch_size, max_msa_rows, max_tokens, num_features,)
            padded = torch.full(shape, pad_value, dtype=dtype)
            for i, ex in enumerate(batch):
                feat = ex["feats"][feat_name]
                num_rows = feat.shape[0]
                num_tokens = feat.shape[1]
                padded[i, :num_rows, :num_tokens] = feat
            collated[feat_name] = padded
        # MSA row mask [batch, max_rows]
        msa_row_mask = torch.zeros((batch_size, max_msa_rows), dtype=torch.bool)
        for i, ex in enumerate(batch):
            num_rows = ex["feats"]["msa_final"].shape[0]
            msa_row_mask[i, :num_rows] = True
        collated["msa_row_mask"] = msa_row_mask
    
    # 6. Padding mask (computed, not from feats)
    padding_mask = torch.zeros((batch_size, max_tokens), dtype=torch.bool)
    for i, ex in enumerate(batch):
        num_tokens = ex["feats"]["coords"].shape[0]
        padding_mask[i, :num_tokens] = True
    collated["padding_mask"] = padding_mask

    # 7. Centroid (scalar, already numpy)
    centroids = [ex["feats"]["centroid_of_atoms_without_coordinate_conditioning"] for ex in batch]
    collated["centroid_of_atoms_without_coordinate_conditioning"] = torch.stack(list(centroids))

    # 8. Unbatched lists
    collated["atom_array_batch"] = [ex["atom_array"] for ex in batch]
    collated["task_batch"] = [ex.get("task") for ex in batch]
    collated["example_id_batch"] = [ex["example_id"] for ex in batch]
    collated["metrics_tags_batch"] = [ex.get("metrics_tags", set()) for ex in batch]
    if "extra_info" in batch[0]:
        collated["extra_info_batch"] = [ex["extra_info"] for ex in batch]

    return collated

from typing import Dict, Optional, Union
from proteinfoundation.datapipes.schema.base import BatchedDataSample, DataSample


def collate_batch(
    batch: list[dict],
    schema: Union[str, type[DataSample]],
    fixed_dimensions: Optional[Dict[str, int]] = None
):
    """Collate a batch of data samples into a single batch with padding.

    Args:
        batch: A list of data samples (dictionaries from pipeline).
        schema: Either a DataSample class or a string path to one
            (e.g. "proteinfoundation.datapipes.schema.proteina.ProteinaDataSample").
        fixed_dimensions: Optional dict of dimension names to fixed sizes.
            E.g. {"I": 512, "S": 128} for fixed token and MSA row counts.
            Useful for torch.compile which needs static shapes.
            If None, pads dynamically to max in batch.

    Returns:
        PaddedBatchedDataSample with padding masks.
    """
    # Resolve string schema path to class
    if isinstance(schema, str):
        import importlib
        module_path, class_name = schema.rsplit(".", 1)
        schema = getattr(importlib.import_module(module_path), class_name)

    samples = [schema.from_dict(item) for item in batch]
    batched_samples = BatchedDataSample(
        samples, default_padding_dimensions=fixed_dimensions
    )
    return batched_samples.pad_and_stack()

