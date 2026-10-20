"""Native output constraints for raw BioIR decoder predictions.

Use actual predicted coordinates, including required slots. Never use input or
reference coordinates as a replacement for decoder output.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
from atomworks.ml.conditions import C_SEQ

from rfproteina.conditions import C_CRDGRP
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.partial_autoencoder.decoder import get_allowed_residue_type_mask


@contextmanager
def capture_presence(decoder: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    """Observe the unpruned presence logits before the class-table intersection."""
    records = []
    if decoder.atom_presence_linear is None:
        raise RuntimeError("Native decoder constraints require atom-presence logits")
    handle = decoder.atom_presence_linear.register_forward_hook(lambda module, args, out: records.append(out))
    try:
        yield records
    finally:
        handle.remove()


def unpack_presence(records: list[torch.Tensor], emit: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    """Invert BioIR.decode's full-batch or per-row emitted-token slicing."""
    if torch.equal(emit, padding):
        if len(records) != 1 or records[0].shape != (*emit.shape, 37):
            raise ValueError("Unexpected full-batch decoder presence shape")
        return records[0]
    indices = [i for i in range(len(emit)) if bool(emit[i].any())]
    if len(records) != len(indices):
        raise ValueError("Decoder presence calls do not match emitted rows")
    result = torch.full((*emit.shape, 37), -torch.inf, device=emit.device)
    for i, record in zip(indices, records, strict=True):
        if record.shape != (1, int(emit[i].sum()), 37):
            raise ValueError("Unexpected per-row decoder presence shape")
        result[i, emit[i]] = record[0].float()
    return result


def constrained_sample(
    native: Any, prediction: Any, features: Mapping[str, torch.Tensor], records: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Mirror native AE final masks and sequence policy, without another decode."""
    emit = features["can_be_encoded"]
    padding = features["padding_mask"]
    presence = unpack_presence(records, emit, padding) > 0
    presence[..., 1] = True
    if C_CRDGRP.full_name in features:
        presence = presence | (features[C_CRDGRP.full_name] >= 0)
    sequence = features[C_SEQ.full_name]
    conditioned = sequence != UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
    if native.inf_cfg.args.get("condition_decoder_on_sequence_prediction", False):
        logits = prediction.sampling.trunk_sequence_logits
        if logits is None or logits.shape[-1] != UNIFIED_ATOM37_ENCODING.n_tokens:
            raise ValueError("Native sequence-conditioned decoding needs canonical trunk logits")
        if "token_type" in features:
            allowed = get_allowed_residue_type_mask(device=logits.device)[features["token_type"].long()]
            logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
        fill = emit & ~conditioned
        sequence = torch.where(fill, logits.argmax(-1), sequence)
        conditioned = conditioned | fill
    raw_types = prediction.residue_types
    raw_types = torch.where(raw_types >= 0, raw_types + prediction.residue_type_offset, raw_types)
    types = torch.where(conditioned & emit, sequence, raw_types)
    legal = native.autoencoder.decoder.residue_atom_mask.to(types.device)[types.clamp_min(0)]
    mask = legal & presence & emit[..., None]
    # Non-latent ligand tokens retain the sampler's actual reference-atom output.
    mask = torch.where(emit[..., None], mask, prediction.atom_mask) & padding[..., None]
    coordinates = torch.where(mask[..., None], prediction.coordinates.float(), 0.0)
    types = torch.where(emit, types, torch.full_like(types, -1))
    return {"coors": coordinates * 10.0, "residue_type": types, "padding_mask": padding, "atom_mask": mask}
