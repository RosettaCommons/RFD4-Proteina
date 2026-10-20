from types import SimpleNamespace

import pytest
import torch
from atomworks.ml.conditions import C_SEQ

from rfproteina.conditions import C_CRDGRP
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENCODING
from rfproteina.nn.bioir_decoder import capture_presence, constrained_sample, unpack_presence
from rfproteina.nn.bioir_full import format_prediction
from rfproteina.partial_autoencoder.autoencoder import AutoEncoder
from rfproteina.partial_autoencoder.decoder import get_atom_mask


@pytest.mark.parametrize(
    "desired,predicted", [("ALA", "GLY"), ("THR", "ALA"), ("LYS", "ASN"), ("GLU", "THR"), ("DG", "DT"), ("DT", "DA")]
)
@pytest.mark.parametrize("use_trunk_sequence", [False, True])
def test_policy_matches_native_ae_without_reference_coordinates(desired, predicted, use_trunk_sequence):
    want = ENCODING.token_to_idx[desired]
    wrong = ENCODING.token_to_idx[predicted]
    sequence = torch.tensor([[want, want]])
    emit = torch.tensor([[True, False]])
    padding = emit.clone()
    coordinates = torch.arange(2 * 37 * 3, dtype=torch.float32).reshape(1, 2, 37, 3) / 37
    presence = torch.full((1, 2, 37), -1.0)
    required = torch.zeros(1, 2, 37, dtype=torch.bool)
    required[0, 0] = get_atom_mask()[want]
    raw_types = torch.tensor([[wrong, 0]])
    raw = {"coors_nm": coordinates, "emit_mask": emit, "aatype_max": raw_types, "atom_presence_logits": presence}

    class Decoder:
        residue_atom_mask = get_atom_mask()

        def __call__(self, batch):
            return raw

    decoder = Decoder()
    native = SimpleNamespace(
        autoencoder=SimpleNamespace(decoder=decoder),
        inf_cfg=SimpleNamespace(args={"condition_decoder_on_sequence_prediction": use_trunk_sequence}),
    )
    expected = AutoEncoder.decode(native.autoencoder, {C_SEQ.full_name: sequence}, required_atom_mask=required)
    logits = torch.full((1, 2, ENCODING.n_tokens), -100.0)
    logits[..., want] = 10
    pred = SimpleNamespace(
        residue_types=raw_types,
        residue_type_offset=0,
        coordinates=coordinates,
        atom_mask=torch.zeros_like(required),
        sampling=SimpleNamespace(trunk_sequence_logits=logits),
    )
    input_sequence = sequence.clone()
    if use_trunk_sequence:
        input_sequence[0, 0] = ENCODING.token_to_idx["<M>"]
    features = {
        "padding_mask": padding,
        "can_be_encoded": emit,
        C_SEQ.full_name: input_sequence,
        C_CRDGRP.full_name: torch.where(required, 0, -1),
    }
    actual = constrained_sample(native, pred, features, [presence])
    torch.testing.assert_close(actual["atom_mask"], expected["atom_mask"])
    torch.testing.assert_close(actual["residue_type"][emit], expected["residue_type"][emit])
    torch.testing.assert_close(actual["coors"] / 10, expected["coors_nm"])
    torch.testing.assert_close(pred.coordinates, coordinates)
    assert not actual["atom_mask"][~padding].any()


def test_sliced_presence_preserves_original_rows():
    emit = torch.tensor([[True, False, True], [False, False, False], [False, True, False]])
    padding = torch.ones_like(emit)
    records = [torch.full((1, 2, 37), 2.0), torch.full((1, 1, 37), 3.0)]
    logits = unpack_presence(records, emit, padding)
    assert (logits[0, emit[0]] == 2).all()
    assert (logits[2, emit[2]] == 3).all()
    assert torch.isneginf(logits[~emit]).all()
    with pytest.raises(ValueError):
        unpack_presence(records[:1], emit, padding)


def test_formatter_preserves_predicted_coordinates_ligand_and_insertion_head():
    desired = ENCODING.token_to_idx["LYS"]
    sequence = torch.tensor([[desired, ENCODING.token_to_idx["<M>"], desired]])
    padding = torch.tensor([[True, True, False]])
    emit = torch.tensor([[True, False, False]])
    coordinates = torch.arange(3 * 37 * 3).reshape(1, 3, 37, 3).float()
    mask = torch.zeros(1, 3, 37, dtype=torch.bool)
    mask[0, 1, 1] = True
    required = torch.zeros_like(mask)
    required[0, 0] = get_atom_mask()[desired]
    insertion = torch.randn(1, 3, 3)
    prediction = SimpleNamespace(
        coordinates=coordinates,
        partial_redesign=None,
        residue_types=torch.tensor([[ENCODING.token_to_idx["GLY"], -1, -1]]),
        residue_type_offset=0,
        atom_mask=mask,
        insertion_logits=insertion,
        final_coordinates=torch.full_like(coordinates, 9999.0),
    )
    calls = []

    def assemble(sample, batch, **kwargs):
        calls.append(kwargs)
        assert kwargs["insertion_logits"] is insertion
        assert sample["residue_type"].tolist() == [[desired, -1, -1]]
        assert torch.equal(sample["atom_mask"], required | mask)
        torch.testing.assert_close(sample["coors"][sample["atom_mask"]], coordinates[required | mask] * 10)
        assert not sample["coors"][~sample["atom_mask"]].any()
        return [{}]

    native = SimpleNamespace(
        autoencoder=SimpleNamespace(decoder=SimpleNamespace(residue_atom_mask=get_atom_mask())),
        inf_cfg=SimpleNamespace(args={}),
        format_sample_prots_to_atom_arrays=assemble,
    )
    features = {
        "padding_mask": padding,
        "can_be_encoded": emit,
        "_native_row_index": torch.tensor([0]),
        C_SEQ.full_name: sequence,
        C_CRDGRP.full_name: torch.where(required, 0, -1),
    }
    result = format_prediction(
        native, prediction, features, {"padding_mask": padding}, decoder_presence=[torch.full((1, 1, 37), -1.0)]
    )
    assert len(calls) == 1 and result[0]["orig_batch_idx"] == 0


def test_presence_hook_is_removed_on_error():
    decoder = SimpleNamespace(atom_presence_linear=torch.nn.Identity())
    logits = torch.ones(1, 2, 37)
    with pytest.raises(RuntimeError, match="test error"), capture_presence(decoder) as records:
        decoder.atom_presence_linear(logits)
        raise RuntimeError("test error")
    assert len(records) == 1 and records[0] is logits
    assert not decoder.atom_presence_linear._forward_hooks
