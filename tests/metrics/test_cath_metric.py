"""Tests for Foldseek CATH annotation and GearNet input assembly.

The Foldseek test requires FOLDSEEK_EXEC, CATH_FOLDSEEK_DB, and CATH_DOMAIN_LIST
and is skipped automatically when its environment variables are absent.

Expected Foldseek result for 5VHT chain A:
  CATH superfamily 1.20.59.10 (Mainly Alpha / Up-down Bundle), type H.
  Ground truth confirmed from CathDomainList.S95.v4.4.0:
    5vhtA00  1  20  59  10  ...
"""

import io
import os
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
import zstandard
from atomworks.io.parser import parse
from atomworks.io.utils.selection import get_residue_starts

from rfproteina.metrics.cath_metric import _assemble_cath_classifier_inputs, annotate_cath
from rfproteina.utils.parse_presets import DISTILLATION

# The AFDB fixtures are annotated CIFs carrying their own CATH label. Training reads these
# datasets the same way: the distillation preset, with load_standard_annotations turned on
# in the dataset config's parser_args (see configs/datasets/train/distill/afdb_monomer_256.yaml).
ANNOTATED_DISTILLATION = DISTILLATION.replace(load_standard_annotations=True)

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CIF_PATH = _DATA_DIR / "5vht_from_file.cif"
AFDB_ZST_PATH = _DATA_DIR / "afdb_cath_examples" / "AF-A0A4P6L4I8-F1-model_v4.cif.zst"

_REQUIRED_FOLDSEEK_ENV = ("FOLDSEEK_EXEC", "CATH_FOLDSEEK_DB", "CATH_DOMAIN_LIST")

requires_foldseek = pytest.mark.skipif(
    any(os.getenv(v) is None for v in _REQUIRED_FOLDSEEK_ENV),
    reason="FOLDSEEK_EXEC / CATH_FOLDSEEK_DB / CATH_DOMAIN_LIST not set",
)


@pytest.fixture(scope="module")
def afdb_array():
    """Load the Zstandard-compressed annotated CIF fixture."""
    cif_text = zstandard.ZstdDecompressor().decompress(AFDB_ZST_PATH.read_bytes()).decode()
    return parse(io.StringIO(cif_text), config=ANNOTATED_DISTILLATION)["asym_unit"][0]


@requires_foldseek
def test_cath_label_5vht_chain_a():
    """5VHT chain A annotates as CATH superfamily 1.20.59.10 (H-level).

    Ground truth: CathDomainList.S95.v4.4.0 lists 5vhtA00 as 1.20.59.10.
    """
    arr = parse(CIF_PATH, config=DISTILLATION)["asym_unit"][0]
    chain_a = arr[(arr.chain_id == "A") & struc.filter_amino_acids(arr)]

    result = annotate_cath(chain_a, threads=1)

    assert result["C"] == "1", f"Expected C='1', got {result['C']!r}"
    assert result["A"] == "1.20", f"Expected A='1.20', got {result['A']!r}"
    assert result["T"] == "1.20.59", f"Expected T='1.20.59', got {result['T']!r}"
    assert result["H"] == "1.20.59.10", f"Expected H='1.20.59.10', got {result['H']!r}"
    assert result["type"] == "H", f"Expected type='H', got {result['type']!r}"
    assert result["evalue"] is not None and result["evalue"] < 1e-5
    assert result["tmscore"] is not None and result["tmscore"] > 0.8


def test_ca_array_reserves_rows_for_missing_residues(afdb_array):
    """``res_id`` gaps open masked rows, so a missing residue is a gap rather than a splice.

    Covers the three ways a residue can be absent from what the classifier sees -- cropped out,
    present but unresolved, and a crop that was renumbered contiguously. The first two must land
    identically (both are "this residue is missing from an otherwise 70-residue chain"); the
    third legitimately has no gap, because renumbered ``res_id``s assert contiguity.
    """
    arr = afdb_array
    coords, mask = _assemble_cath_classifier_inputs(arr)
    n_res = coords.shape[0]
    assert int(mask.sum()) == n_res, "fixture should be fully resolved"

    missing = np.arange(21, 41)  # res_ids, 20 residues
    expected_holes = np.arange(20, 40)  # zero-based rows

    cut = arr[~np.isin(arr.res_id, missing)]
    cut_coords, cut_mask = _assemble_cath_classifier_inputs(cut)
    assert cut_coords.shape[0] == n_res, "rows must still span the chain's full res_id extent"
    assert np.array_equal(np.flatnonzero(~cut_mask.numpy()), expected_holes)

    unresolved = arr.copy()
    unresolved.coord[np.isin(unresolved.res_id, missing)] = np.nan
    _, unresolved_mask = _assemble_cath_classifier_inputs(unresolved)
    assert np.array_equal(
        unresolved_mask.numpy(), cut_mask.numpy()
    ), "a cropped-out residue and an unresolved one must reach the model identically"

    renumbered = cut.copy()
    _, counts = np.unique(cut.res_id, return_counts=True)
    renumbered.res_id = np.repeat(np.arange(1, len(counts) + 1), counts)
    renum_coords, renum_mask = _assemble_cath_classifier_inputs(renumbered)
    assert renum_coords.shape[0] == n_res - len(missing)
    assert renum_mask.all(), "contiguous res_ids assert no gaps, so none should be invented"


def test_ca_array_keeps_rejoined_excisions_contiguous(afdb_array):
    """A res_id jump across a *bonded* pair is not a gap -- the backbone there is continuous.

    Excising an unindexed region re-joins the residues it left adjacent, so their res_ids jump
    while the chain stays unbroken. Taking res_id literally would open a gap the structure does
    not have, and gaps measurably move the classifier's predictions.
    """
    arr = afdb_array
    arr = arr[struc.filter_amino_acids(arr)]
    n_res = struc.get_residue_count(arr)

    # Excise a section and then re-bond the flanking residues,
    # as excising an unindexed region does.
    excised = arr[~np.isin(arr.res_id, np.arange(21, 41))]
    starts = get_residue_starts(excised)
    res_ids = excised.res_id[starts]
    prev_res, next_res = int(np.where(res_ids == 20)[0][0]), int(np.where(res_ids == 41)[0][0])
    c_idx = starts[prev_res] + int(np.where(excised.atom_name[starts[prev_res] : starts[prev_res + 1]] == "C")[0][0])
    n_idx = starts[next_res] + int(np.where(excised.atom_name[starts[next_res] : starts[next_res + 1]] == "N")[0][0])
    rejoined = excised.copy()
    rejoined.bonds.add_bond(c_idx, n_idx, 1)

    _, rejoined_mask = _assemble_cath_classifier_inputs(rejoined)
    assert len(rejoined_mask) == n_res - 20, "a re-bonded excision must not leave a hole"
    assert rejoined_mask.all()
