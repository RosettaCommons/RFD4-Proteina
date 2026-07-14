"""Test for CathMetric: CATH superfamily annotation via Foldseek.

Requires FOLDSEEK_EXEC, CATH_FOLDSEEK_DB, and CATH_DOMAIN_LIST environment
variables to be set.  The test is skipped automatically when they are absent.

Expected result for 5VHT chain A:
  CATH superfamily 1.20.59.10 (Mainly Alpha / Up-down Bundle), type H.
  Ground truth confirmed from CathDomainList.S95.v4.4.0:
    5vhtA00  1  20  59  10  ...
"""

import os

import biotite.structure as struc
import pytest
from atomworks.io.parser import parse

from rfproteina.utils.parse_presets import DISTILLATION

CIF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "5vht_from_file.cif")

_REQUIRED_ENV = ("FOLDSEEK_EXEC", "CATH_FOLDSEEK_DB", "CATH_DOMAIN_LIST")

pytestmark = pytest.mark.skipif(
    any(os.getenv(v) is None for v in _REQUIRED_ENV),
    reason="FOLDSEEK_EXEC / CATH_FOLDSEEK_DB / CATH_DOMAIN_LIST not set",
)


def test_cath_label_5vht_chain_a():
    """5VHT chain A annotates as CATH superfamily 1.20.59.10 (H-level).

    Ground truth: CathDomainList.S95.v4.4.0 lists 5vhtA00 as 1.20.59.10.
    """
    from rfproteina.metrics.cath_metric import annotate_cath

    arr = parse(CIF_PATH, config=DISTILLATION)["asym_unit"][0]
    chain_a = arr[(arr.chain_id == "A") & struc.filter_amino_acids(arr)]

    result = annotate_cath(chain_a)

    assert result["C"] == "1", f"Expected C='1', got {result['C']!r}"
    assert result["A"] == "1.20", f"Expected A='1.20', got {result['A']!r}"
    assert result["T"] == "1.20.59", f"Expected T='1.20.59', got {result['T']!r}"
    assert result["H"] == "1.20.59.10", f"Expected H='1.20.59.10', got {result['H']!r}"
    assert result["type"] == "H", f"Expected type='H', got {result['type']!r}"
    assert result["evalue"] is not None and result["evalue"] < 1e-5
    assert result["tmscore"] is not None and result["tmscore"] > 0.8
