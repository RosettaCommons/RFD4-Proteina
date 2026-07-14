"""Tests for the TMol block-pair metric (ligand params prepared from AtomArray)."""

from pathlib import Path

import pytest
from atomworks.io import parse
from atomworks.io.config import ParseConfig

# tmol ships as a torch-version-pinned wheel (see the `tmol` extra) and is only present in
# the container image; skip rather than fail collection for the whole suite when absent.
pytest.importorskip("tmol")

from proteinfoundation.metrics.physics.tmol_block_pair_metric import TMolBlockPairMetric  # noqa: E402

SAM_BINDER_CIF = Path(__file__).resolve().parents[1] / "data" / "sam_binder_generated_minimized.cif"

# Snapshot from a generated sm_binder sample (SAM ligand), cart-minimized once up front so
# relax="none" below scores a clash-free, deterministic pose; wide tolerance for TMol/version drift.
_EXPECTED_DDG = -13.6
_EXPECTED_HBONDS = 2.0

# fast_relax repacks rotamers (stochastic), so its ddG/energy vary run to run; wide tolerances.
_EXPECTED_RELAXED_DDG = -38.0
_EXPECTED_RELAXED_HBONDS = 3.0


@pytest.fixture
def sam_binder_atom_array():
    if not SAM_BINDER_CIF.is_file():
        pytest.skip(f"Missing fixture: {SAM_BINDER_CIF}")
    config = ParseConfig(model=1, build_assembly=None, add_missing_atoms=False)
    return parse(str(SAM_BINDER_CIF), config=config)["asym_unit"][0]


@pytest.mark.gpu
def test_sam_binder_block_pair_interface_metrics(sam_binder_atom_array):
    metric = TMolBlockPairMetric(
        region_a={"res_name": "SAM"},
        region_b={"chain_id": "A"},
        device="cuda",
        no_opt_h=False,
        term_names=["hbond"],
        term_cutoffs={"n_interface_hbonds": ("hbond", -0.3)},
    )
    metric.setup()
    out = metric.compute(predicted_atom_array=sam_binder_atom_array)
    metric.teardown()

    assert out["interface_ddg"] == pytest.approx(_EXPECTED_DDG, abs=2.0)
    assert out["n_interface_hbonds"] == pytest.approx(_EXPECTED_HBONDS, abs=1.0)
    assert out["hbond"] < 0.0


@pytest.mark.gpu
def test_sam_binder_fast_relax_block_pair_interface_metrics(sam_binder_atom_array):
    metric = TMolBlockPairMetric(
        region_a={"res_name": "SAM"},
        region_b={"chain_id": "A"},
        device="cuda",
        relax="fast_relax",
        no_opt_h=False,
        term_names=["hbond"],
        term_cutoffs={"n_interface_hbonds": ("hbond", -0.3)},
    )
    metric.setup()
    out = metric.compute(predicted_atom_array=sam_binder_atom_array)
    metric.teardown()

    assert out["interface_ddg"] == pytest.approx(_EXPECTED_RELAXED_DDG, abs=8.0)
    assert out["n_interface_hbonds"] == pytest.approx(_EXPECTED_RELAXED_HBONDS, abs=1.0)
    assert out["hbond"] < 0.0
