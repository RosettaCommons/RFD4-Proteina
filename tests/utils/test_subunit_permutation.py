"""Symmetry invariance for a generated TNF-alpha binder and PDB 4KPY."""

from functools import cache
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
from atomworks.io import parse
from atomworks.io.transforms.atom_array import ensure_atom_array_stack

import rfproteina.conditions  # noqa: F401
from rfproteina.metrics.gt_structure_comparison import StructureComparisonMetric
from rfproteina.metrics.lddt_metric import AllAtomLDDT
from rfproteina.utils.parse_presets import ANNOTATED_CIF
from rfproteina.utils.subunit_permutation import permute_gt_subunits

DATA = Path(__file__).parents[1] / "data/subunit_permutation"
CASES = [("tnfa.cif.zst", (1, 2, 3)), ("4kpy_tmp.cif.zst", (1, 2)), ("4kpy_mn.cif.zst", (1, 2))]


@cache
def load_structure(filename: str) -> struc.AtomArray:
    config = ANNOTATED_CIF.replace(build_assembly="first", add_id_and_entity_annotations=True)
    return parse(DATA / filename, config=config)["assemblies"]["1"][0]


def permuted(array: struc.AtomArray, molecule_iids: tuple[int, ...], atom_swaps: bool = False) -> struc.AtomArray:
    pred = array.copy()
    indices = [np.flatnonzero(array.molecule_iid == iid) for iid in molecule_iids]
    assert all(len(index) == len(indices[0]) > 0 for index in indices)
    assert len(np.unique(array.molecule_entity[np.concatenate(indices)])) == 1
    for destination, source in zip(indices, indices[1:] + indices[:1], strict=True):
        pred.coord[destination] = array.coord[source]
        if atom_swaps:
            oxygen = destination[np.isin(array.atom_name[destination], ["O1P", "O2P"])]
            pred.coord[oxygen] = pred.coord[oxygen[::-1]]
    return pred


@pytest.mark.parametrize("filename,molecule_iids", CASES)
@pytest.mark.parametrize("missing", [(), ("molecule_entity",), ("molecule_iid",), ("molecule_entity", "molecule_iid")])
def test_permutation_invariance(filename: str, molecule_iids: tuple, missing: tuple) -> None:
    """Swapping equivalent molecules preserves zero RMSD with present or missing annotations."""
    original = load_structure(filename)
    gt = original.copy()
    pred = permuted(gt, molecule_iids)
    for annotation in missing:
        gt.del_annotation(annotation)
    if len(missing) == 2:
        for array in (gt, pred):
            array.del_annotation("chain_iid")
            array.transformation_id = array.chain_id.copy()
            array.chain_id[:] = "A"
    annotations = gt.get_annotation_categories()
    metric = StructureComparisonMetric(align_mask="is_polymer")
    assert metric.compute(gt, pred)["rmsd"] < 1e-4
    metric = StructureComparisonMetric(align_mask="is_polymer", resolve_symmetry=("atom",))
    assert metric.compute(gt, pred)["rmsd"] > 0.01
    assert gt.get_annotation_categories() == annotations
    np.testing.assert_array_equal(gt.coord, original.coord)


def test_tnfa_target_aligned_binder() -> None:
    """Permuting TNF-alpha subunits preserves binder RMSD when aligning only on the target."""
    gt = load_structure("tnfa.cif.zst")
    pred = permuted(gt, (1, 2, 3))
    metric = StructureComparisonMetric(align_mask="molecule_iid != 0", subset_mask="molecule_iid == 0")
    assert metric.compute(gt, pred)["rmsd"] < 1e-4


@pytest.mark.parametrize("levels", [("subunit", "atom"), ("atom", "subunit")])
def test_subunit_then_atom(levels: tuple) -> None:
    """Combined molecule and atom swaps resolve in subunit-first order for RMSD and LDDT."""
    gt = load_structure("4kpy_tmp.cif.zst").copy()
    gt.set_annotation("group", np.zeros(len(gt), dtype=int))
    pred = permuted(gt, (1, 2), atom_swaps=True)
    metric = StructureComparisonMetric(
        align_mask="is_polymer", subset_mask="~is_polymer", resolve_symmetry=levels, subdivide_by_annotation="group"
    )
    assert metric.compute(gt, pred)["rmsd"] < 1e-4
    assert AllAtomLDDT(resolve_symmetry=levels).compute(pred, gt)["all_atom_lddt"] == 1
    assert StructureComparisonMetric(resolve_symmetry=("subunit",)).compute(gt, pred)["rmsd"] > 0.01


@pytest.mark.parametrize("constraint", ["entity", "selection", "missing"])
def test_forbidden_swaps(constraint: str) -> None:
    """Different entities, incompatible selections, or missing coordinates prevent a swap."""
    original = load_structure("4kpy_tmp.cif.zst")
    gt, pred = original.copy(), permuted(original, (1, 2))
    selected = gt.molecule_iid == 1
    score = gt.res_name == "TMP"
    if constraint == "entity":
        gt.molecule_entity[selected] = gt.molecule_entity.max() + 1
    elif constraint == "selection":
        score = selected
    else:
        gt.coord[np.flatnonzero(selected)[0]] = np.nan
    result = permute_gt_subunits(pred, gt, frame_mask=gt.is_polymer, score_mask=score)
    np.testing.assert_array_equal(result.coord[0], gt.coord)


@pytest.mark.parametrize("metric_class", [StructureComparisonMetric, AllAtomLDDT])
def test_permutation_cap(metric_class: type, monkeypatch: pytest.MonkeyPatch) -> None:
    """Twenty identical ions raise an unsupported-search error before candidate allocation."""
    crowded = struc.concatenate([load_structure("4kpy_mn.cif.zst")[-1:]] * 20)
    crowded.molecule_iid = np.arange(20)
    monkeypatch.setattr(np, "tile", lambda *_a, **_k: pytest.fail("Must reject before allocating permutations"))
    with pytest.raises(NotImplementedError, match="above 720 permutations is not supported"):
        metric_class().compute(crowded, crowded)


def test_broadcast_per_model_annotations() -> None:
    """One GT model broadcasts its annotations while matching each prediction independently."""
    gt = ensure_atom_array_stack(load_structure("4kpy_tmp.cif.zst"))
    gt.set_per_stack_annotation("b_factor", np.full((1, gt.array_length()), 17.0))
    pred = ensure_atom_array_stack([gt[0], permuted(gt[0], (1, 2))])
    resolved = permute_gt_subunits(pred, gt)
    np.testing.assert_array_equal(resolved.get_per_stack_annotation("b_factor"), np.full((2, gt.array_length()), 17.0))
    assert StructureComparisonMetric().compute(gt, pred)["rmsd"] < 1e-4
    assert AllAtomLDDT().compute(pred, gt)["all_atom_lddt"] == 1


@pytest.mark.parametrize("filename,molecule_iids", CASES)
def test_independent_residue_numbering(filename: str, molecule_iids: tuple[int, ...]) -> None:
    """Equivalent copies match despite different residue offsets and gaps, without renumbering."""
    gt = load_structure(filename).copy()
    for i, iid in enumerate(molecule_iids):
        selected = gt.molecule_iid == iid
        gt.res_id[selected] = gt.res_id[selected] * (i + 2) + 100 * (i + 1)
    pred = permuted(gt, molecule_iids)
    residue_ids = gt.res_id.copy()
    metric = StructureComparisonMetric(align_mask="is_polymer")
    assert metric.compute(gt, pred)["rmsd"] < 1e-4
    assert AllAtomLDDT().compute(pred, gt)["all_atom_lddt"] == 1
    np.testing.assert_array_equal(gt.res_id, residue_ids)
    np.testing.assert_array_equal(pred.res_id, residue_ids)


@pytest.mark.parametrize("displacement", [0.0, 0.5])
def test_independently_numbered_manganese_ions(displacement: float) -> None:
    """Swapped single-atom ligands retain their actual displacement after backbone alignment."""
    gt = load_structure("4kpy_mn.cif.zst").copy()
    ions = np.flatnonzero(~gt.is_polymer)
    assert len(ions) == 2
    assert (gt.res_id[ions] == 101).all()
    gt.res_id[ions[1]] = 102
    pred = gt.copy()
    pred.coord[ions] = gt.coord[ions[::-1]]
    pred.coord[ions[0], 0] += displacement
    metric = StructureComparisonMetric(align_mask="is_polymer", subset_mask="~is_polymer")
    assert metric.compute(gt, pred)["rmsd"] == pytest.approx(displacement / np.sqrt(2), abs=1e-4)


@pytest.mark.parametrize("mismatch", ["atom_order", "element"])
def test_strict_correspondence(mismatch: str) -> None:
    """Equivalent molecule copies must keep ordered atom identities."""
    gt = load_structure("4kpy_tmp.cif.zst").copy()
    indices = np.flatnonzero(gt.molecule_iid == 1)
    if mismatch == "atom_order":
        order = np.arange(len(gt))
        order[indices[:2]] = indices[:2][::-1]
        gt = gt[order]
    else:
        gt.element[indices[0]] = "Fe"
    with pytest.raises(ValueError, match="identical chain boundaries and ordered atoms"):
        StructureComparisonMetric().compute(gt, gt)


def test_small_and_empty_selections() -> None:
    """Two-atom alignment and empty scoring selections retain their metric semantics."""
    gt = load_structure("4kpy_mn.cif.zst")
    ions = gt[~gt.is_polymer]
    assert len(ions) == 2
    assert AllAtomLDDT(inclusion_radius=1000).compute(ions, ions)["all_atom_lddt"] == 1
    assert StructureComparisonMetric(align_mask="~is_polymer").compute(ions, ions)["rmsd"] < 1e-4
    metric = StructureComparisonMetric(align_mask="is_polymer", subset_mask="res_name == 'XXX'")
    assert np.isnan(metric.compute(gt, gt)["rmsd"])
