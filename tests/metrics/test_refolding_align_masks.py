"""The shipped PPI align masks must resolve under either kind of coordinate conditioning.

Regression test for `ppi.pass` reporting 0 for every binder built the way
`docs/how-to/input_prep/examples/condition_spec_ppi.yaml` and the protein-binder skill
recommend. Those inputs condition the target with internal coordinates (`C_CRDGRP`), while the
PPI align masks named only absolute coordinate conditioning (`sa_mask('C_CRD')`), which selects
no atoms for such a target. Alignment then found nothing to superimpose,
`refolded_target_backbone_aligned_binder_rmsd.rmsd` was never computed, and the `ppi` pass
criterion treated the absent key as indeterminate and forced `pass=0` regardless of design
quality. The shipped PPI benchmarks condition their targets absolutely, so CI only ever
exercised the branch the mask supported.

These tests read the masks from the shipped config rather than restating them, so a future edit
that drops conditioning-kind coverage fails here.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_CRD, C_SEQ
from biotite.structure import AtomArray
from hydra.utils import instantiate
from omegaconf import OmegaConf

from rfproteina.conditions import C_CRDGRP, C_INTCRD
from rfproteina.inference_utils.condition_spec import (
    build_annotated_atom_array_from_condition_spec,
    load_spec_file,
)
from rfproteina.transforms.inference_annotations import AddSrcComponentAnnotation
from rfproteina.transforms.unindexing import annotate_unindexed_insertion
from rfproteina.utils.mask_spec import resolve_mask
from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY
from rfproteina.utils.standard_annotations import SA_CRDSRC
from tests.helpers import CONFIGS_DIR, PROJECT_ROOT

REFOLDING_METRICS_CFG = CONFIGS_DIR / "metrics" / "refolding_metrics" / "all_refolding_metrics.yaml"

# The refolding metrics that align a design onto its coordinate-conditioned target. Both feed a
# pass criterion (`ppi` and `antibody` in `configs/metrics/pass_criteria/all_pass_criteria.yaml`).
TARGET_ALIGNED_METRICS = ("refolded_target_aligned_binder_rmsd", "refolded_target_backbone_aligned_binder_rmsd")

BACKBONE_ATOM_NAMES = ("N", "CA", "C", "O")
_ELEMENT_OF = {"N": "N", "CA": "C", "C": "C", "O": "O"}


def _metric_cfg(metric_name: str):
    return OmegaConf.load(REFOLDING_METRICS_CFG)[metric_name]


def _ppi_complex(conditioning: str, seed: int = 0) -> tuple[AtomArray, np.ndarray]:
    """A minimal binder/target complex: chain A is the given target, chain B the designed binder.

    Args:
        conditioning: ``"absolute"`` pins the target with ``C_CRD`` (coordinate group 0, what the
            PPI benchmarks ship); ``"internal"`` pins it with one ``C_CRDGRP`` internal-coordinate
            group and no absolute coordinates (what the input-prep docs and skill recommend).
        seed: Seed for the backbone coordinates.

    Returns:
        The complex and a per-atom mask marking the target chain.
    """
    residues = [("A", i) for i in range(1, 7)] + [("B", i) for i in range(1, 5)]
    n_atoms = len(residues) * len(BACKBONE_ATOM_NAMES)
    atom_array = as_atom_array_plus(AtomArray(n_atoms))
    for i, ((chain_id, res_id), atom_name) in enumerate(
        (residue, atom_name) for residue in residues for atom_name in BACKBONE_ATOM_NAMES
    ):
        atom_array.chain_id[i] = chain_id
        atom_array.res_id[i] = res_id
        atom_array.res_name[i] = "ALA"
        atom_array.atom_name[i] = atom_name
        atom_array.element[i] = _ELEMENT_OF[atom_name]
    atom_array.coord = (np.random.default_rng(seed).normal(size=(n_atoms, 3)) * 5).astype(np.float32)

    is_target = atom_array.chain_id == "A"
    atom_array.set_annotation("chain_iid", np.where(is_target, "A_1", "B_1"))
    atom_array.set_annotation("molecule_iid", (~is_target).astype(int))
    atom_array.set_annotation("is_polymer", np.ones(n_atoms, dtype=bool))
    atom_array.set_annotation("atomize", np.zeros(n_atoms, dtype=bool))
    # The target's sequence is given and the binder's is generated, which is what makes the binder
    # the chain of interest for `is_chain_of_interest`.
    C_SEQ.set_annotation(atom_array, is_target)

    C_CRD.set_annotation_from_ground_truth(
        atom_array, is_target if conditioning == "absolute" else np.zeros(n_atoms, dtype=bool)
    )
    if conditioning == "absolute":
        groups = np.where(is_target, C_CRDGRP.KNOWN_ABSOLUTE_COORDS, C_CRDGRP.UNKNOWN_COORDS)
    else:
        groups = np.where(is_target, 1, C_CRDGRP.UNKNOWN_COORDS)
    C_CRDGRP.set_annotation(atom_array, groups)
    if conditioning == "internal":
        C_INTCRD.set_annotation(atom_array, np.where(is_target[:, None], atom_array.coord, np.nan))

    # Staged over coordinate-conditioned atoms of either kind by the unindexing transforms, and
    # saved to the design CIF, so the metrics see it on the parsed design.
    src = SA_CRDSRC.default_annotation(atom_array)
    src[is_target] = "A"
    SA_CRDSRC.set_annotation(atom_array, src)

    return atom_array, is_target


@pytest.mark.parametrize("conditioning", ["absolute", "internal"])
@pytest.mark.parametrize("metric_name", TARGET_ALIGNED_METRICS)
def test_ppi_align_mask_selects_the_conditioned_target(metric_name: str, conditioning: str) -> None:
    """Each mask selects the whole conditioned target backbone, however the target was conditioned."""
    atom_array, is_target = _ppi_complex(conditioning)

    selected = np.asarray(resolve_mask(atom_array, _metric_cfg(metric_name).align_mask))

    np.testing.assert_array_equal(selected, is_target)


@pytest.mark.parametrize("conditioning", ["absolute", "internal"])
def test_ppi_metric_reports_an_rmsd_for_either_conditioning(conditioning: str) -> None:
    """The `ppi` criterion's third threshold key is produced, so the criterion can be evaluated.

    An empty alignment selection used to return ``{}`` here, which is what left
    `refolded_target_backbone_aligned_binder_rmsd.rmsd` out of the metrics and made
    `PassCriterionMetric` report `all_metrics_present=0` and `pass=0`.
    """
    design, _ = _ppi_complex(conditioning)
    refold = design.copy()
    refold.coord = refold.coord + 0.1

    metric = instantiate(_metric_cfg("refolded_target_backbone_aligned_binder_rmsd"))
    result = metric.compute(ground_truth_atom_array=design, predicted_atom_array=refold)

    assert "rmsd" in result, "the `ppi` pass criterion cannot be evaluated without this key"
    assert np.isfinite(result["rmsd"])


def test_absolute_conditioning_aligns_on_exactly_the_atoms_it_did_before() -> None:
    """The fix only adds atoms where the old mask had none, so the benchmarks are unaffected.

    `sa_mask('SA_CRDSRC')` is unioned into masks that were `sa_mask('C_CRD')`-only, and
    `SA_CRDSRC` covers every absolutely-conditioned atom, so an absolute-coordinate input
    resolves the identical selection.
    """
    atom_array, _ = _ppi_complex("absolute")

    for metric_name in TARGET_ALIGNED_METRICS:
        align_mask = _metric_cfg(metric_name).align_mask
        before = np.asarray(
            resolve_mask(atom_array, align_mask.replace("sa_mask('C_CRD') | sa_mask('SA_CRDSRC')", "sa_mask('C_CRD')"))
        )
        after = np.asarray(resolve_mask(atom_array, align_mask))
        np.testing.assert_array_equal(after, before, err_msg=f"{metric_name} changed on absolute conditioning")


def test_unconditioned_context_is_not_aligned_on() -> None:
    """Generated context in a non-designed chain stays out of the alignment.

    Alignment must use only atoms whose coordinates were given, so a mask keyed on coordinate-group
    membership is not a substitute: every atom carries a coordinate group
    (``CoordinateGroup.mask_from_annotation`` is all-True), so ``sa_mask('C_CRDGRP')`` would pull
    model-generated context into the frame.
    """
    atom_array, is_target = _ppi_complex("internal")
    # Only the first three target residues are conditioned; the rest is generated context.
    conditioned = is_target & (atom_array.res_id <= 3)
    C_INTCRD.set_annotation(atom_array, np.full((atom_array.array_length(), 3), np.nan))
    C_CRDGRP.set_annotation(
        atom_array,
        np.where(conditioned, 1, C_CRDGRP.UNKNOWN_COORDS),
        on_internal_coord_mismatch="clear",
    )
    C_INTCRD.set_annotation(atom_array, np.where(conditioned[:, None], atom_array.coord, np.nan))
    src = SA_CRDSRC.default_annotation(atom_array)
    src[conditioned] = "A"
    SA_CRDSRC.set_annotation(atom_array, src)

    for metric_name in TARGET_ALIGNED_METRICS:
        selected = np.asarray(resolve_mask(atom_array, _metric_cfg(metric_name).align_mask))
        np.testing.assert_array_equal(selected, conditioned, err_msg=f"{metric_name} aligned on generated context")


def test_shipped_ppi_spec_stages_the_annotation_the_masks_rely_on() -> None:
    """End-to-end on the documented PPI input: the fixed masks select the conditioned target.

    The unit tests above set ``SA_CRDSRC`` directly, so they cannot catch the staging that makes it
    mean anything: ``AddSrcComponentAnnotation`` (the inference pipeline's first transform) labels
    every residue, and ``annotate_unindexed_insertion`` copies those labels onto the
    coordinate-conditioned atoms only. If that coupling breaks, ``SA_CRDSRC`` silently resolves to
    nothing and `ppi.pass` goes back to being 0 for every design -- which is what this asserts
    against, on the real `condition_spec_ppi.yaml`, through the CIF round-trip the metrics read.
    """
    spec_path = PROJECT_ROOT / "docs" / "how-to" / "input_prep" / "examples" / "condition_spec_ppi.yaml"
    ((example_id, example),) = load_spec_file(spec_path).items()
    built = build_annotated_atom_array_from_condition_spec(example, spec_dir=spec_path.parent, example_id=example_id)
    atom_array = built["atom_array"]

    # The target is conditioned internally, so absolute-coordinate conditioning selects nothing.
    internally_conditioned = np.asarray(C_CRDGRP.is_internal(atom_array))
    assert internally_conditioned.any()
    assert not C_CRD.mask(atom_array, default="generate").any()

    atom_array = AddSrcComponentAnnotation()({"atom_array": atom_array})["atom_array"]
    design = annotate_unindexed_insertion(atom_array.copy(), {})
    np.testing.assert_array_equal(np.asarray(SA_CRDSRC.mask(design)), internally_conditioned)

    with tempfile.TemporaryDirectory() as tmp:
        cif_path = Path(tmp) / f"{example_id}.cif"
        to_cif_file(design, cif_path)
        reparsed = parse(cif_path, config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]

    for metric_name in TARGET_ALIGNED_METRICS:
        selected = np.asarray(resolve_mask(reparsed, _metric_cfg(metric_name).align_mask))
        assert selected.sum() == internally_conditioned.sum(), f"{metric_name} lost the conditioned target"


def test_an_empty_align_mask_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    """A mask that resolves to nothing names itself in the log instead of failing silently."""
    atom_array, _ = _ppi_complex("internal")
    metric = instantiate(
        OmegaConf.merge(
            _metric_cfg("refolded_target_backbone_aligned_binder_rmsd"),
            {"align_mask": "sa_mask('C_CRD') & is_protein_backbone"},  # selects nothing: no absolute coords
        )
    )

    with caplog.at_level("WARNING"):
        result = metric.compute(ground_truth_atom_array=atom_array, predicted_atom_array=atom_array.copy())

    assert result == {}
    assert "selected no atoms" in caplog.text
    assert "sa_mask('C_CRD') & is_protein_backbone" in caplog.text
