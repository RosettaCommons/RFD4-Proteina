"""Joint-frame motif geometry diagnostics for the indexed, apo M-CSA curriculum."""

import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus_stack
from atomworks.ml.conditions import C_SEQ
from biotite.structure import AtomArray, AtomArrayStack, superimpose

from rfproteina.metrics.metric import Metric
from rfproteina.utils.standard_annotations import SA_CRDSRC


def score_geometry(
    design: np.ndarray,
    predicted: np.ndarray,
    reference: np.ndarray,
    scaffold_ca: np.ndarray,
    motif_bb: np.ndarray,
    catalytic: np.ndarray,
    segment_ids: np.ndarray,
) -> dict[str, float]:
    """Fit all motif segments together; score each segment in that same frame.

    Catalytic atoms use literal atom names. Equivalent atom permutations are not
    optimized, making this a conservative geometry screen rather than AME scoring.
    """
    if not (design.shape == predicted.shape == reference.shape):
        raise ValueError("Design, prediction, and reference must have identical atom correspondence")
    if scaffold_ca.sum() < 3 or motif_bb.sum() < 3 or not catalytic.any():
        raise ValueError("Missing scaffold, motif backbone, or catalytic atoms")
    selected = scaffold_ca | motif_bb | catalytic
    if not np.isfinite(predicted[selected]).all() or not np.isfinite(design[scaffold_ca]).all():
        raise ValueError("Missing predicted or designed coordinates")
    if not np.isfinite(reference[motif_bb | catalytic]).all():
        raise ValueError("Missing reference coordinates")
    scaffold_fit, _ = superimpose(design[scaffold_ca], predicted[scaffold_ca])
    _, transform = superimpose(reference[motif_bb], predicted[motif_bb])
    aligned = transform.apply(predicted)
    errors = np.linalg.norm(aligned - reference, axis=-1)
    segment_rmsds = [
        np.sqrt(np.mean(errors[motif_bb & (segment_ids == group)] ** 2)) for group in np.unique(segment_ids[motif_bb])
    ]
    target_cat = reference[catalytic]
    predicted_cat = predicted[catalytic]
    distances = np.linalg.norm(target_cat[:, None] - target_cat[None, :], axis=-1)
    predicted_distances = np.linalg.norm(predicted_cat[:, None] - predicted_cat[None, :], axis=-1)
    return {
        "scaffold_ca_rmsd": float(np.sqrt(np.mean(np.sum((scaffold_fit - design[scaffold_ca]) ** 2, axis=-1)))),
        "motif_bb_rmsd": float(np.sqrt(np.mean(errors[motif_bb] ** 2))),
        "worst_segment_bb_rmsd": float(max(segment_rmsds)),
        "catalytic_rmsd": float(np.sqrt(np.mean(errors[catalytic] ** 2))),
        "max_catalytic_displacement": float(errors[catalytic].max()),
        "max_catalytic_distance_error": float(np.abs(distances - predicted_distances).max()),
    }


class MotifGeometryMetric(Metric):
    """Compare refolds with the original coordinate-conditioned motif, without per-island fits."""

    @property
    def kwargs_to_compute_args(self) -> dict[str, str]:
        return {
            "ground_truth_atom_array": "ground_truth_atom_array",
            "predicted_atom_array": "predicted_atom_array",
        }

    def compute(
        self,
        ground_truth_atom_array: AtomArray | AtomArrayStack,
        predicted_atom_array: AtomArray | AtomArrayStack,
    ) -> dict[str, float]:
        gt = as_atom_array_plus_stack(ground_truth_atom_array)
        pred = as_atom_array_plus_stack(predicted_atom_array)
        ensure_annotations(gt, "is_protein")
        if gt.array_length() != pred.array_length():
            raise ValueError("Refold lost atoms; cannot score the full motif")
        for annotation in ("chain_id", "res_id", "res_name", "atom_name", "ins_code"):
            if not np.array_equal(gt.get_annotation(annotation), pred.get_annotation(annotation)):
                raise ValueError(f"Refold atom correspondence differs: {annotation}")
        conditioned = SA_CRDSRC.mask(gt, default="raise") & gt.is_protein
        motif_bb = conditioned & np.isin(gt.atom_name, ["N", "CA", "C"])
        catalytic = conditioned & C_SEQ.mask(gt, default="raise") & ~np.isin(gt.atom_name, ["N", "CA", "C", "O", "OXT"])
        scaffold_ca = gt.is_protein & (gt.atom_name == "CA")
        segment_ids = np.full(gt.array_length(), -1, dtype=int)
        keys = list(zip(gt.chain_id, gt.res_id, gt.ins_code, strict=True))
        previous = None
        group = -1
        for i in np.flatnonzero(motif_bb):
            key = keys[i]
            if key != previous:
                if previous is None or key[0] != previous[0] or key[1] != previous[1] + 1:
                    group += 1
                previous = key
            segment_ids[i] = group
        annotation = "annotation_coords-with-insertion_1_atom"
        if annotation in gt.get_per_stack_annotation_categories():
            references = gt.get_per_stack_annotation(annotation)
        else:
            references = gt.get_annotation(annotation)[None]
        results = []
        for i, prediction in enumerate(pred.coord):
            gt_index = 0 if gt.stack_depth() == 1 else i
            reference = references[0 if len(references) == 1 else i]
            results.append(
                score_geometry(gt.coord[gt_index], prediction, reference, scaffold_ca, motif_bb, catalytic, segment_ids)
            )
        # Match StructureComparisonMetric's reduction across structure-stack members.
        return {key: float(np.mean([r[key] for r in results])) for key in results[0]}
