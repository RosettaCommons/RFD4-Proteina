import ast
from collections import defaultdict
from collections.abc import Callable

import numpy as np
from atomworks.enums import ChainType
from atomworks.io.utils.atom_array_plus import AtomArrayPlus, AtomArrayPlusStack, as_atom_array_plus_stack
from atomworks.io.utils.query import QueryExpression
from atomworks.ml.conditions import C_DIS
from atomworks.ml.conditions.annotator import ANNOTATOR_REGISTRY, ensure_annotations
from beartype.typing import Any, List, Literal
from biotite.structure import AtomArray, AtomArrayStack, rmsd, superimpose

from foundry.metrics.metric import Metric
from foundry.utils.logging import RankedLogger

ranked_logger = RankedLogger(name=__name__, rank_zero_only=False)


def _extract_annotation_names(query: str) -> list[str]:
    """Extract annotation names from query string by walking QueryExpression AST."""
    qe = QueryExpression(query)
    names = []
    for node in ast.walk(qe.tree):
        if isinstance(node, ast.Name) and node.id in ANNOTATOR_REGISTRY:
            names.append(node.id)
    return names


def align_structures_with_masks(
    gt_array_stack: AtomArrayStack,
    pred_array_stack: AtomArrayStack,
    gt_masks: List[np.ndarray],
    pred_masks: List[np.ndarray] | None = None,
) -> tuple[AtomArrayStack, AtomArrayStack]:
    """
    Align predicted structure to ground truth using specified masks (boolean arrays).
    If pred_masks is None, gt_masks are only applied to the groud truth structure and used blindly on the
    predicted structure

    Args:
        gt_array: Ground truth atom array
        pred_array: Predicted atom array
        gt_masks: List of masks to use for alignment.
        pred_masks: List of masks to use for alignment on predicted structure. If None, gt_masks are used.

    Returns:
        Tuple of (aligned_gt, aligned_pred)
    """

    gt_align_mask = np.logical_and.reduce(gt_masks)
    if pred_masks is None:
        pred_align_mask = gt_align_mask
    else:
        pred_align_mask = np.logical_and.reduce(pred_masks)

    gt_atoms_to_align = gt_array_stack[:, gt_align_mask]
    pred_atoms_to_align = pred_array_stack[:, pred_align_mask]

    assert len(gt_atoms_to_align) == len(
        pred_atoms_to_align
    ), "Alignment subsets of ground truth and predicted atom array stacks must be the same length"

    if len(gt_atoms_to_align) == 0:
        return None, None
    assert gt_atoms_to_align.mask("(~has_nan_coord())").all(), "Ground truth atoms to align have nan coords"
    assert pred_atoms_to_align.mask("(~has_nan_coord())").all(), "Predicted atoms to align have nan coords"
    if not all(len(gt_atoms_to_align[i]) == len(pred_atoms_to_align[i]) for i in range(len(gt_atoms_to_align))):
        ranked_logger.warning(
            "Ground truth and predicted atom arrays to align are different lengths. None values will be returned."
        )
        return None, None

    # align
    _, transform = superimpose(gt_atoms_to_align.coord, pred_atoms_to_align.coord)
    pred_array_stack.coord = transform.apply(pred_array_stack.coord)

    return gt_array_stack, pred_array_stack


def align_structures_by_annotations(
    gt_array_stack: AtomArrayStack,
    pred_array_stack: AtomArrayStack,
    align_to: str | None = None,
    apply_query_to_pred: bool = True,
) -> tuple[AtomArrayStack, AtomArrayStack]:
    """
    Align predicted structure to ground truth using query string.
    Note that nan coords are always ignored when aligning.

    Args:
        gt_array: Ground truth atom array
        pred_array: Predicted atom array
        align_to: Query string using AtomWorks syntax (e.g., "is_protein_backbone | is_nucleic_acid").
            If empty or None, no alignment is performed.
        apply_query_to_pred: If True, the `align_to` query will be independently applied to the predicted structure.
            Otherwise, the query is only applied to the ground truth structure and used blindly on the
            predicted structure.

    Returns:
        Tuple of (aligned_gt, aligned_pred)
    """

    if not align_to:
        return gt_array_stack, pred_array_stack

    # Extract and ensure annotations from query
    annotations_to_ensure = _extract_annotation_names(align_to)
    ensure_annotations(gt_array_stack, *annotations_to_ensure)

    # Get alignment mask and nan mask
    gt_align_mask = gt_array_stack.mask(align_to)
    gt_nan_mask = gt_array_stack.mask("~has_nan_coord()")
    gt_combined_mask = gt_align_mask & gt_nan_mask

    if apply_query_to_pred:
        pred_align_mask = pred_array_stack.mask(align_to)
        pred_nan_mask = pred_array_stack.mask("~has_nan_coord()")
        pred_combined_mask = pred_align_mask & pred_nan_mask
    else:
        pred_combined_mask = gt_combined_mask

    return align_structures_with_masks(
        gt_array_stack, pred_array_stack, gt_masks=[gt_combined_mask], pred_masks=[pred_combined_mask]
    )


def compute_rmsd(gt_array_stack: AtomArrayStack, pred_array_stack: AtomArrayStack, **kwargs) -> float:
    """
    Compute RMSD between two atom array stacks. Returns the RMSD for each atom array in the stack.

    Args:
        gt_array_stack: Ground truth atom array stack
        pred_array_stack: Predicted atom array stack

    Returns:
        dict with key "rmsd" and value RMSD values for each atom array in the stack
    """

    assert len(gt_array_stack) == len(
        pred_array_stack
    ), "Ground truth and predicted atom array stacks must be the same length"
    if len(gt_array_stack) == 0 or len(pred_array_stack) == 0:
        return np.nan

    assert all(
        len(gt_array_stack[i]) == len(pred_array_stack[i]) for i in range(len(gt_array_stack))
    ), "Ground truth and predicted atom arrays must be the same length"

    rmsds = []
    for i in range(len(gt_array_stack)):
        if len(gt_array_stack[i]) == 0:
            rmsds.append(np.nan)
            continue
        rmsds.append(float(rmsd(gt_array_stack[i].coord, pred_array_stack[i].coord)))

    return {"rmsd": np.asarray(rmsds)}


def compute_distance_mae(
    gt_array_stack: AtomArrayStack,
    pred_array_stack: AtomArrayStack,
    extra_distance_types: List[Literal["tip_atom", "inter_chain", "intra_chain", "intra_chain_protein"]] = [],
    aggregation_fn: Literal["max", "min", "mean"] = "mean",
) -> float:
    """
    Compute distance MAE between two atom array stacks. Returns the MAE for each atom array in the stack.
    Will check gt_array_stack for 2D distance annotation and use the mask to mask some distances.
    Args:
        gt_array_stack: Ground truth atom array stack
        pred_array_stack: Predicted atom array stack
        extra_distance_types: List of extra distance types to compute. Can be "tip_atom", "inter_chain",
            "intra_chain", or "intra_chain_protein".
            "tip_atom" will compute the MAE of conditioned distances between protein atoms and non-protein atoms.
        aggregation_fn: Function to aggregate the MAE values for each pair in an atom array.

    Returns:
        dict with keys "mae_all_pairs", "mae_pairs_with_2d_condition", "mae_pairs_no_2d_condition" and values as
        numpy arrays
    """
    assert len(gt_array_stack) == len(
        pred_array_stack
    ), "Ground truth and predicted atom array stacks must be the same length"

    assert len(gt_array_stack) > 0, "Ground truth atom array stack is empty"

    assert all(
        len(gt_array_stack[i]) == len(pred_array_stack[i]) for i in range(len(gt_array_stack))
    ), "Ground truth and predicted atom arrays must be the same length"

    if not all(
        distance_type in ["tip_atom", "inter_chain", "intra_chain", "intra_chain_protein"]
        for distance_type in extra_distance_types
    ):
        raise ValueError(
            f"One or more distance types are invalid: {extra_distance_types}. "
            "Must be one of: tip_atom, inter_chain, intra_chain, intra_chain_protein"
        )

    if aggregation_fn == "max":
        aggregation_fn = np.max
    elif aggregation_fn == "min":
        aggregation_fn = np.min
    elif aggregation_fn == "mean":
        aggregation_fn = np.mean
    else:
        raise ValueError(f"Invalid aggregation function: {aggregation_fn}. Must be one of: max, min, mean")

    # ========== Convenience ==========
    def mae(
        pred_dists: np.ndarray,
        gt_dists: np.ndarray,
        mask: np.ndarray,
        aggregation_fn: Callable[[np.ndarray], float] = aggregation_fn,
    ) -> float:
        if mask.any():
            return float(aggregation_fn(np.abs(pred_dists[mask.astype(bool)] - gt_dists[mask.astype(bool)])))
        else:
            return np.nan

    # =================================

    metrics = {
        "all_pairs": [],
        "pairs_with_2d_condition": [],
        "pairs_short": [],
        "pairs_medium": [],
        "pairs_long": [],
        "pairs_no_2d_condition": [],
    } | {
        f"{distance_type}_{length_type}": []
        for distance_type in extra_distance_types
        for length_type in ["short", "medium", "long", "all_lengths"]
    }

    for i in range(len(gt_array_stack)):
        # Handle empty arrays
        if len(gt_array_stack[i]) == 0:
            for metric in metrics:
                metrics[metric].append(np.nan)
            continue

        # Compute pairwise distances
        gt_diff = gt_array_stack[i].coord[:, None, :] - gt_array_stack[i].coord[None, :, :]
        gt_dists = np.linalg.norm(gt_diff, axis=2)

        pred_diff = pred_array_stack[i].coord[:, None, :] - pred_array_stack[i].coord[None, :, :]
        pred_dists = np.linalg.norm(pred_diff, axis=2)

        dist_conditioning_mask = C_DIS.mask(gt_array_stack[i]).as_dense_array()

        metrics["all_pairs"].append(mae(pred_dists, gt_dists, np.ones(len(pred_dists))))
        metrics["pairs_with_2d_condition"].append(mae(pred_dists, gt_dists, dist_conditioning_mask))
        metrics["pairs_short"].append(mae(pred_dists, gt_dists, dist_conditioning_mask & (gt_dists < 4.0)))
        metrics["pairs_medium"].append(
            mae(
                pred_dists,
                gt_dists,
                dist_conditioning_mask & (gt_dists >= 4.0) & (gt_dists < 8.0),
            )
        )
        metrics["pairs_long"].append(mae(pred_dists, gt_dists, dist_conditioning_mask & (gt_dists >= 8.0)))
        metrics["pairs_no_2d_condition"].append(mae(pred_dists, gt_dists, ~dist_conditioning_mask))

        for distance_type in extra_distance_types:
            for length_type in ["short", "medium", "long", "all_lengths"]:
                if length_type == "all_lengths":
                    length_mask = np.ones_like(gt_dists, dtype=bool)
                elif length_type == "short":
                    length_mask = gt_dists < 4.0
                elif length_type == "medium":
                    length_mask = (gt_dists >= 4.0) & (gt_dists < 8.0)
                elif length_type == "long":
                    length_mask = gt_dists >= 8.0
                if distance_type == "tip_atom":
                    protein_mask = np.isin(
                        gt_array_stack[i].chain_type,
                        [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L],
                    )
                    tipatom_mask_2d = protein_mask[:, None] & ~protein_mask[None, :]
                    tipatom_mask_2d = tipatom_mask_2d | tipatom_mask_2d.T  # symmetric
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            tipatom_mask_2d & dist_conditioning_mask & length_mask,
                        )
                    )
                elif distance_type == "inter_chain":
                    inter_chain_mask = gt_array_stack[i].chain_id[:, None] != gt_array_stack[i].chain_id[None, :]
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            inter_chain_mask & dist_conditioning_mask & length_mask,
                        )
                    )
                elif distance_type == "intra_chain":
                    intra_chain_mask = gt_array_stack[i].chain_id[:, None] == gt_array_stack[i].chain_id[None, :]
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            intra_chain_mask & dist_conditioning_mask & length_mask,
                        )
                    )
                elif distance_type == "intra_chain_protein":
                    protein_mask = np.isin(
                        gt_array_stack[i].chain_type,
                        [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L],
                    )
                    protein_mask_2d = protein_mask[:, None] & protein_mask[None, :]
                    intra_chain_mask = gt_array_stack[i].chain_id[:, None] == gt_array_stack[i].chain_id[None, :]
                    metrics[f"{distance_type}_{length_type}"].append(
                        mae(
                            pred_dists,
                            gt_dists,
                            intra_chain_mask & protein_mask_2d & dist_conditioning_mask & length_mask,
                        )
                    )

    return {f"mae_{metric}": np.asarray(metrics[metric]) for metric in metrics}


def structure_comparison_metric(
    ground_truth_atom_array_stack: AtomArray
    | AtomArrayStack
    | AtomArrayPlus
    | AtomArrayPlusStack
    | list[AtomArray | AtomArrayPlus],
    predicted_atom_array_stack: AtomArray
    | AtomArrayStack
    | AtomArrayPlus
    | AtomArrayPlusStack
    | list[AtomArray | AtomArrayPlus],
    subset_mask: str | None = None,
    align_mask: str | None = None,
    metric_fn: Callable[[AtomArray, AtomArray], float] = compute_rmsd,
    verbose_naming: bool = False,
    _return_n_atoms: bool = False,
    **metric_kwargs: Any,
) -> dict[str, float]:
    """Compute structure comparison metrics between ground truth and predicted structures.

    Args:
        ground_truth_atom_array_stack: Ground truth atom array or stack.
        predicted_atom_array_stack: Predicted atom array or stack.
        subset_mask: Query string to subset atom arrays (e.g., ``"is_backbone & is_polymer"``).
        align_mask: Query string for alignment (e.g., ``"is_protein_backbone | is_nucleic_acid"``).
        metric_fn: Callable computing the metric between two aligned atom arrays.
        verbose_naming: Emit verbose metric keys
            (e.g. ``rmsd_align=<is_backbone>_subset=<is_backbone>`` instead of ``rmsd``).
        _return_n_atoms: Internal flag. When ``True``, include ``"n_atoms"`` in the returned dict
            (used by :py:meth:`StructureComparisonMetric._compute_per_group` for weighted averaging).
        metric_kwargs: Additional keyword arguments forwarded to ``metric_fn``.

    Returns:
        Dictionary with comparison results. Contains ``"n_atoms"`` only when ``_return_n_atoms=True``.
    """
    # Convert inputs to stacks for uniform handling
    gt_stack = as_atom_array_plus_stack(ground_truth_atom_array_stack)
    pred_stack = as_atom_array_plus_stack(predicted_atom_array_stack)

    # ... copy to avoid modifying the original stacks
    pred_stack = pred_stack.copy()
    gt_stack = gt_stack.copy()

    # Extract and ensure annotations from queries
    annotations_to_ensure = []
    if subset_mask:
        annotations_to_ensure.extend(_extract_annotation_names(subset_mask))
    if align_mask:
        annotations_to_ensure.extend(_extract_annotation_names(align_mask))
    if annotations_to_ensure:
        ensure_annotations(gt_stack, *annotations_to_ensure)
        ensure_annotations(pred_stack, *annotations_to_ensure)

    # align
    gt_stack, pred_stack = align_structures_by_annotations(gt_stack, pred_stack, align_mask, apply_query_to_pred=True)

    # No metrics are returned if alignment failed
    if gt_stack is None or pred_stack is None:
        return {}

    # subset
    if subset_mask:
        final_gt_subset_mask = gt_stack.mask(subset_mask) & gt_stack.mask("~has_nan_coord()")
        final_pred_subset_mask = pred_stack.mask(subset_mask) & pred_stack.mask("~has_nan_coord()")
    else:
        final_gt_subset_mask = gt_stack.mask("~has_nan_coord()")
        final_pred_subset_mask = pred_stack.mask("~has_nan_coord()")

    gt_stack = gt_stack[:, final_gt_subset_mask]
    pred_stack = pred_stack[:, final_pred_subset_mask]

    if (final_gt_subset_mask.sum() == 0 or final_pred_subset_mask.sum() == 0) and not _return_n_atoms:
        ranked_logger.warning(
            f"The subset mask {subset_mask} resulted in an empty selection after filtering out atoms with NaN "
            "coordinates. This Metric will return NaN values."
        )

    # No metrics are returned if lengths differ after subsetting
    if not all(len(gt_stack[i]) == len(pred_stack[i]) for i in range(len(gt_stack))):
        ranked_logger.warning(
            "After applying subset masks, ground truth and predicted atom arrays were different lengths. "
            "No metrics will be returned."
        )
        return {}

    # compute metric
    metric_result = metric_fn(gt_stack, pred_stack, **metric_kwargs)

    # Format result keys
    align_str = align_mask.replace(" ", "") if align_mask else "all"
    subset_str = subset_mask.replace(" ", "") if subset_mask else "all"
    metric_result = {
        (f"{metric}_align=<{align_str}>_subset=<{subset_str}>" if verbose_naming else metric): np.mean(result)
        for metric, result in metric_result.items()
    }

    if _return_n_atoms:
        metric_result["n_atoms"] = gt_stack.array_length()

    return metric_result


class StructureComparisonMetric(Metric):
    """Computes structure comparison metrics between ground truth and predicted structures.

    Supports RMSD and distance MAE, with optional subsetting, alignment, and
    per-annotation-group alignment.

    Args:
        subset_mask: Query string to subset the atom arrays (e.g., ``"is_backbone & is_polymer"``).
        align_mask: Query string for alignment (e.g., ``"is_backbone"``).
        subdivide_by_annotation: Name of an annotation whose distinct values
            define independent align + score groups.
            ``subset_mask`` / ``align_mask`` are applied within each slice.
            Off by default (``None``).
        gt_coord_annotation_name: Per-atom annotation to use as ground-truth coordinates
            instead of ``coord``. Useful when the true target is stored in a separate annotation.
        metric_type: Comparison metric: ``"rmsd"`` or ``"dist_mae"``.
        metric_kwargs: Extra kwargs forwarded to the metric function, keyed by metric
            type (e.g. ``{"dist_mae": {"extra_distance_types": ["tip_atom"]}}``).
        input_kwarg_names: Maps ``compute()`` parameter names to pipeline kwarg names.
            Defaults to ``{"ground_truth_atom_array": "ground_truth_atom_array",
            "predicted_atom_array": "predicted_atom_array"}``.
        verbose_naming: Emit verbose metric keys
            (e.g. ``rmsd_align=<is_backbone>_subset=<is_backbone>`` instead of ``rmsd``).

    Examples:
        >>> metric = StructureComparisonMetric(
        ...     subset_mask="is_backbone & is_polymer",
        ...     align_mask="is_backbone",
        ...     metric_type="rmsd",
        ... )
        >>> result = metric.compute(ground_truth_atom_array=gt, predicted_atom_array=pred)
        >>> print(result)
        {'rmsd': 1.23}
    """

    def __init__(
        self,
        subset_mask: str | None = None,
        align_mask: str | None = None,
        subdivide_by_annotation: str | None = None,
        gt_coord_annotation_name: str = "coord",
        metric_type: Literal["rmsd", "dist_mae"] = "rmsd",
        metric_kwargs: dict[str, Any] | None = None,
        input_kwarg_names: dict[str, str] | None = None,
        verbose_naming: bool = False,
        **kwargs,
    ):
        self._input_kwarg_names = input_kwarg_names or {
            "ground_truth_atom_array": "ground_truth_atom_array",
            "predicted_atom_array": "predicted_atom_array",
        }

        super().__init__(**kwargs)

        self.subset_mask = subset_mask
        self.align_mask = align_mask
        self.subdivide_by_annotation = subdivide_by_annotation
        self.gt_coord_annotation_name = gt_coord_annotation_name
        self.metric_type = metric_type
        self.metric_kwargs = metric_kwargs or {}
        self.verbose_naming = verbose_naming

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return self._input_kwarg_names

    def _prepare_gt_stack(
        self,
        ground_truth_atom_array: AtomArray
        | AtomArrayStack
        | AtomArrayPlus
        | AtomArrayPlusStack
        | list[AtomArray | AtomArrayPlus],
    ) -> AtomArrayPlusStack:
        """Convert GT to stack and substitute coordinates from annotation if configured."""
        gt_stack = as_atom_array_plus_stack(ground_truth_atom_array).copy()

        if self.gt_coord_annotation_name != "coord":
            if self.gt_coord_annotation_name in gt_stack.get_per_stack_annotation_categories():
                updated_gt_coords = gt_stack.get_per_stack_annotation(self.gt_coord_annotation_name)
            else:
                updated_gt_coords = gt_stack.get_annotation(self.gt_coord_annotation_name)
            gt_stack.coord = updated_gt_coords if updated_gt_coords.ndim == 3 else updated_gt_coords[None]

        return gt_stack

    def _compute_per_group(
        self,
        gt_stack: AtomArrayPlusStack,
        pred_stack: AtomArrayPlusStack,
        metric_kwargs: dict[str, Any],
    ) -> dict[str, float]:
        """Slice by annotation value, score each group, return aggregated result."""
        annot = gt_stack.get_annotation(self.subdivide_by_annotation)
        unique_vals = np.unique(annot)

        weighted: dict[str, float] = defaultdict(float)
        total_atoms = 0
        any_result = False

        # We need to know how to aggregate the per-group metrics
        if self.metric_type not in ["rmsd", "dist_mae"]:
            raise ValueError(f"Unsupported metric type for per-group computation: {self.metric_type}")

        for val in unique_vals:
            mask = annot == val
            gt_sub = gt_stack[:, mask]
            pred_sub = pred_stack[:, mask]

            result = structure_comparison_metric(
                gt_sub,
                pred_sub,
                subset_mask=self.subset_mask,
                align_mask=self.align_mask,
                metric_fn=STRUCTURE_COMPARISON_METRICS[self.metric_type],
                verbose_naming=self.verbose_naming,
                _return_n_atoms=True,
                **metric_kwargs,
            )

            if not result:
                continue

            n = result.pop("n_atoms")

            if n == 0:
                continue

            total_atoms += n
            for k, v in result.items():
                if self.metric_type == "rmsd":
                    weighted[k] += v**2 * n
                else:
                    weighted[k] += v * n
            any_result = True

        if not any_result:
            ranked_logger.warning(
                f"No groups produced a valid result for annotation '{self.subdivide_by_annotation}'. "
                "Returning empty dict."
            )
            return {}

        if self.metric_type == "rmsd":
            return {k: float(np.sqrt(v / total_atoms)) for k, v in weighted.items()}
        else:
            return {k: float(v / total_atoms) for k, v in weighted.items()}

    def compute(
        self,
        ground_truth_atom_array: AtomArray
        | AtomArrayStack
        | AtomArrayPlus
        | AtomArrayPlusStack
        | list[AtomArray | AtomArrayPlus],
        predicted_atom_array: AtomArray
        | AtomArrayStack
        | AtomArrayPlus
        | AtomArrayPlusStack
        | list[AtomArray | AtomArrayPlus],
    ) -> dict[str, float]:
        """Compute structure comparison metrics.

        Args:
            ground_truth_atom_array: Ground truth atom array or stack.
            predicted_atom_array: Predicted atom array or stack.

        Returns:
            Dictionary with comparison results.
        """
        metric_kwargs = self.metric_kwargs.get(self.metric_type, {})
        gt_stack = self._prepare_gt_stack(ground_truth_atom_array)

        if self.subdivide_by_annotation:
            pred_stack = as_atom_array_plus_stack(predicted_atom_array).copy()
            return self._compute_per_group(gt_stack, pred_stack, metric_kwargs)

        return structure_comparison_metric(
            ground_truth_atom_array_stack=gt_stack,
            predicted_atom_array_stack=predicted_atom_array,
            subset_mask=self.subset_mask,
            align_mask=self.align_mask,
            metric_fn=STRUCTURE_COMPARISON_METRICS[self.metric_type],
            verbose_naming=self.verbose_naming,
            **metric_kwargs,
        )


STRUCTURE_COMPARISON_METRICS = {
    "rmsd": compute_rmsd,
    "dist_mae": compute_distance_mae,
}
