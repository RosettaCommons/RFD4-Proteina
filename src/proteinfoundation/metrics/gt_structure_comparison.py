import ast
from collections.abc import Callable
from collections import defaultdict

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
from proteinfoundation.conditions import C_CRDGRP, C_INTCRD

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
    If pred_masks is None, gt_masks are only applied to the groud truth structure and used blindly on the predicted structure

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

    assert len(gt_atoms_to_align) > 0, "No atoms found to align"
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
    apply_query_to_pred: bool = False,
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
            Otherwise, the query is only applied to the ground truth structure and used blindly on the predicted structure.

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
        extra_distance_types: List of extra distance types to compute. Can be "tip_atom", "inter_chain", "intra_chain", or "intra_chain_protein".
            "tip_atom" will compute the MAE of conditioned distances between protein atoms and non-protein atoms.
        aggregation_fn: Function to aggregate the MAE values for each pair in an atom array.

    Returns:
        dict with keys "mae_all_pairs", "mae_pairs_with_2d_condition", "mae_pairs_no_2d_condition" and values as numpy arrays
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
            f"One or more distance types are invalid: {extra_distance_types}. Must be one of: tip_atom, inter_chain, intra_chain, intra_chain_protein"
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
    apply_mask_queries_to_pred: bool = False,
    metric_fn: Callable[[AtomArray, AtomArray], float] = compute_rmsd,
    verbose_naming: bool = False,
    keep_aligned_predicted_structure: bool = False,
    **metric_kwargs: Any,
) -> dict[str, float]:
    """
    Compute structure comparison metrics between ground truth and predicted structures.
    Note that we always ignore nan coords when comparing structures. You do not need to include that in your subset mask.

    Args:
        ground_truth_atom_array_stack: Ground truth atom array or stack
        predicted_atom_array_stack: Predicted atom array or stack (optional if predicted_coords provided)
        subset_mask: Query string to subset the atom arrays (e.g., "is_backbone & is_polymer")
        align_mask: Query string to use for alignment (e.g., "is_protein_backbone | is_nucleic_acid")
        metric_type: Callable that computes the metric between two atom arrays. These methods assume everything is subset and aligned properly.
        apply_mask_queries_to_pred: Whether to apply the subset and align queries to the predicted structure independently, or simply use
            the masks obtained from querying the ground-truth AtomArray. Must be True if the AtomArrays are different lengths.
        verbose_naming: Whether to use verbose naming  for the metric keys (e.g. rmsd_align=<is_protein>_subset=<is_protein> vs rmsd)
        keep_aligned_predicted_structure: Whether to include the aligned predicted structure in the output dict.
        metric_kwargs: Additional keyword arguments to pass to the metric function.

    Returns:
        Dictionary with comparison results
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

    # align
    gt_stack, pred_stack = align_structures_by_annotations(
        gt_stack, pred_stack, align_mask, apply_query_to_pred=apply_mask_queries_to_pred
    )
    if keep_aligned_predicted_structure:
        aligned_pred_stack = pred_stack.copy()

    # No metrics are returned if alignment failed
    if gt_stack is None or pred_stack is None:
        return {}

    # subset
    if subset_mask:
        final_gt_subset_mask = gt_stack.mask(subset_mask) & gt_stack.mask("~has_nan_coord()")
        if apply_mask_queries_to_pred:
            final_pred_subset_mask = pred_stack.mask(subset_mask) & pred_stack.mask("~has_nan_coord()")
        else:
            final_pred_subset_mask = final_gt_subset_mask
    else:
        final_gt_subset_mask = gt_stack.mask("~has_nan_coord()")
        if apply_mask_queries_to_pred:
            final_pred_subset_mask = pred_stack.mask("~has_nan_coord()")
        else:
            final_pred_subset_mask = final_gt_subset_mask

    gt_stack = gt_stack[:, final_gt_subset_mask]
    pred_stack = pred_stack[:, final_pred_subset_mask]

    if final_gt_subset_mask.sum() == 0 or final_pred_subset_mask.sum() == 0:
        ranked_logger.warning(
            f"The subset mask {subset_mask} resulted in an empty selection after filtering out atoms with NaN coordinates."
            "This Metric will return NaN values."
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
    
    if keep_aligned_predicted_structure:
        metric_result["aligned_predicted_structure"] = aligned_pred_stack

    return metric_result


class StructureComparisonMetric(Metric):
    """
    Computes structure comparison metrics between ground truth and predicted structures.

    Supports various comparison types including RMSD and distance MAE, with optional
    subsetting and alignment capabilities.

    Args:
        subset_mask: Query string to subset the atom arrays (e.g., "is_backbone & is_polymer")
        align_mask: Query string to use for alignment (e.g., "is_backbone")
        gt_coord_annotation_name: Name of the coordinate annotation to use for the ground truth.
            NOTE: These new coordinates will not be symmetry-resolved after RF3 unless passed to it
            in `extra_annots_to_symm_resolve`.
        metric_type: Type of comparison metric ("rmsd" or "dist_mae")
        metric_kwargs: Additional keyword arguments to pass to the metric function. Must be in the format:
            {
                "dist_mae": {  # key must be the name of the metric type ("dist_mae" or "rmsd")
                    "extra_distance_types": ["tip_atom", "inter_chain", "intra_chain"]
                }
            }
        input_kwarg_names: Mapping from compute() parameter names to input kwarg names.
            Defaults to {"ground_truth_atom_array_stack": "ground_truth_atom_array",
            "predicted_atom_array_stack": "predicted_atom_array"}.
        apply_mask_queries_to_pred: Whether to apply the subset and align queries to the predicted structure independently, or simply use
            the masks obtained from querying the ground-truth AtomArray. Must be True if the AtomArrays are different lengths.
        verbose_naming: Whether to name metric keys verbosely (e.g. rmsd_align=<is_protein>_subset=<is_protein> vs rmsd).
        keep_aligned_predicted_structure: Whether to include the aligned predicted structure in the output dict.

    Example:
        >>> metric = StructureComparisonMetric(
        ...     subset_mask="is_backbone & is_polymer",
        ...     align_mask="is_backbone",
        ...     metric_type="rmsd",
        ... )
        >>> result = metric.compute(
        ...     ground_truth_atom_array_stack=gt, predicted_atom_array_stack=pred
        ... )
        >>> print(result)
        {'rmsd_align=<is_backbone>_subset=<is_backbone&is_polymer>': 1.23}
    """

    def __init__(
        self,
        subset_mask: str | None = None,
        align_mask: str | None = None,
        gt_coord_annotation_name: str = "coord",
        metric_type: Literal["rmsd", "dist_mae"] = "rmsd",
        metric_kwargs: dict[str, Any] = {},
        input_kwarg_names: dict[str, str] | None = None,
        apply_mask_queries_to_pred: bool = False,
        verbose_naming: bool = False,
        keep_aligned_predicted_structure: bool = False,
        **kwargs,
    ):
        self._input_kwarg_names = input_kwarg_names or {
            "ground_truth_atom_array_stack": "ground_truth_atom_array",
            "predicted_atom_array_stack": "predicted_atom_array",
        }

        super().__init__(**kwargs)

        self.subset_mask = subset_mask
        self.align_mask = align_mask
        self.gt_coord_annotation_name = gt_coord_annotation_name
        self.metric_type = metric_type
        self.metric_kwargs = metric_kwargs
        self.apply_mask_queries_to_pred = apply_mask_queries_to_pred
        self.verbose_naming = verbose_naming
        self.keep_aligned_predicted_structure = keep_aligned_predicted_structure

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return self._input_kwarg_names

    def compute(
        self,
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
    ) -> dict[str, float]:
        """
        Compute structure comparison metrics.

        Args:
            ground_truth_atom_array_stack: Ground truth atom array or stack
            predicted_atom_array_stack: Predicted atom array or stack

        Returns:
            Dictionary with comparison results
        """
        metric_kwargs = self.metric_kwargs.get(self.metric_type, {})

        # Optionally substitute ground truth coordinates
        updated_gt_stack = as_atom_array_plus_stack(ground_truth_atom_array_stack).copy()
        if self.gt_coord_annotation_name != "coord":
            if self.gt_coord_annotation_name in updated_gt_stack.get_per_stack_annotation_categories():
                updated_gt_coords = updated_gt_stack.get_per_stack_annotation(self.gt_coord_annotation_name)
            else:
                updated_gt_coords = updated_gt_stack.get_annotation(self.gt_coord_annotation_name)
            updated_gt_stack.coord = updated_gt_coords if updated_gt_coords.ndim == 3 else updated_gt_coords[None]

        return structure_comparison_metric(
            ground_truth_atom_array_stack=updated_gt_stack,
            predicted_atom_array_stack=predicted_atom_array_stack,
            subset_mask=self.subset_mask,
            align_mask=self.align_mask,
            metric_fn=STRUCTURE_COMPARISON_METRICS[self.metric_type],
            apply_mask_queries_to_pred=self.apply_mask_queries_to_pred,
            verbose_naming=self.verbose_naming,
            keep_aligned_predicted_structure=self.keep_aligned_predicted_structure,
            **metric_kwargs,
        )


class InternalCoordAdherenceMetric(Metric):
    """Computes mean adherence across all given internal coordinate groups.

    Detects these groups, then computes the rmsd analogously to the StructureComparisonMetric.

    Args:
        metric_type: Type of comparison metric ("rmsd" or "dist_mae")
        metric_kwargs: Additional keyword arguments to pass to the metric function. Must be in the format:
            {
                "dist_mae": {  # key must be the name of the metric type ("dist_mae" or "rmsd")
                    "extra_distance_types": ["tip_atom", "inter_chain", "intra_chain"]
                }
            }
        input_kwarg_names: Mapping from compute() parameter names to input kwarg names.
            Defaults to {"ground_truth_atom_array_stack": "ground_truth_atom_array",
            "predicted_atom_array_stack": "predicted_atom_array"}.
        additional_align_mask: Additional query string to AND with the internal coordinate group mask for alignment.
        additional_subset_mask: Additional query string to AND with the internal coordinate group mask for subsetting.
        apply_mask_queries_to_pred: Whether to apply the subset and align queries to the predicted structure independently, or simply use
            the masks obtained from querying the ground-truth AtomArray. Must be True if the AtomArrays are different lengths.
        verbose_naming: Whether to name metric keys verbosely (e.g. rmsd_align=<is_protein>_subset=<is_protein> vs rmsd).
        save_per_group_metrics: Whether to save metrics for each internal coordinate group separately.
    """

    def __init__(
        self,
        metric_type: Literal["rmsd", "dist_mae"] = "rmsd",
        metric_kwargs: dict[str, Any] = {},
        input_kwarg_names: dict[str, str] | None = None,
        additional_align_mask: str | None = None,
        additional_subset_mask: str | None = None,
        apply_mask_queries_to_pred: bool = False,
        verbose_naming: bool = False,
        save_per_group_metrics: bool = False,
        **kwargs,
    ):
        self._input_kwarg_names = input_kwarg_names or {
            "ground_truth_atom_array_stack": "ground_truth_atom_array",
            "predicted_atom_array_stack": "predicted_atom_array",
        }

        super().__init__(**kwargs)

        self.metric_type = metric_type
        self.metric_kwargs = metric_kwargs
        self.apply_mask_queries_to_pred = apply_mask_queries_to_pred
        self.verbose_naming = verbose_naming
        self.additional_align_mask = additional_align_mask
        self.additional_subset_mask = additional_subset_mask
        self.save_per_group_metrics = save_per_group_metrics

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return self._input_kwarg_names

    def compute(
        self,
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
    ) -> dict[str, float]:
        """
        Compute structure comparison metrics for each internal coordinate group.

        Args:
            ground_truth_atom_array_stack: Ground truth atom array or stack
            predicted_atom_array_stack: Predicted atom array or stack

        Returns:
            Dictionary with comparison results, both per-group and overall.
        """
        metric_kwargs = self.metric_kwargs.get(self.metric_type, {})

        # Substitute in internal coord group conditioning for the ground-truth coordinates
        updated_gt_stack = as_atom_array_plus_stack(ground_truth_atom_array_stack).copy()
        internal_coords_mask = C_INTCRD.mask(updated_gt_stack, default="generate")
        internal_coord_conditioning = C_INTCRD.annotation(updated_gt_stack, default="generate")
        updated_gt_stack.coord[:, internal_coords_mask, :] = internal_coord_conditioning[internal_coords_mask, :]

        coord_groups = C_CRDGRP.annotation(updated_gt_stack, default="generate")
        unique_coord_groups = set(np.unique(coord_groups))
        unique_internal_coord_groups = unique_coord_groups - {C_CRDGRP.UNKNOWN_COORDS, C_CRDGRP.KNOWN_ABSOLUTE_COORDS}

        if len(unique_internal_coord_groups) == 0:
            ranked_logger.warning(
                "No internal coordinate groups found in the ground truth data. This Metric will return an empty dict."
            )
            return {}

        metrics_dict = {}
        groups_dict = {}
        for group in unique_internal_coord_groups:
            mask_str = f"{C_CRDGRP.full_name} == {group}"
            subset_mask = (
                f"({mask_str}) & ({self.additional_subset_mask})"
                if self.additional_subset_mask
                else mask_str
            )
            align_mask = (
                f"({mask_str}) & ({self.additional_align_mask})"
                if self.additional_align_mask
                else mask_str
            )
            group_output = structure_comparison_metric(
                ground_truth_atom_array_stack=updated_gt_stack,
                predicted_atom_array_stack=predicted_atom_array_stack,
                subset_mask=subset_mask,
                align_mask=align_mask,
                metric_fn=STRUCTURE_COMPARISON_METRICS[self.metric_type],
                apply_mask_queries_to_pred=self.apply_mask_queries_to_pred,
                verbose_naming=self.verbose_naming,
                **metric_kwargs,
            )
            for k, v in group_output.items():
                new_key = k if self.verbose_naming else f"{k}_coord_group_{group}"
                metrics_dict[new_key] = v
                groups_dict[new_key] = group

        # Compute overall mean adherence, weighted by group size
        means_dict = defaultdict(float)
        total_length = np.isin(C_CRDGRP.annotation(updated_gt_stack), list(unique_internal_coord_groups)).sum()
        for k, v in metrics_dict.items():
            metric_name = k.split("_align=")[0] if self.verbose_naming else k.split("_coord_group_")[0]
            group_length = (C_CRDGRP.annotation(updated_gt_stack) == groups_dict[k]).sum()
            means_dict[metric_name] += v * group_length / total_length

        output_dict = means_dict
        if self.save_per_group_metrics:
            output_dict |= metrics_dict

        return output_dict


STRUCTURE_COMPARISON_METRICS = {
    "rmsd": compute_rmsd,
    "dist_mae": compute_distance_mae,
}
