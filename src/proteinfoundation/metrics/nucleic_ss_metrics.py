import numpy as np
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.ml.conditions import C_CRD, C_DIS, C_IDX
from atomworks.ml.utils.token import apply_token_wise, get_token_starts, spread_token_wise
from biotite.structure import AtomArray, AtomArrayStack

from foundry.metrics.metric import Metric
from proteinfoundation.conditions import C_BP, C_INTCRD, C_NA_SS_STATE, BasePairEnum
from proteinfoundation.metrics.metrics_utils import normalize_to_atom_array_list
from proteinfoundation.transforms.utils.na_geom_utils import annotate_na_ss


def _has_na_ss_conditions(atom_array: AtomArray) -> bool:
    """Return True if the array carries canonical NA-SS conditions."""
    atom_array = as_atom_array_plus(atom_array)
    categories_1d = atom_array.get_annotation_categories()
    categories_2d = atom_array.get_annotation_2d_categories()
    has_pair = C_BP.full_name in categories_2d
    has_state = C_NA_SS_STATE.full_name in categories_1d
    return has_pair and has_state


def _ensure_na_ss_conditions(atom_array: AtomArray) -> AtomArray:
    """Return the array if canonical NA-SS conditions exist, else raise."""
    atom_array = as_atom_array_plus(atom_array)
    if _has_na_ss_conditions(atom_array):
        return atom_array
    raise ValueError("atom_array missing canonical NA-SS conditions")


def _has_any_na_ss_annotation(atom_array: AtomArray) -> bool:
    """Return True if the array already carries canonical NA-SS conditions."""
    categories_1d = atom_array.get_annotation_categories()
    categories_2d = atom_array.get_annotation_2d_categories()
    return C_NA_SS_STATE.full_name in categories_1d or C_BP.full_name in categories_2d


def _get_candidate_token_ids(
    atom_array: AtomArray,
    *,
    restrict_to_nucleic: bool,
    compute_for_diffused_region_only: bool,
) -> set[int]:
    """Return a set of token_ids to include for scoring."""
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids = np.asarray(token_level_array.token_id, dtype=int)

    token_mask = np.ones(len(token_ids), dtype=bool)

    if restrict_to_nucleic:
        ensure_annotations(token_level_array, "is_rna", "is_dna")
        is_rna = np.asarray(token_level_array.is_rna, dtype=bool)
        is_dna = np.asarray(token_level_array.is_dna, dtype=bool)
        token_mask &= (is_rna | is_dna) if (is_rna.any() or is_dna.any()) else token_mask

    if compute_for_diffused_region_only:
        ensure_annotations(atom_array, "token_id")
        is_unindexed = ~C_IDX.mask(atom_array, default="generate")
        has_fixed_coord = C_CRD.mask(atom_array, default="generate")
        has_dist = C_DIS.mask(atom_array, default="generate").as_dense_array(False).any(axis=0)
        has_internal_coords = C_INTCRD.mask(atom_array, default="generate")
        has_coord_or_dist_conditioning = has_fixed_coord | has_dist | has_internal_coords

        token_is_unindexed = apply_token_wise(atom_array, is_unindexed, np.any)
        token_has_conditioning = apply_token_wise(atom_array, has_coord_or_dist_conditioning, np.any)
        token_mask &= ~token_is_unindexed & ~token_has_conditioning

    return {int(t) for t in token_ids[token_mask].tolist()}


def _extract_loop_and_pair_info(
    atom_array: AtomArray,
    *,
    allowed_token_ids: set[int],
) -> tuple[set[int], set[int], set[tuple[int, int]]]:
    """Identify loop tokens, paired tokens, and explicit base-pair edges.

    Args:
        atom_array: AtomArray carrying canonical NA-SS conditions.
        allowed_token_ids: Token IDs to consider (e.g., restrict to nucleic
            tokens or diffused region). Tokens outside this set are ignored.

    Returns:
        tuple[set[int], set[int], set[tuple[int, int]]]:
        (loop_token_ids, paired_token_ids, bp_pairs) within the allowed set.
        ``loop_token_ids`` are tokens explicitly marked unpaired.
        ``paired_token_ids`` participate in at least one base pair.
        ``bp_pairs`` stores sorted token-id pairs.
    """
    atom_array = _ensure_na_ss_conditions(atom_array)
    token_starts = get_token_starts(atom_array)
    token_level_array = atom_array[token_starts]
    token_ids = np.asarray(token_level_array.token_id, dtype=int)
    atom_to_token = spread_token_wise(atom_array, np.arange(len(token_starts), dtype=np.int32))

    loop_token_ids: set[int] = set()
    paired_token_ids: set[int] = set()
    bp_pairs: set[tuple[int, int]] = set()
    state_values = C_NA_SS_STATE.annotation(atom_array, default="raise")[token_starts].astype(np.int16)
    loop_mask = state_values == BasePairEnum.LOOP.value
    for i_tid in token_ids[loop_mask]:
        if int(i_tid) in allowed_token_ids:
            loop_token_ids.add(int(i_tid))

    pair_annotation = C_BP.annotation(atom_array, default="raise")
    if len(pair_annotation.pairs) > 0:
        token_pairs = atom_to_token[pair_annotation.pairs]
        valid_pairs = np.all(token_pairs >= 0, axis=1)
        token_pairs = token_pairs[valid_pairs]
        pair_values = np.asarray(pair_annotation.values)[valid_pairs]
        for (i_pos, j_pos), value in zip(token_pairs.tolist(), pair_values.tolist(), strict=False):
            if int(value) != BasePairEnum.PAIR.value:
                continue
            i_tid = int(token_ids[int(i_pos)])
            j_tid = int(token_ids[int(j_pos)])
            if i_tid in allowed_token_ids and j_tid in allowed_token_ids and i_tid != j_tid:
                paired_token_ids.add(i_tid)
                paired_token_ids.add(j_tid)
                a, b = (i_tid, j_tid) if i_tid < j_tid else (j_tid, i_tid)
                bp_pairs.add((a, b))

    return loop_token_ids, paired_token_ids, bp_pairs


def _safe_f1_from_sizes(intersection_n: int, pred_n: int, gt_n: int) -> float:
    """Return F1 with sensible empty-set handling."""
    if pred_n == 0 and gt_n == 0:
        return 1.0

    precision = float(intersection_n / pred_n) if pred_n > 0 else 0.0
    recall = float(intersection_n / gt_n) if gt_n > 0 else 0.0

    if precision + recall == 0.0:
        return 0.0

    return float(2.0 * precision * recall / (precision + recall))


def _f1_statistics_from_two_arr(
    gt_arr: AtomArray,
    pred_arr: AtomArray,
    restrict_to_nucleic: bool = True,
    compute_for_diffused_region_only: bool = False,
) -> tuple[float | None, float | None, float | None]:
    """
    Compute F1 statistics for base-pair and loop prediction from two AtomArrays with canonical NA-SS conditions.

    Args:
    - gt_arr: AtomArray representing the ground truth or user specification.
    - pred_arr: AtomArray representing the prediction to evaluate.
    - restrict_to_nucleic: If True, only consider tokens annotated as nucleic acids for scoring.
    - compute_for_diffused_region_only: If True, only consider tokens in diffused regions (non-motif) for scoring.

    Returns:
    A tuple of (pair_f1, loop_f1, weighted_f1) where:
    - pair_f1: F1 score for correctly predicted base pairs.
    - loop_f1: F1 score for correctly predicted explicitly unpaired loop tokens.
    - weighted_f1: Combined F1 score weighted by the number of paired vs. loop
      tokens in the ground truth, reflecting overall secondary structure adherence.

    """

    # Restrict to token_ids that are valid in both arrays.
    gt_allowed = _get_candidate_token_ids(
        gt_arr,
        restrict_to_nucleic=restrict_to_nucleic,
        compute_for_diffused_region_only=compute_for_diffused_region_only,
    )
    pred_allowed = _get_candidate_token_ids(
        pred_arr,
        restrict_to_nucleic=restrict_to_nucleic,
        compute_for_diffused_region_only=compute_for_diffused_region_only,
    )
    allowed = gt_allowed & pred_allowed

    if len(allowed) == 0:
        return None, None, None

    gt_loop, gt_paired_tokens, gt_pairs = _extract_loop_and_pair_info(gt_arr, allowed_token_ids=allowed)
    pred_loop, _pred_paired_tokens, pred_pairs = _extract_loop_and_pair_info(pred_arr, allowed_token_ids=allowed)

    pair_tp = len(gt_pairs & pred_pairs)
    pair_pred_n = len(pred_pairs)
    pair_gt_n = len(gt_pairs)

    loop_tp = len(gt_loop & pred_loop)
    loop_pred_n = len(pred_loop)
    loop_gt_n = len(gt_loop)

    pair_f1 = _safe_f1_from_sizes(pair_tp, pair_pred_n, pair_gt_n)
    loop_f1 = _safe_f1_from_sizes(loop_tp, loop_pred_n, loop_gt_n)

    pair_weight = len(gt_paired_tokens)
    loop_weight = len(gt_loop)
    total_weight = pair_weight + loop_weight
    if total_weight == 0:
        weighted_f1 = 1.0
    else:
        weighted_f1 = float((pair_weight * pair_f1 + loop_weight * loop_f1) / total_weight)

    return pair_f1, loop_f1, weighted_f1


class NucleicSSSimilarityMetrics(Metric):
    """Evaluate base-pair secondary-structure adherence against NA-SS targets.

    By default, annotation/scoring is not restricted to nucleic-acid residues.
    This is intentional: planar non-NA residues are allowed when they satisfy
    the same geometry/H-bond criteria as a base-paired interaction.
    """

    def __init__(
        self,
        compute_for_diffused_region_only: bool = False,
        annotate_predicted_fresh: bool = False,
        annotation_restrict_to_nucleic: bool = False,
        annotation_planar_only: bool = True,
        annotation_NA_only: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compute_for_diffused_region_only = compute_for_diffused_region_only
        self.annotate_predicted_fresh = annotate_predicted_fresh
        # Backward-compatible alias: `annotation_NA_only` previously controlled
        # whether annotation/scoring should be restricted to nucleic-acid tokens.
        if annotation_NA_only is not None:
            annotation_restrict_to_nucleic = annotation_NA_only
        self.annotation_restrict_to_nucleic = annotation_restrict_to_nucleic
        self.annotation_planar_only = annotation_planar_only

    @property
    def kwargs_to_compute_args(self) -> dict[str, tuple[str, ...]]:
        return {
            "predicted_atom_array_stack": ("predicted_atom_array",),
            "ground_truth_atom_array_stack": ("ground_truth_atom_array",),
        }

    def compute(
        self,
        *,
        predicted_atom_array_stack: AtomArray | AtomArrayStack,
        ground_truth_atom_array_stack: AtomArray | AtomArrayStack,
    ) -> dict[str, float]:
        """Compute F1 scores for nucleic acid secondary structure adherence.

        Args:
          predicted_atom_array_stack: structures with NA-SS feats generated from annotation of prediction.
          ground_truth_atom_array_stack: structures with NA-SS feats generated from user-specification or GT annotation.

        Returns:
          Dictionary with F1 scores.
            - ``pair_f1``: f1 similarity of predicted vs. specified base pair positions..
            - ``loop_f1``: f1 similarity of predicted vs. specified loop positions (explicitly unpaired tokens).
            - ``weighted_f1``: weighted sum of pair_f1 and loop_f1, weighted by
              the relative fraction of positions with each feature type in the target.
            - ``n_valid_samples``: number of valid samples.
        """

        if ground_truth_atom_array_stack is None or predicted_atom_array_stack is None:
            return {}

        predicted_atom_arrays = normalize_to_atom_array_list(predicted_atom_array_stack)
        ground_truth_atom_arrays = normalize_to_atom_array_list(ground_truth_atom_array_stack)

        pair_f1_list: list[float] = []
        loop_f1_list: list[float] = []
        weighted_f1_list: list[float] = []

        # count valid atom_arrays in stack
        n_valid = 0

        for gt_arr, pred_arr in zip(ground_truth_atom_arrays, predicted_atom_arrays, strict=False):
            if not _has_na_ss_conditions(gt_arr):
                continue
            gt_arr = as_atom_array_plus(gt_arr)

            # Important: predicted AtomArrays are built from a template AtomArray.
            # If that template already carries canonical NA-SS conditions derived
            # from the target, the prediction can inherit them and yield
            # artificially perfect scores. Optionally recompute conditions from the
            # predicted coordinates directly.
            if self.annotate_predicted_fresh or not _has_any_na_ss_annotation(pred_arr):
                pred_arr = annotate_na_ss(
                    pred_arr,
                    NA_only=self.annotation_restrict_to_nucleic,
                    planar_only=self.annotation_planar_only,
                )

            if not _has_na_ss_conditions(pred_arr):
                continue
            pred_arr = as_atom_array_plus(pred_arr)

            # Get the F1 statistics for these samples, split by feature type and weighted.
            pair_f1, loop_f1, weighted_f1 = _f1_statistics_from_two_arr(
                gt_arr,
                pred_arr,
                restrict_to_nucleic=self.annotation_restrict_to_nucleic,
                compute_for_diffused_region_only=self.compute_for_diffused_region_only,
            )

            # Exit if we couldn't compute valid scores, e.g. no valid tokens after filtering.
            # (dict returned below expects floats, so we can't have None values in the lists)
            if pair_f1 is None:
                continue

            # Base-pair F1
            pair_f1_list.append(pair_f1)
            # Loop F1 (explicitly unpaired tokens)
            loop_f1_list.append(loop_f1)
            # weighted sum of scores (based on SS element distribution in GT/user-specification)
            weighted_f1_list.append(weighted_f1)
            # If we've reached this point, then we have a valid sample and can count it as such.
            n_valid += 1

        if n_valid == 0:
            return {}

        return {
            "pair_f1": float(np.mean(pair_f1_list)),
            "loop_f1": float(np.mean(loop_f1_list)),
            "weighted_f1": float(np.mean(weighted_f1_list)),
            "n_valid_samples": int(n_valid),
        }
