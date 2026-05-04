"""Secondary structure evaluation metrics for generated protein structures."""

import itertools
from typing import Any

import numpy as np
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.atom_array_plus import as_atom_array_plus, stack_atom_array_plus
from atomworks.ml.transforms.dssp import SSEnum
from atomworks.ml.utils.token import get_token_starts, spread_token_wise
from biotite.structure import AtomArray, AtomArrayStack, annotate_sse

from foundry.metrics import Metric
from proteinfoundation.conditions import C_SEC


def annotate_secondary_structure(
    atom_array: AtomArray,
    annotation_name: str = "sse",
    is_valid_annotation_name: str = "sse_is_valid",
    inplace: bool = False,
) -> AtomArray:
    """Annotate secondary structure using P-SEA algorithm.

    Only protein residues are assigned secondary structure; all others are
    set to SSEnum.NONE.

    Args:
      atom_array: AtomArray to annotate.
      annotation_name: Name for the SSE annotation. Defaults to ``"sse"``.
      is_valid_annotation_name: Name for validity annotation. Defaults to ``"sse_is_valid"``.
      inplace: If True, annotate ``atom_array`` directly. Defaults to False.

    Returns:
      AtomArray with secondary structure annotations added. When ``inplace=True``
      this is the same object that was passed in.
    """
    atom_array = as_atom_array_plus(atom_array)
    if not inplace:
        atom_array = atom_array.copy()

    # Get token-level information
    token_starts = get_token_starts(atom_array)
    n_tokens = len(token_starts)

    # Default all tokens to NONE
    sse = np.full(n_tokens, SSEnum.NONE, dtype=np.int8)
    is_valid = np.zeros(n_tokens, dtype=bool)

    try:
        # Run P-SEA on the entire structure
        # annotate_sse returns array of 'a', 'b', 'c', or '' for each residue
        psea_codes = annotate_sse(atom_array)

        # Map P-SEA codes to SSEnum values
        for i, code in enumerate(psea_codes):
            if code == "a":  # alpha helix
                sse[i] = SSEnum.HELIX
                is_valid[i] = True
            elif code == "b":  # beta sheet
                sse[i] = SSEnum.SHEET
                is_valid[i] = True
            elif code == "c":  # coil/loop
                sse[i] = SSEnum.LOOP
                is_valid[i] = True
            # Empty string '' means non-amino acid or missing CA -> leave as NONE

    except Exception:
        # P-SEA failed - return with all NONE values
        pass

    # Spread token-wise to all atoms
    sse_spread = spread_token_wise(atom_array, sse)
    is_valid_spread = spread_token_wise(atom_array, is_valid)

    # Set annotations
    atom_array.set_annotation(annotation_name, sse_spread)
    atom_array.set_annotation(is_valid_annotation_name, is_valid_spread)

    return atom_array


def compute_sse_distribution(atom_array: AtomArray) -> dict[str, float]:
    """Compute secondary structure distribution using P-SEA algorithm.

    Args:
      atom_array: Protein structure to analyze.

    Returns:
      Dict with keys:
        - helix_fraction: Fraction of tokens in alpha helix
        - sheet_fraction: Fraction of tokens in beta sheet
        - loop_fraction: Fraction of tokens in coil/loop
        - num_ss_elements: Number of contiguous SSE regions

      Returns NaN values if annotation fails or no valid annotations exist.
    """
    atom_array = as_atom_array_plus(atom_array)

    # Run P-SEA annotation
    try:
        atom_array = annotate_secondary_structure(
            atom_array, annotation_name="sse", is_valid_annotation_name="sse_is_valid"
        )
    except Exception:
        # Annotation failed - return NaN
        return {
            "helix_fraction": np.nan,
            "sheet_fraction": np.nan,
            "loop_fraction": np.nan,
            "num_ss_elements": np.nan,
        }

    # Get token-level annotations
    token_starts = get_token_starts(atom_array)
    token_sse = atom_array.sse[token_starts]
    token_is_valid = atom_array.sse_is_valid[token_starts]

    # Filter to valid tokens only
    valid_mask = token_is_valid
    if not valid_mask.any():
        # No valid annotations
        return {
            "helix_fraction": np.nan,
            "sheet_fraction": np.nan,
            "loop_fraction": np.nan,
            "num_ss_elements": np.nan,
        }

    valid_sse = token_sse[valid_mask]
    n_valid = len(valid_sse)

    # Compute fractions
    helix_fraction = (valid_sse == SSEnum.HELIX).sum() / n_valid
    sheet_fraction = (valid_sse == SSEnum.SHEET).sum() / n_valid
    loop_fraction = (valid_sse == SSEnum.LOOP).sum() / n_valid

    # Count contiguous SSE elements
    num_ss_elements = count_contiguous_sse_elements(valid_sse)

    return {
        "helix_fraction": float(helix_fraction),
        "sheet_fraction": float(sheet_fraction),
        "loop_fraction": float(loop_fraction),
        "num_ss_elements": float(num_ss_elements),
    }


def count_contiguous_sse_elements(sse_array: np.ndarray) -> int:
    """Count number of contiguous secondary structure elements.

    Args:
      sse_array: Array of SSEnum values (token-level).

    Returns:
      Number of contiguous SSE regions.

    Examples:
      >>> sse = np.array([1, 1, 2, 2, 3, 3, 1])  # helix, sheet, loop, helix
      >>> count_contiguous_sse_elements(sse)
      4
    """
    if len(sse_array) == 0:
        return 0

    # Count transitions using groupby
    # Each group represents a contiguous SSE element
    groups = [(k, len(list(g))) for k, g in itertools.groupby(sse_array)]
    return len(groups)


def compute_sse_accuracy(atom_array: AtomArray) -> dict[str, float]:
    """Compute secondary structure conditioning accuracy.

    Compares C_SEC annotation (conditioning) against P-SEA-annotated SSE on the
    same structure. Measures how well the generated structure matches its own
    secondary structure conditioning annotations.

    Args:
      atom_array: Generated structure with C_SEC annotations to validate.

    Returns:
      Dict with keys:
        - overall_accuracy: Fraction of correct predictions
        - helix_recall: Recall for helix conditioning
        - sheet_recall: Recall for sheet conditioning
        - loop_recall: Recall for loop conditioning
        - n_conditioned_atoms: Number of atoms with C_SEC conditioning

      Returns empty dict if no C_SEC annotation present.
    """
    atom_array = as_atom_array_plus(atom_array)

    # Check if C_SEC annotation exists
    if C_SEC.full_name not in atom_array.get_annotation_categories():
        return {}

    c_sec_annotation = C_SEC.annotation(atom_array)

    # Check if there are any conditioned atoms
    conditioned_mask = c_sec_annotation != SSEnum.NONE
    if not conditioned_mask.any():
        return {}

    # Run P-SEA on the structure
    try:
        atom_array = annotate_secondary_structure(
            atom_array,
            annotation_name="sse",
            is_valid_annotation_name="sse_is_valid",
        )
    except Exception:
        # Annotation failed
        return {
            "overall_accuracy": np.nan,
            "helix_recall": np.nan,
            "sheet_recall": np.nan,
            "loop_recall": np.nan,
            "n_conditioned_atoms": int(conditioned_mask.sum()),
        }

    # Filter to atoms that are:
    # 1. Conditioned (C_SEC != NONE)
    # 2. Have valid annotations
    valid_comparison_mask = conditioned_mask & atom_array.sse_is_valid

    if not valid_comparison_mask.any():
        # No valid comparisons possible
        return {
            "overall_accuracy": np.nan,
            "helix_recall": np.nan,
            "sheet_recall": np.nan,
            "loop_recall": np.nan,
            "n_conditioned_atoms": int(conditioned_mask.sum()),
        }

    # Extract conditioned and predicted SSE for valid atoms
    c_sec_conditioned = c_sec_annotation[valid_comparison_mask]
    sse_predicted = atom_array.sse[valid_comparison_mask]

    # Overall accuracy
    overall_accuracy = (c_sec_conditioned == sse_predicted).mean()

    # Per-class recall
    def compute_recall(sse_class: int) -> float:
        """Compute recall for a specific SSE class."""
        conditioned_as_class = c_sec_conditioned == sse_class
        if not conditioned_as_class.any():
            return np.nan
        return (sse_predicted[conditioned_as_class] == sse_class).mean()

    helix_recall = compute_recall(SSEnum.HELIX)
    sheet_recall = compute_recall(SSEnum.SHEET)
    loop_recall = compute_recall(SSEnum.LOOP)

    return {
        "overall_accuracy": float(overall_accuracy),
        "helix_recall": float(helix_recall),
        "sheet_recall": float(sheet_recall),
        "loop_recall": float(loop_recall),
        "n_conditioned_atoms": int(valid_comparison_mask.sum()),
    }


class SecondaryStructureDistribution(Metric):
    """Computes secondary structure distribution of generated protein structures.

    Uses P-SEA to annotate generated structures and reports the fraction of
    residues in helix, sheet, and loop conformations at the token level.
    Only considers tokens with valid annotations.

    Args:
      atom_array_input_key: Name of the input kwarg containing atom arrays.
        Use ``"generated_atom_array"`` for design contexts,
        ``"predicted_atom_array"`` for structure prediction contexts.
        Defaults to ``"generated_atom_array"``.
    """

    def __init__(self, atom_array_input_key: str = "predicted_atom_array", **kwargs):
        """Initialize SSE distribution metric."""
        self.atom_array_input_key = atom_array_input_key
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {"atom_array": self.atom_array_input_key}

    def compute(self, atom_array: AtomArray | AtomArrayStack | list) -> dict[str, float]:
        """Compute SSE distribution for generated structure(s).

        Args:
          atom_array: AtomArray, AtomArrayStack, or list of AtomArrays to analyze.

        Returns:
          Dict with SSE distribution values:
            - helix_fraction: Fraction of tokens in helix
            - sheet_fraction: Fraction of tokens in sheet
            - loop_fraction: Fraction of tokens in loop
            - num_ss_elements: Number of contiguous SSE regions

          For AtomArrayStack inputs, returns nanmean across all structures.
        """
        if isinstance(atom_array, list):
            atom_array = stack_atom_array_plus(atom_array)

        atom_array_stack = ensure_atom_array_stack(atom_array)
        results = [compute_sse_distribution(arr) for arr in atom_array_stack]
        return {
            "helix_fraction": float(np.nanmean([r["helix_fraction"] for r in results])),
            "sheet_fraction": float(np.nanmean([r["sheet_fraction"] for r in results])),
            "loop_fraction": float(np.nanmean([r["loop_fraction"] for r in results])),
            "num_ss_elements": float(np.nanmean([r["num_ss_elements"] for r in results])),
        }


class SecondaryStructureAccuracy(Metric):
    """Computes accuracy of secondary structure conditioning vs generated structures.

    Compares C_SEC conditioning annotation (carried through from generation) against
    P-SEA-annotated secondary structure of the same generated structures. Validates
    that the generated structure matches its own conditioning annotations.

    Args:
      atom_array_input_key: Name of the input kwarg containing atom arrays.

    """

    def __init__(self, atom_array_input_key: str = "predicted_atom_array", **kwargs):
        """Initialize SSE accuracy metric.

        Args:
          atom_array_input_key: Input kwarg name for atom array.
          **kwargs: Additional arguments passed to parent Metric class.
        """
        self.atom_array_input_key = atom_array_input_key
        super().__init__(**kwargs)

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        """Map validation inputs to compute() parameters."""
        return {"atom_array": self.atom_array_input_key}

    def compute(self, atom_array: AtomArray | AtomArrayStack) -> dict[str, float]:
        """Compute SSE conditioning accuracy.

        Args:
          atom_array: AtomArray, AtomArrayStack, or list of AtomArrays with C_SEC annotations.

        Returns:
          Dict with accuracy values (only if C_SEC present):
            - overall_accuracy: Fraction of correct predictions
            - helix_recall: Recall for helix conditioning
            - sheet_recall: Recall for sheet conditioning
            - loop_recall: Recall for loop conditioning
            - n_conditioned_atoms: Number of conditioned atoms

          For AtomArrayStack inputs, returns nanmean across all structures.
          Returns empty dict if no C_SEC annotations found.
        """
        if isinstance(atom_array, list):
            atom_array = stack_atom_array_plus(atom_array)

        atom_array_stack = ensure_atom_array_stack(atom_array)
        results = [compute_sse_accuracy(arr) for arr in atom_array_stack]

        # Filter out empty results (no C_SEC annotation)
        non_empty_results = [r for r in results if r]
        if not non_empty_results:
            return {}

        return {
            "overall_accuracy": float(np.nanmean([r["overall_accuracy"] for r in non_empty_results])),
            "helix_recall": float(np.nanmean([r["helix_recall"] for r in non_empty_results])),
            "sheet_recall": float(np.nanmean([r["sheet_recall"] for r in non_empty_results])),
            "loop_recall": float(np.nanmean([r["loop_recall"] for r in non_empty_results])),
            "n_conditioned_atoms": int(np.nanmean([r["n_conditioned_atoms"] for r in non_empty_results])),
        }
