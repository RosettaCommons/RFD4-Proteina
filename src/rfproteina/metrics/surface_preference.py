"""Surface chemistry preferences for discrete search and differentiable guidance.

Discrete search uses the XTR chemistry preference vector. Gradient guidance
also offers a published side-chain entropy-loss proxy. Neither objective alone
establishes improved stability or solvent tolerance.
"""

import hashlib
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from atomworks.io import parse
from biotite.structure import AtomArray, superimpose

from rfproteina.encoding import UNIFIED_ATOM37_ENCODING
from rfproteina.metrics.metric import Metric
from rfproteina.surface_engineering import (
    AA_NAMES,
    SIDECHAIN_TDS,
    XTR_BIASES,
    ResidueKey,
    atom_residue_keys,
    selection_mask,
)
from rfproteina.utils.parse_presets import ANNOTATED_CIF


class SurfacePreferenceMetric(Metric):
    """Negative mean XTR preference on exact reference residue keys; minimise it."""

    def __init__(self, selected_keys: Sequence[Sequence[str | int]], reference_file: str | None = None) -> None:
        super().__init__()
        keys: list[ResidueKey] = []
        for key in selected_keys:
            if (
                len(key) != 3
                or not isinstance(key[0], str)
                or not isinstance(key[1], int)
                or not isinstance(key[2], str)
            ):
                raise ValueError("selected_keys must contain chain, residue number and insertion code")
            keys.append((key[0], key[1], key[2]))
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Surface selection must be nonempty and unique")
        self.selected_keys = keys
        self.reference = (
            None if reference_file is None else parse(Path(reference_file), config=ANNOTATED_CIF)["assemblies"]["1"][0]
        )

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, str]:
        """Read the native decoded candidate, including chain and residue IDs."""
        return {"predicted_atom_array": "predicted_atom_array"}

    # Metric routes named arguments through a **kwargs base method; concrete
    # signatures expose those names to its runtime input resolver.
    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:  # type: ignore[override]
        """Count residues once, regardless of their side-chain atom inventory."""
        if not np.isfinite(predicted_atom_array.coord).all():
            raise ValueError("Surface search candidate coordinates must be finite")
        selected = selection_mask(predicted_atom_array, self.selected_keys)
        residues = {
            key: str(name)
            for key, name, keep in zip(
                atom_residue_keys(predicted_atom_array), predicted_atom_array.res_name, selected, strict=True
            )
            if keep
        }
        if any(name not in AA_NAMES for name in residues.values()):
            raise ValueError("Surface preference encountered an unresolved/noncanonical identity")
        result = {
            "cost": -float(np.mean([XTR_BIASES.get(name, 0.0) for name in residues.values()])),
            "n_surface": float(len(residues)),
        }
        # A deterministic, sequence-independent uniform tie breaker supplies an
        # equal-branch-budget control without changing native search machinery.
        digest = hashlib.sha256(np.asarray(predicted_atom_array.coord, dtype=np.float32).tobytes()).digest()
        result["random_score"] = int.from_bytes(digest[:8], "little") / 2**64
        if self.reference is not None:
            expected = {
                key: str(name)
                for key, name in zip(atom_residue_keys(self.reference), self.reference.res_name, strict=True)
                if name in AA_NAMES
            }
            actual = {
                key: str(name)
                for key, name in zip(
                    atom_residue_keys(predicted_atom_array), predicted_atom_array.res_name, strict=True
                )
                if name in AA_NAMES
            }
            if expected.keys() != actual.keys():
                raise ValueError("Candidate and reference protein residue mappings differ")
            if any(actual[key] != name for key, name in expected.items() if key not in self.selected_keys):
                raise ValueError("Protected protein identity changed before search ranking")
            left = {
                k: c
                for k, n, c in zip(
                    atom_residue_keys(self.reference), self.reference.atom_name, self.reference.coord, strict=True
                )
                if n == "CA"
            }
            right = {
                k: c
                for k, n, c in zip(
                    atom_residue_keys(predicted_atom_array),
                    predicted_atom_array.atom_name,
                    predicted_atom_array.coord,
                    strict=True,
                )
                if n == "CA"
            }
            if left.keys() != right.keys():
                raise ValueError("Candidate and reference backbone residue mappings differ")
            ref = np.array([left[k] for k in left])
            candidate, _ = superimpose(ref, np.array([right[k] for k in left]))
            result["backbone_rmsd"] = float(np.sqrt(np.mean(np.sum((ref - candidate) ** 2, axis=-1))))
        return result


class ExpectedSurfacePreferenceMetric(Metric):
    """Expected chemistry or side-chain entropy cost under AE probabilities.

    Use the AE sequence-decision path explicitly, with
    condition_decoder_on_sequence_prediction=false, and configure matching
    sequence_guidance_token_indices so other state leaves are detached before
    decoding. This metric is only valid for the native full-generation path.
    """

    differentiable = True

    def __init__(self, token_indices: Sequence[int], objective: str = "xtr") -> None:
        super().__init__()
        if not token_indices or len(set(token_indices)) != len(token_indices) or any(i < 0 for i in token_indices):
            raise ValueError("Expected-cost selection must contain unique nonnegative token indices")
        self.indices = list(token_indices)
        if objective not in {"xtr", "sidechain_entropy"}:
            raise ValueError("Objective must be xtr or sidechain_entropy")
        self.costs = [
            SIDECHAIN_TDS[name] if objective == "sidechain_entropy" else -XTR_BIASES.get(name, 0.0) for name in AA_NAMES
        ]

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, tuple[str, str]]:
        """Request logits rather than differentiating through discrete identities."""
        return {"sequence_logits": ("decoded", "sequence_logits")}

    # Same named-input routing contract as the other concrete native metrics.
    def compute(self, sequence_logits: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """Return a finite scalar whose gradient follows selected soft probabilities."""
        if not sequence_logits.requires_grad:
            raise ValueError("Expected surface cost requires graph-preserving logits with a gradient")
        if sequence_logits.ndim != 3 or max(self.indices) >= sequence_logits.shape[1]:
            raise ValueError("Token indices do not map into decoded sequence logits")
        aa_indices = [UNIFIED_ATOM37_ENCODING.token_to_idx[name] for name in AA_NAMES]
        probabilities = sequence_logits[:, self.indices][:, :, aa_indices].float().softmax(dim=-1)
        costs = probabilities.new_tensor(self.costs)
        cost = (probabilities * costs).sum(dim=-1).mean()
        if not torch.isfinite(cost):
            raise ValueError("Expected surface cost is non-finite")
        return {"cost": cost}
