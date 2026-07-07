"""Configurable atom-pair distance metric for catalytic geometry constraints."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from foundry.metrics.metric import Metric
from foundry.utils.ddp import RankedLogger
from proteinfoundation.metrics.atom_selection import atom_selection_from_spec
from proteinfoundation.metrics.soft_threshold import apply_soft_bounds_to_outputs

logger = RankedLogger(__name__, rank_zero_only=True)


class AtomPairDistanceMetric(Metric):
    """Evaluate atom-pair upper-bound distance constraints on one structure.

    Each constraint checks that the distance between an "A" atom and a "B" atom
    stays at or below a hard upper bound (``max_distance_angstrom``) — useful for
    catalytic geometry and motif-contact checks.

    Config (per constraint in ``constraints``):

    - ``name`` — output prefix (optional; default ``constraint_{i}``).
    - ``max_distance_angstrom`` — hard **upper** bound in Å (required).
    - ``atom_a`` / ``atom_b`` — one atom spec dict **or** a list of spec dicts
      (both required). When a list is given, the constraint uses the **minimum**
      distance over all A-to-B pairs (OR semantics over alternative atoms).
    - Atom specs are dicts passed to :func:`atom_selection_from_spec` (``query``,
      ``chain_id``, ``res_name``, ``res_id``, ``atom_name``). Each spec must
      resolve to exactly one atom; specs that resolve to zero or multiple atoms
      are skipped.

    If neither endpoint resolves any atom, the distance is ``inf`` so the
    constraint always reads as violated — constrained atoms that are missing from
    a design should never count as satisfied.

    Outputs per constraint ``{name}``:

    - ``{name}_distance``, ``{name}_violation``, ``{name}_satisfied``
    - ``{name}_soft_score`` (linear decay from 1 at the bound to 0 at ``soft_margin`` above it)

    Aggregate: ``soft_min_score``, ``n_satisfied``, ``n_violated``, ``all_satisfied``,
    ``total_violation``, ``worst_violation``.
    """

    def __init__(
        self,
        constraints: list[dict[str, Any]],
        soft_margin: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.constraints = constraints
        self.soft_margin = 1.0 if soft_margin is None else float(soft_margin)

    @cached_property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {"predicted_atom_array": "predicted_atom_array"}

    @staticmethod
    def _endpoint_specs(constraint: dict[str, Any], key: str) -> list[dict[str, Any]]:
        """Return the atom spec dict(s) for endpoint ``key`` (``atom_a`` / ``atom_b``)."""
        if key not in constraint:
            raise ValueError(f"Constraint '{constraint.get('name', '?')}' is missing required '{key}'")
        val = constraint[key]
        return [val] if isinstance(val, dict) else list(val)

    @classmethod
    def _resolve_coords(cls, atom_array: AtomArray, specs: list[dict[str, Any]]) -> list[np.ndarray]:
        """Resolve each spec to the coordinate of its single matching atom (skipping non-unique specs)."""
        coords: list[np.ndarray] = []
        for spec in specs:
            idxs = np.flatnonzero(atom_selection_from_spec(atom_array, spec))
            if len(idxs) != 1:
                if len(idxs) > 1:
                    logger.warning(f"Spec matched {len(idxs)} atoms, expected 1 — returning None: {spec}")
                return []
            coords.append(atom_array.coord[idxs[0]])
        return coords

    @classmethod
    def _min_pair_distance(cls, atom_array: AtomArray, constraint: dict[str, Any]) -> float:
        a_coords = cls._resolve_coords(atom_array, cls._endpoint_specs(constraint, "atom_a"))
        b_coords = cls._resolve_coords(atom_array, cls._endpoint_specs(constraint, "atom_b"))
        if not a_coords or not b_coords:
            logger.warning(
                f"No atoms resolved for constraint '{constraint.get('name', '?')}' "
                f"(a={len(a_coords)}, b={len(b_coords)}); distance is inf (violated)"
            )
            return float("inf")
        return float(min(np.linalg.norm(a - b) for a in a_coords for b in b_coords))

    def compute(self, predicted_atom_array: AtomArray) -> dict[str, float]:
        results: dict[str, float] = {}
        violations: list[float] = []
        soft_bounds: list[dict[str, Any]] = []
        n_satisfied = 0
        margin = self.soft_margin

        for idx, constraint in enumerate(self.constraints):
            name = constraint.get("name", f"constraint_{idx}")
            max_dist = float(constraint["max_distance_angstrom"])
            distance = self._min_pair_distance(predicted_atom_array, constraint)
            violation = max(0.0, distance - max_dist)
            satisfied = float(distance <= max_dist)
            distance_key = f"{name}_distance"

            results[distance_key] = distance
            results[f"{name}_violation"] = violation
            results[f"{name}_satisfied"] = satisfied
            soft_bounds.append({"key": name, "value_key": distance_key, "upper_bound": max_dist, "margin": margin})

            violations.append(violation)
            n_satisfied += int(satisfied)

        if soft_bounds:
            results.update(apply_soft_bounds_to_outputs(results, soft_bounds))

        total = len(self.constraints)
        n_violated = total - n_satisfied
        results.update(
            {
                "n_satisfied": float(n_satisfied),
                "n_violated": float(n_violated),
                "all_satisfied": float(n_violated == 0),
                "total_violation": float(np.sum(violations)) if violations else 0.0,
                "worst_violation": float(np.max(violations)) if violations else 0.0,
            }
        )
        return results
