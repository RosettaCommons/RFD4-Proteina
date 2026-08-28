"""Configurable atom-pair distance metric for catalytic geometry constraints."""

from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
from biotite.structure import AtomArray

from rfproteina.metrics.metric import Metric
from rfproteina.metrics.utils.soft_threshold import apply_soft_bounds_to_outputs
from rfproteina.utils.ddp import RankedLogger
from rfproteina.utils.mask_spec import resolve_mask

logger = RankedLogger(__name__, rank_zero_only=True)


class AtomPairDistanceMetric(Metric):
    """Evaluate atom-pair upper-bound distance constraints on one structure.

    Each constraint checks that the distance between an "A" atom and a "B" atom
    stays at or below a hard upper bound (``max_distance_angstrom``) — useful for
    catalytic geometry and motif-contact checks.

    Config (per constraint in ``constraints``):

    - ``name`` — output prefix (optional; default ``constraint_{i}``).
    - ``max_distance_angstrom`` — hard **upper** bound in Å (required).
    - ``atom_a`` / ``atom_b`` — one AtomWorks query string **or** a list of query
      strings (both required). When a list is given, the constraint uses the
      **minimum** distance over all A-to-B pairs (OR semantics over alternative
      atoms). Each query (see :py:func:`~rfproteina.utils.mask_spec.resolve_mask`)
      must resolve to exactly one atom; queries that resolve to zero or multiple
      atoms are skipped.

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
    def _endpoint_queries(constraint: dict[str, Any], key: str) -> list[str]:
        """Return the query string(s) for endpoint ``key`` (``atom_a`` / ``atom_b``)."""
        if key not in constraint:
            raise ValueError(f"Constraint '{constraint.get('name', '?')}' is missing required '{key}'")
        val = constraint[key]
        if isinstance(val, str):
            return [val]
        if isinstance(val, list | tuple) and all(isinstance(v, str) for v in val):
            return list(val)
        raise ValueError(
            f"Constraint '{constraint.get('name', '?')}' field '{key}' must be an AtomWorks query "
            f"string or a list of them, got {val!r}. The old dict form is no longer supported -- "
            "rewrite it as a query, e.g. \"res_id == 31 and atom_name == 'NE2'\" "
            "(see docs/workflow/metrics.md)."
        )

    @classmethod
    def _resolve_coords(cls, atom_array: AtomArray, queries: list[str]) -> list[np.ndarray]:
        """Resolve each query to the coordinate of its single matching atom (skipping non-unique queries).

        The queries for one endpoint are *alternatives* -- the metric takes the minimum
        distance over them -- so a query that resolves to zero or several atoms only
        drops that one alternative. Bailing out on the whole endpoint instead would let
        a single inapplicable alternative (e.g. an atom name absent from this design)
        discard the perfectly resolvable ones alongside it.
        """
        coords: list[np.ndarray] = []
        for query in queries:
            idxs = np.flatnonzero(resolve_mask(atom_array, query))
            if len(idxs) != 1:
                if len(idxs) > 1:
                    logger.warning(f"Query matched {len(idxs)} atoms, expected 1 — skipping: {query!r}")
                continue
            coords.append(atom_array.coord[idxs[0]])
        return coords

    @classmethod
    def _min_pair_distance(cls, atom_array: AtomArray, constraint: dict[str, Any]) -> float:
        a_coords = cls._resolve_coords(atom_array, cls._endpoint_queries(constraint, "atom_a"))
        b_coords = cls._resolve_coords(atom_array, cls._endpoint_queries(constraint, "atom_b"))
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
