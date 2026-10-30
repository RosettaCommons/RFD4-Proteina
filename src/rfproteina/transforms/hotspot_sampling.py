"""Sample separated target hotspots around the centroid of annotated seed atoms."""

from typing import Literal

import numpy as np
from atomworks.ml.transforms.base import Transform
from biotite.structure import get_residue_starts, sasa

from rfproteina.conditions import C_HOT


class SampleNearbyHotspots(Transform):
    """Sample 2-6 exposed atoms on distinct target residues near the seed centroid.

    Excludes seed residues and enforces ``min_distance`` separation (Angstroms).
    ``patch_diameter`` sets the sphere diameter or oval minor diameters;
    ``major_diameter`` sets the oval's long axis in a random local surface direction.
    ``min_atom_sasa`` is in square Angstroms (1.4 A probe). Uses NumPy's seeded RNG,
    copies the input, and raises if the sampled count cannot fit; never reduces it.
    """

    def __init__(
        self,
        min_distance: float = 5.0,
        patch_diameter: float = 15.0,
        min_atom_sasa: float = 1.0,
        num_hotspots: tuple[int, ...] = (2, 3),
        shape: Literal["sphere", "oval"] = "sphere",
        major_diameter: float = 30.0,
        orientation_attempts: int = 32,
    ) -> None:
        if not np.isfinite([min_distance, patch_diameter, major_diameter, min_atom_sasa]).all():
            raise ValueError("Hotspot distances and surface-area threshold must be finite.")
        if not 0 < min_distance <= patch_diameter <= major_diameter or min_atom_sasa < 0:
            raise ValueError("Require 0 < min_distance <= patch_diameter <= major_diameter and min_atom_sasa >= 0.")
        if not num_hotspots or any(not isinstance(n, int) or not 2 <= n <= 6 for n in num_hotspots):
            raise ValueError("num_hotspots must contain integers from 2 through 6.")
        if shape not in ("sphere", "oval") or not isinstance(orientation_attempts, int) or orientation_attempts < 1:
            raise ValueError("Require shape='sphere' or 'oval' and a positive integer orientation_attempts.")
        self.min_distance = min_distance
        self.patch_diameter = patch_diameter
        self.min_atom_sasa = min_atom_sasa
        self.num_hotspots = tuple(num_hotspots)
        self.shape = shape
        self.major_diameter = major_diameter
        self.orientation_attempts = orientation_attempts

    @staticmethod
    def _sample_clique(compatible: np.ndarray, count: int) -> list[int] | None:
        """Find a separated set by randomized backtracking, pruning short branches."""

        def search(available: np.ndarray, needed: int) -> list[int] | None:
            if needed == 0:
                return []
            for position in range(len(available) - needed + 1):
                first, rest = int(available[position]), available[position + 1 :]
                selected = search(rest[compatible[first, rest]], needed - 1)
                if selected is not None:
                    return [first, *selected]
            return None

        return search(np.random.permutation(len(compatible)), count)

    def forward(self, data: dict) -> dict:
        """Resample hotspots around one or more seeds on a single target chain."""
        array = data["atom_array"]
        seeds = C_HOT.mask(array, default="generate")
        chains = array.chain_iid if "chain_iid" in array.get_annotation_categories() else array.chain_id
        if not seeds.any() or len(np.unique(chains[seeds])) != 1 or not np.isfinite(array.coord[seeds]).all():
            raise ValueError("Provide at least one finite hotspot seed, all on the same target chain.")
        target = (chains == chains[seeds][0]) & ~np.isin(array.element, ["H", "D"])
        target &= np.isfinite(array.coord).all(axis=1)
        starts = get_residue_starts(array, add_exclusive_stop=True)
        residues = np.repeat(np.arange(len(starts) - 1), np.diff(starts))
        offsets = array.coord.astype(np.float64) - array.coord[seeds].astype(np.float64).mean(axis=0)
        radii = np.full(3, self.patch_diameter / 2)
        if self.shape == "oval":
            radii[0] = self.major_diameter / 2
        eligible = target & ~np.isin(residues, residues[seeds]) & (np.linalg.norm(offsets, axis=1) <= radii[0])
        if self.min_atom_sasa > 0:
            eligible[target] &= sasa(array[target], probe_radius=1.4) >= self.min_atom_sasa
        candidates = np.flatnonzero(eligible)
        xyz = offsets[candidates]
        count = int(np.random.choice(self.num_hotspots))
        separation = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
        compatible = (separation >= self.min_distance) & (residues[candidates, None] != residues[candidates][None, :])
        if self.shape == "oval" and len(candidates) >= count:
            centered = xyz - xyz.mean(axis=0)
            _, axes = np.linalg.eigh(centered.T @ centered)
        for _ in range(self.orientation_attempts if self.shape == "oval" else 1):
            if len(candidates) < count:
                break
            inside = np.arange(len(candidates))
            if self.shape == "oval":
                angle = np.random.uniform(0, 2 * np.pi)
                major = axes[:, 1:] @ [np.cos(angle), np.sin(angle)]
                frame = np.column_stack([major, np.cross(axes[:, 0], major), axes[:, 0]])
                scaled = (xyz @ frame) / radii
                inside = np.flatnonzero(np.sum(scaled**2, axis=1) <= 1)
            selected = self._sample_clique(compatible[np.ix_(inside, inside)], count)
            if selected is not None:
                output = array.copy()
                mask = np.zeros(len(array), dtype=bool)
                mask[candidates[inside[selected]]] = True
                C_HOT.set_annotation(output, mask)
                return {**data, "atom_array": output}
        raise ValueError(
            f"{data.get('example_id', 'Input')}: cannot fit {count} separated hotspots in the {self.shape} patch."
        )
