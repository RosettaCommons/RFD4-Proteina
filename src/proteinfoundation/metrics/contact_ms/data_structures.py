"""Internal struct-of-arrays containers for CMS computation.

These are numpy-backed data structures optimised for the molecular surface
algorithm. They grow dynamically via ``append`` / ``extend`` and should be
finalised (trimmed) before vectorised computation.

.. note::
   ``SurfaceAtomArray`` is *not* related to ``biotite.structure.AtomArray``.
   It is an internal container used only inside the surface calculator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from jaxtyping import Float, Int

# ── Attention / burial flag constants ─────────────────────────────────────────

ATTEN_BLOCKER = 1
ATTEN_BURIED = 5
ATTEN_6 = 6


# ── Result bookkeeping ────────────────────────────────────────────────────────


@dataclass
class SurfaceStats:
    """Bookkeeping counters accumulated during surface generation."""

    n_atoms: int = 0
    n_buried_atoms: int = 0
    n_blocked_atoms: int = 0
    convex_dots: int = 0
    toroidal_dots: int = 0
    concave_dots: int = 0

    per_molecule: list[dict[str, int]] = field(default_factory=lambda: [{}, {}])

    def init_molecule(self, mol: int, n: int) -> None:
        self.per_molecule[mol] = {
            "nAtoms": n,
            "nBuriedAtoms": 0,
            "nBlockedAtoms": 0,
        }


# ── SurfaceAtomArray ──────────────────────────────────────────────────────────


class SurfaceAtomArray:
    """Struct-of-arrays container for atoms in CMS computation.

    Grows dynamically via :meth:`extend_from_arrays` and must be trimmed
    with :meth:`finalize` before vectorised operations.

    Fields per atom:
        - ``xyz``: coordinates ``(N, 3)``
        - ``radius``: atom radius
        - ``density``: dot density
        - ``molecule``: ``0`` (binder) or ``1`` (target)
        - ``atten``: attention/burial flag
        - ``access``: accessibility flag
        - ``natom``: global atom index
        - ``nresidue``: residue index (unused in numpy path)
        - ``atom_name``, ``residue_name``: string labels (optional)
    """

    _INITIAL_CAP = 256
    _FLOAT_FIELDS = ("radius", "density")
    _INT8_FIELDS = ("molecule", "atten", "access")
    _INT32_FIELDS = ("natom", "nresidue")

    def __init__(self) -> None:
        cap = self._INITIAL_CAP
        self._n = 0
        self._cap = cap

        for f in self._FLOAT_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.float64))
        for f in self._INT8_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.int8))
        for f in self._INT32_FIELDS:
            setattr(self, f, np.zeros(cap, dtype=np.int32))

        self.atom_name: list[str] = [""] * cap
        self.residue_name: list[str] = [""] * cap
        self.xyz = np.zeros((cap, 3), dtype=np.float64)

    # ── growth ────────────────────────────────────────────────────────────

    def _grow(self) -> None:
        new_cap = self._cap * 2
        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            old = getattr(self, f)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[: self._n] = old[: self._n]
            setattr(self, f, new)
        self.atom_name = self.atom_name + [""] * self._cap
        self.residue_name = self.residue_name + [""] * self._cap
        xyz_old = self.xyz
        self.xyz = np.zeros((new_cap, 3), dtype=np.float64)
        self.xyz[: self._n] = xyz_old[: self._n]
        self._cap = new_cap

    # ── bulk insert ───────────────────────────────────────────────────────

    def extend_from_arrays(
        self,
        xyz: Float[np.ndarray, "N 3"],
        radii: Float[np.ndarray, N],
        density: float | Float[np.ndarray, N],
        molecule: int,
    ) -> None:
        """Bulk-append *N* atoms from numpy arrays.

        Args:
            xyz: Coordinates, shape ``(N, 3)``.
            radii: Radii, shape ``(N,)``.
            density: Dot density (scalar broadcast to all atoms).
            molecule: ``0`` (binder) or ``1`` (target).
        """
        n = len(radii)
        if n == 0:
            return
        while self._n + n > self._cap:
            self._grow()
        i = self._n
        self.xyz[i : i + n] = xyz
        self.radius[i : i + n] = radii
        self.density[i : i + n] = density
        self.molecule[i : i + n] = molecule
        self.natom[i : i + n] = np.arange(i, i + n, dtype=np.int32)
        self.access[i : i + n] = 0
        self._n += n

    # ── access ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        return self._n > 0

    def __getitem__(self, idx: slice | np.ndarray | list[int]) -> SurfaceAtomArray:
        """Slice, boolean mask, or integer array -> new ``SurfaceAtomArray``."""
        if isinstance(idx, slice):
            indices = np.arange(self._n)[idx]
        else:
            indices = np.asarray(idx)
            if indices.dtype == bool:
                if indices.shape[0] != self._n:
                    raise IndexError("Boolean index wrong length")
                indices = np.nonzero(indices)[0]

        indices = np.where(indices < 0, indices + self._n, indices)
        if np.any((indices < 0) | (indices >= self._n)):
            raise IndexError("SurfaceAtomArray index out of range")

        new = SurfaceAtomArray()
        new._n = len(indices)
        new._cap = new._n

        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            setattr(new, f, getattr(self, f)[indices])

        new.xyz = self.xyz[indices]
        new.atom_name = [self.atom_name[i] for i in indices]
        new.residue_name = [self.residue_name[i] for i in indices]
        new.finalize()
        return new

    def finalize(self) -> None:
        """Trim all backing arrays to ``[0:n]``. Call once after all inserts."""
        n = self._n
        for f in self._FLOAT_FIELDS + self._INT8_FIELDS + self._INT32_FIELDS:
            setattr(self, f, getattr(self, f)[:n])
        self.xyz = self.xyz[:n]
        self.atom_name = self.atom_name[:n]
        self.residue_name = self.residue_name[:n]
        self._cap = n


# ── DotArray ──────────────────────────────────────────────────────────────────


class DotArray:
    """Struct-of-arrays for molecular surface dots.

    Fields per dot:
        - ``coor_xyz``: surface point coordinates ``(N, 3)``
        - ``outnml_xyz``: outward normal ``(N, 3)``
        - ``area``: area contribution
        - ``buried``: ``1`` if buried by the other molecule
        - ``type_``: ``1`` convex, ``2`` toroidal, ``3`` concave
        - ``atom_idx``: index into :class:`SurfaceAtomArray`
    """

    _INITIAL_CAP = 4096

    def __init__(self) -> None:
        cap = self._INITIAL_CAP
        self._n = 0
        self._cap = cap

        self.coor_xyz = np.zeros((cap, 3), dtype=np.float64)
        self.outnml_xyz = np.zeros((cap, 3), dtype=np.float64)
        self.area = np.zeros(cap, dtype=np.float64)
        for f in ("buried", "type_"):
            setattr(self, f, np.zeros(cap, dtype=np.int8))
        self.atom_idx = np.zeros(cap, dtype=np.int32)

    def _grow(self) -> None:
        new_cap = self._cap * 2
        for f in ("coor_xyz", "outnml_xyz", "area", "buried", "type_", "atom_idx"):
            old = getattr(self, f)
            new = np.zeros((new_cap,) + old.shape[1:], dtype=old.dtype)
            new[: self._n] = old[: self._n]
            setattr(self, f, new)
        self._cap = new_cap

    def extend(
        self,
        coor_xyz: Float[np.ndarray, "D 3"],
        outnml_xyz: Float[np.ndarray, "D 3"],
        area: Float[np.ndarray, D],
        buried: Int[np.ndarray, D],
        type_: Int[np.ndarray, D],
        atom_idx: Int[np.ndarray, D],
    ) -> None:
        """Batch-append *D* dots."""
        n = len(coor_xyz)
        if n == 0:
            return
        while self._n + n > self._cap:
            self._grow()
        i = self._n
        self.coor_xyz[i : i + n] = coor_xyz
        self.outnml_xyz[i : i + n] = outnml_xyz
        self.area[i : i + n] = area
        self.buried[i : i + n] = buried
        self.type_[i : i + n] = type_
        self.atom_idx[i : i + n] = atom_idx
        self._n += n

    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        return self._n > 0

    def finalize(self) -> None:
        """Trim backing arrays to true size."""
        n = self._n
        for f in ("coor_xyz", "outnml_xyz", "area", "buried", "type_", "atom_idx"):
            setattr(self, f, getattr(self, f)[:n])


# ── ProbeArray ────────────────────────────────────────────────────────────────


class ProbeArray:
    """Struct-of-arrays for probe spheres placed at atom triplets.

    Fields per probe:
        - ``atom_idx_0/1/2``: the three generating atoms
        - ``height``: probe height above the torus circle
        - ``point_xyz``: probe centre coordinates ``(N, 3)``
        - ``alt_xyz``: alternate axis direction ``(N, 3)``
    """

    _INITIAL_CAP = 1024

    def __init__(self) -> None:
        cap = self._INITIAL_CAP
        self._n = 0
        self._cap = cap

        self.height = np.zeros(cap, dtype=np.float64)
        self.atom_idx_0 = np.zeros(cap, dtype=np.int32)
        self.atom_idx_1 = np.zeros(cap, dtype=np.int32)
        self.atom_idx_2 = np.zeros(cap, dtype=np.int32)
        self.point_xyz = np.zeros((cap, 3), dtype=np.float64)
        self.alt_xyz = np.zeros((cap, 3), dtype=np.float64)

    def _grow(self) -> None:
        new_cap = self._cap * 2
        for f in ("height", "atom_idx_0", "atom_idx_1", "atom_idx_2"):
            old = getattr(self, f)
            new = np.zeros(new_cap, dtype=old.dtype)
            new[: self._n] = old[: self._n]
            setattr(self, f, new)
        for f in ("point_xyz", "alt_xyz"):
            old = getattr(self, f)
            new = np.zeros((new_cap, 3), dtype=np.float64)
            new[: self._n] = old[: self._n]
            setattr(self, f, new)
        self._cap = new_cap

    def extend_from_arrays(
        self,
        atom_idx_0: Int[np.ndarray, P],
        atom_idx_1: Int[np.ndarray, P],
        atom_idx_2: Int[np.ndarray, P],
        height: Float[np.ndarray, P],
        point_xyz: Float[np.ndarray, "P 3"],
        alt_xyz: Float[np.ndarray, "P 3"],
    ) -> None:
        """Bulk-append *P* probes from numpy arrays."""
        n = len(height)
        if n == 0:
            return
        while self._n + n > self._cap:
            self._grow()
        i = self._n
        self.atom_idx_0[i : i + n] = atom_idx_0
        self.atom_idx_1[i : i + n] = atom_idx_1
        self.atom_idx_2[i : i + n] = atom_idx_2
        self.height[i : i + n] = height
        self.point_xyz[i : i + n] = point_xyz
        self.alt_xyz[i : i + n] = alt_xyz
        self._n += n

    def __len__(self) -> int:
        return self._n

    def __bool__(self) -> bool:
        return self._n > 0

    def finalize(self) -> None:
        """Trim backing arrays to true size."""
        n = self._n
        for f in (
            "height",
            "point_xyz",
            "alt_xyz",
            "atom_idx_0",
            "atom_idx_1",
            "atom_idx_2",
        ):
            setattr(self, f, getattr(self, f)[:n])


# ── NeighborArray ─────────────────────────────────────────────────────────────


class NeighborArray:
    """Pre-computed neighbor information for each atom.

    Stores up to *K* neighbors per atom with their coordinates,
    radii, and original atom indices. Padding entries have ``natom == -1``
    and ``NaN`` coordinates/radii.

    Attributes:
        xyz: Neighbor coordinates, shape ``(N, K, 3)``.
        radius: Neighbor radii, shape ``(N, K)``.
        natom: Neighbor atom indices, shape ``(N, K)``.
            ``-1`` marks padding.
        nneighbors: Count of real neighbors per atom, shape ``(N,)``.
    """

    def __init__(self, n_atoms: int, max_neighbors: int) -> None:
        self.xyz = np.full((n_atoms, max_neighbors, 3), np.nan, dtype=np.float64)
        self.radius = np.full((n_atoms, max_neighbors), np.nan, dtype=np.float64)
        self.natom = np.full((n_atoms, max_neighbors), -1, dtype=np.int32)
        self.nneighbors = np.zeros(n_atoms, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.nneighbors)

    def __bool__(self) -> bool:
        return len(self.nneighbors) > 0
