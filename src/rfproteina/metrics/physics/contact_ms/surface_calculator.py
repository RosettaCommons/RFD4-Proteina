"""Molecular surface calculator for Contact Molecular Surface (CMS).

Implements the Connolly surface algorithm (convex + toroidal + concave dots)
and scores interfacial contact as ``CMS = Σ area x exp(-weight x dist^2)``.

This is the NumPy/SciPy reference implementation.  For production use,
prefer ``TorchCMS`` from ``gpu_cms.py`` which is faster on both GPU and CPU.

See ``README.md`` in this package for algorithm details.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from .data_structures import (
    ATTEN_6,
    ATTEN_BLOCKER,
    ATTEN_BURIED,
    DotArray,
    NeighborArray,
    ProbeArray,
    SurfaceAtomArray,
    SurfaceStats,
)
from .geometry import (
    check_point_collisions,
    check_probe_collisions,
    distance_points_to_line,
    subdivide_arc,
    subdivide_circle,
)


@dataclass
class SurfaceSettings:
    """Parameters controlling surface generation and CMS scoring."""

    probe_radius: float = 1.7
    density: float = 15.0
    separation: float = 8.0
    weight: float = 0.5


class _ToroidQueue:
    """Parallel-array accumulator for atom pairs that need toroidal surface dots."""

    __slots__ = ("between", "na1", "na2", "rij", "tij", "uij")

    def __init__(self) -> None:
        self.na1: list[int] = []
        self.na2: list[int] = []
        self.uij: list[np.ndarray] = []
        self.tij: list[np.ndarray] = []
        self.rij: list[float] = []
        self.between: list[bool] = []

    def extend(
        self,
        na1: list[int],
        na2: list[int],
        uij: np.ndarray,
        tij: np.ndarray,
        rij: list[float],
        between: list[bool],
    ) -> None:
        self.na1.extend(na1)
        self.na2.extend(na2)
        self.uij.append(uij)
        self.tij.append(tij)
        self.rij.extend(rij)
        self.between.extend(between)

    def __bool__(self) -> bool:
        return len(self.na1) > 0

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.na1, dtype=np.int32),
            np.array(self.na2, dtype=np.int32),
            np.concatenate(self.uij) if self.uij else np.empty((0, 3)),
            np.concatenate(self.tij) if self.tij else np.empty((0, 3)),
            np.array(self.rij, dtype=np.float64),
            np.array(self.between, dtype=bool),
        )


class _RunState:
    """Mutable state for a single CMS computation."""

    __slots__ = (
        "atoms",
        "buried_array",
        "dots",
        "good_atoms",
        "neighbor_array",
        "probes",
        "radmax",
        "stats",
        "toroid_queue",
        "trimmed_dots",
    )

    def __init__(self) -> None:
        self.radmax: float = 0.0
        self.atoms = SurfaceAtomArray()
        self.dots: list[DotArray] = [DotArray(), DotArray()]
        self.trimmed_dots: list[DotArray] = [DotArray(), DotArray()]
        self.probes = ProbeArray()
        self.neighbor_array: NeighborArray | None = None
        self.buried_array: NeighborArray | None = None
        self.toroid_queue = _ToroidQueue()
        self.stats = SurfaceStats()
        self.good_atoms: np.ndarray | None = None


class MolecularSurfaceCalculator:
    """Contact Molecular Surface calculator (NumPy/SciPy).

    Reference implementation that mirrors the pipeline structure of
    ``TorchCMS`` in ``gpu_cms.py``.

    Example::

        calc = MolecularSurfaceCalculator()
        calc.add_binder_and_target(binder_xyz, binder_radii, target_xyz, target_radii)
        cms, per_atom_cms = calc.compute()

        # Both sides at once:
        (t_cms, t_pa), (b_cms, b_pa) = calc.compute_both_sides()

        # Binder-side only (backward compat):
        binder_cms, _ = calc.compute_contact_ms(target_side=False)
    """

    def __init__(self, settings: SurfaceSettings | None = None) -> None:
        self.settings = settings or SurfaceSettings()
        self.run = _RunState()

    # ── Public API ─────────────────────────────────────────────────────────

    def add_binder_and_target(
        self,
        binder_xyz: Float[np.ndarray, "Nb 3"],
        binder_radii: Float[np.ndarray, Nb],
        target_xyz: Float[np.ndarray, "Nt 3"],
        target_radii: Float[np.ndarray, Nt],
    ) -> None:
        """Load binder (molecule 0) and target (molecule 1) atoms."""
        self.run = _RunState()
        self.add_molecule(0, binder_xyz, binder_radii)
        self.add_molecule(1, target_xyz, target_radii)

    def add_molecule(
        self,
        molecule: int,
        xyz: Float[np.ndarray, "N 3"],
        radii: Float[np.ndarray, N],
    ) -> None:
        """Load atoms for one molecule. Atoms with ``radii <= 0`` are skipped."""
        mol_val = 1 if molecule == 1 else 0
        mask = np.asarray(radii) > 0
        xyz_f, radii_f = np.asarray(xyz)[mask], np.asarray(radii)[mask]
        self.run.atoms.extend_from_arrays(xyz_f, radii_f, self.settings.density, mol_val)
        self.run.stats.n_atoms += len(radii_f)
        self.run.stats.init_molecule(mol_val, len(radii_f))

    def compute(self) -> tuple[float, Float[np.ndarray, Nt]]:
        """Run the full CMS pipeline. Returns ``(total_cms, per_atom_target_cms)``."""
        assert len(self.run.atoms) > 0
        self._run_pipeline()
        return self._score_cms(target_side=True)

    def compute_both_sides(
        self,
    ) -> tuple[
        tuple[float, Float[np.ndarray, Nt]],
        tuple[float, Float[np.ndarray, Nb]],
    ]:
        """Run pipeline once, score both sides.

        Returns:
            ``((target_cms, target_per_atom), (binder_cms, binder_per_atom))``
        """
        assert len(self.run.atoms) > 0
        self._run_pipeline()
        target = self._score_cms(target_side=True)
        binder = self._score_cms(target_side=False)
        return target, binder

    def compute_max_cms(self) -> tuple[float, Float[np.ndarray, N]]:
        """Compute maximum possible CMS (total surface area) for molecule 0."""
        assert len(self.run.atoms) > 0
        self._run_pipeline(all_atoms=True)
        return self._compute_max_possible_cms()

    def compute_contact_ms(
        self,
        target_side: bool = True,
    ) -> tuple[float, Float[np.ndarray, Nq]]:
        """Score CMS from pre-computed surface dots.

        Call after ``compute()`` to score a different side without
        regenerating the surface.
        """
        return self._score_cms(target_side=target_side)

    # ── Pipeline orchestrator ──────────────────────────────────────────────

    def _run_pipeline(self, all_atoms: bool = False) -> None:
        """Execute stages 1-6, populating surface dots.

        Stage 7 (``_score_cms``) is called separately so that both sides
        can be scored from the same set of dots.
        """
        self._setup_atoms(all_atoms=all_atoms)
        self._build_neighbors()
        self._process_atom_pairs()
        self._generate_convex_dots()
        self._generate_toroidal_dots()
        self.run.probes.finalize()
        if self.settings.probe_radius > 0:
            self._generate_concave_dots()
        self.run.dots[0].finalize()
        self.run.dots[1].finalize()

    # ── Stage 1: Setup ─────────────────────────────────────────────────────

    def _setup_atoms(self, all_atoms: bool = False) -> None:
        """Finalize atom arrays, compute radmax, and assign attention flags.

        Atoms far from the other molecule become blockers — they contribute
        to the Connolly surface but not to the CMS score.
        """
        atoms = self.run.atoms
        atoms.finalize()
        self.run.radmax = atoms.radius.max()

        mol = atoms.molecule

        if all_atoms:
            atoms.atten[:] = ATTEN_BURIED
            for m in range(2):
                self.run.stats.per_molecule[m]["nBuriedAtoms"] = int((mol == m).sum())
            return

        mask0, mask1 = mol == 0, mol == 1
        if mask0.any() and mask1.any():
            d = cdist(atoms.xyz[mask0], atoms.xyz[mask1])
            min0, min1 = d.min(axis=1), d.min(axis=0)
        else:
            min0, min1 = np.full(mask0.sum(), 1e5), np.full(mask1.sum(), 1e5)

        sep = self.settings.separation
        for mask_mol, min_d, mol_id in [(mask0, min0, 0), (mask1, min1, 1)]:
            idx = np.where(mask_mol)[0]
            far = min_d >= sep
            atoms.atten[idx[far]] = ATTEN_BLOCKER
            atoms.atten[idx[~far]] = ATTEN_BURIED
            self.run.stats.per_molecule[mol_id]["nBlockedAtoms"] = int(far.sum())
            self.run.stats.per_molecule[mol_id]["nBuriedAtoms"] = int((~far).sum())

    # ── Stage 2: Neighbour lists ───────────────────────────────────────────

    def _build_neighbors(self) -> None:
        """Build sorted neighbour and burial arrays for all atoms.

        Stores ``self.run.good_atoms`` — boolean mask of active atoms with
        at least one same-molecule neighbour.
        """
        atoms = self.run.atoms
        n = len(atoms)
        rp = self.settings.probe_radius
        xyz, radius, mol, atten, natom = (
            atoms.xyz,
            atoms.radius,
            atoms.molecule,
            atoms.atten,
            atoms.natom,
        )

        d2 = cdist(xyz, xyz, metric="sqeuclidean")  # (N, N)
        bridge2 = (radius[:, None] + radius[None, :] + 2.0 * rp) ** 2  # (N, N)

        active = atten > 0  # (N,)
        eye_off = ~np.eye(n, dtype=bool)  # (N, N)
        same_mol = mol[:, None] == mol[None, :]  # (N, N)
        diff_mol = ~same_mol  # (N, N)
        buried_eligible = atten >= ATTEN_BURIED  # (N,)

        coincident = same_mol & eye_off & active[:, None] & active[None, :] & (d2 <= 1e-4)
        if coincident.any():
            i0, j0 = map(int, np.argwhere(coincident)[0])
            raise RuntimeError(f"Coincident atoms {natom[i0]} and {natom[j0]}")

        neighbor_mask = active[:, None] & active[None, :] & same_mol & eye_off & (d2 < bridge2)  # (N, N)
        buried_mask = active[:, None] & buried_eligible[None, :] & diff_mol & (d2 < bridge2)  # (N, N)

        bb2 = (4.0 * self.run.radmax + 4.0 * rp) ** 2
        nbb = (active[:, None] & buried_eligible[None, :] & diff_mol & (d2 < bb2)).sum(axis=1)  # (N,)

        n_neigh = neighbor_mask.sum(axis=1)  # (N,)
        atten6_no_nbb = (atten == ATTEN_6) & (nbb == 0)  # (N,)
        atoms.access[:n] |= (active & ~atten6_no_nbb & (n_neigh == 0)).astype(np.int8)
        self.run.good_atoms = active & ~atten6_no_nbb & (n_neigh > 0)  # (N,)

        max_n = max(int(n_neigh.max()), 1) if n_neigh.any() else 1
        na = NeighborArray(n, max_n)
        n_bur = buried_mask.sum(axis=1)  # (N,)
        max_b = max(int(n_bur.max()), 1) if n_bur.any() else 1
        ba = NeighborArray(n, max_b)

        sorted_n = np.argsort(np.where(neighbor_mask, d2, np.inf), axis=1)[:, :max_n]
        sorted_b = np.argsort(np.where(buried_mask, d2, np.inf), axis=1)[:, :max_b]

        for i in np.where(active & (n_neigh > 0))[0]:
            k = int(n_neigh[i])
            idx = sorted_n[i, :k]
            na.xyz[i, :k] = xyz[idx]
            na.radius[i, :k] = radius[idx]
            na.natom[i, :k] = natom[idx]
            na.nneighbors[i] = k

        for i in np.where(active & (n_bur > 0))[0]:
            k = int(n_bur[i])
            idx = sorted_b[i, :k]
            ba.xyz[i, :k] = xyz[idx]
            ba.radius[i, :k] = radius[idx]
            ba.natom[i, :k] = natom[idx]
            ba.nneighbors[i] = k

        self.run.neighbor_array, self.run.buried_array = na, ba

    # ── Stage 3: Atom-pair processing ──────────────────────────────────────

    def _process_atom_pairs(self) -> None:
        """Process forward atom pairs: toroid queue + probe triplets.

        For each pair ``(atom_i, atom_j)`` with ``j > i`` whose extended
        spheres overlap, queue toroid pairs and find probe triplet positions.

        Mutates: ``access``, ``toroid_queue``.
        """
        ai = np.where(self.run.good_atoms)[0]  # (G,)
        if len(ai) == 0:
            return

        rp = self.settings.probe_radius
        na = self.run.neighbor_array
        atoms = self.run.atoms

        nneigh = na.nneighbors[ai]  # (G,)
        n_xyz, n_rad, n_nat = na.xyz[ai], na.radius[ai], na.natom[ai]  # (G,K,3), (G,K), (G,K)
        a_xyz, a_rad, a_att = atoms.xyz[ai], atoms.radius[ai], atoms.atten[ai]  # (G,3), (G,), (G,)

        fwd = n_nat > ai[:, None]  # (G, K)
        diff = n_xyz - a_xyz[:, None, :]  # (G, K, 3)
        dij_sq = np.einsum("nki,nki->nk", diff, diff)  # (G, K)
        dij = np.sqrt(np.where(fwd, dij_sq, 1.0))  # (G, K)

        eri, erj = a_rad + rp, n_rad + rp  # (G,), (G,K)

        asymm = np.where(fwd, (eri[:, None] ** 2 - erj**2) / dij, 0.0)  # (G, K)
        between = np.abs(asymm) < dij  # (G, K)
        tij = (a_xyz[:, None, :] + n_xyz) * 0.5 + (diff / dij[..., None]) * (asymm * 0.5)[..., None]  # (G, K, 3)

        far_sq = np.where(fwd, (eri[:, None] + erj) ** 2 - dij_sq, -1.0)  # (G, K)
        cont_sq = np.where(
            fwd,
            dij_sq - (a_rad[:, None] - n_rad) ** 2,
            -1.0,
        )  # (G, K)
        geom_ok = fwd & (far_sq > 0.0) & (cont_sq > 0.0)  # (G, K)

        shortcut = (nneigh <= 1) & geom_ok.any(axis=1)  # (G,)
        if shortcut.any():
            atoms.access[ai[shortcut]] = 1
            first_k = np.argmax(geom_ok[shortcut], axis=1)
            atoms.access[n_nat[shortcut][np.arange(shortcut.sum()), first_k].astype(np.int32)] = 1

        pair_mask = geom_ok & ~shortcut[:, None]  # (G, K)
        i_idx, k_idx = np.where(pair_mask)
        if len(i_idx) == 0:
            return

        na1 = ai[i_idx]  # (P,)
        na2 = n_nat[i_idx, k_idx].astype(np.int32)  # (P,)
        dij_f = dij[i_idx, k_idx]  # (P,)
        uij_f = diff[i_idx, k_idx] / dij_f[:, None]  # (P, 3)
        tij_f = tij[i_idx, k_idx]  # (P, 3)
        rij_f = 0.5 * np.sqrt(far_sq[i_idx, k_idx]) * np.sqrt(cont_sq[i_idx, k_idx]) / dij_f  # (P,)
        bet_f = between[i_idx, k_idx]  # (P,)

        a1_att = a_att[i_idx]  # (P,)
        a2_att = atoms.atten[na2]  # (P,)
        need_tor = (a1_att > ATTEN_BLOCKER) | ((a2_att > ATTEN_BLOCKER) & (rp > 0.0))
        ti = np.where(need_tor)[0]
        if len(ti) > 0:
            self.run.toroid_queue.extend(
                na1[ti].tolist(),
                na2[ti].tolist(),
                uij_f[ti],
                tij_f[ti],
                rij_f[ti].tolist(),
                bet_f[ti].tolist(),
            )

        access_idx = self._find_probes(na1, na2, uij_f, tij_f, rij_f)
        if len(access_idx):
            atoms.access[access_idx] = 1

    # ── Stage 3b: Probe finding ────────────────────────────────────────────

    def _find_probes(
        self,
        na1: Int[np.ndarray, P],
        na2: Int[np.ndarray, P],
        uij: Float[np.ndarray, "P 3"],
        tij: Float[np.ndarray, "P 3"],
        rij: Float[np.ndarray, P],
    ) -> Int[np.ndarray, R]:
        """Find probe-sphere positions at atom triplets.

        For each ``(atom1, atom2)`` pair, search atom1's neighbour list for
        ``atom3 > atom2`` forming a valid triple.  Returns atom indices that
        gained accessibility.
        """
        m = len(na1)
        rp = self.settings.probe_radius
        atoms = self.run.atoms
        na = self.run.neighbor_array

        eri = atoms.radius[na1] + rp  # (P,)
        erj = atoms.radius[na2] + rp  # (P,)
        n_xyz, n_rad, n_nat = na.xyz[na1], na.radius[na1], na.natom[na1]  # (P,K,3),(P,K),(P,K)

        valid = (n_nat > na2[:, None]) & (n_nat >= 0)  # (P, K)
        erk = np.where(valid, n_rad + rp, np.inf)  # (P, K)

        a1_xyz, a2_xyz = atoms.xyz[na1], atoms.xyz[na2]  # (P,3),(P,3)

        djk = np.linalg.norm(n_xyz - a2_xyz[:, None, :], axis=-1)  # (P, K)
        valid &= djk < (erj[:, None] + erk)
        dik = np.linalg.norm(n_xyz - a1_xyz[:, None, :], axis=-1)  # (P, K)
        valid &= dik < (eri[:, None] + erk)

        safe_na3 = np.where(n_nat >= 0, n_nat, 0)  # (P, K)
        all_blocked = (
            (atoms.atten[na1][:, None] <= ATTEN_BLOCKER)
            & (atoms.atten[na2][:, None] <= ATTEN_BLOCKER)
            & (atoms.atten[safe_na3] <= ATTEN_BLOCKER)
        )  # (P, K)
        valid &= ~all_blocked

        if not valid.any():
            return np.empty(0, dtype=np.int32)

        pi, ki = np.where(valid)

        a1q, a3q = a1_xyz[pi], n_xyz[pi, ki]  # (Q,3),(Q,3)
        na1q, na2q, na3q = na1[pi], na2[pi], n_nat[pi, ki]  # (Q,) each
        eriq, erkq = eri[pi], n_rad[pi, ki] + rp  # (Q,) each
        dikq = dik[pi, ki]  # (Q,)
        uijq, tijq, rijq = uij[pi], tij[pi], rij[pi]  # (Q,3),(Q,3),(Q,)

        uikq = (a3q - a1q) / dikq[:, None]  # (Q, 3)
        dt = np.einsum("qi,qi->q", uijq, uikq)  # (Q,)
        wijk = np.arccos(np.clip(dt, -1.0, 1.0))  # (Q,)
        swijk = np.sin(wijk)  # (Q,)

        degen = (dt >= 1.0) | (dt <= -1.0) | (wijk <= 0.0) | (swijk <= 0.0)
        keep = ~degen  # (Q,)

        if degen.any():
            dtijk_sq = np.sum((tijq[degen] - a3q[degen]) ** 2, axis=-1)
            rkp_sq = erkq[degen] ** 2 - rijq[degen] ** 2
            kills = np.zeros(len(pi), dtype=bool)
            kills[degen] = dtijk_sq < rkp_sq
            if kills.any():
                kill_pair = np.zeros(m, dtype=bool)
                kill_pair[pi[kills]] = True
                keep &= ~kill_pair[pi]

        if not keep.any():
            return np.empty(0, dtype=np.int32)

        a1q, a3q = a1q[keep], a3q[keep]
        na1q, na2q, na3q = na1q[keep], na2q[keep], na3q[keep]
        eriq, erkq, dikq = eriq[keep], erkq[keep], dikq[keep]
        uijq, tijq, rijq = uijq[keep], tijq[keep], rijq[keep]
        uikq, swijk = uikq[keep], swijk[keep]
        pi = pi[keep]
        q = len(pi)

        uijk = np.cross(uijq, uikq) / swijk[:, None]  # (Q', 3)
        utb = np.cross(uijk, uijq)  # (Q', 3)
        asymm = (eriq**2 - erkq**2) / dikq  # (Q',)
        tik = (a1q + a3q) * 0.5 + uikq * (asymm * 0.5)[:, None]  # (Q', 3)
        dt_b = np.einsum("qi,qi->q", uikq, tik - tijq)  # (Q',)
        bijk = tijq + utb * (dt_b / swijk)[:, None]  # (Q', 3)

        h2 = eriq**2 - np.einsum("qi,qi->q", bijk - a1q, bijk - a1q)  # (Q',)
        vh = h2 > 0.0  # (Q',)
        if not vh.any():
            return np.empty(0, dtype=np.int32)

        bijk, uijk = bijk[vh], uijk[vh]
        h = np.sqrt(h2[vh])  # (R,)
        na1q, na2q, na3q = na1q[vh], na2q[vh], na3q[vh]
        q = int(vh.sum())

        bijk2 = np.concatenate([bijk, bijk])  # (2R, 3)
        uijk2 = np.concatenate([uijk, uijk])  # (2R, 3)
        h2x = np.concatenate([h, h])  # (2R,)
        na1x = np.concatenate([na1q, na1q])  # (2R,)
        na2x = np.concatenate([na2q, na2q])  # (2R,)
        na3x = np.concatenate([na3q, na3q])  # (2R,)
        sign = np.concatenate([np.ones(q), -np.ones(q)])  # (2R,)

        probe_xyz = bijk2 + uijk2 * (h2x * sign)[:, None]  # (2R, 3)

        c_xyz = na.xyz[na1x]  # (2R, K, 3)
        c_rad = na.radius[na1x]  # (2R, K)
        c_nat = na.natom[na1x]  # (2R, K)

        exclude = (c_nat == na2x[:, None]) | (c_nat == na3x[:, None]) | (c_nat < 0)
        rows, cols = np.where(~exclude)
        diff_c = probe_xyz[rows] - c_xyz[rows, cols]  # (H, 3)
        d2c = np.einsum("ij,ij->i", diff_c, diff_c)  # (H,)
        hit = d2c <= (c_rad[rows, cols] + rp) ** 2

        collision = np.zeros(len(probe_xyz), dtype=bool)  # (2R,)
        collision[rows[hit]] = True

        ok = ~collision  # (2R,)
        if not ok.any():
            return np.empty(0, dtype=np.int32)

        a0 = np.where(sign > 0, na1x, na2x)[ok].astype(np.int32)
        a1 = np.where(sign > 0, na2x, na1x)[ok].astype(np.int32)
        a2 = na3x[ok].astype(np.int32)
        self.run.probes.extend_from_arrays(
            a0,
            a1,
            a2,
            h2x[ok],
            probe_xyz[ok],
            (uijk2 * sign[:, None])[ok],
        )
        return np.unique(np.concatenate([a0, a1, a2]))

    # ── Stage 4: Convex dots ───────────────────────────────────────────────

    def _generate_convex_dots(self) -> None:
        """Generate convex surface dots for accessible atoms.

        Selects atoms that are accessible, have buried-eligible attention, and
        have at least one burial neighbour.  Subdivides each atom's exposed
        sphere via latitude arcs + longitude circles.
        """
        good_idx = np.where(self.run.good_atoms)[0]  # (G,)
        if len(good_idx) == 0:
            return

        atoms = self.run.atoms
        atten = atoms.atten[good_idx]  # (G,)
        access = atoms.access[good_idx].astype(bool)  # (G,)
        n_bur = self.run.buried_array.nneighbors[good_idx]  # (G,)
        ai = good_idx[access & (atten > ATTEN_BLOCKER) & ~((atten == ATTEN_6) & (n_bur == 0))]  # (C,)
        n = len(ai)
        if n == 0:
            return

        na = self.run.neighbor_array
        rp = self.settings.probe_radius

        a_xyz, ri = atoms.xyz[ai], atoms.radius[ai]  # (C,3), (C,)
        eri = ri + rp  # (C,)
        a_den, a_mol = atoms.density[ai], atoms.molecule[ai]  # (C,), (C,)

        north = np.tile([0.0, 0.0, 1.0], (n, 1))  # (C, 3)
        south = np.tile([0.0, 0.0, -1.0], (n, 1))  # (C, 3)
        eqvec = np.tile([1.0, 0.0, 0.0], (n, 1))  # (C, 3)

        has_neigh = na.nneighbors[ai] > 0  # (C,)
        if has_neigh.any():
            n_xyz = na.xyz[ai][:, 0]  # (C, 3)
            n_rad = na.radius[ai][:, 0]  # (C,)

            nv = a_xyz - n_xyz
            nv /= np.linalg.norm(nv, axis=-1, keepdims=True)
            north[has_neigh] = nv[has_neigh]

            vt = np.stack(
                [
                    north[:, 1] ** 2 + north[:, 2] ** 2,
                    north[:, 0] ** 2 + north[:, 2] ** 2,
                    north[:, 0] ** 2 + north[:, 1] ** 2,
                ],
                axis=-1,
            )
            vt /= np.linalg.norm(vt, axis=-1, keepdims=True)
            vt[np.abs((vt * north).sum(-1)) > 0.99] = [1.0, 0.0, 0.0]
            eq = np.cross(north, vt)
            eq /= np.linalg.norm(eq, axis=-1, keepdims=True)
            eqvec[has_neigh] = eq[has_neigh]

            vql = np.cross(eqvec, north)
            erj = n_rad + rp
            dij = np.linalg.norm(n_xyz - a_xyz, axis=-1)
            u_ij = (n_xyz - a_xyz) / dij[:, None]
            asymm = (eri * eri - erj * erj) / dij
            t_ij = (a_xyz + n_xyz) * 0.5 + u_ij * (asymm[:, None] * 0.5)
            r_ij = 0.5 * np.sqrt((eri + erj) ** 2 - dij**2) * np.sqrt(dij**2 - (ri - n_rad) ** 2) / dij
            south[has_neigh] = (((t_ij + vql * r_ij[:, None]) - a_xyz) / eri[:, None])[has_neigh]

        cs, lats = subdivide_arc(
            np.zeros((n, 3)),
            ri,
            eqvec,
            a_den,
            north,
            south,
        )  # (C,), (C, M, 3)

        dt = (lats * north[:, None, :]).sum(-1)  # (C, M)
        cen = a_xyz[:, None, :] + dt[..., None] * north[:, None, :]  # (C, M, 3)
        rad_sq = ri[:, None] ** 2 - dt**2  # (C, M)
        rad = np.sqrt(np.maximum(rad_sq, 0.0))  # (C, M)
        rad[rad_sq <= 0] = 0.0

        M = lats.shape[1]
        flat_rad = rad.ravel()  # (C*M,)
        active = flat_rad > 0  # (C*M,)
        if not active.any():
            return

        ps_a, pts_a = subdivide_circle(
            cen.reshape(-1, 3)[active],
            flat_rad[active],
            np.repeat(north, M, axis=0)[active],
            np.repeat(a_den, M)[active],
        )
        K = pts_a.shape[1]
        full_pts = np.full((n * M, K, 3), np.nan)
        full_pts[active] = pts_a
        full_ps = np.zeros(n * M)
        full_ps[active] = ps_a

        points = full_pts.reshape(n, M, K, 3)  # (C, M, K, 3)
        ps = full_ps.reshape(n, M)  # (C, M)
        valid = ~np.isnan(points[..., 0])  # (C, M, K)
        area = ps * cs[:, None]  # (C, M)

        pts_flat = points[valid]  # (D, 3)
        atom_i, lat_i, _ = np.where(valid)

        pcen = a_xyz[atom_i] + (pts_flat - a_xyz[atom_i]) * (eri[atom_i] / ri[atom_i])[:, None]  # (D, 3)

        coll = check_point_collisions(
            pcen[..., None, :],
            na.xyz[ai][atom_i],
            na.radius[ai][atom_i],
            rp,
        )  # (D,)

        keep = ~coll  # (D,)
        if not keep.any():
            return

        self.run.stats.convex_dots += int(keep.sum())
        self._write_dots(
            a_mol[atom_i[keep]],
            1,
            pts_flat[keep],
            area[atom_i[keep], lat_i[keep]],
            pcen[keep],
            ai[atom_i[keep]],
        )

    # ── Stage 5: Toroidal dots ─────────────────────────────────────────────

    def _generate_toroidal_dots(self) -> None:
        """Generate toroidal (saddle) surface dots between atom pairs.

        Reads queued pairs from ``self.run.toroid_queue``, subdivides the
        torus circle, and generates dots on both halves of the saddle.
        Updates atom accessibility directly.
        """
        if not self.run.toroid_queue:
            return

        na1, na2, uij, tij, rij, between = self.run.toroid_queue.as_arrays()
        n = len(na1)
        rp = self.settings.probe_radius
        atoms = self.run.atoms
        na = self.run.neighbor_array

        n_xyz = na.xyz[na1]  # (T, K, 3)
        n_rad = na.radius[na1]  # (T, K)
        n_nat = na.natom[na1]  # (T, K)
        a1_xyz, a2_xyz = atoms.xyz[na1], atoms.xyz[na2]  # (T, 3) each
        a1_r, a2_r = atoms.radius[na1], atoms.radius[na2]  # (T,) each
        a1_den, a2_den = atoms.density[na1], atoms.density[na2]  # (T,) each
        a1_att, a2_att = atoms.atten[na1], atoms.atten[na2]  # (T,) each
        a1_mol = atoms.molecule[na1]  # (T,)

        density = (a1_den + a2_den) * 0.5  # (T,)
        eri, erj = a1_r + rp, a2_r + rp  # (T,) each

        rci = rij * a1_r / eri  # (T,)
        rcj = rij * a2_r / erj  # (T,)
        rs = (rci + 2 * np.maximum(rij - rp, 0.0) + rcj) * 0.25  # (T,)
        edens = (rs / rij) ** 2 * density  # (T,)

        ts, subs = subdivide_circle(tij, rij, uij, edens)  # (T,), (T, S, 3)
        valid_sub = ~np.isnan(subs[..., 0])  # (T, S)
        if not valid_sub.any():
            return

        ai, si = np.where(valid_sub)
        pij = subs[ai, si]  # (V, 3)

        nxf, nrf, nnf = n_xyz[ai], n_rad[ai], n_nat[ai]  # (V,K,3),(V,K),(V,K)
        exclude = (nnf < 0) | (nnf == na2[ai, None])  # (V, K)
        rows, cols = np.where(~exclude)
        diff_a = pij[rows] - nxf[rows, cols]  # (H, 3)
        d2a = np.einsum("ij,ij->i", diff_a, diff_a)  # (H,)
        too_close = np.zeros(len(pij), dtype=bool)  # (V,)
        too_close[rows[d2a < (nrf[rows, cols] + rp) ** 2]] = True

        ok = ~too_close & ~(
            (a1_att[ai] == ATTEN_6) & (a2_att[ai] == ATTEN_6) & (self.run.buried_array.nneighbors[na1[ai]] == 0)
        )  # (V,)
        if not ok.any():
            return

        pij, ai = pij[ok], ai[ok]
        a1_access = np.zeros(n, dtype=bool)  # (T,)
        a2_access = np.zeros(n, dtype=bool)  # (T,)
        a1_access[ai] = True
        a2_access[ai] = True

        pi = (a1_xyz[ai] - pij) / eri[ai, None]  # (V', 3)
        pj = (a2_xyz[ai] - pij) / erj[ai, None]  # (V', 3)
        axis = np.cross(pi, pj)  # (V', 3)
        axis /= np.linalg.norm(axis, axis=-1, keepdims=True)

        dtq = rp**2 - rij[ai] ** 2  # (V',)
        pcusp = (dtq > 0) & between[ai]  # (V',)

        pqi, pqj = np.empty_like(pi), np.empty_like(pi)  # (V',3) each
        if pcusp.any():
            ds = np.sqrt(dtq[pcusp])
            pqi[pcusp] = (tij[ai[pcusp]] - uij[ai[pcusp]] * ds[:, None] - pij[pcusp]) / rp
            pqj[pcusp] = 0.0
        if (~pcusp).any():
            tmp = pi[~pcusp] + pj[~pcusp]
            tmp /= np.linalg.norm(tmp, axis=-1, keepdims=True)
            pqi[~pcusp] = tmp
            pqj[~pcusp] = tmp

        dt1, dt2 = (pqi * pi).sum(-1), (pqj * pj).sum(-1)  # (V',) each
        va = (np.abs(dt1) < 1.0) & (np.abs(dt2) < 1.0)  # (V',)
        if not va.any():
            atoms.access[na1] |= a1_access.astype(np.int8)
            atoms.access[na2] |= a2_access.astype(np.int8)
            return

        pij = pij[va]
        axis, pi, pj, pqi, pqj = axis[va], pi[va], pj[va], pqi[va], pqj[va]
        ai = ai[va]

        for start, end, nat_src in [(pi, pqi, na1), (pqj, pj, na2)]:
            mask = atoms.atten[nat_src[ai]] > ATTEN_BLOCKER  # (V'',)
            if not mask.any():
                continue

            ps, pts = subdivide_arc(
                pij[mask],
                np.full(int(mask.sum()), rp),
                axis[mask],
                density[ai[mask]],
                start[mask],
                end[mask],
            )
            dist = distance_points_to_line(
                tij[ai[mask]][..., None, :],
                uij[ai[mask]][..., None, :],
                pts,
            )
            exp_pij = np.broadcast_to(pij[mask][:, None, :], pts.shape).copy()
            areas = ps[:, None] * ts[ai[mask], None] * dist / rij[ai[mask], None]
            v = ~np.isnan(areas)  # (..., S)

            k = int(v.sum())
            self.run.stats.toroidal_dots += k
            if k > 0:
                mol = np.broadcast_to(a1_mol[ai][mask][:, None], v.shape)[v]
                nat = np.broadcast_to(nat_src[ai][mask][:, None], v.shape)[v]
                self._write_dots(mol, 2, pts[v], areas[v], exp_pij[v], nat)

        atoms.access[na1] |= a1_access.astype(np.int8)
        atoms.access[na2] |= a2_access.astype(np.int8)

    # ── Stage 6: Concave dots ──────────────────────────────────────────────

    def _generate_concave_dots(self) -> None:
        """Generate concave surface dots inside probe spheres.

        Each probe sits at an atom triplet.  The concave (inward-facing)
        part of the probe sphere is subdivided via latitude arcs + longitude
        circles.  Dots are rejection-tested against half-planes and
        collision-checked against nearby low probes.
        """
        probes = self.run.probes
        if not probes:
            return

        rp = self.settings.probe_radius
        rp2 = rp * rp
        P = len(probes)
        atoms = self.run.atoms

        pijk, uijk, hijk = probes.point_xyz, probes.alt_xyz, probes.height
        idx3 = [probes.atom_idx_0, probes.atom_idx_1, probes.atom_idx_2]

        a_xyz = np.stack([atoms.xyz[i] for i in idx3], axis=1)  # (P, 3, 3)
        a_rad = np.stack([atoms.radius[i] for i in idx3], axis=-1)  # (P, 3)
        a_den = np.stack([atoms.density[i] for i in idx3], axis=-1)  # (P, 3)
        a_att = np.stack([atoms.atten[i] for i in idx3], axis=-1)  # (P, 3)
        a_mol = np.stack([atoms.molecule[i] for i in idx3], axis=-1)  # (P, 3)

        density = a_den.mean(axis=1)  # (P,)

        if (a_att == ATTEN_6).all(axis=1).all():
            return

        low = hijk < rp  # (P,)
        low_pts = pijk[low]  # (L, 3)
        nears_mask = None
        if low.any():
            d2 = ((pijk[:, None, :] - low_pts[None, :, :]) ** 2).sum(-1)  # (P, L)
            nears_mask = d2 <= 4 * rp2  # (P, L)
            nears_mask[np.where(low)[0], np.arange(low.sum())] = False

        vp = a_xyz - pijk[:, None, :]  # (P, 3, 3)
        vp /= np.linalg.norm(vp, axis=-1, keepdims=True)

        vectors = np.stack(
            [
                np.cross(vp[:, 0], vp[:, 1]),
                np.cross(vp[:, 1], vp[:, 2]),
                np.cross(vp[:, 2], vp[:, 0]),
            ],
            axis=1,
        )  # (P, 3, 3)
        vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)

        mm = np.argmax((uijk[:, None, :] * vp).sum(-1), axis=1)  # (P,)
        s = -uijk  # (P, 3)
        ax = np.cross(vp[np.arange(P), mm], s)  # (P, 3)
        ax /= np.linalg.norm(ax, axis=-1, keepdims=True)

        cs, lats = subdivide_arc(
            np.zeros_like(pijk),
            np.full(P, rp),
            ax,
            density,
            vp[np.arange(P), mm],
            s,
        )  # (P,), (P, M, 3)
        dt_lat = (lats * s[:, None, :]).sum(-1)  # (P, M)
        cen = s[:, None, :] * dt_lat[..., None]  # (P, M, 3)
        rad = np.sqrt(np.clip(rp2 - dt_lat**2, 0, None))  # (P, M)

        M = lats.shape[1]
        flat_rad = rad.ravel()  # (P*M,)
        active = flat_rad > 0  # (P*M,)

        if active.any():
            ps_a, pts_a = subdivide_circle(
                cen.reshape(-1, 3)[active],
                flat_rad[active],
                np.repeat(s, M, axis=0)[active],
                np.repeat(density, M)[active],
            )
            K = pts_a.shape[1]
            full_pts = np.full((P * M, K, 3), np.nan)
            full_pts[active] = pts_a
            full_ps = np.zeros(P * M)
            full_ps[active] = ps_a
        else:
            K = 0
            full_pts = np.empty((P * M, 0, 3))
            full_ps = np.zeros(P * M)

        points = full_pts.reshape(P, M, K, 3)  # (P, M, K, 3)
        area = full_ps.reshape(P, M) * cs[:, None]  # (P, M)

        ip, il, _ = np.where(~np.isnan(points[..., 0]))
        pts = points[ip, il, _]  # (D, 3)

        bail = ((pts[:, None, :] * vectors[ip]).sum(-1) >= 0).any(axis=1)  # (D,)
        pts, ip, il = pts[~bail], ip[~bail], il[~bail]
        pts = pts + pijk[ip[np.arange(len(pts))]]  # (D', 3)

        if nears_mask is not None:
            near_any = nears_mask[ip].any(axis=1)  # (D',)
            coll = check_probe_collisions(pts, low_pts, nears_mask[ip], rp2)
            reject = (hijk[ip] < rp) & near_any & coll  # (D',)
            pts, ip, il = pts[~reject], ip[~reject], il[~reject]

        d = np.linalg.norm(pts[:, None, :] - a_xyz[ip], axis=-1) - a_rad[ip]  # (D'', 3)
        mc = np.argmin(d, axis=1)  # (D'',)

        self.run.stats.concave_dots += len(pts)
        nat_stack = np.stack(idx3, axis=1)  # (P, 3)
        self._write_dots(
            a_mol[ip, mc],
            3,
            pts,
            area[ip, il],
            pijk[ip],
            nat_stack[ip, mc],
        )

    # ── Stage 7: CMS scoring ──────────────────────────────────────────────

    def _score_cms(
        self,
        target_side: bool = True,
    ) -> tuple[float, Float[np.ndarray, Nq]]:
        """Score CMS from accumulated surface dots.

        For each buried dot on the *query* side, find the nearest buried
        dot on the *reference* side and sum
        ``area_q x exp(-weight x nearest_dist^2)``.

        Args:
            target_side: If ``True``, query = target (mol 1) and
                         reference = binder (mol 0).
        """
        atoms = self.run.atoms
        mol = atoms.molecule
        n_mol0 = int((mol == 0).sum())
        n_mol1 = int((mol == 1).sum())

        if target_side:
            dots_q, dots_r, n_q, offset = (
                self.run.dots[1],
                self.run.dots[0],
                n_mol1,
                n_mol0,
            )
        else:
            dots_q, dots_r, n_q, offset = (
                self.run.dots[0],
                self.run.dots[1],
                n_mol0,
                0,
            )

        if len(dots_r) == 0:
            return 0.0, np.zeros(n_q, dtype=np.float64)

        buried_q = dots_q.buried.astype(bool)  # (Dq,)
        buried_r = dots_r.buried.astype(bool)  # (Dr,)
        if not buried_q.any() or not buried_r.any():
            return 0.0, np.zeros(n_q, dtype=np.float64)

        tree = cKDTree(dots_r.coor_xyz[buried_r])
        min_dist, _ = tree.query(dots_q.coor_xyz[buried_q], k=1)

        area_q = dots_q.area[buried_q]  # (Bq,)
        per_dot = area_q * np.exp(-(min_dist**2) * self.settings.weight)  # (Bq,)

        local_idx = dots_q.atom_idx[buried_q] - offset  # (Bq,)
        per_atom = np.bincount(
            local_idx,
            weights=per_dot,
            minlength=n_q,
        ).astype(np.float64)  # (Nq,)
        return float(per_atom.sum()), per_atom

    def _compute_max_possible_cms(self) -> tuple[float, np.ndarray]:
        """Sum all dot areas for molecule 0 (no burial/distance weighting)."""
        n_mol0 = int((self.run.atoms.molecule == 0).sum())
        dots = self.run.dots[0]
        if len(dots) == 0:
            return 0.0, np.zeros(n_mol0, dtype=np.float64)
        per_atom = np.bincount(
            dots.atom_idx,
            weights=dots.area,
            minlength=n_mol0,
        ).astype(np.float64)
        return float(per_atom.sum()), per_atom

    # ── Dot writing helper ─────────────────────────────────────────────────

    def _write_dots(
        self,
        molecule: Int[np.ndarray, D],
        dot_type: int,
        coor: Float[np.ndarray, "D 3"],
        area: Float[np.ndarray, D],
        pcen: Float[np.ndarray, "D 3"],
        atom_idx: Int[np.ndarray, D],
    ) -> None:
        """Write surface dots to per-molecule DotArrays with burial flags."""
        if len(coor) == 0:
            return

        pr = self.settings.probe_radius
        atom_idx = np.asarray(atom_idx, dtype=np.int32)
        molecule = np.asarray(molecule, dtype=np.int8)

        outnml = (pcen - coor) / pr if pr > 0 else coor - self.run.atoms.xyz[atom_idx]

        bxyz = self.run.buried_array.xyz[atom_idx]  # (D, Kb, 3)
        brad = self.run.buried_array.radius[atom_idx]  # (D, Kb)
        d2 = np.square(bxyz - pcen[:, None, :]).sum(-1)  # (D, Kb)
        buried = (d2 <= (brad + pr) ** 2).any(axis=-1).astype(np.int8)  # (D,)

        type_arr = np.full(len(coor), dot_type, dtype=np.int8)
        for mol_id in (0, 1):
            mask = molecule == mol_id
            if mask.any():
                self.run.dots[mol_id].extend(
                    coor[mask],
                    outnml[mask],
                    area[mask],
                    buried[mask],
                    type_arr[mask],
                    atom_idx[mask],
                )
