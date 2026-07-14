"""Exact Contact Molecular Surface on GPU (PyTorch).

Produces results matching the CPU/NumPy reference within float32 precision.
Optimised for minimal kernel launches and zero unnecessary CPU-GPU sync.

The ``compute`` method runs a 7-stage pipeline (see ``_run_pipeline``):

1. **Setup** — concatenate binder + target, assign attention flags.
2. **Neighbors** — build sorted same-molecule neighbor and cross-molecule
   burial arrays via vectorised argsort + gather.
3. **Atom pairs** — process forward pairs, find probe triplets, queue
   toroid pairs.
4. **Convex dots** — latitude-longitude subdivision of exposed atom surfaces.
5. **Toroidal dots** — saddle-surface dots between bonded atom pairs.
6. **Concave dots** — dots inside probe spheres at atom triplets.
7. **Score** — distance-weighted CMS from buried surface dots.

Usage::

    calc = TorchCMS(device="cuda")
    cms, per_atom = calc.compute(binder_xyz, binder_radii, target_xyz, target_radii)

    # Both sides at once (single pipeline run):
    (t_cms, t_pa), (b_cms, b_pa) = calc.compute_both_sides(
        binder_xyz,
        binder_radii,
        target_xyz,
        target_radii,
    )
"""

from __future__ import annotations

import math

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .data_structures import ATTEN_6, ATTEN_BLOCKER, ATTEN_BURIED

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_SUBDIV = 100
_TWO_PI = 2.0 * math.pi
_EPS = 1e-12
_PARALLEL_THRESH = 0.99
_MAX_NEIGHBORS = 128


def _dot(a: Float[Tensor, "* D"], b: Float[Tensor, "* D"]) -> Float[Tensor, "*"]:
    """Batched dot product along the last dimension."""
    return torch.einsum("...i,...i->...", a, b)


# ═══════════════════════════════════════════════════════════════════════
# Geometry — fused helpers with minimal kernel launches
# ═══════════════════════════════════════════════════════════════════════


def _safe_norm(v: Float[Tensor, "* D"]) -> Float[Tensor, "* 1"]:
    """L2 norm along last dim, clamped to avoid division by zero."""
    return v.norm(dim=-1, keepdim=True).clamp(min=_EPS)


def _normalize(v: Float[Tensor, "* D"]) -> Float[Tensor, "* D"]:
    return v / _safe_norm(v)


def _cross_norm(a: Float[Tensor, "* 3"], b: Float[Tensor, "* 3"]) -> Float[Tensor, "* 3"]:
    """Cross product + normalize — 2 kernels instead of 4."""
    c = torch.linalg.cross(a, b)
    return c / _safe_norm(c)


def _build_ortho_basis(
    axis: Float[Tensor, "* 3"],
) -> tuple[Float[Tensor, "* 3"], Float[Tensor, "* 3"]]:
    """Build orthonormal (x, y) basis perpendicular to *axis*.

    Fused: replaces 4x normalize + 2x cross_norm + 1x cross in old code
    with a single function doing it all.
    """
    axis = _normalize(axis)
    v = torch.stack(
        [
            axis[..., 1] ** 2 + axis[..., 2] ** 2,
            axis[..., 0] ** 2 + axis[..., 2] ** 2,
            axis[..., 0] ** 2 + axis[..., 1] ** 2,
        ],
        dim=-1,
    )
    v = _normalize(v)
    parallel = _dot(v, axis).abs() > _PARALLEL_THRESH
    fallback = torch.tensor([1.0, 0.0, 0.0], device=axis.device, dtype=axis.dtype)
    v = torch.where(parallel[..., None], fallback, v)
    y_basis = _cross_norm(axis, v)
    x_basis = _cross_norm(axis, y_basis)
    return x_basis, torch.linalg.cross(axis, x_basis)


def _subdivide_angle(
    center: Float[Tensor, "* 3"],
    radius: Float[Tensor, "*"],
    x: Float[Tensor, "* 3"],
    y: Float[Tensor, "* 3"],
    angle: Float[Tensor, "*"],
    density: Float[Tensor, "*"],
    use_bf16: bool = False,
) -> tuple[Float[Tensor, "*"], Float[Tensor, "* K 3"]]:
    """Evenly space dots along an angular sweep.

    One unavoidable ``.item()`` sync to determine output K dimension.
    """
    dev = radius.device

    delta = 1.0 / (density.sqrt() * radius)  # (...,)
    raw = (angle / delta).nan_to_num(0.0).clamp(min=0.0)  # (...,)
    max_k = min(_MAX_SUBDIV, max(1, int(raw.max().item())))

    slots = torch.arange(max_k, device=dev, dtype=radius.dtype)  # (K,)
    a = delta[..., None] * (slots + 0.5)  # (..., K)
    valid = a <= angle[..., None]  # (..., K)

    if use_bf16 and dev.type == "cuda":
        a16 = a.bfloat16()
        cos_a = a16.cos().to(radius.dtype)
        sin_a = a16.sin().to(radius.dtype)
    else:
        cos_a = a.cos()
        sin_a = a.sin()

    r_cos = (radius[..., None] * cos_a)[..., None]  # (..., K, 1)
    r_sin = (radius[..., None] * sin_a)[..., None]  # (..., K, 1)
    points = center[..., None, :] + x[..., None, :] * r_cos + y[..., None, :] * r_sin  # (..., K, 3)

    points.masked_fill_(~valid[..., None], float("nan"))

    counts = valid.sum(dim=-1)  # (...,)
    spacing = torch.where(
        counts > 0,
        radius * angle / counts.clamp(min=1).to(radius.dtype),
        torch.zeros_like(radius),
    )  # (...,)
    return spacing, points


def _subdivide_arc(
    center: Float[Tensor, "* 3"],
    radius: Float[Tensor, "*"],
    axis: Float[Tensor, "* 3"],
    density: Float[Tensor, "*"],
    start: Float[Tensor, "* 3"],
    end: Float[Tensor, "* 3"],
    use_bf16: bool = False,
) -> tuple[Float[Tensor, "*"], Float[Tensor, "* K 3"]]:
    """Subdivide the arc from *start* to *end* around *axis*."""
    y = torch.linalg.cross(axis, start)
    cos_angle = _dot(end, start)
    sin_angle = _dot(end, y)
    angle = torch.atan2(sin_angle, cos_angle)
    angle = torch.where(angle < 0, angle + _TWO_PI, angle)
    near_zero = cos_angle.abs().lt(1e-6) & sin_angle.abs().lt(1e-6)
    angle.masked_fill_(near_zero, float("nan"))
    return _subdivide_angle(center, radius, start, y, angle, density, use_bf16)


def _subdivide_circle(
    center: Float[Tensor, "* 3"],
    radius: Float[Tensor, "*"],
    axis: Float[Tensor, "* 3"],
    density: Float[Tensor, "*"],
    use_bf16: bool = False,
) -> tuple[Float[Tensor, "*"], Float[Tensor, "* K 3"]]:
    """Subdivide a full 2pi circle perpendicular to *axis*."""
    x, y = _build_ortho_basis(axis)
    angle = torch.full_like(radius, _TWO_PI)
    return _subdivide_angle(center, radius, x, y, angle, density, use_bf16)


def _dist_to_line(
    origin: Float[Tensor, "* 3"],
    direction: Float[Tensor, "* 3"],
    points: Float[Tensor, "* 3"],
) -> Float[Tensor, "*"]:
    """Perpendicular distance from *points* to a line."""
    v = points - origin
    proj = _dot(v, direction)
    return (v.square().sum(-1) - proj * proj).clamp(min=0).sqrt()


def _collision_skip_first(
    points: Float[Tensor, "D 1 3"],
    neigh_xyz: Float[Tensor, "D K 3"],
    neigh_rad: Float[Tensor, "D K"],
    probe_r: float,
) -> Bool[Tensor, D]:
    """Dense collision check, skipping first neighbor (C++ convention)."""
    d2 = (neigh_xyz - points).square().sum(dim=-1)
    hit = (d2 <= (neigh_rad + probe_r) ** 2) & ~neigh_rad.isnan()
    return hit[..., 1:].any(dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# Padded neighbor builder — fully vectorised, zero Python loops
# ═══════════════════════════════════════════════════════════════════════


def _gather_neighbors(
    xyz: Float[Tensor, "N 3"],
    radius: Float[Tensor, N],
    natom: Int[Tensor, N],
    mask: Bool[Tensor, "N N"],
    counts: Int[Tensor, N],
    max_k: int,
    dist_sq: Float[Tensor, "N N"],
) -> tuple[
    Float[Tensor, "N K 3"],
    Float[Tensor, "N K"],
    Int[Tensor, "N K"],
    Int[Tensor, N],
]:
    """Build padded neighbor arrays via vectorised argsort + gather.

    Args:
        xyz:      Atom coordinates.               ``(N, 3)``
        radius:   Atom radii.                      ``(N,)``
        natom:    Atom indices.                     ``(N,)``
        mask:     Boolean adjacency mask.           ``(N, N)``
        counts:   Neighbour count per atom.         ``(N,)``
        max_k:    Max neighbour slots to keep.
        dist_sq:  Squared pairwise distances.       ``(N, N)``

    Returns:
        ``(out_xyz, out_rad, out_idx, counts)`` padded with NaN / -1.
    """
    dev = xyz.device

    sorted_idx = dist_sq.masked_fill(~mask, float("inf")).argsort(dim=1)[:, :max_k]  # (N, K)
    out_xyz = xyz[sorted_idx]  # (N, K, 3)
    out_rad = radius[sorted_idx]  # (N, K)
    out_nat = natom[sorted_idx]  # (N, K)

    padding = torch.arange(max_k, device=dev)[None, :] >= counts[:, None]  # (N, K)
    out_rad.masked_fill_(padding, float("nan"))
    out_nat.masked_fill_(padding, -1)
    out_xyz.masked_fill_(padding.unsqueeze(-1), float("nan"))

    return out_xyz, out_rad, out_nat, counts


def _cat_or_empty(
    lst: list[Tensor],
    dim2: int | None = None,
    device: torch.device | str = "cpu",
) -> Float[Tensor, ...]:
    """Concatenate a list of tensors, returning an empty tensor if empty."""
    if lst:
        return torch.cat(lst)
    return torch.zeros((0, dim2) if dim2 else (0,), device=device)


# ═══════════════════════════════════════════════════════════════════════
# Pipeline state — threaded through computation stages
# ═══════════════════════════════════════════════════════════════════════


class _State:
    """Mutable state populated incrementally by ``TorchCMS._run_pipeline``.

    Not for external use.  Attributes are set by each pipeline stage;
    accessing one before its stage has run raises ``AttributeError``.
    """

    __slots__ = (
        "access",
        "atom_density",
        "atom_idx",
        "atten",
        "burial_count",
        "burial_rad",
        "burial_xyz",
        # per-atom arrays, all length N
        "coords",
        "dev",
        "dot_areas",
        "dot_buried",
        # accumulated surface dots, split by molecule [mol0, mol1]
        "dot_coords",
        "dot_owners",
        # boolean mask of atoms with neighbours
        "good_atoms",
        "mol_id",
        "n",
        "n_binder",
        "neigh_count",
        "neigh_idx",
        "neigh_rad",
        # padded neighbour / burial arrays
        "neigh_xyz",
        "probe_data",
        "probe_idx",
        "radii",
        "toroid_geom",
        # intermediate geometry for toroidal / concave stages
        "toroid_pairs",
    )


# ═══════════════════════════════════════════════════════════════════════
# Main calculator
# ═══════════════════════════════════════════════════════════════════════


class TorchCMS:
    """Exact Contact Molecular Surface calculator (PyTorch).

    Runs on GPU or CPU.  On CUDA the subdivision trig can optionally use
    bfloat16 for a modest speed-up (``use_bf16=True``, the default).

    Example::

        calc = TorchCMS(device="cuda")
        cms, per_atom = calc.compute(binder_xyz, binder_radii, target_xyz, target_radii)

        # Both sides in one pipeline run:
        (t_cms, t_pa), (b_cms, b_pa) = calc.compute_both_sides(
            binder_xyz,
            binder_radii,
            target_xyz,
            target_radii,
        )
    """

    def __init__(
        self,
        probe_radius: float = 1.7,
        density: float = 15.0,
        separation: float = 8.0,
        weight: float = 0.5,
        device: str | torch.device = "cpu",
        use_bf16: bool = True,
    ) -> None:
        self.rp = probe_radius
        self.density = density
        self.separation = separation
        self.weight = weight
        self.device = torch.device(device)
        self.bf16 = use_bf16 and self.device.type == "cuda"

    # ── Public API ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute(
        self,
        binder_xyz: Float[Tensor, "Nb 3"],
        binder_radii: Float[Tensor, Nb],
        target_xyz: Float[Tensor, "Nt 3"],
        target_radii: Float[Tensor, Nt],
    ) -> tuple[Float[Tensor, ""], Float[Tensor, Nt]]:
        """Compute target-side CMS.

        Args:
            binder_xyz:   Binder atom coordinates.  ``(Nb, 3)``
            binder_radii: Binder atom radii.         ``(Nb,)``
            target_xyz:   Target atom coordinates.   ``(Nt, 3)``
            target_radii: Target atom radii.          ``(Nt,)``

        Returns:
            ``(total_cms, per_target_atom_cms)`` with per-atom shape ``(Nt,)``.
        """
        s = self._run_pipeline(binder_xyz, binder_radii, target_xyz, target_radii)
        return self._score_cms(s, target_side=True)

    @torch.no_grad()
    def compute_both_sides(
        self,
        binder_xyz: Float[Tensor, "Nb 3"],
        binder_radii: Float[Tensor, Nb],
        target_xyz: Float[Tensor, "Nt 3"],
        target_radii: Float[Tensor, Nt],
    ) -> tuple[
        tuple[Float[Tensor, ""], Float[Tensor, Nt]],
        tuple[Float[Tensor, ""], Float[Tensor, Nb]],
    ]:
        """Compute CMS for both sides in a single pipeline run.

        The surface-generation pipeline runs once; only the lightweight
        scoring step is executed twice (one per side).

        Args:
            binder_xyz:   Binder atom coordinates.  ``(Nb, 3)``
            binder_radii: Binder atom radii.         ``(Nb,)``
            target_xyz:   Target atom coordinates.   ``(Nt, 3)``
            target_radii: Target atom radii.          ``(Nt,)``

        Returns:
            ``((target_cms, target_per_atom), (binder_cms, binder_per_atom))``
            where target_per_atom is ``(Nt,)`` and binder_per_atom is ``(Nb,)``.
        """
        s = self._run_pipeline(binder_xyz, binder_radii, target_xyz, target_radii)
        target = self._score_cms(s, target_side=True)
        binder = self._score_cms(s, target_side=False)
        return target, binder

    @torch.no_grad()
    def compute_max_cms(
        self,
        xyz: Float[Tensor, "N 3"],
        radii: Float[Tensor, N],
    ) -> tuple[Float[Tensor, ""], Float[Tensor, N]]:
        """Compute maximum possible CMS (total surface area) for one molecule.

        Generates the full Connolly surface for a single molecule and sums
        all dot areas without distance weighting or burial filtering.

        Args:
            xyz:   Atom coordinates.  ``(N, 3)``
            radii: Atom radii.        ``(N,)``

        Returns:
            ``(total_area, per_atom_area)`` with per-atom shape ``(N,)``.
        """
        s = self._setup_atoms_single(xyz, radii)
        self._build_neighbors(s)
        self._process_atom_pairs(s)
        self._generate_convex_dots(s)
        self._generate_toroidal_dots(s)
        self._generate_concave_dots(s)
        return self._compute_max_area(s)

    # ── Pipeline orchestrator ──────────────────────────────────────────

    def _run_pipeline(
        self,
        binder_xyz: Float[Tensor, "Nb 3"],
        binder_radii: Float[Tensor, Nb],
        target_xyz: Float[Tensor, "Nt 3"],
        target_radii: Float[Tensor, Nt],
    ) -> _State:
        """Execute stages 1-6, populating ``_State`` with surface dots.

        Stage 7 (``_score_cms``) is called separately so that both sides
        can be scored from the same set of dots.
        """
        s = self._setup_atoms(binder_xyz, binder_radii, target_xyz, target_radii)
        self._build_neighbors(s)
        self._process_atom_pairs(s)
        self._generate_convex_dots(s)
        self._generate_toroidal_dots(s)
        self._generate_concave_dots(s)
        return s

    # ── Stage 1: Setup ─────────────────────────────────────────────────

    def _init_state(self, coords: Tensor, radii: Tensor, n_binder: int) -> _State:
        """Shared initialisation for ``_setup_atoms`` and ``_setup_atoms_single``."""
        s = _State()
        s.dev = self.device
        s.n_binder = n_binder
        s.coords = coords.to(s.dev)  # (N, 3)
        s.radii = radii.to(s.dev)  # (N,)
        s.n = s.coords.shape[0]

        s.mol_id = torch.zeros(s.n, dtype=torch.int64, device=s.dev)  # (N,)
        s.atom_density = torch.full((s.n,), self.density, device=s.dev)  # (N,)
        s.atom_idx = torch.arange(s.n, dtype=torch.int64, device=s.dev)  # (N,)
        s.atten = torch.full(
            (s.n,),
            ATTEN_BURIED,
            dtype=torch.int64,
            device=s.dev,
        )  # (N,)
        s.access = torch.zeros(s.n, dtype=torch.int64, device=s.dev)  # (N,)

        s.dot_coords: list[list[Tensor]] = [[], []]
        s.dot_areas: list[list[Tensor]] = [[], []]
        s.dot_buried: list[list[Tensor]] = [[], []]
        s.dot_owners: list[list[Tensor]] = [[], []]
        s.toroid_pairs: list[Tensor] = []
        s.toroid_geom: list[Tensor] = []
        s.probe_idx: list[Tensor] = []
        s.probe_data: list[Tensor] = []
        return s

    def _setup_atoms(
        self,
        binder_xyz: Float[Tensor, "Nb 3"],
        binder_radii: Float[Tensor, Nb],
        target_xyz: Float[Tensor, "Nt 3"],
        target_radii: Float[Tensor, Nt],
    ) -> _State:
        """Concatenate binder + target and assign attention flags.

        Atoms further than ``self.separation`` from any cross-molecule atom
        become *blockers* — they contribute to the Connolly surface but not
        to the CMS score.

        Populates: ``coords``, ``radii``, ``mol_id``, ``atten``, ``access``,
        ``atom_density``, ``atom_idx``, and dot-accumulator lists.
        """
        s = self._init_state(
            torch.cat([binder_xyz, target_xyz]),
            torch.cat([binder_radii, target_radii]),
            binder_xyz.shape[0],
        )
        s.mol_id[s.n_binder :] = 1

        cross_dist = torch.cdist(
            s.coords[: s.n_binder],
            s.coords[s.n_binder :],
        )  # (Nb, Nt)
        s.atten[: s.n_binder].masked_fill_(
            cross_dist.min(dim=1).values >= self.separation,
            ATTEN_BLOCKER,
        )
        s.atten[s.n_binder :].masked_fill_(
            cross_dist.min(dim=0).values >= self.separation,
            ATTEN_BLOCKER,
        )
        return s

    def _setup_atoms_single(
        self,
        xyz: Float[Tensor, "N 3"],
        radii: Float[Tensor, N],
    ) -> _State:
        """Set up state for a single molecule (used by ``compute_max_cms``).

        All atoms are assigned to molecule 0 with ``atten = ATTEN_BURIED``
        (no cross-molecule distance check needed).
        """
        return self._init_state(xyz, radii, xyz.shape[0])

    # ── Stage 2: Neighbour lists ───────────────────────────────────────

    def _build_neighbors(self, s: _State) -> None:
        """Build sorted neighbour and burial arrays for all atoms.

        *Same-molecule* neighbours within bonding distance → ``neigh_*``.
        *Cross-molecule* buried-eligible atoms within bonding distance
        → ``burial_*``.

        Also marks isolated active atoms as accessible and identifies
        *good atoms* (active, non-dead, with ≥1 neighbour).

        Populates: ``neigh_*``, ``burial_*``, ``good_atoms``.
        Mutates: ``access``.

        Note: Allocates O(N^2) memory for the pairwise distance matrix.
        For ~1000 atoms this is ~4 MB; for ~5000 atoms it is ~100 MB.
        The NumPy backend uses ``scipy.spatial.cKDTree`` for O(N log N)
        lookups.
        """
        rp = self.rp
        n, dev = s.n, s.dev

        dist_sq = torch.cdist(s.coords, s.coords).square()  # (N, N)
        bridge_sq = (s.radii[:, None] + s.radii[None, :] + 2.0 * rp) ** 2  # (N, N)

        active = s.atten > 0  # (N,)
        eye_off = ~torch.eye(n, dtype=torch.bool, device=dev)  # (N, N)
        same_mol = s.mol_id[:, None] == s.mol_id[None, :]  # (N, N)
        diff_mol = ~same_mol  # (N, N)
        burial_eligible = s.atten >= ATTEN_BURIED  # (N,)

        neigh_mask = active[:, None] & active[None, :] & same_mol & eye_off & (dist_sq < bridge_sq)  # (N, N)
        burial_mask = active[:, None] & burial_eligible[None, :] & diff_mol & (dist_sq < bridge_sq)  # (N, N)

        bb_sq = (4.0 * s.radii.max() + 4.0 * rp) ** 2
        n_cross = (active[:, None] & burial_eligible[None, :] & diff_mol & (dist_sq < bb_sq)).sum(dim=1)  # (N,)

        s.neigh_count = neigh_mask.sum(dim=1).to(torch.int64)  # (N,)
        s.burial_count = burial_mask.sum(dim=1).to(torch.int64)  # (N,)
        dead = (s.atten == ATTEN_6) & (n_cross == 0)  # (N,)
        s.access.masked_fill_(active & ~dead & (s.neigh_count == 0), 1)
        s.good_atoms = active & ~dead & (s.neigh_count > 0)  # (N,)

        max_neigh = min(n - 1, _MAX_NEIGHBORS)
        max_burial = min(n, _MAX_NEIGHBORS)
        s.neigh_xyz, s.neigh_rad, s.neigh_idx, _ = _gather_neighbors(
            s.coords,
            s.radii,
            s.atom_idx,
            neigh_mask,
            s.neigh_count,
            max_neigh,
            dist_sq,
        )
        s.burial_xyz, s.burial_rad, _, _ = _gather_neighbors(
            s.coords,
            s.radii,
            s.atom_idx,
            burial_mask,
            s.burial_count,
            max_burial,
            dist_sq,
        )

    # ── Stage 3: Atom-pair processing ──────────────────────────────────

    def _process_atom_pairs(self, s: _State) -> None:
        """Process forward atom pairs: toroid queue + probe triplets.

        For each pair ``(atom_i, atom_j)`` with ``j > i`` whose extended
        spheres overlap:

        * Queue pairs needing toroidal surface dots → ``s.toroid_*``.
        * Find atom_k forming valid probe triplets → ``s.probe_idx/data``.

        Mutates: ``access``, ``toroid_pairs``, ``toroid_geom``, ``probe_idx``, ``probe_data``.
        """
        dev, rp = s.dev, self.rp
        good_idx = s.good_atoms.nonzero(as_tuple=True)[0]  # (G,)
        if good_idx.numel() == 0:
            return

        g_neigh_xyz, g_neigh_rad, g_neigh_idx = (
            s.neigh_xyz[good_idx],
            s.neigh_rad[good_idx],
            s.neigh_idx[good_idx],
        )  # (G,K,3), (G,K), (G,K)
        g_coords, g_radii, g_atten, g_ncount = (
            s.coords[good_idx],
            s.radii[good_idx],
            s.atten[good_idx],
            s.neigh_count[good_idx],
        )  # (G,3), (G,), (G,), (G,)

        forward = g_neigh_idx > good_idx[:, None]  # (G, K)
        diff_vec = g_neigh_xyz - g_coords[:, None, :]  # (G, K, 3)
        pair_dist_sq = _dot(diff_vec, diff_vec)  # (G, K)
        pair_dist = torch.where(
            forward,
            pair_dist_sq,
            torch.ones_like(pair_dist_sq),
        ).sqrt()  # (G, K)

        ext_self = g_radii + rp  # (G,)
        ext_neigh = g_neigh_rad + rp  # (G, K)

        asymmetry = torch.where(
            forward,
            (ext_self[:, None] ** 2 - ext_neigh**2) / pair_dist,
            torch.zeros_like(pair_dist),
        )  # (G, K)
        is_between = asymmetry.abs() < pair_dist  # (G, K)
        torus_center = (g_coords[:, None, :] + g_neigh_xyz) * 0.5 + (diff_vec / pair_dist[..., None]) * (
            asymmetry * 0.5
        )[..., None]  # (G, K, 3)

        far_sq = torch.where(
            forward,
            (ext_self[:, None] + ext_neigh) ** 2 - pair_dist_sq,
            -torch.ones_like(pair_dist_sq),
        )  # (G, K)
        contain_sq = torch.where(
            forward,
            pair_dist_sq - (g_radii[:, None] - g_neigh_rad) ** 2,
            -torch.ones_like(pair_dist_sq),
        )  # (G, K)
        geom_valid = forward & (far_sq > 0) & (contain_sq > 0)  # (G, K)

        shortcut = (g_ncount <= 1) & geom_valid.any(dim=1)  # (G,)
        if shortcut.any():
            s.access[good_idx[shortcut]] = 1
            first_k = geom_valid[shortcut].byte().argmax(dim=1)
            s.access[g_neigh_idx[shortcut][torch.arange(shortcut.sum(), device=dev), first_k]] = 1

        flat_mask = geom_valid & ~shortcut[:, None]  # (G, K)
        row, col = flat_mask.nonzero(as_tuple=True)
        if row.numel() == 0:
            return

        pair_atom1 = good_idx[row]  # (P,)
        pair_atom2 = g_neigh_idx[row, col]  # (P,)
        pair_d = pair_dist[row, col]  # (P,)
        pair_dir = diff_vec[row, col] / pair_d[:, None]  # (P, 3)
        pair_tc = torus_center[row, col]  # (P, 3)
        pair_tr = 0.5 * far_sq[row, col].clamp(min=0).sqrt() * contain_sq[row, col].clamp(min=0).sqrt() / pair_d  # (P,)
        pair_between = is_between[row, col]  # (P,)

        # Queue pairs needing toroidal surface dots
        att1 = g_atten[row]  # (P,)
        att2 = s.atten[pair_atom2]  # (P,)
        needs_toroid = (att1 > ATTEN_BLOCKER) | ((att2 > ATTEN_BLOCKER) & (rp > 0))  # (P,)
        tor = needs_toroid.nonzero(as_tuple=True)[0]
        if tor.numel() > 0:
            s.toroid_pairs.append(
                torch.stack([pair_atom1[tor], pair_atom2[tor]], dim=-1),
            )  # (M, 2)
            s.toroid_geom.append(
                torch.cat(
                    [
                        pair_dir[tor],
                        pair_tc[tor],
                        pair_tr[tor, None],
                        pair_between[tor, None].float(),
                        torch.zeros(tor.numel(), 1, device=dev),
                    ],
                    dim=-1,
                )
            )  # (M, 9)

        self._find_probes(s, pair_atom1, pair_atom2, pair_dir, pair_tc, pair_tr)

    # ── Stage 3b: Probe finding ────────────────────────────────────────

    def _find_probes(
        self,
        s: _State,
        pair_atom1: Int[Tensor, P],
        pair_atom2: Int[Tensor, P],
        pair_dir: Float[Tensor, "P 3"],
        pair_tc: Float[Tensor, "P 3"],
        pair_tr: Float[Tensor, P],
    ) -> None:
        """Find probe-sphere positions at atom triplets.

        For each ``(atom1, atom2)`` pair, search atom1's neighbour list for
        an ``atom3 > atom2`` forming a valid triple.  Two candidate probes
        per triple (``± height`` above the torus plane) are collision-checked.

        Args:
            pair_atom1: First atom index per pair.    ``(P,)``
            pair_atom2: Second atom index per pair.   ``(P,)``
            pair_dir:   Unit vector atom1 → atom2.    ``(P, 3)``
            pair_tc:    Torus circle centre.           ``(P, 3)``
            pair_tr:    Torus circle radius.           ``(P,)``

        Mutates: ``access``, ``probe_idx``, ``probe_data``.
        """
        dev, rp = s.dev, self.rp
        n_pairs = pair_atom1.shape[0]

        ext1, ext2 = s.radii[pair_atom1] + rp, s.radii[pair_atom2] + rp  # (P,), (P,)
        tri_neigh_xyz, tri_neigh_rad, tri_neigh_idx = (
            s.neigh_xyz[pair_atom1],
            s.neigh_rad[pair_atom1],
            s.neigh_idx[pair_atom1],
        )  # (P,K,3), (P,K), (P,K)

        tri_valid = (tri_neigh_idx > pair_atom2[:, None]) & (tri_neigh_idx >= 0)  # (P, K)
        ext3 = torch.where(tri_valid, tri_neigh_rad + rp, float("inf"))  # (P, K)

        atom1_xyz = s.coords[pair_atom1]  # (P, 3)
        atom2_xyz = s.coords[pair_atom2]  # (P, 3)
        dist_23 = (tri_neigh_xyz - atom2_xyz[:, None, :]).norm(dim=-1)  # (P, K)
        tri_valid &= dist_23 < (ext2[:, None] + ext3)
        dist_13 = (tri_neigh_xyz - atom1_xyz[:, None, :]).norm(dim=-1)  # (P, K)
        tri_valid &= dist_13 < (ext1[:, None] + ext3)

        safe_idx3 = tri_neigh_idx.clamp(min=0)  # (P, K)
        all_blocked = (
            (s.atten[pair_atom1][:, None] <= ATTEN_BLOCKER)
            & (s.atten[pair_atom2][:, None] <= ATTEN_BLOCKER)
            & (s.atten[safe_idx3] <= ATTEN_BLOCKER)
        )  # (P, K)
        tri_valid &= ~all_blocked

        tri_row, tri_col = tri_valid.nonzero(as_tuple=True)
        if tri_row.numel() == 0:
            return

        t1_xyz, t3_xyz = atom1_xyz[tri_row], tri_neigh_xyz[tri_row, tri_col]  # (T,3) each
        t_idx = torch.stack(
            [
                pair_atom1[tri_row],
                pair_atom2[tri_row],
                tri_neigh_idx[tri_row, tri_col],
            ],
            dim=-1,
        )  # (T, 3)
        t_ext1, t_ext3 = ext1[tri_row], tri_neigh_rad[tri_row, tri_col] + rp  # (T,) each
        t_d13, t_dir, t_tc, t_tr = (
            dist_13[tri_row, tri_col],
            pair_dir[tri_row],
            pair_tc[tri_row],
            pair_tr[tri_row],
        )  # (T,), (T,3), (T,3), (T,)

        dir13 = (t3_xyz - t1_xyz) / t_d13[:, None]  # (T, 3)
        cos_angle = _dot(t_dir, dir13)  # (T,)
        arc_angle = torch.acos(cos_angle.clamp(-1, 1))  # (T,)
        sin_arc = arc_angle.sin()  # (T,)

        degen = (cos_angle >= 1) | (cos_angle <= -1) | (arc_angle <= 0) | (sin_arc <= 0)  # (T,)
        degen_dist_sq = (t_tc - t3_xyz).pow(2).sum(-1)  # (T,)
        kills = degen & (degen_dist_sq < (t_ext3**2 - t_tr**2))
        kill_pair_mask = torch.zeros(n_pairs, dtype=torch.bool, device=dev)
        kill_pair_mask.scatter_(0, tri_row[kills], True)
        keep = ~degen & ~kill_pair_mask[tri_row]  # (T,)

        t1_xyz, t3_xyz = t1_xyz[keep], t3_xyz[keep]
        t_idx = t_idx[keep]
        scalars = torch.stack(
            [
                t_ext1[keep],
                t_ext3[keep],
                t_d13[keep],
                t_tr[keep],
                sin_arc[keep],
            ],
            dim=-1,
        )  # (Q, 5)
        vecs = torch.stack(
            [t_dir[keep], t_tc[keep], dir13[keep]],
            dim=1,
        )  # (Q, 3, 3)
        q = t1_xyz.shape[0]
        if q == 0:
            return

        ext1_q, ext3_q, d13_q, tr_q, sin_q = scalars.unbind(-1)
        dir_q, tc_q, dir13_q = vecs.unbind(1)

        cross_ij_ik = torch.linalg.cross(dir_q, dir13_q) / sin_q[:, None]  # (Q, 3)
        cross_perp = torch.linalg.cross(cross_ij_ik, dir_q)  # (Q, 3)

        asym = (ext1_q**2 - ext3_q**2) / d13_q  # (Q,)
        mid13 = (t1_xyz + t3_xyz) * 0.5 + dir13_q * (asym * 0.5)[:, None]  # (Q, 3)
        proj = _dot(dir13_q, mid13 - tc_q)  # (Q,)
        base = tc_q + cross_perp * (proj / sin_q)[:, None]  # (Q, 3)

        height_sq = ext1_q**2 - _dot(base - t1_xyz, base - t1_xyz)  # (Q,)
        valid_h = height_sq > 0  # (Q,)
        base, cross_ij_ik, t_idx = base[valid_h], cross_ij_ik[valid_h], t_idx[valid_h]
        height = height_sq[valid_h].clamp(min=0).sqrt()  # (Q',)
        q = height.shape[0]
        if q == 0:
            return

        sign = torch.cat(
            [
                torch.ones(q, device=dev),
                -torch.ones(q, device=dev),
            ]
        )  # (2Q,)
        probe_pos = base.repeat(2, 1) + cross_ij_ik.repeat(2, 1) * (height.repeat(2) * sign)[:, None]  # (2Q, 3)

        idx1, idx2, idx3_t = (
            t_idx[:, 0].repeat(2),
            t_idx[:, 1].repeat(2),
            t_idx[:, 2].repeat(2),
        )  # (2Q,) each

        coll_xyz, coll_rad, coll_idx_t = (
            s.neigh_xyz[idx1],
            s.neigh_rad[idx1],
            s.neigh_idx[idx1],
        )  # (2Q,K,3), (2Q,K), (2Q,K)
        exclude = (coll_idx_t == idx2[:, None]) | (coll_idx_t == idx3_t[:, None]) | (coll_idx_t < 0)  # (2Q, K)
        coll_d2 = (probe_pos[:, None, :] - coll_xyz).square().sum(-1)  # (2Q, K)
        collision = ((coll_d2 <= (coll_rad + rp) ** 2) & ~exclude).any(dim=1)  # (2Q,)

        ok = ~collision  # (2Q,)
        pos_sign = sign > 0  # (2Q,)
        pa0 = torch.where(pos_sign, idx1, idx2)[ok]  # (R,)
        pa1 = torch.where(pos_sign, idx2, idx1)[ok]  # (R,)
        pa2 = idx3_t[ok]  # (R,)

        if pa0.numel() > 0:
            alt_dir = cross_ij_ik.repeat(2, 1) * sign[:, None]  # (2Q, 3)
            s.probe_idx.append(
                torch.stack([pa0, pa1, pa2], dim=-1),
            )  # (R, 3) int64
            s.probe_data.append(
                torch.cat(
                    [
                        height.repeat(2)[ok, None],
                        probe_pos[ok],
                        alt_dir[ok],
                    ],
                    dim=-1,
                )
            )  # (R, 7)
            s.access[pa0] = 1
            s.access[pa1] = 1
            s.access[pa2] = 1

    # ── Stage 4: Convex dots ───────────────────────────────────────────

    def _generate_convex_dots(self, s: _State) -> None:
        """Generate convex surface dots for accessible atoms.

        For each atom that is accessible, has buried-eligible attention,
        and has at least one burial neighbour, subdivide the exposed sphere
        via latitude arcs + longitude circles.  Dots are projected to the
        probe-contact surface and collision-checked against neighbours.

        Mutates: ``dot_*`` accumulators.
        """
        dev, rp = s.dev, self.rp
        good_idx = s.good_atoms.nonzero(as_tuple=True)[0]  # (G,)
        conv_atten = s.atten[good_idx]  # (G,)
        conv_access = s.access[good_idx] > 0  # (G,)
        conv_burial_n = s.burial_count[good_idx]  # (G,)
        conv_sel = (
            conv_access & (conv_atten > ATTEN_BLOCKER) & ~((conv_atten == ATTEN_6) & (conv_burial_n == 0))
        )  # (G,)
        conv_atoms = good_idx[conv_sel]  # (C,)
        if conv_atoms.numel() == 0:
            return

        nc = conv_atoms.shape[0]
        c_xyz, c_rad, c_den, c_mol = (
            s.coords[conv_atoms],
            s.radii[conv_atoms],
            s.atom_density[conv_atoms],
            s.mol_id[conv_atoms],
        )  # (C,3), (C,), (C,), (C,)
        c_ext = c_rad + rp  # (C,)

        north = torch.tensor([0.0, 0.0, 1.0], device=dev).repeat(nc, 1)  # (C, 3)
        south = torch.tensor([0.0, 0.0, -1.0], device=dev).repeat(nc, 1)  # (C, 3)
        eq = torch.tensor([1.0, 0.0, 0.0], device=dev).repeat(nc, 1)  # (C, 3)

        has_n = s.neigh_count[conv_atoms] > 0  # (C,)
        n1_xyz = s.neigh_xyz[conv_atoms][:, 0]  # (C, 3)
        n1_rad = s.neigh_rad[conv_atoms][:, 0]  # (C,)
        nv = _normalize(c_xyz - n1_xyz)  # (C, 3)
        north[has_n] = nv[has_n]

        vt = _normalize(
            torch.stack(
                [
                    north[:, 1] ** 2 + north[:, 2] ** 2,
                    north[:, 0] ** 2 + north[:, 2] ** 2,
                    north[:, 0] ** 2 + north[:, 1] ** 2,
                ],
                dim=-1,
            )
        )  # (C, 3)
        vt[_dot(vt, north).abs() > 0.99] = torch.tensor([1.0, 0.0, 0.0], device=dev)
        perp = _cross_norm(north, vt)  # (C, 3)
        eq[has_n] = perp[has_n]

        vql = torch.linalg.cross(eq, north)  # (C, 3)
        n1_ext = n1_rad + rp  # (C,)
        d_to_n1 = (n1_xyz - c_xyz).norm(dim=-1)  # (C,)
        u_to_n1 = (n1_xyz - c_xyz) / d_to_n1[:, None]  # (C, 3)
        asym_c = (c_ext**2 - n1_ext**2) / d_to_n1  # (C,)
        tc_c = (c_xyz + n1_xyz) * 0.5 + u_to_n1 * (asym_c[:, None] * 0.5)  # (C, 3)
        tr_c = (
            0.5
            * ((c_ext + n1_ext) ** 2 - d_to_n1**2).clamp(min=0).sqrt()
            * (d_to_n1**2 - (c_rad - n1_rad) ** 2).clamp(min=0).sqrt()
            / d_to_n1
        )  # (C,)
        sv = ((tc_c + vql * tr_c[:, None]) - c_xyz) / c_ext[:, None]  # (C, 3)
        south[has_n] = sv[has_n]

        arc_sp, lats = _subdivide_arc(
            torch.zeros(nc, 3, device=dev),
            c_rad,
            eq,
            c_den,
            north,
            south,
            self.bf16,
        )  # (C,), (C, M, 3)
        dt = _dot(lats, north[:, None, :])  # (C, M)
        centers = c_xyz[:, None, :] + dt[..., None] * north[:, None, :]  # (C, M, 3)
        lat_rad_sq = c_rad[:, None] ** 2 - dt**2  # (C, M)
        lat_rad = lat_rad_sq.clamp(min=0).sqrt()  # (C, M)
        lat_rad[lat_rad_sq <= 0] = 0.0

        M = lats.shape[1]
        flat_r = lat_rad.reshape(-1)  # (C*M,)
        active_lat = flat_r > 0  # (C*M,)
        if not active_lat.any():
            return

        circ_sp, circ_pts = _subdivide_circle(
            centers.reshape(-1, 3)[active_lat],
            flat_r[active_lat],
            north.repeat_interleave(M, dim=0)[active_lat],
            c_den.repeat_interleave(M)[active_lat],
            self.bf16,
        )  # (...,), (..., K, 3)
        K = circ_pts.shape[1]
        all_pts = torch.full((nc * M, K, 3), float("nan"), device=dev)
        all_pts[active_lat] = circ_pts
        all_sp = torch.zeros(nc * M, device=dev)
        all_sp[active_lat] = circ_sp

        pts_3d = all_pts.reshape(nc, M, K, 3)  # (C, M, K, 3)
        sp_2d = all_sp.reshape(nc, M)  # (C, M)
        valid_pts = ~pts_3d[..., 0].isnan()  # (C, M, K)
        area = sp_2d * arc_sp[:, None]  # (C, M)

        flat_pts = pts_3d[valid_pts]  # (D, 3)
        ai, li, _ = valid_pts.nonzero(as_tuple=True)

        projected = c_xyz[ai] + (flat_pts - c_xyz[ai]) * (c_ext[ai] / c_rad[ai])[:, None]  # (D, 3)
        coll = _collision_skip_first(
            projected[:, None, :],
            s.neigh_xyz[conv_atoms][ai],
            s.neigh_rad[conv_atoms][ai],
            rp,
        )  # (D,)
        surviving = ~coll  # (D,)
        if surviving.any():
            self._write_dots(
                s,
                c_mol[ai[surviving]],
                flat_pts[surviving],
                area[ai[surviving], li[surviving]],
                projected[surviving],
                conv_atoms[ai[surviving]],
            )

    # ── Stage 5: Toroidal dots ─────────────────────────────────────────

    def _generate_toroidal_dots(self, s: _State) -> None:
        """Generate toroidal (saddle) surface dots between atom pairs.

        For each queued ``(atom1, atom2)`` pair, subdivide the torus circle
        and generate dots on both halves of the saddle.  Dots are
        collision-checked against neighbours and split by molecule.

        Mutates: ``access``, ``dot_*`` accumulators.
        """
        if not s.toroid_pairs:
            return

        dev, rp = s.dev, self.rp

        tp = torch.cat(s.toroid_pairs)  # (T, 2)
        tg = torch.cat(s.toroid_geom)  # (T, 9)
        t_a1, t_a2 = tp[:, 0], tp[:, 1]  # (T,) each
        t_uij, t_tij = tg[:, :3], tg[:, 3:6]  # (T, 3) each
        t_rij = tg[:, 6]  # (T,)
        t_between = tg[:, 7] > 0.5  # (T,)

        t_n_xyz, t_n_rad, t_n_idx = (
            s.neigh_xyz[t_a1],
            s.neigh_rad[t_a1],
            s.neigh_idx[t_a1],
        )  # (T,K,3), (T,K), (T,K)
        a1_xyz, a1_rad, a1_den, a1_att, a1_mol = (
            s.coords[t_a1],
            s.radii[t_a1],
            s.atom_density[t_a1],
            s.atten[t_a1],
            s.mol_id[t_a1],
        )  # (T,3), (T,), (T,), (T,), (T,)
        a2_xyz, a2_rad, a2_den, a2_att = (
            s.coords[t_a2],
            s.radii[t_a2],
            s.atom_density[t_a2],
            s.atten[t_a2],
        )  # (T,3), (T,), (T,), (T,)

        tor_density = (a1_den + a2_den) * 0.5  # (T,)
        ext1_t = a1_rad + rp  # (T,)
        ext2_t = a2_rad + rp  # (T,)

        rci = t_rij * a1_rad / ext1_t  # (T,)
        rcj = t_rij * a2_rad / ext2_t  # (T,)
        rs = (rci + 2 * (t_rij - rp).clamp(min=0) + rcj) * 0.25  # (T,)
        eff_density = (rs / t_rij) ** 2 * tor_density  # (T,)

        torus_sp, torus_pts = _subdivide_circle(
            t_tij,
            t_rij,
            t_uij,
            eff_density,
            self.bf16,
        )  # (T,), (T, S, 3)
        valid_tor = ~torus_pts[..., 0].isnan()  # (T, S)
        if not valid_tor.any():
            return

        ti, si = valid_tor.nonzero(as_tuple=True)
        surf_pt = torus_pts[ti, si]  # (V, 3)

        tn_xyz, tn_rad, tn_idx = t_n_xyz[ti], t_n_rad[ti], t_n_idx[ti]  # (V,K,3), (V,K), (V,K)
        exclude_t = (tn_idx < 0) | (tn_idx == t_a2[ti, None])  # (V, K)
        d2_tor = (surf_pt[:, None, :] - tn_xyz).square().sum(-1)  # (V, K)
        too_close = ((d2_tor < (tn_rad + rp) ** 2) & ~exclude_t).any(dim=1)  # (V,)

        ok = ~too_close & ~((a1_att[ti] == ATTEN_6) & (a2_att[ti] == ATTEN_6) & (s.burial_count[t_a1[ti]] == 0))  # (V,)
        if not ok.any():
            return

        surf_pt, ti = surf_pt[ok], ti[ok]
        s.access[t_a1[ti]] = 1
        s.access[t_a2[ti]] = 1

        pi = (a1_xyz[ti] - surf_pt) / ext1_t[ti, None]  # (V', 3)
        pj = (a2_xyz[ti] - surf_pt) / ext2_t[ti, None]  # (V', 3)
        arc_axis = _cross_norm(pi, pj)  # (V', 3)

        dtq = rp**2 - t_rij[ti] ** 2  # (V',)
        is_cusp = (dtq > 0) & t_between[ti]  # (V',)
        mid_dir = _normalize(pi + pj)  # (V', 3)
        cusp_ds = dtq.clamp(min=0).sqrt()  # (V',)
        cusp_val = (t_tij[ti] - t_uij[ti] * cusp_ds[:, None] - surf_pt) / rp  # (V', 3)
        pqi = torch.where(is_cusp[:, None], cusp_val, mid_dir)  # (V', 3)
        pqj = torch.where(
            is_cusp[:, None],
            torch.zeros_like(mid_dir),
            mid_dir,
        )  # (V', 3)

        dt1 = _dot(pqi, pi)  # (V',)
        dt2 = _dot(pqj, pj)  # (V',)
        angle_ok = (dt1.abs() < 1.0) & (dt2.abs() < 1.0)  # (V',)
        if not angle_ok.any():
            return

        surf_pt, arc_axis, ti = surf_pt[angle_ok], arc_axis[angle_ok], ti[angle_ok]
        pi, pj = pi[angle_ok], pj[angle_ok]
        pqi, pqj = pqi[angle_ok], pqj[angle_ok]

        for arc_start, arc_end, idx_src in [
            (pi, pqi, t_a1),
            (pqj, pj, t_a2),
        ]:
            eligible = s.atten[idx_src[ti]] >= 2  # (V'',)
            if not eligible.any():
                continue
            e = eligible
            sp, pts = _subdivide_arc(
                surf_pt[e],
                torch.full((e.sum(),), rp, device=dev),
                arc_axis[e],
                tor_density[ti[e]],
                arc_start[e],
                arc_end[e],
                self.bf16,
            )  # (...,), (..., S, 3)
            dist = _dist_to_line(
                t_tij[ti[e], None, :],
                t_uij[ti[e], None, :],
                pts,
            )  # (..., S)
            pcen = surf_pt[e][:, None, :].expand_as(pts)  # (..., S, 3)
            areas = sp[:, None] * torus_sp[ti[e], None] * dist / t_rij[ti[e], None]  # (..., S)
            v = ~areas.isnan()  # (..., S)
            if v.any():
                self._write_dots(
                    s,
                    a1_mol[ti][e][:, None].expand_as(v)[v],
                    pts[v],
                    areas[v],
                    pcen[v],
                    idx_src[ti][e][:, None].expand_as(v)[v],
                )

    # ── Stage 6: Concave dots ──────────────────────────────────────────

    def _generate_concave_dots(self, s: _State) -> None:
        """Generate concave surface dots inside probe spheres.

        Each probe sits at an atom triplet.  The concave (inward-facing)
        part of the probe sphere is subdivided via latitude arcs + longitude
        circles.  Dots are rejection-tested against half-planes of the
        three generating atoms and collision-checked against nearby low
        probes.

        Mutates: ``dot_*`` accumulators.
        """
        if not s.probe_idx:
            return

        dev, rp = s.dev, self.rp
        idx3 = torch.cat(s.probe_idx)  # (P, 3) int64
        data = torch.cat(s.probe_data)  # (P, 7)
        p_height, p_xyz, p_alt = data[:, 0], data[:, 1:4], data[:, 4:7]  # (P,), (P,3), (P,3)
        P = idx3.shape[0]
        probe_atom_xyz, probe_atom_rad, probe_atom_den = (
            s.coords[idx3],
            s.radii[idx3],
            s.atom_density[idx3],
        )  # (P,3,3), (P,3), (P,3)
        probe_atom_att, probe_atom_mol = s.atten[idx3], s.mol_id[idx3]  # (P,3), (P,3)

        c_den_p = probe_atom_den.float().mean(dim=1)  # (P,)

        skip_all = (probe_atom_att == ATTEN_6).all(dim=1).all()
        if skip_all:
            return

        rp2 = rp * rp
        low = p_height < rp  # (P,)
        low_pts = p_xyz[low]  # (L, 3)
        nears_mask: Tensor | None = None
        if low.any():
            dd = torch.cdist(p_xyz, low_pts).square()  # (P, L)
            nears_mask = dd <= 4 * rp2  # (P, L)
            low_i = low.nonzero(as_tuple=True)[0]
            nears_mask[low_i, torch.arange(low_i.shape[0], device=dev)] = False

        vp = _normalize(probe_atom_xyz - p_xyz[:, None, :])  # (P, 3, 3)
        rejection_planes = _normalize(
            torch.stack(
                [
                    torch.linalg.cross(vp[:, 0], vp[:, 1]),
                    torch.linalg.cross(vp[:, 1], vp[:, 2]),
                    torch.linalg.cross(vp[:, 2], vp[:, 0]),
                ],
                dim=1,
            )
        )  # (P, 3, 3)

        highest = _dot(p_alt[:, None, :], vp).argmax(dim=1)  # (P,)
        south_dir = -p_alt  # (P, 3)
        start_dir = vp[torch.arange(P, device=dev), highest]  # (P, 3)
        ax = _cross_norm(start_dir, south_dir)  # (P, 3)

        cs_c, lats_c = _subdivide_arc(
            torch.zeros_like(p_xyz),
            torch.full((P,), rp, device=dev),
            ax,
            c_den_p,
            start_dir,
            south_dir,
            self.bf16,
        )  # (P,), (P, M, 3)
        lat_proj = _dot(lats_c, south_dir[:, None, :])  # (P, M)
        cen_c = south_dir[:, None, :] * lat_proj[..., None]  # (P, M, 3)
        rad_c = (rp2 - lat_proj**2).clamp(min=0).sqrt()  # (P, M)

        Mc = lats_c.shape[1]
        flat_r_c = rad_c.reshape(-1)  # (P*M,)
        act_c = flat_r_c > 0  # (P*M,)
        if not act_c.any():
            return

        sp_c, pts_c = _subdivide_circle(
            cen_c.reshape(-1, 3)[act_c],
            flat_r_c[act_c],
            south_dir.repeat_interleave(Mc, dim=0)[act_c],
            c_den_p.repeat_interleave(Mc)[act_c],
            self.bf16,
        )  # (...,), (..., K, 3)
        Kc = pts_c.shape[1]
        full_pts = torch.full((P * Mc, Kc, 3), float("nan"), device=dev)
        full_pts[act_c] = pts_c
        full_sp = torch.zeros(P * Mc, device=dev)
        full_sp[act_c] = sp_c

        all_pts_c = full_pts.reshape(P, Mc, Kc, 3)  # (P, M, K, 3)
        area_c = full_sp.reshape(P, Mc) * cs_c[:, None]  # (P, M)

        valid_c = ~all_pts_c[..., 0].isnan()  # (P, M, K)
        ip, il, ik = valid_c.nonzero(as_tuple=True)
        pts = all_pts_c[ip, il, ik]  # (D, 3)

        bail = (_dot(pts[:, None, :], rejection_planes[ip]) >= 0).any(dim=1)  # (D,)
        pts, ip, il = pts[~bail], ip[~bail], il[~bail]
        pts = pts + p_xyz[ip]  # (D', 3)

        if nears_mask is not None and pts.shape[0] > 0:
            near_any = nears_mask[ip].any(dim=1)  # (D',)
            d2_low = torch.cdist(pts, low_pts).square()  # (D', L)
            coll_low = ((d2_low < rp2) & nears_mask[ip]).any(dim=1)  # (D',)
            reject = (p_height[ip] < rp) & near_any & coll_low  # (D',)
            pts, ip, il = pts[~reject], ip[~reject], il[~reject]

        if pts.shape[0] > 0:
            d_to_atoms = (pts[:, None, :] - probe_atom_xyz[ip]).norm(dim=-1) - probe_atom_rad[ip]  # (D'', 3)
            closest = d_to_atoms.argmin(dim=1)  # (D'',)
            self._write_dots(
                s,
                probe_atom_mol[ip, closest],
                pts,
                area_c[ip, il],
                p_xyz[ip],
                idx3[ip, closest],
            )

    # ── Stage 7: CMS scoring ──────────────────────────────────────────

    def _score_cms(
        self,
        s: _State,
        target_side: bool = True,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, Nq]]:
        """Score CMS from accumulated surface dots.

        For each buried dot on the *query* side, find the nearest buried
        dot on the *reference* side and sum
        ``area_q x exp(-weight x nearest_dist^2)``.

        Args:
            s:           Pipeline state with populated dot accumulators.
            target_side: If ``True``, query = target (mol 1) and
                         reference = binder (mol 0).
                         If ``False``, query = binder, reference = target.

        Returns:
            ``(total_cms, per_query_atom_cms)`` with per-atom shape
            ``(Nt,)`` (target side) or ``(Nb,)`` (binder side).
        """
        dev = s.dev

        if target_side:
            q_idx, r_idx = 1, 0
            n_query = s.n - s.n_binder
            owner_offset = s.n_binder
        else:
            q_idx, r_idx = 0, 1
            n_query = s.n_binder
            owner_offset = 0

        q_coor, q_area, q_bur, q_own = (
            _cat_or_empty(s.dot_coords[q_idx], 3, dev),
            _cat_or_empty(s.dot_areas[q_idx], device=dev),
            _cat_or_empty(s.dot_buried[q_idx], device=dev),
            _cat_or_empty(s.dot_owners[q_idx], device=dev),
        )  # (Dq,3), (Dq,), (Dq,), (Dq,)
        r_coor = _cat_or_empty(s.dot_coords[r_idx], 3, dev)  # (Dr, 3)
        r_bur = _cat_or_empty(s.dot_buried[r_idx], device=dev)  # (Dr,)

        zero = torch.tensor(0.0, device=dev)
        zero_per_atom = torch.zeros(n_query, device=dev)

        if r_coor.shape[0] == 0 or q_coor.shape[0] == 0:
            return zero, zero_per_atom

        bq, br = q_bur.bool(), r_bur.bool()
        if not bq.any() or not br.any():
            return zero, zero_per_atom

        buried_q_coor = q_coor[bq]  # (Bq, 3)
        buried_r_coor = r_coor[br]  # (Br, 3)
        buried_q_area = q_area[bq]  # (Bq,)

        nearest_dist = (
            torch.cdist(
                buried_q_coor,
                buried_r_coor,
            )
            .min(dim=1)
            .values
        )  # (Bq,)
        per_dot = buried_q_area * torch.exp(
            -(nearest_dist**2) * self.weight,
        )  # (Bq,)

        local_owner = q_own[bq] - owner_offset  # (Bq,)
        per_atom = torch.zeros(
            n_query,
            device=dev,
            dtype=per_dot.dtype,
        )  # (Nq,)
        per_atom.scatter_add_(0, local_owner, per_dot)

        return per_atom.sum(), per_atom

    def _compute_max_area(
        self,
        s: _State,
    ) -> tuple[Float[Tensor, ""], Float[Tensor, N]]:
        """Sum all dot areas for molecule 0 (no burial/distance weighting).

        Used by ``compute_max_cms`` to get total Connolly surface area.
        """
        dev = s.dev
        all_areas = _cat_or_empty(s.dot_areas[0], device=dev)  # (D,)
        all_owners = _cat_or_empty(s.dot_owners[0], device=dev)  # (D,)

        if all_areas.shape[0] == 0:
            return torch.tensor(0.0, device=dev), torch.zeros(s.n, device=dev)

        per_atom = torch.zeros(s.n, device=dev, dtype=all_areas.dtype)  # (N,)
        per_atom.scatter_add_(0, all_owners, all_areas)
        return per_atom.sum(), per_atom

    # ── Dot writing helper ─────────────────────────────────────────────

    def _write_dots(
        self,
        s: _State,
        mol: Int[Tensor, D],
        coor: Float[Tensor, "D 3"],
        area: Float[Tensor, D],
        pcen: Float[Tensor, "D 3"],
        owners: Int[Tensor, D],
    ) -> None:
        """Compute burial flags and accumulate surface dots by molecule.

        A dot is *buried* if any cross-molecule burial neighbour's extended
        sphere (radius + probe_radius) covers the dot's probe-contact point.

        Args:
            s:      Pipeline state.
            mol:    Molecule assignment per dot.  ``(D,)``
            coor:   Surface point coordinates.     ``(D, 3)``
            area:   Area contribution per dot.     ``(D,)``
            pcen:   Probe-contact point.            ``(D, 3)``
            owners: Generating-atom index per dot. ``(D,)``
        """
        if coor.shape[0] == 0:
            return
        d2 = (s.burial_xyz[owners] - pcen[:, None, :]).square().sum(-1)  # (D, Kb)
        is_buried = (d2 <= (s.burial_rad[owners] + self.rp) ** 2).any(dim=-1)  # (D,)
        for m in (0, 1):
            sel = mol == m  # (D,)
            if sel.any():
                s.dot_coords[m].append(coor[sel])
                s.dot_areas[m].append(area[sel])
                s.dot_buried[m].append(is_buried[sel])
                s.dot_owners[m].append(owners[sel])
