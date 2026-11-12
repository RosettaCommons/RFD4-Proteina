"""3D chiral point-group symmetry definitions and exact SO(3) constructors."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import torch
from torch import Tensor

GroupFamily = Literal["C", "D", "T", "O", "I"]
DecodeMode = Literal["asu_medoid", "asu_first", "chordal_mean"]

VALID_DECODE_MODES: tuple[str, ...] = ("asu_medoid", "asu_first", "chordal_mean")
_CYCLIC_RE = re.compile(r"^C(\d+)$", re.IGNORECASE)
_DIHEDRAL_RE = re.compile(r"^D(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SymmetrySpec:
    """Canonical 3D point-group specification in SO(3).

    Attributes:
        name: Canonical uppercase group symbol (e.g. ``"C1"``, ``"C3"``, ``"D2"``, ``"T"``, ``"O"``, ``"I"``).
        family: Point-group family identifier (``"C"``, ``"D"``, ``"T"``, ``"O"``, or ``"I"``).
        order: Group order ``K = |G| >= 1``.
        rotations_f64: Canonical CPU ``float64`` rotation matrices of shape ``[K, 3, 3]`` with ``R[0] = I_3``.
        radial_dir_f64: Collision-free CPU ``float64`` unit vector of shape ``[3]`` for ASU radial placement.
    """

    name: str
    family: GroupFamily
    order: int
    rotations_f64: Tensor
    radial_dir_f64: Tensor

    @property
    def is_nontrivial(self) -> bool:
        """Return True iff the group order exceeds 1."""
        return self.order > 1

    def rotations(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return the ``[K, 3, 3]`` SO(3) rotation matrices on the target device and dtype."""
        return self.rotations_f64.to(device=device, dtype=dtype)

    def radial_dir(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return the ``[3]`` collision-free unit ASU radial offset vector on target device and dtype."""
        return self.radial_dir_f64.to(device=device, dtype=dtype)

    def reynolds_operator(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return the ``[3, 3]`` Reynolds projection ``(1 / K) * sum_g R_g`` onto Fix(G)."""
        proj = self.rotations_f64.mean(dim=0)
        return proj.to(device=device, dtype=dtype)

    def cayley_table(self, atol: float = 1e-8) -> Tensor:
        """Compute and validate the ``[K, K]`` Cayley multiplication table ``T[i, j] = k`` where ``R_i R_j = R_k``."""
        rot = self.rotations_f64
        products = torch.einsum("iaf,jfb->ijab", rot, rot)  # [K, K, 3, 3]
        diffs = torch.amax(torch.abs(products.unsqueeze(2) - rot.view(1, 1, self.order, 3, 3)), dim=(-2, -1))
        min_err, indices = torch.min(diffs, dim=-1)
        if bool(torch.any(min_err > atol)):
            max_violation = float(min_err.max().item())
            raise ValueError(f"Group closure violated for {self.name}: max error {max_violation:.3e} > {atol:.3e}")
        return indices.to(dtype=torch.long)

    def inverse_indices(self, atol: float = 1e-8) -> Tensor:
        """Compute and validate ``[K]`` indices ``inv[i] = j`` where ``R_i^T = R_j``."""
        rot = self.rotations_f64
        transposed = rot.transpose(-1, -2)  # [K, 3, 3]
        diffs = torch.amax(torch.abs(transposed.unsqueeze(1) - rot.unsqueeze(0)), dim=(-2, -1))
        min_err, indices = torch.min(diffs, dim=-1)
        if bool(torch.any(min_err > atol)):
            max_violation = float(min_err.max().item())
            raise ValueError(f"Group inverse violated for {self.name}: max error {max_violation:.3e} > {atol:.3e}")
        return indices.to(dtype=torch.long)


@dataclass(frozen=True)
class SymmetryConfig:
    """Configuration for point-group symmetry generation and ASU expansion.

    Attributes:
        group: Point-group name (e.g. ``"C3"``, ``"D2"``, ``"T"``, ``"O"``, ``"I"``), ``"C1"``, or ``None``.
        radius: Optional initial ASU radial offset in Angstroms (``>= 0.0``).
        input_is_asu: If True, input structure/contig specifies a single ASU to be expanded ``K`` times.
        tie_sequence: If True, tie homomeric sequence identity and logits across symmetric copies.
        decode_mode: All-atom AE decode symmetrization mode (``"asu_medoid"``, ``"asu_first"``, or ``"chordal_mean"``).
    """

    group: str | None = None
    radius: float = 0.0
    input_is_asu: bool = True
    tie_sequence: bool = True
    decode_mode: str = "asu_medoid"

    def __post_init__(self) -> None:
        if self.group is None:
            norm_group: str | None = None
        elif isinstance(self.group, str):
            stripped = self.group.strip()
            if not stripped or stripped.lower() in ("none", "null", "false"):
                norm_group = None
            else:
                norm_group = parse_symmetry(stripped).name
        else:
            raise ValueError(f"SymmetryConfig.group must be str or None, got {type(self.group).__name__}")

        rad = float(self.radius)
        if not math.isfinite(rad) or rad < 0.0:
            raise ValueError(f"SymmetryConfig.radius must be finite and >= 0.0, got {self.radius!r}")

        mode = str(self.decode_mode).strip().lower()
        if mode not in VALID_DECODE_MODES:
            raise ValueError(f"Invalid decode_mode {self.decode_mode!r}; expected one of {VALID_DECODE_MODES}")

        object.__setattr__(self, "group", norm_group)
        object.__setattr__(self, "radius", rad)
        object.__setattr__(self, "input_is_asu", bool(self.input_is_asu))
        object.__setattr__(self, "tie_sequence", bool(self.tie_sequence))
        object.__setattr__(self, "decode_mode", mode)

    @property
    def spec(self) -> SymmetrySpec:
        """Return the parsed :class:`SymmetrySpec` (``C1`` when ``group`` is ``None``)."""
        return parse_symmetry(self.group)

    @property
    def is_active(self) -> bool:
        """Return True iff nontrivial point-group symmetry (``order > 1``) is enabled."""
        return self.spec.is_nontrivial

    @property
    def order(self) -> int:
        """Return the point-group order ``K = |G|``."""
        return self.spec.order

    @property
    def radius_nm(self) -> float:
        """Return the ASU radial offset converted from Angstroms to nanometers (``radius / 10.0``)."""
        return self.radius / 10.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain JSON/YAML-compatible dictionary."""
        return {
            "group": self.group,
            "order": self.order,
            "radius": self.radius,
            "input_is_asu": self.input_is_asu,
            "tie_sequence": self.tie_sequence,
            "decode_mode": self.decode_mode,
        }

    @classmethod
    def from_any(
        cls,
        val: Any,
        *,
        default_radius: float | None = None,
    ) -> SymmetryConfig:
        """Normalize ``None``, ``str``, ``SymmetrySpec``, ``SymmetryConfig``, or ``Mapping`` to ``SymmetryConfig``."""
        fallback_radius = 0.0 if default_radius is None else float(default_radius)

        if val is None or val is False:
            return cls(group=None, radius=fallback_radius)

        if isinstance(val, cls):
            if default_radius is not None and val.radius == 0.0 and fallback_radius != 0.0:
                return cls(
                    group=val.group,
                    radius=fallback_radius,
                    input_is_asu=val.input_is_asu,
                    tie_sequence=val.tie_sequence,
                    decode_mode=val.decode_mode,
                )
            return val

        if isinstance(val, SymmetrySpec):
            return cls(group=val.name, radius=fallback_radius)

        if isinstance(val, str):
            return cls(group=val, radius=fallback_radius)

        if isinstance(val, Mapping):
            raw_group = val.get("group")
            if raw_group is None and "symmetry" in val:
                nested = val["symmetry"]
                if isinstance(nested, Mapping):
                    return cls.from_any(
                        nested,
                        default_radius=val.get("sym_radius", default_radius),
                    )
                raw_group = nested

            if "radius" in val and val["radius"] is not None:
                raw_radius = float(val["radius"])
            elif "sym_radius" in val and val["sym_radius"] is not None:
                raw_radius = float(val["sym_radius"])
            elif "radius_nm" in val and val["radius_nm"] is not None:
                raw_radius = float(val["radius_nm"]) * 10.0
            else:
                raw_radius = fallback_radius

            raw_input_is_asu = val.get("input_is_asu", True)
            raw_tie_seq = val.get("tie_sequence", True)
            raw_decode_mode = val.get("decode_mode", "asu_medoid")
            return cls(
                group=raw_group,
                radius=raw_radius,
                input_is_asu=True if raw_input_is_asu is None else bool(raw_input_is_asu),
                tie_sequence=True if raw_tie_seq is None else bool(raw_tie_seq),
                decode_mode="asu_medoid" if raw_decode_mode is None else str(raw_decode_mode),
            )

        raise TypeError(f"Cannot construct SymmetryConfig from {type(val).__name__}: {val!r}")


def _rot_z_f64(theta: float) -> Tensor:
    """Construct an exact float64 right-handed rotation matrix around +z by angle ``theta``."""
    if theta == 0.0:
        return torch.eye(3, dtype=torch.float64)
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )


def _build_cyclic_f64(n: int) -> Tensor:
    """Construct the ``[n, 3, 3]`` float64 rotation matrices of ``C_n`` around the +z axis."""
    mats = [_rot_z_f64(2.0 * math.pi * k / n) for k in range(n)]
    return torch.stack(mats, dim=0)


def _build_dihedral_f64(n: int) -> Tensor:
    """Construct the ``[2*n, 3, 3]`` float64 rotation matrices of ``D_n`` (+z principal, +x C2)."""
    cn = _build_cyclic_f64(n)
    c2x = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))
    secondary = torch.matmul(cn, c2x.unsqueeze(0))
    return torch.cat([cn, secondary], dim=0)


def _signed_perm_matrix_f64(signs: tuple[float, float, float], perm: tuple[int, int, int]) -> Tensor:
    """Construct a 3x3 signed permutation matrix ``(M x)_i = signs[i] * x[perm[i]]``."""
    mat = torch.zeros((3, 3), dtype=torch.float64)
    for row_idx in range(3):
        mat[row_idx, perm[row_idx]] = signs[row_idx]
    return mat


def _build_tetrahedral_f64() -> Tensor:
    """Construct the ``[12, 3, 3]`` exact signed permutation matrices of chiral group ``T``."""
    even_perms: list[tuple[int, int, int]] = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    even_signs: list[tuple[float, float, float]] = [
        (1.0, 1.0, 1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
    ]
    mats = [_signed_perm_matrix_f64(s, p) for s in even_signs for p in even_perms]
    return torch.stack(mats, dim=0)


def _build_octahedral_f64() -> Tensor:
    """Construct the ``[24, 3, 3]`` exact signed permutation matrices of chiral group ``O``."""
    tet = _build_tetrahedral_f64()
    odd_perms: list[tuple[int, int, int]] = [(0, 2, 1), (1, 0, 2), (2, 1, 0)]
    odd_signs: list[tuple[float, float, float]] = [
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, -1.0),
        (-1.0, -1.0, -1.0),
    ]
    odd_mats = torch.stack([_signed_perm_matrix_f64(s, p) for s in odd_signs for p in odd_perms], dim=0)
    return torch.cat([tet, odd_mats], dim=0)


def _build_icosahedral_f64() -> Tensor:
    """Construct the ``[60, 3, 3]`` float64 rotation matrices of chiral group ``I = T * <R_5>``.

    Preserves the golden-box regular icosahedron with vertices ``cyc(0, +-1, +-phi)``.
    Because ``|T| = 12``, ``|<R_5>| = 5``, and ``gcd(12, 5) = 1``, the 60 coset products
    ``g @ R_5^k`` are guaranteed pairwise distinct with ``R[0] = I_3``.
    """
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    inv_phi = phi - 1.0  # Exact 1 / phi

    r5_1 = 0.5 * torch.tensor(
        [
            [inv_phi, -phi, 1.0],
            [phi, 1.0, inv_phi],
            [-1.0, inv_phi, phi],
        ],
        dtype=torch.float64,
    )
    r5_2 = 0.5 * torch.tensor(
        [
            [-phi, -1.0, inv_phi],
            [1.0, -inv_phi, phi],
            [-inv_phi, phi, 1.0],
        ],
        dtype=torch.float64,
    )
    c5_powers = [
        torch.eye(3, dtype=torch.float64),
        r5_1,
        r5_2,
        r5_2.T.clone(),
        r5_1.T.clone(),
    ]
    tet = _build_tetrahedral_f64()
    cosets = [torch.matmul(tet, r5_k.unsqueeze(0)) for r5_k in c5_powers]
    return torch.cat(cosets, dim=0)


def _build_radial_dir_f64(family: GroupFamily, n: int, rotations_f64: Tensor) -> Tensor:
    """Construct a deterministic unit vector avoiding all non-identity rotation axes of G."""
    if rotations_f64.shape[0] == 1 or family == "C":
        u = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    elif family == "D":
        angle = math.pi / (2.0 * n)
        raw = torch.tensor([math.cos(angle), math.sin(angle), 0.5], dtype=torch.float64)
        u = raw / torch.linalg.norm(raw)
    else:
        raw = torch.tensor([1.0, 2.0, 3.5], dtype=torch.float64)
        u = raw / torch.linalg.norm(raw)

    if rotations_f64.shape[0] > 1:
        displaced = torch.einsum("kij,j->ki", rotations_f64[1:], u) - u.unsqueeze(0)
        min_sep = float(torch.linalg.norm(displaced, dim=-1).min().item())
        if min_sep <= 1e-4:
            raise RuntimeError(f"Internal error: radial direction lies near a symmetry axis (min_sep={min_sep:.3e})")
    return u


@lru_cache(maxsize=128)
def _cached_parse_symmetry(norm_key: str) -> SymmetrySpec:
    """Build and cache canonical :class:`SymmetrySpec` instances keyed by uppercase group symbol."""
    if norm_key == "C1":
        rot = torch.eye(3, dtype=torch.float64).unsqueeze(0)
        rad = _build_radial_dir_f64("C", 1, rot)
        return SymmetrySpec(name="C1", family="C", order=1, rotations_f64=rot, radial_dir_f64=rad)

    if norm_key == "T":
        rot = _build_tetrahedral_f64()
        rad = _build_radial_dir_f64("T", 0, rot)
        return SymmetrySpec(name="T", family="T", order=12, rotations_f64=rot, radial_dir_f64=rad)

    if norm_key == "O":
        rot = _build_octahedral_f64()
        rad = _build_radial_dir_f64("O", 0, rot)
        return SymmetrySpec(name="O", family="O", order=24, rotations_f64=rot, radial_dir_f64=rad)

    if norm_key == "I":
        rot = _build_icosahedral_f64()
        rad = _build_radial_dir_f64("I", 0, rot)
        return SymmetrySpec(name="I", family="I", order=60, rotations_f64=rot, radial_dir_f64=rad)

    m_c = _CYCLIC_RE.match(norm_key)
    if m_c is not None:
        n = int(m_c.group(1))
        if n < 2:
            raise ValueError(f"Cyclic symmetry order must be >= 1 (got {norm_key!r})")
        rot = _build_cyclic_f64(n)
        rad = _build_radial_dir_f64("C", n, rot)
        return SymmetrySpec(name=f"C{n}", family="C", order=n, rotations_f64=rot, radial_dir_f64=rad)

    m_d = _DIHEDRAL_RE.match(norm_key)
    if m_d is not None:
        n = int(m_d.group(1))
        if n < 2:
            raise ValueError(f"Dihedral symmetry requires n >= 2 (got {norm_key!r}; use 'C2' instead of 'D1')")
        rot = _build_dihedral_f64(n)
        rad = _build_radial_dir_f64("D", n, rot)
        return SymmetrySpec(name=f"D{n}", family="D", order=2 * n, rotations_f64=rot, radial_dir_f64=rad)

    raise ValueError(
        f"Unsupported point-group symmetry specification {norm_key!r}. "
        "Expected 'C1', 'C<n>' (n >= 2), 'D<n>' (n >= 2), 'T', 'O', 'I', or None."
    )


def parse_symmetry(spec: str | SymmetrySpec | SymmetryConfig | None) -> SymmetrySpec:
    """Parse a case-insensitive point-group specification into a canonical :class:`SymmetrySpec`.

    Supported groups:
        - ``None``, ``""``, ``"none"``, ``"C1"`` -> trivial identity group ``C1`` (order 1)
        - ``"C<n>"`` (``n >= 2``) -> cyclic group ``C_n`` (order ``n``)
        - ``"D<n>"`` (``n >= 2``) -> dihedral group ``D_n`` (order ``2n``)
        - ``"T"`` -> chiral tetrahedral group (order 12)
        - ``"O"`` -> chiral octahedral group (order 24)
        - ``"I"`` -> chiral icosahedral group (order 60)
    """
    if isinstance(spec, SymmetrySpec):
        return spec
    if isinstance(spec, SymmetryConfig):
        return _cached_parse_symmetry("C1" if spec.group is None else spec.group)
    if spec is None:
        return _cached_parse_symmetry("C1")
    if not isinstance(spec, str):
        raise ValueError(f"Symmetry specification must be str, SymmetrySpec, SymmetryConfig, or None; got {spec!r}")

    cleaned = spec.strip()
    if not cleaned or cleaned.lower() == "none":
        return _cached_parse_symmetry("C1")

    upper = cleaned.upper()
    if upper == "C1":
        return _cached_parse_symmetry("C1")
    return _cached_parse_symmetry(upper)


def chain_index_to_id(index: int) -> str:
    """Convert a 0-based integer chain index into an mmCIF base-26 label (``A..Z``, ``AA..ZZ``, ``AAA...``)."""
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"Chain index must be a non-negative integer, got {index!r}")
    chars: list[str] = []
    n = index
    while True:
        n, rem = divmod(n, 26)
        chars.append(chr(ord("A") + rem))
        if n == 0:
            break
        n -= 1
    return "".join(reversed(chars))


def get_symmetry_chain_ids(
    order_or_spec: int | str | SymmetrySpec | SymmetryConfig | None,
    chains_per_asu: int = 1,
) -> list[str]:
    """Return ``total_chains = K * chains_per_asu`` unique base-26 mmCIF chain IDs (``A..Z, AA..BH, ...``).

    Args:
        order_or_spec: Either a positive integer chain/subunit count ``K``, or a symmetry spec/config/name.
        chains_per_asu: Number of chains per asymmetric unit (default ``1``).
    """
    if chains_per_asu < 1:
        raise ValueError(f"chains_per_asu must be >= 1, got {chains_per_asu}")
    if isinstance(order_or_spec, int) and not isinstance(order_or_spec, bool):
        if order_or_spec < 1:
            raise ValueError(f"Chain count / symmetry order must be >= 1, got {order_or_spec}")
        order = order_or_spec
    else:
        order = parse_symmetry(order_or_spec).order

    total_chains = order * chains_per_asu
    return [chain_index_to_id(i) for i in range(total_chains)]
