"""Pure geometry functions for molecular surface dot generation.

All functions are stateless and operate on numpy arrays. They implement
the subdivision of arcs and circles used to tile the molecular surface
with dots, plus collision-detection helpers.
"""

from __future__ import annotations

import math

import numpy as np
from jaxtyping import Bool, Float

MAX_SUBDIV = 100
_PARALLEL_THRESH = 0.99


def subdivide_arc(
    center: Float[np.ndarray, "* 3"],
    radius: Float[np.ndarray, "*"],
    axis: Float[np.ndarray, "* 3"],
    density: Float[np.ndarray, "*"],
    start: Float[np.ndarray, "* 3"],
    end: Float[np.ndarray, "* 3"],
) -> tuple[Float[np.ndarray, "*"], Float[np.ndarray, "* K 3"]]:
    """Subdivide an arc from *start* to *end* around *axis*.

    Args:
        center: Arc centres, shape ``(*, 3)``.
        radius: Arc radii, shape ``(*,)``.
        axis: Rotation axes, shape ``(*, 3)``.
        density: Dot density per unit area, shape ``(*,)``.
        start: Start direction vectors, shape ``(*, 3)``.
        end: End direction vectors, shape ``(*, 3)``.

    Returns:
        ``(arc_spacing, points)`` where *arc_spacing* has shape ``(*,)``
        and *points* has shape ``(*, K, 3)`` with ``K <= MAX_SUBDIV``.
        Invalid entries are ``NaN``.
    """
    y = np.cross(axis, start)
    dt1 = np.sum(end * start, axis=-1)
    dt2 = np.sum(end * y, axis=-1)

    angle = np.arctan2(dt2, dt1)
    angle = np.where(angle < 0.0, angle + 2 * np.pi, angle)
    angle = np.where(np.isclose(dt1, 0) & np.isclose(dt2, 0), np.nan, angle)

    return _subdivide_angle(center, radius, start, y, angle, density)


def subdivide_circle(
    center: Float[np.ndarray, "* 3"],
    radius: Float[np.ndarray, "*"],
    axis: Float[np.ndarray, "* 3"],
    density: Float[np.ndarray, "*"],
) -> tuple[Float[np.ndarray, "*"], Float[np.ndarray, "* K 3"]]:
    """Subdivide a full circle perpendicular to *axis*.

    Args:
        center: Circle centres, shape ``(*, 3)``.
        radius: Circle radii, shape ``(*,)``.
        axis: Normal axes, shape ``(*, 3)``.
        density: Dot density, shape ``(*,)``.

    Returns:
        ``(arc_spacing, points)`` — same layout as :func:`subdivide_arc`.
    """
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)

    v1 = np.stack(
        [
            axis[..., 1] ** 2 + axis[..., 2] ** 2,
            axis[..., 0] ** 2 + axis[..., 2] ** 2,
            axis[..., 0] ** 2 + axis[..., 1] ** 2,
        ],
        axis=-1,
    )
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)

    dt = np.sum(v1 * axis, axis=-1)
    replace = np.abs(dt) > _PARALLEL_THRESH
    v1 = np.where(replace[..., None], np.array([1.0, 0.0, 0.0]), v1)

    v2 = np.cross(axis, v1)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

    x = np.cross(axis, v2)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)

    y = np.cross(axis, x)

    angle = np.full_like(radius, 2 * np.pi)
    return _subdivide_angle(center, radius, x, y, angle, density)


def _subdivide_angle(
    center: Float[np.ndarray, "* 3"],
    radius: Float[np.ndarray, "*"],
    x: Float[np.ndarray, "* 3"],
    y: Float[np.ndarray, "* 3"],
    angle: Float[np.ndarray, "*"],
    density: Float[np.ndarray, "*"],
) -> tuple[Float[np.ndarray, "*"], Float[np.ndarray, "* K 3"]]:
    """Core subdivision: evenly space dots along an angular sweep.

    Args:
        center: Shape ``(*, 3)``.
        radius: Shape ``(*,)``.
        x, y: Orthonormal basis vectors in the arc plane, ``(*, 3)``.
        angle: Total sweep angle in radians, ``(*,)``.
        density: Dot density, ``(*,)``.

    Returns:
        ``(arc_spacing, points)`` where *arc_spacing* is ``(*,)`` and
        *points* is ``(*, K, 3)`` with ``NaN`` for invalid entries.
    """
    radius = np.asarray(radius)
    density = np.asarray(density)
    angle = np.asarray(angle)
    base_shape = radius.shape

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 1.0 / (np.sqrt(density) * radius)
        raw = angle / delta

    valid_raw = np.isfinite(raw) & (raw > 0)
    if not valid_raw.any():
        return np.zeros(base_shape), np.full(base_shape + (0, 3), np.nan)

    max_count = int(min(MAX_SUBDIV, math.ceil(float(raw[valid_raw].max()))))
    if max_count == 0:
        return np.zeros(base_shape), np.full(base_shape + (0, 3), np.nan)

    i = np.arange(max_count)
    a = delta[..., None] * (i + 0.5)
    mask = a <= angle[..., None]

    cos_a = np.cos(a)
    sin_a = np.sin(a)

    points = (
        center[..., None, :]
        + x[..., None, :] * (radius[..., None] * cos_a)[..., None]
        + y[..., None, :] * (radius[..., None] * sin_a)[..., None]
    )
    points[~mask] = np.nan

    counts = mask.sum(axis=-1)
    arc_spacing = np.where(
        counts > 0,
        radius * angle / counts.clip(min=1),
        0.0,
    )
    return arc_spacing, points


def distance_points_to_line(
    center: Float[np.ndarray, "* 3"],
    axis: Float[np.ndarray, "* 3"],
    points: Float[np.ndarray, "* 3"],
) -> Float[np.ndarray, "*"]:
    """Perpendicular distance from *points* to a line through *center*.

    Args:
        center: Line origin, shape ``(*, 3)``.
        axis: Line direction (unit vector), shape ``(*, 3)``.
        points: Query points, shape ``(*, 3)``.

    Returns:
        Distances, shape ``(*,)``.
    """
    vec = points - center
    dt = (vec * axis).sum(axis=-1)
    d2 = np.square(vec).sum(axis=-1) - dt * dt
    return np.where(d2 < 0.0, 0.0, np.sqrt(d2))


def check_point_collisions(
    points: Float[np.ndarray, "D 1 3"],
    neighbor_xyz: Float[np.ndarray, "D K 3"],
    neighbor_radii: Float[np.ndarray, "D K"],
    probe_radius: float,
) -> Bool[np.ndarray, "D"]:
    """Check whether projected surface points collide with neighbor atoms.

    Skips the first neighbor (matching the C++ convention where
    ``begin()+1`` is used).

    Args:
        points: Query points, shape ``(D, 1, 3)``.
        neighbor_xyz: Neighbor coordinates, shape ``(D, K, 3)``.
        neighbor_radii: Neighbor radii, shape ``(D, K)``.
        probe_radius: Probe sphere radius.

    Returns:
        Boolean array of shape ``(D,)``. ``True`` = collision.
    """
    dists2 = np.square(neighbor_xyz - points).sum(axis=-1)
    collision = (dists2 <= (neighbor_radii + probe_radius) ** 2) & ~np.isnan(
        neighbor_radii
    )
    return collision[..., 1:].any(axis=-1)


def check_probe_collisions(
    points: Float[np.ndarray, "D 3"],
    near_points: Float[np.ndarray, "L 3"],
    near_mask: Bool[np.ndarray, "D L"],
    r_squared: float,
) -> Bool[np.ndarray, "D"]:
    """Check whether surface points collide with nearby low probes.

    Args:
        points: Surface dot coordinates, shape ``(D, 3)``.
        near_points: Low-probe positions, shape ``(L, 3)``.
        near_mask: Boolean mask ``(D, L)`` of which pairs to check.
        r_squared: Collision threshold (squared distance).

    Returns:
        Boolean array of shape ``(D,)``. ``True`` = collision.
    """
    n = points.shape[0]
    rows, cols = np.where(near_mask)
    if rows.size == 0:
        return np.zeros(n, dtype=bool)

    diff = points[rows] - near_points[cols]
    d2 = np.einsum("ij,ij->i", diff, diff)
    colliding = d2 < r_squared

    if not np.any(colliding):
        return np.zeros(n, dtype=bool)

    result = np.zeros(n, dtype=bool)
    result[rows[colliding]] = True
    return result
