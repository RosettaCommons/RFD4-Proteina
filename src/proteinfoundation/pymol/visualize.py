"""Custom PyMOL view functions for proteinfoundation conditions.

Provides color-by-group views for :py:data:`~proteinfoundation.conditions.C_CRDGRP`
and :py:data:`~proteinfoundation.conditions.C_IDXGRP`, registered as
entries in :py:mod:`atomworks.pymol_plugin.views`.

Only imported inside a running PyMOL process.
"""

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from pymol import cmd

from proteinfoundation.conditions import C_CRDGRP, C_IDXGRP

# Colours for named CoordinateGroup constants and cycling relative groups
_CRDGRP_NAMED_COLORS = {
    -1: "gray60",  # UNKNOWN_COORDS
    0: "lightblue",  # KNOWN_ABSOLUTE_COORDS
}
_PALETTE = ["red", "green", "orange", "purple", "cyan", "yellow", "magenta", "salmon"]


def color_by_coord_group(arr: AtomArrayPlus, obj_name: str) -> None:
    """Color *obj_name* by :py:data:`~proteinfoundation.conditions.C_CRDGRP`.

    No-op if the annotation is not present on *arr*.
    """
    groups = C_CRDGRP.annotation(arr, default="generate")
    _color_by_group(arr, obj_name, groups, named_colors=_CRDGRP_NAMED_COLORS)


def color_by_index_group(arr: AtomArrayPlus, obj_name: str) -> None:
    """Color *obj_name* by :py:data:`~proteinfoundation.conditions.C_IDXGRP`.

    Always generates a default if not explicitly annotated, since
    :py:class:`~proteinfoundation.conditions.IndexGroup` is derived from
    chain structure and is always well-defined.
    """
    groups = C_IDXGRP.annotation(arr, default="generate")
    _color_by_group(arr, obj_name, groups, named_colors={})


def _color_by_group(
    arr: AtomArrayPlus,
    obj_name: str,
    groups: np.ndarray,
    named_colors: dict[int, str],
) -> None:
    """Shared implementation: write group ids into b-factor, then color per group."""
    lookup: dict[tuple[str, str, str], int] = {}
    for i in range(len(arr)):
        key = (arr.chain_iid[i], str(arr.res_id[i]), arr.atom_name[i])
        lookup[key] = int(groups[i])

    def _get_group(chain: str, resi: str, name: str) -> int:
        resi_clean = resi.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") or "0"
        return lookup.get((chain, resi_clean, name), -999)

    cmd.alter(obj_name, "b = _g(chain, resi, name)", space={"_g": _get_group})

    unique_groups = sorted({int(g) for g in groups})
    cycle_idx = 0
    for group_val in unique_groups:
        color = named_colors.get(group_val)
        if color is None:
            color = _PALETTE[cycle_idx % len(_PALETTE)]
            cycle_idx += 1
        cmd.color(color, f"{obj_name} and b={group_val}")
