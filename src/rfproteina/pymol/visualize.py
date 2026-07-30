"""Custom PyMOL view functions for rfproteina conditions.

Provides color-by-group views for :py:data:`~rfproteina.conditions.C_CRDGRP`
and :py:data:`~rfproteina.conditions.C_IDXGRP`, registered as
entries in :py:mod:`atomworks.pymol_plugin.views`.

Only imported inside a running PyMOL process.
"""

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from pymol import cmd

from rfproteina.conditions import C_CRDGRP, C_IDXGRP

# Colours for named CoordinateGroup constants and cycling relative groups
_CRDGRP_NAMED_COLORS = {
    -1: "gray60",  # UNKNOWN_COORDS
    0: "lightblue",  # KNOWN_ABSOLUTE_COORDS
}
# 20 maximally-distinct group colors (based on Trubetskoy's distinct-color set),
# registered as custom PyMOL colors so their appearance is consistent
_PALETTE_RGB: list[tuple[str, tuple[float, float, float]]] = [
    ("aw_grp_00", (0.902, 0.098, 0.294)),  # red
    ("aw_grp_01", (0.235, 0.706, 0.294)),  # green
    ("aw_grp_02", (0.000, 0.000, 0.459)),  # navy
    ("aw_grp_03", (0.259, 0.831, 0.957)),  # cyan
    ("aw_grp_04", (1.000, 0.847, 0.694)),  # apricot
    ("aw_grp_05", (0.863, 0.745, 1.000)),  # lavender
    ("aw_grp_06", (0.604, 0.388, 0.141)),  # brown
    ("aw_grp_07", (0.263, 0.388, 0.847)),  # blue
    ("aw_grp_08", (0.941, 0.196, 0.902)),  # magenta
    ("aw_grp_09", (0.502, 0.000, 0.000)),  # maroon
    ("aw_grp_10", (0.275, 0.600, 0.565)),  # teal
    ("aw_grp_11", (0.502, 0.502, 0.000)),  # olive
    ("aw_grp_12", (0.667, 1.000, 0.765)),  # mint
    ("aw_grp_13", (0.663, 0.663, 0.663)),  # grey
    ("aw_grp_14", (1.000, 0.882, 0.098)),  # yellow
    ("aw_grp_15", (0.569, 0.118, 0.706)),  # purple
    ("aw_grp_16", (0.961, 0.510, 0.192)),  # orange
    ("aw_grp_17", (0.749, 0.937, 0.271)),  # lime
    ("aw_grp_18", (0.980, 0.745, 0.831)),  # pink
    ("aw_grp_19", (1.000, 0.980, 0.784)),  # beige
]
_PALETTE = [name for name, _ in _PALETTE_RGB]


def _ensure_palette_registered() -> None:
    """Register the custom group colors with PyMOL (idempotent)."""
    for name, rgb in _PALETTE_RGB:
        cmd.set_color(name, list(rgb))


def color_by_coord_group(arr: AtomArrayPlus, obj_name: str) -> None:
    """Color *obj_name* by :py:data:`~rfproteina.conditions.C_CRDGRP`.

    No-op if the annotation is not present on *arr*.
    """
    groups = C_CRDGRP.annotation(arr, default="generate")
    _color_by_group(arr, obj_name, groups, named_colors=_CRDGRP_NAMED_COLORS)


def color_by_index_group(arr: AtomArrayPlus, obj_name: str) -> None:
    """Color *obj_name* by :py:data:`~rfproteina.conditions.C_IDXGRP`.

    Always generates a default if not explicitly annotated, since
    :py:class:`~rfproteina.conditions.IndexGroup` is derived from
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
    _ensure_palette_registered()

    unique_groups = sorted({int(g) for g in groups})
    cycled = [g for g in unique_groups if g not in named_colors]
    if len(cycled) > len(_PALETTE):
        cmd.color("grey50", obj_name)
        _warn_too_many_groups(len(cycled), len(_PALETTE))
        return

    color_for: dict[int, str] = dict(named_colors)
    for palette_color, group_val in zip(_PALETTE, cycled, strict=False):
        color_for[group_val] = palette_color

    lookup: dict[tuple[str, str, str], int] = {}
    for i in range(len(arr)):
        key = (arr.chain_iid[i], str(arr.res_id[i]), arr.atom_name[i])
        lookup[key] = int(groups[i])

    def _get_group(chain: str, resi: str, name: str) -> int:
        resi_clean = resi.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") or "0"
        return lookup.get((chain, resi_clean, name), -999)

    cmd.alter(obj_name, "b = _g(chain, resi, name)", space={"_g": _get_group})

    for group_val in unique_groups:
        cmd.color(color_for[group_val], f"{obj_name} and b={group_val}")


def _warn_too_many_groups(n_groups: int, n_colors: int) -> None:
    """Warn (console + popup) that there are more groups than distinct colors."""
    msg = (
        f"{n_groups} groups need distinct colors but only {n_colors} are available. "
        "Reusing colors would make groups ambiguous, so the structure is shown in grey. "
        "Reduce the number of groups to color them distinctly."
    )
    print(f"AtomWorks: {msg}")
    try:
        from pymol.Qt import QtWidgets

        QtWidgets.QMessageBox.warning(None, "Too many groups to color", msg)
    except Exception:
        pass


def _warn_too_many_chains_for_distinct_colors(n_chains: int, n_colors: int) -> None:
    """Warn (console + popup) that some chains will share a marker color."""
    msg = (
        f"{n_chains} chains need distinct colors but only {n_colors} are available. "
        "Some chains' centroid markers will share a color (labels still disambiguate them)."
    )
    print(f"AtomWorks: {msg}")
    try:
        from pymol.Qt import QtWidgets

        QtWidgets.QMessageBox.warning(None, "Too many chains to color distinctly", msg)
    except Exception:
        pass
