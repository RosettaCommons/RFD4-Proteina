"""Custom PyMOL visualizer for proteinfoundation conditions.

Wraps the default atomworks visualizer and adds coloring for
:py:data:`~proteinfoundation.conditions.C_CRDGRP`.

Only imported inside a running PyMOL process.
"""

from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.ml.utils.visualize import get_atom_array_style_cmd
from pymol import cmd

from proteinfoundation.conditions import C_CRDGRP

# Colours for named CoordinateGroup constants and cycling relative groups
_CRDGRP_NAMED_COLORS = {
    -1: "gray60",  # UNKNOWN_COORDS
    0: "lightblue",  # KNOWN_ABSOLUTE_COORDS
}
_CRDGRP_CYCLE = ["red", "green", "orange", "purple", "cyan", "yellow", "magenta", "salmon"]


def visualize(arr: AtomArrayPlus, obj_name: str) -> None:
    """Run the full visualization pipeline for a loaded structure.

    Executes the default atomworks style command then overlays
    coordinate-group coloring if :py:data:`~proteinfoundation.conditions.C_CRDGRP`
    is annotated on the array.

    Args:
        arr: The active AtomArrayPlus.
        obj_name: The PyMOL object name.
    """
    cmd.do(get_atom_array_style_cmd(arr, obj_name))
    _apply_crdgrp_coloring(arr, obj_name)


def _apply_crdgrp_coloring(arr: AtomArrayPlus, obj_name: str) -> None:
    """Overlay per-group coloring for the CoordinateGroup annotation.

    Uses ``cmd.alter`` to write group IDs into b-factors, then colors
    each unique group value distinctly.  No-ops if the annotation is
    not present on the array.

    Args:
        arr: The active AtomArrayPlus.
        obj_name: The PyMOL object name.
    """

    if C_CRDGRP.full_name not in arr.get_annotation_categories():
        return

    groups = C_CRDGRP.annotation(arr, default="generate")

    # Build (chain, res_id_str, atom_name) -> group lookup for cmd.alter
    lookup: dict[tuple[str, str, str], int] = {}
    for i in range(len(arr)):
        key = (arr.chain_id[i], str(arr.res_id[i]), arr.atom_name[i])
        lookup[key] = int(groups[i])

    def _get_group(chain: str, resi: str, name: str) -> int:
        # resi may include insertion-code suffix; strip non-digit tail
        resi_clean = resi.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz") or "0"
        return lookup.get((chain, resi_clean, name), -999)

    cmd.alter(obj_name, "b = _g(chain, resi, name)", space={"_g": _get_group})

    unique_groups = sorted(set(groups))
    cycle_idx = 0
    for group_val in unique_groups:
        color = _CRDGRP_NAMED_COLORS.get(group_val)
        if color is None:
            color = _CRDGRP_CYCLE[cycle_idx % len(_CRDGRP_CYCLE)]
            cycle_idx += 1
        cmd.color(color, f"{obj_name} and b={group_val}")
