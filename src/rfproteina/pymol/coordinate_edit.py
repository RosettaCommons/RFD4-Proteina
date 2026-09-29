"""Confirmation and undo for edits that can discard saved coordinate targets."""

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.ml.conditions import C_CRD
from atomworks.pymol_plugin.editors.widgets import BaseAnnotationEditor
from pymol.Qt import QtWidgets

from rfproteina.conditions import C_CRDGRP, C_INTCRD


def commit_coordinate_edit(
    editor: BaseAnnotationEditor, obj_name: str, original: AtomArrayPlus, edited: AtomArrayPlus
) -> bool:
    """Confirm saved-target loss before taking an undo snapshot and committing a prepared edit."""
    if all(
        np.array_equal(
            cls.annotation(original, default="generate"), cls.annotation(edited, default="generate"), equal_nan=True
        )
        for cls in (C_CRD, C_CRDGRP, C_INTCRD)
    ):
        return False
    before = C_INTCRD.annotation(original, default="generate")
    after = C_INTCRD.annotation(edited, default="generate")
    lost = np.isfinite(before).any(axis=1) & ~np.isfinite(after).any(axis=1)
    if lost.any():
        reply = QtWidgets.QMessageBox.warning(
            editor,
            "Discard saved internal coordinate targets?",
            f"This edit clears saved targets for {int(lost.sum())} atoms in affected coordinate groups, "
            "including group members outside the selection. New targets will be taken from the structure "
            "when model inputs are prepared. Undo restores the saved targets.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return False
    editor._state.push_snapshot(obj_name)
    editor._state.set_array(obj_name, edited)
    return True
