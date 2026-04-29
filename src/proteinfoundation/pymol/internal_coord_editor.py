"""Annotation Wizard editor for the Internal Coordinate condition.

Under the hood this calls
:py:func:`proteinfoundation.transforms.utils.conditioning_utils.convert_to_internal_coords`,
which transfers atoms from absolute coordinate conditioning (``C_CRD``) to
internal coordinate conditioning (``C_INTCRD``) and creates a new coord
group for the region.

The selection must first be coord-conditioned (``C_CRD`` mask = True) — this
editor sets that prerequisite automatically before calling the converter.
"""

import logging

import numpy as np
from atomworks.io.utils.standard_annotations.base import StandardAnnotationBase
from atomworks.pymol_plugin.editors.widgets import BaseAnnotationEditor
from atomworks.pymol_plugin.state import PluginState
from pymol.Qt import QtWidgets

from proteinfoundation.conditions import C_CRD
from proteinfoundation.transforms.utils.conditioning_utils import convert_to_internal_coords

logger = logging.getLogger("atomworks.pymol_plugin")


class InternalCoordEditor(QtWidgets.QWidget, BaseAnnotationEditor):
    """Mask editor that converts the selection to internal-coord conditioning."""

    def __init__(
        self,
        annotation_cls: StandardAnnotationBase,
        state: PluginState,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._annotation_cls = annotation_cls
        self._state = state

        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(f"<b>{annotation_cls.name}</b>")
        layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Convert the selected atoms from absolute to internal coordinate "
            "conditioning. The selection will be placed in its own coord group "
            "and its absolute-coord mask (C_CRD) will be cleared."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(8)

        btn_convert = QtWidgets.QPushButton("Convert selection to internal coords")
        btn_unset = QtWidgets.QPushButton("Unset internal-coord mask on selection")
        layout.addWidget(btn_convert)
        layout.addWidget(btn_unset)

        layout.addSpacing(6)
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

        btn_convert.clicked.connect(self._convert)
        btn_unset.clicked.connect(lambda: self._set_mask(False))

    def _convert(self) -> None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return
        indices = self._get_selection_indices()
        if not indices:
            QtWidgets.QMessageBox.warning(self, "No selection", "Make a selection in PyMOL first.")
            return

        arr = self._state.arrays[obj_name]

        crd_mask = C_CRD.mask(arr, default="generate").copy()
        crd_mask[indices] = True
        C_CRD.set_mask(arr, crd_mask)

        mask = np.zeros(len(arr), dtype=bool)
        mask[indices] = True

        self._state.push_snapshot(obj_name)
        try:
            convert_to_internal_coords(arr, mask)
        except Exception as e:
            self._state.pop_snapshot(obj_name)
            QtWidgets.QMessageBox.critical(self, "Conversion failed", str(e))
            return

        self._status.setText(f"Converted {len(indices)} atoms to internal coords.")
        self._reload_or_recolor()

    def apply_to_selection(self) -> None:
        self._convert()
