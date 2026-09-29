"""Annotation Wizard editor for integer-per-atom group annotations.

Provides :py:class:`GroupEditor`, compatible with the standard
``editor_cls(annotation_cls, state)`` signature used by every other
entry in the Annotation Wizard. Used for both
:py:data:`~rfproteina.conditions.C_IDXGRP` and
:py:data:`~rfproteina.conditions.C_CRDGRP`.
"""

import logging

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.standard_annotations.base import StandardAnnotationBase
from atomworks.pymol_plugin.bridge import reload_structure_in_pymol
from atomworks.pymol_plugin.editors.widgets import BaseAnnotationEditor
from atomworks.pymol_plugin.state import PluginState
from pymol.Qt import QtCore, QtWidgets

from rfproteina.conditions import C_CRDGRP, CoordinateGroup
from rfproteina.pymol.coordinate_edit import commit_coordinate_edit

logger = logging.getLogger("atomworks.pymol_plugin")


def _defaults_for(annotation_cls: StandardAnnotationBase) -> tuple[int, dict[int, str]]:
    """Return ``(unassigned_value, reserved_values)`` for a group condition."""
    if annotation_cls is C_CRDGRP:
        return (
            CoordinateGroup.UNKNOWN_COORDS,
            {
                CoordinateGroup.UNKNOWN_COORDS: "UNKNOWN",
                CoordinateGroup.KNOWN_ABSOLUTE_COORDS: "KNOWN_ABSOLUTE",
            },
        )
    return -1, {}


class GroupEditor(QtWidgets.QWidget, BaseAnnotationEditor):
    """Annotation-wizard editor for integer-per-atom group annotations.

    Supports listing existing groups, adding a PyMOL selection to an existing
    group, creating a new group from a selection, removing a selection from
    any group (assigning the unassigned value), and deleting a group.
    """

    def __init__(
        self,
        annotation_cls: StandardAnnotationBase,
        state: PluginState,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        self._annotation_cls = annotation_cls
        self._state = state
        self._unassigned, self._reserved = _defaults_for(annotation_cls)

        layout = QtWidgets.QVBoxLayout(self)

        cls = annotation_cls
        header = QtWidgets.QLabel(f"<b>{cls.name}</b> &nbsp; <i>(level: {cls.level})</i>")
        header.setWordWrap(True)
        layout.addWidget(header)

        if cls.__doc__:
            doc = QtWidgets.QLabel(cls.__doc__.strip().split("\n")[0])
            doc.setWordWrap(True)
            layout.addWidget(doc)

        layout.addSpacing(6)

        # Groups dropdown + delete
        group_row = QtWidgets.QHBoxLayout()
        group_row.addWidget(QtWidgets.QLabel("Group:"))
        self._group_combo = QtWidgets.QComboBox()
        group_row.addWidget(self._group_combo, stretch=1)
        self._btn_delete = QtWidgets.QPushButton("Delete")
        self._btn_delete.setToolTip("Reassign all atoms in this group to the unassigned value.")
        group_row.addWidget(self._btn_delete)
        btn_refresh = QtWidgets.QPushButton("↻")
        btn_refresh.setToolTip("Refresh group list")
        btn_refresh.setMaximumWidth(30)
        group_row.addWidget(btn_refresh)
        layout.addLayout(group_row)

        layout.addSpacing(4)

        layout.addWidget(QtWidgets.QLabel("<b>Selection actions</b>"))
        self._btn_add = QtWidgets.QPushButton("Add selection to group")
        self._btn_new = QtWidgets.QPushButton("New group from selection")
        self._btn_remove = QtWidgets.QPushButton("Remove selection from any group")
        layout.addWidget(self._btn_add)
        layout.addWidget(self._btn_new)
        layout.addWidget(self._btn_remove)

        layout.addSpacing(6)
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        layout.addStretch()

        self._btn_add.clicked.connect(self._add_to_group)
        self._btn_new.clicked.connect(self._new_group_from_selection)
        self._btn_remove.clicked.connect(self._remove_from_group)
        self._btn_delete.clicked.connect(self._delete_group)
        btn_refresh.clicked.connect(self.refresh)
        self._group_combo.installEventFilter(self)

        self.refresh()

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        """Refresh the dropdown when the user opens it so it reflects current state."""
        if obj is self._group_combo and event.type() == QtCore.QEvent.MouseButtonPress:
            self.refresh()
        return super().eventFilter(obj, event)

    # -- helpers ---------------------------------------------------------

    def _active(self) -> tuple[str, AtomArrayPlus] | None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return None
        return obj_name, self._state.get_synced_array(obj_name)

    def _read_groups(self, arr: AtomArrayPlus) -> np.ndarray:
        return np.asarray(self._annotation_cls.annotation(arr, default="generate")).astype(int)

    def _write_groups(self, obj_name: str, arr: AtomArrayPlus, values: np.ndarray) -> bool:
        edited = arr.copy()
        if self._annotation_cls is C_CRDGRP:
            C_CRDGRP.set_annotation(edited, values, on_internal_coord_mismatch="clear")
            return commit_coordinate_edit(self, obj_name, arr, edited)
        if np.array_equal(self._read_groups(arr), values):
            return False
        self._annotation_cls.set_annotation(edited, values)
        self._state.push_snapshot(obj_name)
        self._state.set_array(obj_name, edited)
        return True

    def _selected_group_id(self) -> int | None:
        data = self._group_combo.currentData()
        return None if data is None else int(data)

    def _post_edit(self, obj_name: str, status: str) -> None:
        self._status.setText(status)
        reload_structure_in_pymol(obj_name)
        self.refresh()

    # -- UI refresh ------------------------------------------------------

    def refresh(self) -> None:
        """Rebuild the group dropdown from the current annotation."""
        selected = self._selected_group_id()
        self._group_combo.blockSignals(True)
        self._group_combo.clear()

        active = self._active()
        if active is None:
            self._group_combo.addItem("(no active object)", None)
            self._group_combo.blockSignals(False)
            return

        _obj, arr = active
        self._displayed = (_obj, self._state.structure_version(_obj))
        groups = self._read_groups(arr)
        uniques, counts = np.unique(groups, return_counts=True)
        for gid, count in zip(uniques, counts, strict=True):
            gid_int = int(gid)
            label = f"Group {gid_int} ({int(count)} atoms)"
            if self._annotation_cls is C_CRDGRP and gid_int <= 0:
                continue
            self._group_combo.addItem(label, gid_int)

        selected_index = self._group_combo.findData(selected)
        if selected_index >= 0:
            self._group_combo.setCurrentIndex(selected_index)
        self._group_combo.blockSignals(False)

    # -- actions ---------------------------------------------------------

    def _add_to_group(self) -> None:
        active = self._active()
        if active is None:
            return
        obj_name, arr = active
        if getattr(self, "_displayed", None) != (obj_name, self._state.structure_version(obj_name)):
            self.refresh()
            self._status.setText("The structure changed. Select the intended group and try again.")
            return
        indices = self._get_selection_indices()
        if not indices:
            QtWidgets.QMessageBox.warning(self, "No selection", "Make a selection in PyMOL first.")
            return
        gid = self._selected_group_id()
        if gid is None or (self._annotation_cls is C_CRDGRP and gid <= 0):
            QtWidgets.QMessageBox.warning(self, "No group", "Pick a group from the dropdown first.")
            return

        try:
            values = self._read_groups(arr).copy()
            values[indices] = gid
            if not self._write_groups(obj_name, arr, values):
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Add to group failed", str(e))
            return
        self._post_edit(obj_name, f"Added {len(indices)} atoms to group {gid}.")

    def _new_group_from_selection(self) -> None:
        active = self._active()
        if active is None:
            return
        obj_name, arr = active
        indices = self._get_selection_indices()
        if not indices:
            QtWidgets.QMessageBox.warning(self, "No selection", "Make a selection in PyMOL first.")
            return

        try:
            values = self._read_groups(arr).copy()
            existing_max = int(values.max()) if len(values) else 0
            new_id = max(existing_max + 1, 1)
            while new_id in self._reserved:
                new_id += 1
            values[indices] = new_id
            if not self._write_groups(obj_name, arr, values):
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "New group failed", str(e))
            return
        self._post_edit(obj_name, f"Created group {new_id} with {len(indices)} atoms.")
        self._group_combo.setCurrentIndex(self._group_combo.findData(new_id))

    def _remove_from_group(self) -> None:
        active = self._active()
        if active is None:
            return
        obj_name, arr = active
        indices = self._get_selection_indices()
        if not indices:
            QtWidgets.QMessageBox.warning(self, "No selection", "Make a selection in PyMOL first.")
            return

        try:
            values = self._read_groups(arr).copy()
            values[indices] = (
                C_CRDGRP.default_annotation(arr)[indices] if self._annotation_cls is C_CRDGRP else self._unassigned
            )
            if not self._write_groups(obj_name, arr, values):
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Remove from group failed", str(e))
            return
        self._post_edit(obj_name, f"Reassigned {len(indices)} atoms to {self._unassigned}.")

    def _delete_group(self) -> None:
        active = self._active()
        if active is None:
            return
        obj_name, arr = active
        if getattr(self, "_displayed", None) != (obj_name, self._state.structure_version(obj_name)):
            self.refresh()
            self._status.setText("The structure changed. Select the intended group and try again.")
            return
        gid = self._selected_group_id()
        if gid is None:
            return
        if gid == self._unassigned:
            QtWidgets.QMessageBox.warning(self, "Cannot delete", f"Group {gid} is the unassigned value.")
            return
        if gid in self._reserved:
            return

        try:
            values = self._read_groups(arr).copy()
            mask = values == gid
            count = int(mask.sum())
            values[mask] = (
                C_CRDGRP.default_annotation(arr)[mask] if self._annotation_cls is C_CRDGRP else self._unassigned
            )
            if not self._write_groups(obj_name, arr, values):
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Delete group failed", str(e))
            return
        self._post_edit(obj_name, f"Deleted group {gid} ({count} atoms reassigned).")

    def apply_to_selection(self) -> None:
        """Default 'apply' action — new group from selection."""
        self._new_group_from_selection()
