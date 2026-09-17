"""CATH label editor for the Annotation Wizard."""

from atomworks.io.utils.standard_annotations.base import StandardAnnotationBase
from atomworks.pymol_plugin.editors.widgets import BaseAnnotationEditor
from atomworks.pymol_plugin.state import PluginState
from pymol.Qt import QtWidgets

from rfproteina.conditions import C_CATH
from rfproteina.utils.condition_resolution import set_cath_condition


class CathEditor(QtWidgets.QWidget, BaseAnnotationEditor):
    """Apply a CATH prefix to selected residues or clear their fold conditioning."""

    def __init__(
        self,
        annotation_cls: StandardAnnotationBase,
        state: PluginState,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._annotation_cls = annotation_cls
        self._state = state
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            "Select whole residues or a chain, then enter a C.A.T label "
            "(class, architecture, topology). Use x for unspecified trailing levels: "
            "1.x.x (mainly alpha), 2.60.40 (immunoglobulin)."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self._label = QtWidgets.QLineEdit("1.x.x")
        layout.addWidget(self._label)
        apply_button = QtWidgets.QPushButton("Set CAT for selection")
        apply_button.clicked.connect(self.apply_to_selection)
        layout.addWidget(apply_button)
        clear_button = QtWidgets.QPushButton("Clear CAT for selection")
        clear_button.clicked.connect(lambda: self._apply_value(C_CATH.NULL_VALUE))
        layout.addWidget(clear_button)
        layout.addStretch()

    def apply_to_selection(self) -> None:
        self._apply_value(self._label.text().strip())

    def _apply_value(self, value: str) -> None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return
        indices = self._require_selection_indices()
        if indices is None:
            return
        arr = self._state.get_synced_array(obj_name)
        self._state.push_snapshot(obj_name)
        try:
            set_cath_condition(arr, indices, value)
        except Exception as exc:
            self._state.pop_snapshot(obj_name)
            QtWidgets.QMessageBox.warning(self, "CAT edit failed", str(exc))
            return
        self._reload_or_recolor()
        self._clear_selection()
