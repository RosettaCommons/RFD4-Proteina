"""Residue probability editor for the Annotation Wizard."""

from atomworks.pymol_plugin.editors.widgets import BaseAnnotationEditor
from atomworks.pymol_plugin.state import PluginState
from pymol.Qt import QtWidgets

from rfproteina.conditions import SequenceProbability
from rfproteina.conditions_reporting import set_sequence_probability_condition


class SequenceProbabilityEditor(QtWidgets.QWidget, BaseAnnotationEditor):
    """Set or clear one identity's probability on selected residues."""

    def __init__(
        self,
        annotation_cls: type[SequenceProbability],
        state: PluginState,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._annotation_cls = annotation_cls
        self._state = state
        layout = QtWidgets.QVBoxLayout(self)
        description = QtWidgets.QLabel(
            f"Probability of {annotation_cls.token_name} on selected residues. "
            "Selecting any atom applies to its whole residue. Zero excludes this identity; "
            "Clear leaves it unspecified. Unspecified identities share the remaining probability. "
            "Exact sequence conditions still take precedence."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self._probability = QtWidgets.QDoubleSpinBox()
        self._probability.setRange(0.0, 1.0)
        self._probability.setDecimals(6)
        self._probability.setSingleStep(0.05)
        layout.addWidget(self._probability)
        apply_button = QtWidgets.QPushButton("Set probability for selected residues")
        apply_button.clicked.connect(self.apply_to_selection)
        layout.addWidget(apply_button)
        clear_button = QtWidgets.QPushButton("Clear probability for selected residues")
        clear_button.clicked.connect(lambda: self._apply_value(None))
        layout.addWidget(clear_button)
        layout.addStretch()

    def apply_to_selection(self) -> None:
        self._apply_value(self._probability.value())

    def _apply_value(self, value: float | None) -> None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return
        indices = self._require_selection_indices()
        if indices is None:
            return
        arr = self._state.get_synced_array(obj_name)
        self._state.push_snapshot(obj_name)
        try:
            set_sequence_probability_condition(arr, indices, self._annotation_cls, value)
        except Exception as exc:
            self._state.pop_snapshot(obj_name)
            QtWidgets.QMessageBox.warning(self, "Sequence probability edit failed", str(exc))
            return
        self._reload_or_recolor()
        self._clear_selection()
