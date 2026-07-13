"""Custom PyMOL editor widgets for proteinfoundation conditions.

Only ever imported inside a running PyMOL process via
:py:func:`proteinfoundation.pymol.editors.register`.
Qt imports at module level are intentional and safe here.
"""

from atomworks.io.utils.standard_annotations.base import StandardAnnotationBase
from atomworks.pymol_plugin.editors.widgets import BaseAnnotationEditor
from atomworks.pymol_plugin.state import PluginState
from pymol.Qt import QtWidgets

from proteinfoundation.conditions import CoordinateGroup

# ---------------------------------------------------------------------------
# ValueButtonEditor — one button per named value
# ---------------------------------------------------------------------------


class ValueButtonEditor(QtWidgets.QWidget, BaseAnnotationEditor):
    """Editor for integer-valued conditions using one button per named value.

    Each button sets the annotation value for the current PyMOL selection.
    An optional spinbox row allows setting arbitrary integer values beyond
    the named set.

    Args:
        annotation_cls: The annotation class to edit.
        state: Plugin state singleton.
        named_values: Sequence of ``(label, int_value)`` pairs, one button each.
        null_value: If provided, setting this value also sets mask=False;
            all other values set mask=True.  Pass ``None`` to skip mask updates
            (e.g. for annotations whose mask is always all-True).
        allow_arbitrary_int: Show an extra spinbox row for arbitrary integers.
        arbitrary_label: Label shown next to the spinbox.
        arbitrary_range: ``(min, max)`` for the spinbox.
    """

    def __init__(
        self,
        annotation_cls: StandardAnnotationBase,
        state: PluginState,
        named_values: list[tuple[str, int]],
        null_value: int | None = None,
        allow_arbitrary_int: bool = False,
        arbitrary_label: str = "Value",
        arbitrary_range: tuple[int, int] = (-9999, 9999),
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        cls = annotation_cls
        header = QtWidgets.QLabel(f"<b>{cls.name}</b> &nbsp; <i>(level: {cls.level})</i>")
        header.setWordWrap(True)
        layout.addWidget(header)

        if cls.__doc__:
            doc = QtWidgets.QLabel(cls.__doc__.strip().split("\n")[0])
            doc.setWordWrap(True)
            layout.addWidget(doc)

        layout.addSpacing(8)

        for label, value in named_values:
            btn = QtWidgets.QPushButton(f"Set {label} for selection")
            btn.clicked.connect(lambda checked, v=value: self._apply_value(v))
            layout.addWidget(btn)

        if allow_arbitrary_int:
            layout.addSpacing(4)
            arb_row = QtWidgets.QHBoxLayout()
            arb_row.addWidget(QtWidgets.QLabel(f"{arbitrary_label}:"))
            self._arb_spin = QtWidgets.QSpinBox()
            self._arb_spin.setRange(*arbitrary_range)
            self._arb_spin.setValue(arbitrary_range[0])
            arb_row.addWidget(self._arb_spin)
            btn_arb = QtWidgets.QPushButton("Set for selection")
            btn_arb.clicked.connect(lambda: self._apply_value(self._arb_spin.value()))
            arb_row.addWidget(btn_arb)
            layout.addLayout(arb_row)

        layout.addSpacing(4)
        btn_reset = QtWidgets.QPushButton("Reset to default")
        btn_reset.clicked.connect(self._reset)
        layout.addWidget(btn_reset)
        layout.addStretch()

        self._annotation_cls = annotation_cls
        self._state = state
        self._null_value = null_value

    def _apply_value(self, value: int) -> None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return
        arr = self._state.get_synced_array(obj_name)
        indices = self._require_selection_indices()
        if indices is None:
            return

        # Read both before modifying either (atomic snapshot prevents stale reads
        # when the same PyMOL selection covers a previously annotated region).
        annot = self._annotation_cls.annotation(arr, default="generate").copy()
        mask = self._annotation_cls.mask(arr, default="generate").copy() if self._null_value is not None else None

        annot[indices] = value
        if mask is not None:
            mask[indices] = value != self._null_value

        self._annotation_cls.set_annotation(arr, annot)
        if mask is not None:
            self._annotation_cls.set_mask(arr, mask)

        self._reload_or_recolor()
        # Force the user to make a fresh selection before the next apply,
        # preventing silent overlap with the previous region. deselect() also
        # clears the on-screen indicators so nothing looks selected afterwards.
        self._clear_selection()

    def _reset(self) -> None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return
        arr = self._state.get_synced_array(obj_name)
        self._annotation_cls.set_annotation(arr, self._annotation_cls.default_annotation(arr))
        if self._null_value is not None:
            self._annotation_cls.set_mask(arr, self._annotation_cls.default_mask(arr))
        self._reload_or_recolor()


# ---------------------------------------------------------------------------
# Concrete editors
# ---------------------------------------------------------------------------


class CoordinateGroupEditor(ValueButtonEditor):
    """Editor for coordinate group assignments.

    Atoms sharing the same group integer are treated as having known relative
    coordinates.  Named constants cover the two special values; arbitrary
    positive integers identify distinct relative groups.
    """

    def __init__(
        self,
        annotation_cls: StandardAnnotationBase,
        state: PluginState,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(
            annotation_cls,
            state,
            named_values=[
                ("UNKNOWN_COORDS", CoordinateGroup.UNKNOWN_COORDS),
                ("KNOWN_ABSOLUTE_COORDS", CoordinateGroup.KNOWN_ABSOLUTE_COORDS),
            ],
            null_value=None,  # mask is always all-True for CoordinateGroup
            allow_arbitrary_int=True,
            arbitrary_label="Relative group ID",
            arbitrary_range=(1, 9999),
            parent=parent,
        )
