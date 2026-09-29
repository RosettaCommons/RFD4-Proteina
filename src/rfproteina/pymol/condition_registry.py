"""Register RFD4-specific condition behavior with the AtomWorks PyMOL plugin.

Showing an annotation in the plugin is opt-in: :py:func:`apply` walks the
*entire* global StandardAnnotation registry (Conditions and plain
StandardAnnotations alike -- every Condition is itself a StandardAnnotation)
and excludes anything *not* declared in ``SUPPORTED_CONDITIONS``, so an
unsupported or newly-added annotation never appears in the Annotation Wizard
(or its auto mask-highlight display) by accident.
"""

import functools
from collections.abc import Callable

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.standard_annotations.base import STANDARD_ANNOTATIONS
from atomworks.ml.conditions import C_CRD, C_IDX
from atomworks.ml.conditions.base import ConditionBase
from atomworks.ml.transforms.dssp import SSEnum
from atomworks.ml.utils.visualize import display_atom_ids
from atomworks.pymol_plugin.editors import exclude_annotation, register_editor
from atomworks.pymol_plugin.editors.widgets import CoordinateEditor, EnumEditor, MaskEditor
from atomworks.pymol_plugin.panel import (
    builtin_annotation_summary,
    register_summary_hook,
    unregister_summary_hook,
)
from atomworks.pymol_plugin.views import register_selection_generator

from rfproteina.conditions import (
    C_CATH,
    C_CENTROID,
    C_CRDGRP,
    C_IDXGRP,
    C_INTCRD,
    C_SEC,
    SEQUENCE_PROBABILITY_CONDITIONS,
)
from rfproteina.conditions_reporting import SUPPORTED_CONDITIONS, summarize_conditions, sync_index_groups
from rfproteina.pymol.cath_editor import CathEditor
from rfproteina.pymol.chain_centroid_editor import ChainCentroidEditor
from rfproteina.pymol.coordinate_edit import commit_coordinate_edit
from rfproteina.pymol.group_editor import GroupEditor
from rfproteina.pymol.sequence_probability_editor import SequenceProbabilityEditor
from rfproteina.transforms.utils.conditioning_utils import set_absolute_coordinate_conditioning

# ---------------------------------------------------------------------------
# _IndexMaskEditor -- lives here (not editors.py) to avoid a circular import,
# since editors.py imports this module to call apply().
# ---------------------------------------------------------------------------


class _IndexMaskEditor(MaskEditor):
    """MaskEditor for the Index condition that keeps :py:data:`C_IDXGRP` in sync.

    Whenever the index mask changes (residues indexed/unindexed), index groups are
    updated so that mandatory split points (molecule boundaries and C_IDX transitions)
    are enforced while preserving any manual group subdivisions the user has made.
    A full reset reverts both the mask and group assignments to their defaults.
    """

    def _set_condition_active(self, value: bool) -> None:
        super()._set_condition_active(value)
        self._sync_index_groups(preserve_manual_edits=True)

    def _reset(self) -> None:
        super()._reset()
        self._sync_index_groups(preserve_manual_edits=False)

    def _sync_index_groups(self, preserve_manual_edits: bool = True) -> None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return
        arr = self._state.get_synced_array(obj_name)
        sync_index_groups(arr, preserve_manual_edits=preserve_manual_edits)
        self._reload_or_recolor()


class _CoordinateEditor(CoordinateEditor):
    """Absolute coordinate editing with group consistency and saved-target confirmation."""

    def __init__(self, annotation_cls: type, state: object, parent: object = None):
        super().__init__(annotation_cls, state, parent)
        from pymol.Qt import QtWidgets

        for button in self.findChildren(QtWidgets.QPushButton):
            if button.text() == "Free design":
                button.setText("Clear absolute coordinate conditioning")

    def _set_condition_active(self, value: bool) -> None:
        obj_name = self._state.active
        if not obj_name or obj_name not in self._state.arrays:
            return
        indices = self._require_selection_indices()
        if indices is None:
            return
        original = self._state.get_synced_array(obj_name)
        edited = original.copy()
        set_absolute_coordinate_conditioning(edited, np.asarray(indices), value)
        if commit_coordinate_edit(self, obj_name, original, edited):
            self._reload_or_recolor()


# ---------------------------------------------------------------------------
# Generic enum selection builder
# ---------------------------------------------------------------------------


def _make_enum_selection_generator(condition_cls: type[ConditionBase]) -> Callable[[AtomArrayPlus, str], None]:
    """Return a generator that creates per-value PyMOL selections for a condition."""
    enum_cls = condition_cls.enum
    assert enum_cls is not None
    prefix = condition_cls.name.lower()

    def _generator(arr: AtomArrayPlus, obj_name: str) -> None:
        from pymol import cmd as _cmd

        default_value = condition_cls.default_value
        # Strict contract: selection values must match the ground truth
        for enum_val in enum_cls:
            if enum_val.value == default_value:
                continue
            _cmd.delete(f"{prefix}_{enum_val.name.lower()}_{obj_name}")

        if not condition_cls.has_annotation(arr):
            return
        annot = condition_cls.annotation(arr, default="raise")
        mask = condition_cls.mask(arr, default="generate")
        atom_ids = display_atom_ids(arr)
        for enum_val in enum_cls:
            if enum_val.value == default_value:
                continue
            atom_mask = mask & (annot == enum_val.value)
            if not atom_mask.any():
                continue
            ids = atom_ids[np.where(atom_mask)[0]]
            id_expr = "+".join(str(int(i)) for i in ids)
            # enable=0: create the selection but do not activate it, so condition
            # highlights never masquerade as the live ``sele`` selection.
            _cmd.select(
                f"{prefix}_{enum_val.name.lower()}_{obj_name}",
                f"model {obj_name} and id {id_expr}",
                enable=0,
            )

    _generator.__name__ = f"_selection_generator_{prefix}"
    return _generator


# ---------------------------------------------------------------------------
# Registration entry point
# ---------------------------------------------------------------------------


def apply() -> None:
    """Register RFD4 edge cases and exclude unsupported annotations.

    Showing an annotation in the plugin is opt-in: anything not listed in
    :py:data:`SUPPORTED_CONDITIONS` never appears in the Annotation Wizard,
    even if a generic auto-fallback editor would technically work, and never
    gets an auto-generated mask-highlight selection either.
    """
    from atomworks.ml.utils.visualize import hide_condition_display

    listed = set(SUPPORTED_CONDITIONS)
    exclude_annotation(C_INTCRD)
    custom_editors = {
        C_IDX: _IndexMaskEditor,
        C_IDXGRP: GroupEditor,
        C_CRDGRP: GroupEditor,
        C_CRD: _CoordinateEditor,
        C_CENTROID: ChainCentroidEditor,
        C_CATH: CathEditor,
    }
    custom_editors[C_SEC] = functools.partial(EnumEditor, exclude=frozenset({SSEnum.NON_PROTEIN.value}))
    custom_editors.update({condition: SequenceProbabilityEditor for condition in SEQUENCE_PROBABILITY_CONDITIONS})
    for condition, editor in custom_editors.items():
        register_editor(condition)(editor)

    for condition in SUPPORTED_CONDITIONS:
        if condition.enum is not None:
            register_selection_generator(_make_enum_selection_generator(condition))

    for condition in (C_IDXGRP, C_CRDGRP):
        hide_condition_display(condition)

    # Opt-in enforcement: anything not explicitly declared above is invisible
    # This is valuable so that new Conditions which don't influence model behavior
    # are not exposed in the plugin, which would be confusing to users
    for annotation_cls in STANDARD_ANNOTATIONS:
        if annotation_cls not in listed:
            exclude_annotation(annotation_cls)
            if issubclass(annotation_cls, ConditionBase):
                hide_condition_display(annotation_cls)

    # `summarize_conditions` covers the full supported vocabulary (AtomWorks-native
    # conditions included), so drop the AtomWorks default to avoid double-reporting.
    unregister_summary_hook(builtin_annotation_summary)
    register_summary_hook(summarize_conditions)
