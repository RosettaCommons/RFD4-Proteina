"""Registers the PyMOL-plugin side of :py:data:`~rfproteina.conditions_reporting.SUPPORTED_CONDITIONS`.

The vocabulary itself (which Condition gets which editor, enum, and summary
treatment) lives in :py:mod:`rfproteina.conditions_reporting`, which must stay
PyMOL-free; this module is the one place that's actually allowed to import
PyMOL, so it's where each spec's lazy ``editor`` factory actually gets called
and where the generic enum-driven editor/selection-generator machinery lives.

Showing an annotation in the plugin is opt-in: :py:func:`apply` walks the
*entire* global StandardAnnotation registry (Conditions and plain
StandardAnnotations alike -- every Condition is itself a StandardAnnotation)
and excludes anything *not* declared in ``SUPPORTED_CONDITIONS``, so an
unsupported or newly-added annotation never appears in the Annotation Wizard
(or its auto mask-highlight display) by accident.
"""

import functools
from collections.abc import Callable
from enum import IntEnum

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.standard_annotations.base import STANDARD_ANNOTATIONS
from atomworks.ml.conditions.base import ConditionBase
from atomworks.ml.utils.visualize import display_atom_ids
from atomworks.pymol_plugin.editors import exclude_annotation, register_editor
from atomworks.pymol_plugin.editors.widgets import MaskEditor
from atomworks.pymol_plugin.panel import register_summary_hook
from atomworks.pymol_plugin.views import register_selection_generator

from rfproteina.conditions_reporting import SUPPORTED_CONDITIONS, ConditionSpec, summarize_conditions, sync_index_groups
from rfproteina.pymol.widgets import ValueButtonEditor

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

    def _set_mask(self, value: bool) -> None:
        super()._set_mask(value)
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


# ---------------------------------------------------------------------------
# Generic editor / summary builders
# ---------------------------------------------------------------------------


def _enum_editor(enum_cls: type[IntEnum], null_value: int, exclude: frozenset[int]) -> Callable[..., ValueButtonEditor]:
    """Build a `editor_cls(annotation_cls, state, parent=None)`-compatible callable."""
    return functools.partial(
        ValueButtonEditor,
        named_values=[(m.name, m.value) for m in enum_cls if m.value not in exclude],
        null_value=null_value,
    )


def _make_enum_selection_generator(spec: ConditionSpec) -> Callable[[AtomArrayPlus, str], None]:
    """Return a generator that creates per-value PyMOL selections for *spec*."""
    condition_cls, enum_cls, null_value = spec.condition, spec.enum_cls, spec.null_value
    prefix = condition_cls.name.lower()

    def _generator(arr: AtomArrayPlus, obj_name: str) -> None:
        from pymol import cmd as _cmd

        # Strict contract: selection values must match the ground truth
        for enum_val in enum_cls:
            if enum_val.value == null_value:
                continue
            _cmd.delete(f"{prefix}_{enum_val.name.lower()}_{obj_name}")

        if condition_cls.full_name not in arr.get_annotation_categories():
            return
        annot = condition_cls.annotation(arr, default="raise")
        mask = condition_cls.mask(arr, default="generate")
        atom_ids = display_atom_ids(arr)
        for enum_val in enum_cls:
            if enum_val.value == null_value:
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
    """Register editors, selection generators, and the summary hook for every
    :py:data:`SUPPORTED_CONDITIONS` entry, then exclude every other globally
    registered StandardAnnotation (Conditions included, since every Condition
    is itself a StandardAnnotation) from the wizard and its auto mask-highlight
    display.

    Showing an annotation in the plugin is opt-in: anything not listed in
    :py:data:`SUPPORTED_CONDITIONS` never appears in the Annotation Wizard,
    even if a generic auto-fallback editor would technically work, and never
    gets an auto-generated mask-highlight selection either -- both are driven
    off the same "is it listed" check, so there's no separate visibility flag
    to remember. There are currently no supported non-Condition
    StandardAnnotations; if one is ever needed, add it to
    ``SUPPORTED_CONDITIONS`` the same way (its ``enum_cls`` field simply won't
    apply).
    """
    from atomworks.ml.utils.visualize import hide_condition_display

    listed = {spec.condition for spec in SUPPORTED_CONDITIONS}

    for spec in SUPPORTED_CONDITIONS:
        editor = spec.editor() if spec.editor is not None else None
        if editor is None and spec.enum_cls is not None:
            editor = _enum_editor(spec.enum_cls, spec.null_value, spec.enum_editor_exclude)
        if editor is not None:
            register_editor(spec.condition)(editor)

        if spec.hide_display:
            hide_condition_display(spec.condition)

        if spec.selection_generator:
            register_selection_generator(_make_enum_selection_generator(spec))

    # Opt-in enforcement: anything not explicitly declared above is invisible
    # This is valuable so that new Conditions which don't influence model behavior
    # are not exposed in the plugin, which would be confusing to users
    for annotation_cls in STANDARD_ANNOTATIONS:
        if annotation_cls not in listed:
            exclude_annotation(annotation_cls)
            if issubclass(annotation_cls, ConditionBase):
                hide_condition_display(annotation_cls)

    register_summary_hook(summarize_conditions)
