"""Single source of truth for which Conditions the plugin supports, and how.

Historically "which conditions get an editor," "which are excluded from the
wizard," "which show up in the summary text," and "enum vs. plain" were each
decided by separate hardcoded lists/checks scattered across ``editors.py``.
That made it easy to add a new Condition (e.g. a new interchain interaction
type) and forget to wire up one of those four concerns.

This module collapses all four into one list of :py:class:`ConditionSpec`
entries. Showing an annotation in the plugin is opt-in: :py:func:`apply` walks
the *entire* global StandardAnnotation registry (Conditions and plain
StandardAnnotations alike -- every Condition is itself a StandardAnnotation)
and excludes anything *not* declared in :py:data:`SUPPORTED_CONDITIONS`, so an
unsupported or newly-added annotation never appears in the Annotation Wizard
(or its auto mask-highlight display) by accident.
"""

import functools
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.standard_annotations.base import STANDARD_ANNOTATIONS
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.conditions.base import ConditionBase
from atomworks.ml.utils.visualize import display_atom_ids
from atomworks.pymol_plugin.editors import exclude_annotation, register_editor
from atomworks.pymol_plugin.editors.widgets import BaseAnnotationEditor, MaskEditor
from atomworks.pymol_plugin.panel import register_summary_hook
from atomworks.pymol_plugin.views import register_selection_generator

from rfproteina.conditions import (
    C_CENTROID,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_RAS,
    HBondEnum,
    RasaEnum,
)
from rfproteina.pymol.chain_centroid_editor import ChainCentroidEditor
from rfproteina.pymol.group_editor import GroupEditor
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
        default_groups = C_IDXGRP.default_annotation(arr)
        if preserve_manual_edits and C_IDXGRP.full_name in arr.get_annotation_categories():
            current_groups = arr.get_annotation(C_IDXGRP.full_name)
            # Enforce mandatory default splits while keeping manual subdivisions:
            # two atoms share a group iff they share both (default_group, current_group).
            combined = np.stack([default_groups, current_groups], axis=1)
            _, new_groups = np.unique(combined, axis=0, return_inverse=True)
        else:
            new_groups = default_groups
        C_IDXGRP.set_annotation(arr, new_groups)
        self._reload_or_recolor()


# ---------------------------------------------------------------------------
# ConditionSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionSpec:
    """Everything the plugin needs to know about one supported Condition.

    Args:
        condition: The Condition class this spec describes.
        editor: Editor class (or ``editor_cls(annotation_cls, state, parent=None)``
            callable) to register. ``None`` means either an enum-driven editor is
            built automatically (see ``enum_cls``), or the editor is registered
            elsewhere (e.g. atomworks core registers ``C_SEQ``/``C_CRD``) and this
            spec exists purely to keep the condition from being auto-excluded.
        enum_cls: IntEnum backing this condition's values, if any. Drives an
            automatic :py:class:`~rfproteina.pymol.widgets.ValueButtonEditor`
            when ``editor`` is unset, an "enum" summary, and (if
            ``selection_generator`` is set) per-value PyMOL highlight selections.
        null_value: The enum member that represents "unconditioned". Required
            whenever ``enum_cls`` is set.
        selection_generator: Register a per-value PyMOL highlight-selection
            generator. Only meaningful with ``enum_cls`` set.
        summary: How to report this condition in the "Summarize changes" dialog.
            ``"mask"``: a single "N atoms" line if the mask has any True.
            ``"enum"``: per-value atom counts (requires ``enum_cls``).
            ``"custom"``: delegate entirely to ``summary_fn``.
            ``None``: not reported here (e.g. already covered by AtomWorks' own
            built-in summary hook for its native conditions).
        summary_label: Display label; defaults to ``condition.name``.
        summary_fn: Required when ``summary="custom"``. Returns summary lines.
        hide_display: Suppress automatic mask-highlight-selection generation
            (e.g. group conditions, whose "mask" is always all-True and thus
            uninformative to highlight).
    """

    condition: type[ConditionBase]
    editor: type | Callable[..., BaseAnnotationEditor] | None = None
    enum_cls: type[IntEnum] | None = None
    null_value: int | None = None
    selection_generator: bool = False
    summary: Literal["mask", "enum", "custom", None] = "mask"
    summary_label: str | None = None
    summary_fn: Callable[[AtomArrayPlus], list[str]] | None = None
    hide_display: bool = False

    def __post_init__(self) -> None:
        if self.selection_generator and (self.enum_cls is None or self.null_value is None):
            raise ValueError(f"{self.condition.name}: selection_generator requires enum_cls and null_value")
        if self.summary == "enum" and (self.enum_cls is None or self.null_value is None):
            raise ValueError(f"{self.condition.name}: summary='enum' requires enum_cls and null_value")
        if self.summary == "custom" and self.summary_fn is None:
            raise ValueError(f"{self.condition.name}: summary='custom' requires summary_fn")

    @property
    def label(self) -> str:
        return self.summary_label or self.condition.name


# ---------------------------------------------------------------------------
# Custom summary functions for conditions whose reporting doesn't fit the
# generic "mask" / "enum" shapes.
# ---------------------------------------------------------------------------


def _index_group_summary(arr: AtomArrayPlus) -> list[str]:
    if C_IDXGRP.full_name not in arr.get_annotation_categories():
        return []
    groups = arr.get_annotation(C_IDXGRP.full_name)
    n_groups = len(np.unique(groups))
    lines = [f'Index groups: {n_groups} groups  (check "Color by index group" view)']
    # Report implied res_id gaps within unindexed groups so the user can confirm
    # that non-contiguous regions were split as intended (rather than collapsed
    # into one group with fixed-size gaps).
    if C_IDX.mask_name in arr.get_annotation_categories():
        is_indexed = C_IDX.mask(arr, default="generate")
        res_id = arr.res_id
        for gid in np.unique(groups):
            in_group = groups == gid
            if is_indexed[in_group].any():
                continue  # indexed group — index offsets are externally defined
            gid_res = np.unique(res_id[in_group])
            if len(gid_res) > 1 and np.any(np.diff(np.sort(gid_res)) > 1):
                lines.append(
                    f"  unindexed group {int(gid)} contains non-sequential res_ids: "
                    "these gaps will be preserved in the output. If this is not desired, "
                    "please adjust the index groups accordingly."
                )
    return ["\n".join(lines)]


def _coord_group_summary(arr: AtomArrayPlus) -> list[str]:
    if C_CRDGRP.full_name not in arr.get_annotation_categories():
        return []
    n_groups = len(np.unique(arr.get_annotation(C_CRDGRP.full_name)))
    return [f'Coordinate groups: {n_groups} groups  (check "Color by coord group" view)']


def _chain_centroid_summary(arr: AtomArrayPlus) -> list[str]:
    if C_CENTROID.full_name not in arr.get_annotation_categories():
        return []
    mask = C_CENTROID.mask(arr, default="generate")
    if not mask.any():
        return []
    annot = C_CENTROID.annotation(arr, default="generate")
    lines = ["Chain centroid:"]
    for chain_id in np.unique(arr.chain_iid[mask]):
        xyz = annot[mask & (arr.chain_iid == chain_id)][0]
        lines.append(f"  {chain_id}: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})")
    return ["\n".join(lines)]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

SUPPORTED_CONDITIONS: list[ConditionSpec] = [
    # AtomWorks-native conditions: editors registered by atomworks core itself
    # (register_custom_editors() in atomworks/pymol_plugin/editors/custom.py).
    # Listed here only so opt-in enforcement doesn't auto-exclude them; their
    # summary is AtomWorks' own built-in hook, not ours.
    ConditionSpec(C_SEQ, summary=None),
    ConditionSpec(C_CRD, summary=None),
    # Index conditioning -- rfproteina overrides the editor to keep
    # index groups in sync; summary stays with AtomWorks' built-in hook.
    ConditionSpec(C_IDX, editor=_IndexMaskEditor, summary=None),
    # Simple boolean mask conditions.
    ConditionSpec(C_HOT, editor=MaskEditor, summary_label="Hotspot"),
    # Enum-valued conditions: editor, summary, and (where useful) PyMOL
    # highlight selections are all built generically from enum_cls/null_value.
    ConditionSpec(
        C_RAS,
        enum_cls=RasaEnum,
        null_value=RasaEnum.UNSPECIFIED.value,
        selection_generator=True,
        summary="enum",
        summary_label="RASA",
    ),
    # ConditionSpec(
    #     C_SEC,
    #     enum_cls=SSEnum,
    #     null_value=SSEnum.NONE.value,
    #     selection_generator=True,
    #     summary="enum",
    #     summary_label="Secondary structure",
    # ),
    ConditionSpec(
        C_HB,
        enum_cls=HBondEnum,
        null_value=HBondEnum.NONE.value,
        selection_generator=True,
        summary="enum",
        summary_label="Hydrogen bonds",
    ),
    # Integer-valued group conditions -- shared GroupEditor, bespoke summary
    # (group count + view hint), hidden from mask-highlighting (mask is
    # trivially all-True and thus uninformative to highlight).
    ConditionSpec(C_IDXGRP, editor=GroupEditor, summary="custom", summary_fn=_index_group_summary, hide_display=True),
    ConditionSpec(C_CRDGRP, editor=GroupEditor, summary="custom", summary_fn=_coord_group_summary, hide_display=True),
    # Internal coordinate conditioning -- wraps convert_to_internal_coords.
    # ConditionSpec(C_INTCRD, editor=InternalCoordEditor, summary_label="Internal coordinate"),
    # Chain centroid conditioning -- manual xyz entry + indicator spheres.
    ConditionSpec(C_CENTROID, editor=ChainCentroidEditor, summary="custom", summary_fn=_chain_centroid_summary),
    # BasePair: excluded for now (2-body enum condition; no dedicated editor
    # yet -- the generic PairValueEditor fallback is built for Angstrom
    # distances and would be actively wrong for PAIR/LOOP semantics).
]


# ---------------------------------------------------------------------------
# Generic editor / summary builders
# ---------------------------------------------------------------------------


def _enum_editor(enum_cls: type[IntEnum], null_value: int) -> Callable[..., ValueButtonEditor]:
    """Build a `editor_cls(annotation_cls, state, parent=None)`-compatible callable."""
    return functools.partial(
        ValueButtonEditor,
        named_values=[(m.name, m.value) for m in enum_cls],
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


def _mask_summary(spec: ConditionSpec, arr: AtomArrayPlus) -> list[str]:
    if not spec.condition.has_mask(arr):
        return []
    mask = spec.condition.mask(arr, default="generate")
    if not mask.any():
        return []
    return [f"{spec.label}: {int(mask.sum())} atoms"]


def _enum_summary(spec: ConditionSpec, arr: AtomArrayPlus) -> list[str]:
    if spec.condition.full_name not in arr.get_annotation_categories():
        return []
    annot = spec.condition.annotation(arr, default="raise")
    mask = spec.condition.mask(arr, default="generate")
    section: list[str] = []
    for enum_val in spec.enum_cls:
        if enum_val.value == spec.null_value:
            continue
        atom_mask = mask & (annot == enum_val.value)
        if not atom_mask.any():
            continue
        section.append(f"  {enum_val.name}: {int(atom_mask.sum())} atoms")
    if not section:
        return []
    return [f"{spec.label}:\n" + "\n".join(section)]


def summarize_conditions(arr: AtomArrayPlus) -> str:
    """Summary hook covering every :py:data:`SUPPORTED_CONDITIONS` entry with ``summary`` set."""
    lines: list[str] = []
    for spec in SUPPORTED_CONDITIONS:
        if spec.summary is None:
            continue
        elif spec.summary == "mask":
            lines.extend(_mask_summary(spec, arr))
        elif spec.summary == "enum":
            lines.extend(_enum_summary(spec, arr))
        elif spec.summary == "custom":
            lines.extend(spec.summary_fn(arr))
    return "\n\n".join(lines)


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
    ``SUPPORTED_CONDITIONS`` the same way (its ``summary``/``enum_cls`` fields
    simply won't apply).
    """
    from atomworks.ml.utils.visualize import hide_condition_display

    listed = {spec.condition for spec in SUPPORTED_CONDITIONS}

    for spec in SUPPORTED_CONDITIONS:
        editor = spec.editor
        if editor is None and spec.enum_cls is not None:
            editor = _enum_editor(spec.enum_cls, spec.null_value)
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
