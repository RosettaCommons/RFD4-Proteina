"""PyMOL editor registrations for proteinfoundation conditions.

This module has no top-level PyMOL or Qt imports and is safe to import
during model training.  It is only executed inside a running PyMOL
process via the AtomWorks extension system.

Register with::

    atomworks pymol add proteinfoundation.pymol.editors
"""

from collections.abc import Callable
from enum import IntEnum

import numpy as np
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.io.utils.standard_annotations import S_ATM
from atomworks.io.utils.standard_annotations.base import StandardAnnotationBase
from atomworks.ml.conditions import C_CTR, C_DIS, C_IDX, C_NTR
from atomworks.ml.utils.visualize import display_atom_ids
from atomworks.pymol_plugin.editors import exclude_annotation, register_editor
from atomworks.pymol_plugin.editors.widgets import MaskEditor
from atomworks.pymol_plugin.views import register_selection_generator, register_view

from foundry.utils.standard_annotations import SA_CRDGRPSRC, SA_CRDSRC, SA_CWI, SA_SRC, SA_UII
from proteinfoundation.conditions import (
    C_ABH,
    C_ABL,
    C_CATH,
    C_CDR,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_INTCRD,
    C_RAS,
    C_SEC,
    HBondEnum,
    RasaEnum,
    SSEnum,
)
from proteinfoundation.pymol.group_editor import GroupEditor
from proteinfoundation.pymol.internal_coord_editor import InternalCoordEditor
from proteinfoundation.pymol.visualize import color_by_coord_group, color_by_index_group
from proteinfoundation.pymol.widgets import (
    HBondEditor,
    RasaEditor,
    SecondaryStructureEditor,
)

# ---------------------------------------------------------------------------
# Selection generators
# ---------------------------------------------------------------------------


def _make_enum_selection_generator(
    condition_cls: StandardAnnotationBase,
    enum_cls: type[IntEnum],
    null_value: int,
    prefix: str,
) -> Callable[[AtomArrayPlus, str], None]:
    """Return a generator that creates per-value PyMOL selections for *condition_cls*."""

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


_rasa_selections = _make_enum_selection_generator(C_RAS, RasaEnum, RasaEnum.UNSPECIFIED.value, "rasa")
_ss_selections = _make_enum_selection_generator(C_SEC, SSEnum, SSEnum.NONE.value, "ss")
_hbond_selections = _make_enum_selection_generator(C_HB, HBondEnum, HBondEnum.NONE.value, "hbond")


# ---------------------------------------------------------------------------
# Index editor — keeps index groups in sync with the index mask
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
        arr = self._state.arrays[obj_name]
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
# Pre-save pipeline validation hook
# ---------------------------------------------------------------------------


def _run_pipeline_validation(arr: AtomArrayPlus) -> list[str]:
    from proteinfoundation.datapipes.pipeline import build_design_validation_pipeline

    pipeline = build_design_validation_pipeline()
    try:
        pipeline({"atom_array": arr.copy()})
        return []
    except ValueError as e:
        msg = str(e)
        if "Batch invariant violation" in msg:
            return [line.strip() for line in msg.splitlines() if line.strip().startswith("[")]
        return [f"[pipeline_error] {msg}"]
    except Exception as e:
        return [f"[pipeline_error] {e}"]


# ---------------------------------------------------------------------------
# Annotation summary hook
# ---------------------------------------------------------------------------


def _annotation_summary(arr: AtomArrayPlus) -> str:
    """Covers proteinfoundation conditions not already summarized by AtomWorks."""
    lines: list[str] = []

    # Enum-valued conditions: show per-value residue/atom counts (skip null value).
    for condition_cls, enum_cls, null_value, label in [
        (C_RAS, RasaEnum, RasaEnum.UNSPECIFIED.value, "RASA"),
        (C_SEC, SSEnum, SSEnum.NONE.value, "Secondary structure"),
        (C_HB, HBondEnum, HBondEnum.NONE.value, "Hydrogen bonds"),
    ]:
        if condition_cls.full_name not in arr.get_annotation_categories():
            continue
        annot = condition_cls.annotation(arr, default="raise")
        mask = condition_cls.mask(arr, default="generate")
        section_lines = []
        for enum_val in enum_cls:
            if enum_val.value == null_value:
                continue
            atom_mask = mask & (annot == enum_val.value)
            if not atom_mask.any():
                continue
            n = int(atom_mask.sum())
            section_lines.append(f"  {enum_val.name}: {n} atoms")
        if section_lines:
            lines.append(f"{label}:\n" + "\n".join(section_lines))

    # Mask-only conditions.
    if C_INTCRD.mask_name in arr.get_annotation_categories():
        intcrd_mask = C_INTCRD.mask(arr, default="generate")
        if intcrd_mask.any():
            lines.append(f"Internal coordinate: {int(intcrd_mask.sum())} atoms")

    # Group conditions.
    if C_IDXGRP.full_name in arr.get_annotation_categories():
        groups = arr.get_annotation(C_IDXGRP.full_name)
        n_groups = len(np.unique(groups))
        section = [f'Index groups: {n_groups} groups  (check "Color by index group" view)']
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
                    section.append(
                        f"  unindexed group {int(gid)} contains non-sequential res_ids: "
                        "these gaps will be preserved in the output. If this is not desired, "
                        "please adjust the index groups accordingly."
                    )
        lines.append("\n".join(section))

    if C_CRDGRP.full_name in arr.get_annotation_categories():
        n_groups = len(np.unique(arr.get_annotation(C_CRDGRP.full_name)))
        lines.append(f'Coordinate groups: {n_groups} groups  (check "Color by coord group" view)')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extension registration entry point
# ---------------------------------------------------------------------------


def register() -> None:
    """Register proteinfoundation condition editors and views with the AtomWorks PyMOL plugin.

    Called once at PyMOL plugin startup after atomworks is confirmed importable.
    """
    from atomworks.ml.utils.visualize import hide_condition_display
    from atomworks.pymol_plugin.panel import register_pre_save_hook, register_summary_hook

    # Views
    register_view("Color by index group", color_by_index_group)
    register_view("Color by coord group", color_by_coord_group)

    # Opt these conditions out of automatic mask-display selection generation.
    # The group masks are always 1 so trivial, and the termini displays are
    # confusing since they will be reset later (in the Transform pipeline)
    for _cond in (C_NTR, C_CTR, C_IDXGRP, C_CRDGRP):
        hide_condition_display(_cond)

    # Boolean mask conditions
    register_editor(C_HOT)(MaskEditor)

    # Index conditioning — custom MaskEditor that keeps index groups in sync.
    register_editor(C_IDX)(_IndexMaskEditor)

    # Enum-valued int conditions — per-value button editors
    register_editor(C_RAS)(RasaEditor)
    register_editor(C_SEC)(SecondaryStructureEditor)
    register_editor(C_HB)(HBondEditor)

    # Integer-valued group conditions — use the shared GroupEditor.
    register_editor(C_IDXGRP)(GroupEditor)
    register_editor(C_CRDGRP)(GroupEditor)

    # Internal coord conditioning — wraps convert_to_internal_coords.
    register_editor(C_INTCRD)(InternalCoordEditor)

    # TODO: Add these Conditions back once they are supported!
    exclude_annotation(C_CATH)
    exclude_annotation(C_CDR)
    exclude_annotation(C_ABH)
    exclude_annotation(C_ABL)
    exclude_annotation(C_DIS)

    # Computed/derived annotations not useful for interactive editing.
    exclude_annotation(SA_CRDSRC)
    exclude_annotation(SA_SRC)
    exclude_annotation(SA_CWI)
    exclude_annotation(SA_CRDGRPSRC)
    exclude_annotation(SA_UII)
    exclude_annotation(S_ATM)
    exclude_annotation(C_NTR)
    exclude_annotation(C_CTR)

    # Per-enum PyMOL selections created after each default view render.
    register_selection_generator(_rasa_selections)
    register_selection_generator(_ss_selections)
    register_selection_generator(_hbond_selections)

    # Pre-save validation hooks.
    register_pre_save_hook(_run_pipeline_validation)

    # Annotation summary hook.
    register_summary_hook(_annotation_summary)
