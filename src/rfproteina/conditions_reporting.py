"""PyMOL-free Condition vocabulary, reporting, and pre-save validation.

Extracted from :py:mod:`rfproteina.pymol.condition_registry` /
:py:mod:`rfproteina.pymol.editors`, which transitively import PyMOL at module
scope and so cannot be imported from ``generate.py``'s environment (PyMOL
lives in a separate conda env -- confirmed by direct import test). This
module has no PyMOL/Qt imports and is safe to import during training/inference,
notably from :py:mod:`rfproteina.inference_utils.condition_spec`.

:py:data:`SUPPORTED_CONDITIONS` is the single vocabulary of which Conditions
the codebase knows about: which gets a PyMOL editor (and how), and how each
is summarized in the "summarize changes" report. These used to be two
separately-maintained lists (an editor list in ``condition_registry.py`` and
a reporting list here) that happened to overlap almost entirely -- easy to
add a Condition to one and forget the other. Since this module must stay
PyMOL-free, ``editor`` is a zero-argument factory that defers its
``pymol``/Qt imports until actually called, which only happens inside a live
PyMOL session, from :py:func:`~rfproteina.pymol.condition_registry.apply`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np
from atomworks.io.transforms.atom_array import (
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
)
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import AtomArrayPlus
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.conditions.base import ConditionBase
from atomworks.ml.transforms.dssp import SSEnum

from rfproteina.conditions import (
    C_CENTROID,
    C_CRDGRP,
    C_HB,
    C_HOT,
    C_IDXGRP,
    C_INTCRD,
    C_RAS,
    C_SEC,
    HBondEnum,
    RasaEnum,
)
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.transforms.invariants import find_index_group_res_id_gaps

# ---------------------------------------------------------------------------
# Editor factories -- each defers its `pymol`/Qt import to call time, since
# this module itself must stay importable with no PyMOL installed. Actually
# calling one of these outside a live PyMOL session will raise ImportError,
# same as importing `rfproteina.pymol.editors` directly would.
# ---------------------------------------------------------------------------


def _mask_editor() -> type:
    from atomworks.pymol_plugin.editors.widgets import MaskEditor

    return MaskEditor


def _index_mask_editor() -> type:
    from rfproteina.pymol.condition_registry import _IndexMaskEditor

    return _IndexMaskEditor


def _group_editor() -> type:
    from rfproteina.pymol.group_editor import GroupEditor

    return GroupEditor


def _internal_coord_editor() -> type:
    from rfproteina.pymol.internal_coord_editor import InternalCoordEditor

    return InternalCoordEditor


def _chain_centroid_editor() -> type:
    from rfproteina.pymol.chain_centroid_editor import ChainCentroidEditor

    return ChainCentroidEditor


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionSpec:
    """Everything the codebase needs to know about one user-facing Condition.

    A single list of these, instead of separate PyMOL-editor and reporting
    vocabularies that happened to overlap almost entirely -- see the module
    docstring.

    Args:
        condition: The Condition class this spec describes.
        editor: Zero-arg factory returning an editor class (or an
            ``editor_cls(annotation_cls, state, parent=None)``-compatible
            callable) to register in the PyMOL plugin. ``None`` means either
            an enum-driven editor is built automatically (see ``enum_cls``),
            or the editor is registered elsewhere (e.g. atomworks core
            registers ``C_SEQ``/``C_CRD``) and this spec exists purely to
            keep the condition from being auto-excluded.
        enum_cls: ``IntEnum`` backing this condition's values, if any. Drives
            an automatic PyMOL value-button editor when ``editor`` is unset,
            a per-value summary line, and (if ``selection_generator``)
            per-value PyMOL highlight selections.
        null_value: The enum member that represents "unconditioned". Required
            whenever ``enum_cls`` is set.
        enum_editor_exclude: Enum values the auto-built PyMOL editor must not
            offer, because the model cannot consume them (e.g.
            ``SSEnum.NON_PROTEIN``, outside the secondary-structure
            embedding's range). Still reported/highlighted if present.
        selection_generator: Register a per-value PyMOL highlight-selection
            generator. Only meaningful with ``enum_cls`` set.
        hide_display: Suppress automatic PyMOL mask-highlight-selection
            generation (e.g. group conditions, whose mask is always all-True
            and thus uninformative to highlight). Independent of reporting.
        summary_label: Display label in the "summarize changes" text;
            defaults to ``condition.name``.
        summary_fn: Custom summary-line builder, for conditions whose
            reporting doesn't fit the generic mask-count / per-enum-value
            shape (e.g. index/coord groups, chain centroid). Overrides the
            default mask/enum-derived summary below.
        report_summary: Whether :py:func:`summarize_conditions` emits a line
            for this condition. ``False`` for conditions whose value is
            redundant with another summary line (``C_IDX``, conveyed by the
            index-group summary); the spec still exists to keep the condition
            from being auto-excluded and to register its editor.
        summarized_by_host: Whether an embedding application reports this
            condition through its *own* summary hook, alongside ours
            (AtomWorks' built-in PyMOL hook does, for ``C_SEQ``/``C_CRD``).
            Such conditions are omitted by default so the PyMOL report doesn't
            list them twice, and included when the caller passes
            ``include_host_summarized=True`` because no host hook is running --
            notably the condition_spec CLI, whose printed report is the only
            confirmation the user gets that these were applied.
    """

    condition: type[ConditionBase]
    editor: Callable[[], type | Callable] | None = None
    enum_cls: type[IntEnum] | None = None
    null_value: int | None = None
    enum_editor_exclude: frozenset[int] = frozenset()
    selection_generator: bool = False
    hide_display: bool = False
    summary_label: str | None = None
    summary_fn: Callable[[AtomArrayPlus], list[str]] | None = None
    report_summary: bool = True
    summarized_by_host: bool = False

    def __post_init__(self) -> None:
        if self.enum_cls is not None and self.null_value is None:
            raise ValueError(f"{self.condition.name}: enum_cls requires null_value")
        if self.selection_generator and self.enum_cls is None:
            raise ValueError(f"{self.condition.name}: selection_generator requires enum_cls")

    @property
    def label(self) -> str:
        return self.summary_label or self.condition.name

    @property
    def summary_kind(self) -> Literal["mask", "enum", "custom"]:
        """How :py:func:`summarize_conditions` reports this condition, derived from the other fields."""
        if self.summary_fn is not None:
            return "custom"
        if self.enum_cls is not None:
            return "enum"
        return "mask"


def _index_group_summary(arr: AtomArrayPlus) -> list[str]:
    if not C_IDXGRP.has_annotation(arr):
        return []
    groups = arr.get_annotation(C_IDXGRP.full_name)
    n_groups = len(np.unique(groups))
    lines = [f'Index groups: {n_groups} groups  (check "Color by index group" in atomworks studio viewer)']
    # Report implied res_id gaps within unindexed groups so the user can confirm
    # that non-contiguous regions were split as intended (rather than collapsed
    # into one group with fixed-size gaps).
    if C_IDX.has_mask(arr):
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
    if not C_CRDGRP.has_annotation(arr):
        return []
    n_groups = len(np.unique(arr.get_annotation(C_CRDGRP.full_name)))
    return [f'Coordinate groups: {n_groups} groups  (check "Color by coord group" view)']


def _chain_centroid_summary(arr: AtomArrayPlus) -> list[str]:
    if not C_CENTROID.has_annotation(arr):
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


SUPPORTED_CONDITIONS: list[ConditionSpec] = [
    # AtomWorks-native conditions: editors registered by atomworks core itself
    # (register_custom_editors() in atomworks/pymol_plugin/editors/custom.py).
    # AtomWorks' own built-in hook reports these inside PyMOL, so they're
    # `summarized_by_host` -- suppressed there, emitted for callers with no
    # host hook of their own (the condition_spec CLI).
    ConditionSpec(C_SEQ, summary_label="Sequence", summarized_by_host=True),
    ConditionSpec(C_CRD, summary_label="Coordinates", summarized_by_host=True),
    # Index conditioning -- rfproteina overrides the editor to keep index groups in sync;
    # the set of indexed atoms is already conveyed by the index-group summary.
    ConditionSpec(C_IDX, editor=_index_mask_editor, summary_label="Index", report_summary=False),
    # Simple boolean mask conditions.
    ConditionSpec(C_HOT, editor=_mask_editor, summary_label="Hotspot"),
    # Enum-valued conditions: editor, summary, and (where useful) PyMOL
    # highlight selections are all built generically from enum_cls/null_value.
    ConditionSpec(
        C_RAS,
        enum_cls=RasaEnum,
        null_value=RasaEnum.UNSPECIFIED.value,
        selection_generator=True,
        summary_label="RASA",
    ),
    # NON_PROTEIN is a DSSP-output sentinel, not a conditioning value: the model's
    # secondary-structure embedding only spans NONE/HELIX/SHEET/LOOP and raises on 4.
    ConditionSpec(
        C_SEC,
        enum_cls=SSEnum,
        null_value=SSEnum.NONE.value,
        enum_editor_exclude=frozenset({SSEnum.NON_PROTEIN.value}),
        selection_generator=True,
        summary_label="Secondary structure",
    ),
    ConditionSpec(
        C_HB,
        enum_cls=HBondEnum,
        null_value=HBondEnum.NONE.value,
        selection_generator=True,
        summary_label="Hydrogen bonds",
    ),
    # Integer-valued group conditions -- shared GroupEditor, bespoke summary
    # (group count + view hint), hidden from mask-highlighting (mask is
    # trivially all-True and thus uninformative to highlight).
    ConditionSpec(C_IDXGRP, editor=_group_editor, hide_display=True, summary_fn=_index_group_summary),
    ConditionSpec(C_CRDGRP, editor=_group_editor, hide_display=True, summary_fn=_coord_group_summary),
    # Internal coordinate conditioning -- wraps convert_to_internal_coords.
    ConditionSpec(C_INTCRD, editor=_internal_coord_editor, summary_label="Internal coordinates"),
    # Chain centroid conditioning -- manual xyz entry + indicator spheres.
    ConditionSpec(C_CENTROID, editor=_chain_centroid_editor, summary_fn=_chain_centroid_summary),
    # BasePair: excluded for now (2-body enum condition; no dedicated editor
    # yet -- the generic PairValueEditor fallback is built for Angstrom
    # distances and would be actively wrong for PAIR/LOOP semantics).
]


def _mask_summary(spec: ConditionSpec, arr: AtomArrayPlus) -> list[str]:
    if not spec.condition.has_mask(arr):
        return []
    mask = spec.condition.mask(arr, default="generate")
    if not mask.any():
        return []
    return [f"{spec.label}: {int(mask.sum())} atoms"]


def _enum_summary(spec: ConditionSpec, arr: AtomArrayPlus) -> list[str]:
    if not spec.condition.has_annotation(arr):
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


def summarize_conditions(arr: AtomArrayPlus, *, include_host_summarized: bool = False) -> str:
    """Summarize every set condition/annotation covered by :py:data:`SUPPORTED_CONDITIONS`.

    Args:
      arr: The annotated structure to summarize.
      include_host_summarized: Also emit lines for conditions an embedding
        application reports through its own summary hook (see
        :py:attr:`ConditionSpec.summarized_by_host`). Leave ``False`` inside
        PyMOL, where AtomWorks' hook already lists them; pass ``True`` when this
        report is the only one the user sees.
    """
    lines: list[str] = []
    for spec in SUPPORTED_CONDITIONS:
        if not spec.report_summary:
            continue
        if spec.summarized_by_host and not include_host_summarized:
            continue
        kind = spec.summary_kind
        if kind == "mask":
            lines.extend(_mask_summary(spec, arr))
        elif kind == "enum":
            lines.extend(_enum_summary(spec, arr))
        else:
            lines.extend(spec.summary_fn(arr))
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-save pipeline validation
# ---------------------------------------------------------------------------


def run_pipeline_validation(arr: AtomArrayPlus) -> list[str]:
    """Dry-run the design validation pipeline against ``arr``, without raising.

    Returns:
      Empty list if valid, otherwise one message per violation.
    """
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


def check_index_group_res_id_gaps(arr: AtomArrayPlus) -> list[str]:
    """Return a violation string for each internal res_id gap within an index group.

    Unlike a per-chain check, discontinuities are only flagged *within the same
    index group* (:py:data:`~rfproteina.conditions.C_IDXGRP`). Gaps that fall on
    an index-group boundary are expected (e.g. unindexed insertions start a new
    group) and are not reported. Intra-group gaps are interpreted by the model as
    breaks within a contiguously-indexed region, so they are usually unintentional.

    Atomized residues (PTMs, ligands, etc.) are skipped: they can share an index
    group with a peptide chain while carrying unrelated res_ids, since atomized
    segments use bond-graph distance -- not res_id -- for relative positional
    encoding. This mirrors the token filtering in
    :py:func:`~rfproteina.transforms.invariants.check_idxgrp_violations`.
    """
    messages = find_index_group_res_id_gaps(arr)
    if not messages:
        return []
    return [
        f"[res_id_gap] Residue numbering has {len(messages)} internal gap(s) within index "
        f"groups: {'; '.join(messages)}. Intra-group gaps are interpreted by the model as "
        "breaks within a contiguously-indexed region -- make sure this is intentional."
    ]


def _molecule_iids(arr: AtomArrayPlus) -> np.ndarray:
    """Return the per-atom ``molecule_iid`` annotation, deriving it if absent.

    Derivation happens on a copy so the caller's array is never mutated. Mirrors
    the molecule bookkeeping in
    :py:meth:`~rfproteina.conditions.IndexGroup.default_annotation`.
    """
    if "molecule_iid" in arr.get_annotation_categories():
        return arr.molecule_iid
    arr = arr.copy()
    ensure_annotations(arr, "is_polymer")
    arr = add_id_and_entity_annotations(arr)
    if "transformation_id" not in arr.get_annotation_categories():
        arr.set_annotation("transformation_id", np.full(arr.array_length(), "1", dtype=str))
    arr = add_molecule_iid_annotation(arr)
    return arr.molecule_iid


def check_index_group_molecule_consistency(arr: AtomArrayPlus) -> list[str]:
    """Return a violation for each molecule whose indexed atoms span multiple index groups.

    Every globally-indexed atom (``C_IDX`` mask true) belonging to the same
    molecule (``molecule_iid``) must share a single index group
    (:py:data:`~rfproteina.conditions.C_IDXGRP`). The model reasons over
    molecules, so a PTM/ligand bonded to a peptide chain is part of that chain's
    molecule and must live in its index group. A molecule whose indexed atoms are
    split across groups means part of a single contiguously-indexed entity is
    being treated as a separate region, which is almost always an annotation
    mistake.
    """
    idx_mask = C_IDX.mask(arr, default="generate")
    if not idx_mask.any():
        return []

    molecule_iids = _molecule_iids(arr)
    index_groups = C_IDXGRP.annotation(arr, default="generate")

    messages: list[str] = []
    for mol_id in np.unique(molecule_iids[idx_mask]):
        groups = np.unique(index_groups[idx_mask & (molecule_iids == mol_id)])
        if groups.size > 1:
            group_list = ", ".join(str(int(g)) for g in groups)
            messages.append(f"molecule {mol_id} spans index groups {{{group_list}}}")

    if not messages:
        return []
    return [
        f"[index_group_molecule] {len(messages)} molecule(s) have indexed atoms split across "
        f"multiple index groups: {'; '.join(messages)}. All indexed atoms of a molecule must "
        "share one index group (a PTM/ligand belongs to its chain's molecule and index group) "
        "-- make sure this is intentional."
    ]


# ---------------------------------------------------------------------------
# Index group sync (pure logic behind the Qt-coupled `_IndexMaskEditor`)
# ---------------------------------------------------------------------------


def sync_index_groups(arr: AtomArrayPlus, preserve_manual_edits: bool = True) -> None:
    """Recompute :py:data:`~rfproteina.conditions.C_IDXGRP` in place on ``arr``.

    Enforces mandatory split points (molecule boundaries and ``C_IDX``
    transitions) while optionally preserving any manual group subdivisions
    already present.

    Args:
      arr: Structure to update in place.
      preserve_manual_edits: If ``True`` (default), two atoms keep sharing a
        group iff they already share both the default group and the current
        group. If ``False``, groups are fully reset to the default split.
    """
    default_groups = C_IDXGRP.default_annotation(arr)
    if preserve_manual_edits and C_IDXGRP.has_annotation(arr):
        current_groups = arr.get_annotation(C_IDXGRP.full_name)
        combined = np.stack([default_groups, current_groups], axis=1)
        _, new_groups = np.unique(combined, axis=0, return_inverse=True)
    else:
        new_groups = default_groups
    C_IDXGRP.set_annotation(arr, new_groups)
