"""PyMOL editor registrations for rfproteina conditions.

This module has no top-level PyMOL or Qt imports and is safe to import
during model training.  It is only executed inside a running PyMOL
process via the AtomWorks extension system.

Which conditions get an editor, which are hidden from the wizard, and how
each is summarized all live in :py:mod:`rfproteina.pymol.condition_registry`
-- this module just wires that registry up alongside views and the pre-save
validation hook.

Register with::

    atomworks pymol add rfproteina.pymol.editors
"""

import numpy as np
from atomworks.io.transforms.atom_array import (
    add_id_and_entity_annotations,
    add_molecule_iid_annotation,
)
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.atom_array_plus import AtomArrayPlus

# Imported for its class-registration side effect (forces StandardAnnotation registration before
# condition_registry.apply() sweeps the registry), so unregistered annotations get excluded
# automatically without individual exclude_annotation() calls.
from atomworks.io.utils.standard_annotations import S_ATM  # noqa: F401
from atomworks.ml.conditions import C_IDX
from atomworks.pymol_plugin.views import register_view

from rfproteina.conditions import C_IDXGRP
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.pymol import condition_registry
from rfproteina.pymol.visualize import color_by_coord_group, color_by_index_group
from rfproteina.transforms.invariants import find_index_group_res_id_gaps
from rfproteina.utils.standard_annotations import (  # noqa: F401
    SA_CRDGRPSRC,
    SA_CRDSRC,
    SA_CWI,
    SA_SRC,
    SA_UII,
)

# ---------------------------------------------------------------------------
# Pre-save pipeline validation hook
# ---------------------------------------------------------------------------


def _run_pipeline_validation(arr: AtomArrayPlus) -> list[str]:
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


def _check_index_group_res_id_gaps(arr: AtomArrayPlus) -> list[str]:
    """Pre-save hook: warn on internal res_id gaps within an index group.

    Gap-finding lives in :py:func:`~rfproteina.transforms.invariants.find_index_group_res_id_gaps`
    (headless, shared with json2cif's check_annotated_cif); this only formats the message.
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


def _check_index_group_molecule_consistency(arr: AtomArrayPlus) -> list[str]:
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
# Extension registration entry point
# ---------------------------------------------------------------------------


def register() -> None:
    """Register rfproteina condition editors and views with the AtomWorks PyMOL plugin.

    Called once at PyMOL plugin startup after atomworks is confirmed importable.
    """
    from atomworks.pymol_plugin.panel import register_pre_save_hook

    # Views
    register_view("Color by index group", color_by_index_group)
    register_view("Color by coord group", color_by_coord_group)

    # Editors, wizard visibility, and the summary hook for every supported
    # annotation -- see condition_registry.SUPPORTED_CONDITIONS. Sweeps the
    # whole StandardAnnotation registry, so the computed/derived annotations
    # imported above (not useful for interactive editing) are excluded
    # automatically for not being listed there.
    condition_registry.apply()

    # Pre-save validation hooks.
    register_pre_save_hook(_run_pipeline_validation)
    register_pre_save_hook(_check_index_group_res_id_gaps)
    register_pre_save_hook(_check_index_group_molecule_consistency)
