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

from atomworks.io.utils.atom_array_plus import AtomArrayPlus

# These imports exist purely to force class registration before
# condition_registry.apply() sweeps the StandardAnnotation registry: anything
# registered by then and not explicitly listed as supported gets excluded
# automatically, so these no longer need individual exclude_annotation() calls.
from atomworks.io.utils.standard_annotations import S_ATM  # noqa: F401
from atomworks.pymol_plugin.views import register_view

from rfproteina.pymol import condition_registry
from rfproteina.pymol.visualize import color_by_coord_group, color_by_index_group
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
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline

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
