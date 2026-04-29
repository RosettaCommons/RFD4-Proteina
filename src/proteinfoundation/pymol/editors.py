"""PyMOL editor registrations for proteinfoundation conditions.

This module has no top-level PyMOL or Qt imports and is safe to import
during model training.  It is only executed inside a running PyMOL
process via the AtomWorks extension system.

Register with::

    atomworks pymol add proteinfoundation.pymol.editors
"""

from atomworks.pymol_plugin.editors import exclude_annotation, register_editor
from atomworks.pymol_plugin.editors.widgets import MaskEditor
from atomworks.pymol_plugin.views import register_view

from foundry.utils.standard_annotations import SA_CRDSRC, SA_CWI, SA_SRC
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
)
from proteinfoundation.pymol.group_editor import GroupEditor
from proteinfoundation.pymol.internal_coord_editor import InternalCoordEditor
from proteinfoundation.pymol.visualize import color_by_coord_group, color_by_index_group
from proteinfoundation.pymol.widgets import (
    HBondEditor,
    RasaEditor,
    SecondaryStructureEditor,
)


def register() -> None:
    """Register proteinfoundation condition editors and views with the AtomWorks PyMOL plugin.

    Called once at PyMOL plugin startup after atomworks is confirmed importable.
    """
    # Views
    register_view("Color by index group", color_by_index_group)
    register_view("Color by coord group", color_by_coord_group)

    # Boolean mask conditions
    register_editor(C_HOT)(MaskEditor)

    # Enum-valued int conditions — per-value button editors
    register_editor(C_RAS)(RasaEditor)
    register_editor(C_SEC)(SecondaryStructureEditor)
    register_editor(C_HB)(HBondEditor)

    # Integer-valued group conditions — use the shared GroupEditor.
    register_editor(C_IDXGRP)(GroupEditor)
    register_editor(C_CRDGRP)(GroupEditor)

    # Internal coord conditioning — wraps convert_to_internal_coords.
    register_editor(C_INTCRD)(InternalCoordEditor)

    exclude_annotation(C_CATH)
    exclude_annotation(C_CDR)
    exclude_annotation(C_ABH)
    exclude_annotation(C_ABL)

    # Computed/derived annotations not useful for interactive editing.
    exclude_annotation(SA_CRDSRC)
    exclude_annotation(SA_SRC)
    exclude_annotation(SA_CWI)
