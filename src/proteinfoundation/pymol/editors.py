"""PyMOL editor registrations for proteinfoundation conditions.

This module has no top-level PyMOL or Qt imports and is safe to import
during model training.  It is only executed inside a running PyMOL
process via the AtomWorks extension system.

Register with::

    atomworks pymol add proteinfoundation.pymol.editors
"""

from atomworks.pymol_plugin.editors import exclude_annotation, register_editor
from atomworks.pymol_plugin.editors.widgets import MaskEditor

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
from proteinfoundation.pymol.widgets import (
    HBondEditor,
    RasaEditor,
    SecondaryStructureEditor,
)


def register() -> None:
    """Register proteinfoundation condition editors and visualizer with the AtomWorks PyMOL plugin.

    Called once at PyMOL plugin startup after atomworks is confirmed importable.
    All Qt and PyMOL imports are deferred inside this function.
    """

    # Boolean mask conditions
    register_editor(C_HOT)(MaskEditor)

    # Enum-valued int conditions — per-value button editors
    register_editor(C_RAS)(RasaEditor)
    register_editor(C_SEC)(SecondaryStructureEditor)
    register_editor(C_HB)(HBondEditor)

    # TODO: Annotations to support soon
    # register_editor(C_CRDGRP)(CoordinateGroupEditor)  # TODO: improve visualization and add back
    exclude_annotation(C_CATH)
    exclude_annotation(C_CDR)
    exclude_annotation(C_ABH)
    exclude_annotation(C_ABL)
    exclude_annotation(C_CRDGRP)

    # Computed/derived annotations not useful for interactive editing.
    exclude_annotation(C_IDXGRP)
    exclude_annotation(C_INTCRD)
    exclude_annotation(SA_CRDSRC)
    exclude_annotation(SA_SRC)
    exclude_annotation(SA_CWI)

    # Register custom visualizer that adds C_CRDGRP coloring on top of the base.
    # register_visualizer(visualize)  # TODO: Same as above
