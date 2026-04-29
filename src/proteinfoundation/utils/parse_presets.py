"""Project-specific parsing presets.

Import these directly or pass to :py:func:`~atomworks.io.parser.parse` as the
``config`` argument:

    from proteinfoundation.utils.parse_presets import RCSB_WITH_METALS
    result = parse(path, config=RCSB_WITH_METALS)
"""

from atomworks.constants import CCD_MIRROR_PATH, CRYSTALLIZATION_AIDS
from atomworks.io.config import ParseConfig

RCSB_WITH_METALS: ParseConfig = ParseConfig(
    # --- PrepareConfig fields ---
    add_missing_atoms=True,
    # Allow covalent + metal coordination bonds (dropout of some metal coordination bonds happens later)
    add_bond_types_from_struct_conn=("covale", "metalc"),
    remove_ccds=tuple(CRYSTALLIZATION_AIDS),
    remove_waters=True,
    fix_ligands_at_symmetry_centers=True,
    fix_arginines=True,
    sanitize_bonds_and_charges=True,
    long_bond_policy="warn",
    struct_conn_distance_policy="keep",  # TODO: Will be updated in the future
    add_id_and_entity_annotations=True,
    convert_mse_to_met=True,
    hydrogen_policy="keep",
    ccd_mirror_path=CCD_MIRROR_PATH,
    # --- ParseConfig-only fields ---
    model=None,
    build_assembly="all",
    extra_fields=None,
    file_type=None,
    # Default altloc handling for determinism while generating non-clashing conformers
    altloc="random_clash_aware",
    altloc_seed=0,
    # Do not load standard annotations by default (we can override for validation parsers)
    load_standard_annotations=False,
    keep_cif_block=False,
    # No caching by default (we can override if/when needed)
    cache_dir=None,
    save_to_cache=False,
    load_from_cache=False,
)
"""Like the ``"rcsb"`` preset, but also keeps metal coordination bonds from ``struct_conn``."""

DISTILLATION: ParseConfig = RCSB_WITH_METALS.replace(
    add_missing_atoms=False,
    fix_arginines=False,
    sanitize_bonds_and_charges=False,
    convert_mse_to_met=False,
    altloc="first",  # AFDB / TED distillation files are PDB format; PDB doesn't support 'random_clash_aware'
)
"""Lightweight config for datasets that require minimal processing; e.g., predicted structures"""

ANNOTATED_CIF: ParseConfig = DISTILLATION.replace(load_standard_annotations=True, remove_waters=False, remove_ccds=())
"""Config for loading annotated cifs that makes minimal changes to structure while loading standard annotations"""


STANDARD_ANNOTATIONS_ONLY: ParseConfig = DISTILLATION.replace(
    load_standard_annotations=True,
    remove_waters=False,
    remove_ccds=(),
)
"""Minimal config that loads StandardAnnotations with minimal intervention.

Used for parsing oracle-predicted structures (RF3, AF3 outputs).
"""


def rcsb_with_metals(**overrides) -> dict:
    """``RCSB_WITH_METALS`` as a parser_args dict, with optional kwarg overrides."""
    return RCSB_WITH_METALS.replace(**overrides).to_dict()


def distillation(**overrides) -> dict:
    """``DISTILLATION`` as a parser_args dict, with optional kwarg overrides."""
    return DISTILLATION.replace(**overrides).to_dict()


def annotated_cif(**overrides) -> dict:
    """``ANNOTATED_CIF`` as a parser_args dict, with optional kwarg overrides."""
    return ANNOTATED_CIF.replace(**overrides).to_dict()
