"""Build the input AF3 folds for a design: the construction RFD4's AF3 oracle does, importable on its own.

:py:class:`~rfproteina.metrics.oracles.af3_oracle.AF3Oracle` hands AF3 a design as a CIF, a *user CCD* and optional
templates. That construction is here, as plain functions of an AtomWorks :class:`~biotite.structure.AtomArray`, so other
projects can produce exactly the input the oracle folds without constructing an oracle, importing its GPU and
scheduler machinery, or reaching for its private methods::

    from atomworks.io.parser import parse
    from rfproteina.af3.af3_input import build_af3_input
    from rfproteina.utils.parse_presets import STANDARD_ANNOTATIONS_ONLY

    atom_array = parse("design.cif", config=STANDARD_ANNOTATIONS_ONLY)["assemblies"]["1"][0]
    af3_input = build_af3_input(atom_array, name="design")
    af3_input.cif  # the design, chains disambiguated, covalent modifiers renamed, NaN atoms dropped
    af3_input.user_ccd  # a dictionary block for every non-standard residue and ligand, from the design's own atoms
    af3_input.template  # {chain_id: {"mmcif", "query_indices", "template_indices"}}, or None

Importing this module does not load the oracle: no ``AF3Oracle``, daemon or apptainer helpers, no metric framework. The
template-query resolver (AtomWorks ML conditions) is imported only when a ``template_query`` is given. ``AF3Oracle``
builds its manifest entries with :py:func:`build_af3_input`, so the two cannot drift.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import biotite.structure as struc
import numpy as np
from atomworks.constants import STANDARD_AA
from atomworks.io.utils.annotator import ensure_annotations
from atomworks.io.utils.bonds import hash_atom_array
from atomworks.io.utils.chain import create_chain_id_generator
from atomworks.io.utils.io_utils import CIFWriteConfig, _to_cif_or_bcif, to_cif_string
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.residues import apply_residue_wise, residue_iter

logger = logging.getLogger(__name__)


# AF3 CCD validator requires _chem_comp_bond.* keys to be present even when a
# residue has no intra-residue bonds (e.g. sulfide ions in Fe-S clusters).
# An empty loop_ with just column headers satisfies the check; AF3's C++ CIF
# parser registers the keys with empty value lists, and parse_bond_data()
# handles empty lists gracefully.
_CHEM_COMP_BOND_EMPTY_LOOP = (
    "loop_\n"
    "_chem_comp_bond.atom_id_1\n"
    "_chem_comp_bond.atom_id_2\n"
    "_chem_comp_bond.value_order\n"
    "_chem_comp_bond.pdbx_aromatic_flag\n"
)


def inject_empty_chem_comp_bond(cif_str: str) -> str:
    """Inject an empty _chem_comp_bond loop into any CIF block that lacks one."""

    def _process_block(block: str) -> str:
        if "_chem_comp_bond.atom_id_1" in block:
            return block
        stripped = block.rstrip()
        if stripped.endswith("#"):
            return stripped[:-1] + _CHEM_COMP_BOND_EMPTY_LOOP + "#\n"
        return stripped + "\n" + _CHEM_COMP_BOND_EMPTY_LOOP

    parts = re.split(r"(?=^data_)", cif_str, flags=re.MULTILINE)
    return "".join(_process_block(p) if p.startswith("data_") else p for p in parts)


_OF3_PROVENANCE_FILENAME = "of3_ported_weights.provenance.json"
# Custom CCD codes shaped like the one tests/e2e uses for its user-CCD heme.
_ATOMIZED_CODE_FORMAT = "L:{index}"


def rename_covalent_modifiers_for_af3(atom_array: AtomArray) -> AtomArray:
    """Give protein-linked modifiers custom CCD codes built from their observed atoms."""
    if atom_array.bonds is None:
        return atom_array

    renamed = atom_array.copy()
    ensure_annotations(renamed, "is_polymer", "is_standard_aa")
    residue_starts = struc.get_residue_starts(renamed, add_exclusive_stop=True)
    atom_to_residue = np.repeat(np.arange(len(residue_starts) - 1), np.diff(residue_starts))

    modifier_residues: set[int] = set()
    for atom_i, atom_j in renamed.bonds.as_array()[:, :2]:
        atom_i, atom_j = int(atom_i), int(atom_j)
        if atom_to_residue[atom_i] == atom_to_residue[atom_j]:
            continue
        for polymer_atom, modifier_atom in ((atom_i, atom_j), (atom_j, atom_i)):
            if (
                renamed.is_polymer[polymer_atom]
                and renamed.is_standard_aa[polymer_atom]
                and not renamed.is_polymer[modifier_atom]
            ):
                modifier_residues.add(int(atom_to_residue[modifier_atom]))
                break

    if not modifier_residues:
        return atom_array

    res_names = np.asarray(renamed.res_name).astype(object)
    existing_codes = set(np.unique(renamed.res_name).astype(str).tolist())
    code_by_graph: dict[str, str] = {}
    next_index = 0
    for residue_index in sorted(modifier_residues):
        start, stop = int(residue_starts[residue_index]), int(residue_starts[residue_index + 1])
        graph_hash = hash_atom_array(
            renamed[start:stop],
            annotations=("element",),
            bond_order=True,
            cast_aromatic_bonds_to_same_type=True,
            use_md5=True,
        )
        if graph_hash not in code_by_graph:
            code = _ATOMIZED_CODE_FORMAT.format(index=next_index)
            next_index += 1
            while code in existing_codes:
                code = _ATOMIZED_CODE_FORMAT.format(index=next_index)
                next_index += 1
            code_by_graph[graph_hash] = code
            existing_codes.add(code)
        res_names[start:stop] = code_by_graph[graph_hash]

    renamed.set_annotation("res_name", res_names.astype(str))
    for name in ("is_standard_aa", "is_standard_dna", "is_standard_rna"):
        if name in renamed.get_annotation_categories():
            renamed.del_annotation(name)
    return renamed


def using_of3_weights() -> bool:
    """Whether ``AF3_MODEL_DIR`` holds converted OpenFold3 parameters, per its provenance stamp.

    Duplicated in ``run_alphafold_batch``, which runs in the AF3 image without ``rfproteina``.
    """
    model_dir = os.environ.get("AF3_MODEL_DIR", "").strip()
    provenance = Path(model_dir) / _OF3_PROVENANCE_FILENAME if model_dir else None
    if provenance is None or not provenance.is_file():
        return False
    # Deliberately unguarded: a corrupt stamp must not quietly fall back to AF3 weights.
    return json.loads(provenance.read_text()).get("weight_family") == "openfold3"


def atomize_ligand_bonded_residues(atom_array: AtomArray) -> AtomArray:
    """Give a custom CCD code to standard residues covalently bonded to a non-polymer.

    OpenFold3 tokenizes those per atom; a custom code is the only lever AF3 exposes.
    Returns a copy, or the input if there are none.
    """
    if atom_array.bonds is None:
        return atom_array
    ensure_annotations(atom_array, "is_protein", "is_nucleic_acid")
    is_polymer = atom_array.is_protein | atom_array.is_nucleic_acid

    targets: set[tuple[str, int]] = set()
    for i, j, _bond_type in atom_array.bonds.as_array():
        if is_polymer[i] == is_polymer[j]:
            continue
        polymer_idx = i if is_polymer[i] else j
        # Non-standard residues are already atomized by AF3, so leave their identity alone.
        if str(atom_array.res_name[polymer_idx]) not in STANDARD_AA:
            continue
        targets.add((str(atom_array.chain_id[polymer_idx]), int(atom_array.res_id[polymer_idx])))

    if not targets:
        return atom_array
    renamed = atom_array.copy()
    # Skip codes already present, including any minted by rename_covalent_modifiers_for_af3,
    # so two structurally different residues never share one custom CCD code.
    existing_codes = set(np.unique(renamed.res_name).astype(str).tolist())
    next_index = 1
    for chain_id, res_id in sorted(targets):
        code = _ATOMIZED_CODE_FORMAT.format(index=next_index)
        next_index += 1
        while code in existing_codes:
            code = _ATOMIZED_CODE_FORMAT.format(index=next_index)
            next_index += 1
        existing_codes.add(code)
        mask = (renamed.chain_id == chain_id) & (renamed.res_id == res_id)
        renamed.res_name[mask] = code
    for name in ("is_standard_aa", "is_standard_dna", "is_standard_rna"):
        if name in renamed.get_annotation_categories():
            renamed.del_annotation(name)
    return renamed


def build_af3_template_dict(
    atom_array: AtomArray, template_mask: np.ndarray, template_query: str | None = None
) -> dict[str, dict] | None:
    """Build AF3 templates using the selection resolved before residue and chain renaming.

    Args:
        atom_array: Prepared structure with finite coordinates.
        template_mask: Boolean selection aligned to ``atom_array`` after filtering.

    Returns:
        Dict mapping chain ID → template dict, or ``None`` if the selection contains no atoms.
    """
    if not np.any(template_mask):
        return None

    ensure_annotations(atom_array, "is_protein")
    non_protein_templated = template_mask & ~atom_array.is_protein
    if np.any(non_protein_templated):
        bad_res = np.unique(atom_array.res_name[non_protein_templated])
        raise ValueError(
            f"Template query {template_query!r} selects non-protein residues: {bad_res}. "
            "AF3 structural templating is only supported for protein chains; "
            "exclude them in the query (e.g. '& ~is_nucleic_acid')."
        )

    template_dict: dict[str, dict] = {}
    for chain_id in np.unique(atom_array.chain_id):
        chain_mask = atom_array.chain_id == chain_id
        chain_atoms = atom_array[chain_mask]
        chain_template_mask = template_mask[chain_mask]
        if not np.any(chain_template_mask):
            continue

        residue_is_templated_all = apply_residue_wise(chain_atoms, chain_template_mask, np.all)
        residue_is_templated_any = apply_residue_wise(chain_atoms, chain_template_mask, np.any)
        if not np.array_equal(residue_is_templated_all, residue_is_templated_any):
            logger.warning(
                f"Some residues on chain {chain_id} are partially templated. "
                "AF3 structural templating only applies to full residues; "
                "any partially-templated residue will be fully templated."
            )

        query_indices = np.where(residue_is_templated_any)[0].tolist()
        config = CIFWriteConfig(
            id=f"template_chain_{chain_id}",
            extra_categories={"pdbx_audit_revision_history": {"revision_date": date.today().isoformat()}},
        )
        template_dict[chain_id] = {
            "mmcif": to_cif_string(chain_atoms, config=config),
            "query_indices": query_indices,
            "template_indices": query_indices,
        }
    return template_dict or None


def build_user_ccd_string(atom_array: AtomArray) -> str | None:
    """Build the user CCD string for AF3 from the input AtomArray.

    Anything that is not a standard polymer residue is added to the user CCD.
    This ensures that reusing residue names from the CCD is handled correctly.

    NOTE: This will not handle the situation where a given residue name means
    two different things within the same AtomArray. That should never be done.

    Args:
        atom_array: Input structure as an AtomArray.

    Returns:
        Multi-block CIF string for non-standard residues, or ``None``.
    """
    ensure_annotations(atom_array, "is_standard_aa", "is_standard_dna", "is_standard_rna")
    nonstandard_mask = ~(atom_array.is_standard_aa | atom_array.is_standard_dna | atom_array.is_standard_rna)
    if not np.any(nonstandard_mask):
        return None

    user_ccd_arrays: dict[str, AtomArray] = {}
    for res_array in residue_iter(atom_array[nonstandard_mask]):
        res_name = res_array.res_name[0]
        user_ccd_arrays.setdefault(res_name, res_array)

    config = CIFWriteConfig(
        include_entity_categories=True,
        id="user_ccd_entry",
        extra_categories={"pdbx_audit_revision_history": {"revision_date": date.today().isoformat()}},
    )
    user_ccd_cif_blocks = {
        name: _to_cif_or_bcif(arr, config=config)["user_ccd_entry"] for name, arr in user_ccd_arrays.items()
    }

    buffer = io.StringIO()
    CIFFile(blocks=user_ccd_cif_blocks).write(buffer)
    return inject_empty_chem_comp_bond(buffer.getvalue())


def disambiguate_chain_ids(atom_array: AtomArray) -> tuple[AtomArray, dict[str, str] | None]:
    """Assign unique chain IDs to chains that share a chain_id but differ by chain_iid.

    Args:
        atom_array: Input structure, optionally with a ``chain_iid`` annotation.

    Returns:
        AtomArray with renamed chains and a mapping from chain_iids to their
        af3 chain_ids (``None`` if no ``chain_iid`` annotation).
    """
    if "chain_iid" not in atom_array.get_annotation_categories():
        return atom_array, None

    seen: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str]] = []
    for chain_id, chain_iid in zip(atom_array.chain_id, atom_array.chain_iid, strict=False):
        pair = (str(chain_id), str(chain_iid))
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)

    chain_id_iids: dict[str, list[str]] = {}
    for chain_id, iid in unique_pairs:
        chain_id_iids.setdefault(chain_id, []).append(iid)
    ambiguous = {chain_id for chain_id, iids in chain_id_iids.items() if len(iids) > 1}
    af3_chain_id_to_original_iid = {
        chain_id: iids[0] for chain_id, iids in chain_id_iids.items() if chain_id not in ambiguous
    }
    if not ambiguous:
        return atom_array, af3_chain_id_to_original_iid

    existing = set(np.unique(atom_array.chain_id).tolist())
    gen = create_chain_id_generator(unavailable_chain_ids=existing)
    pair_to_new: dict[tuple[str, str], str] = {}
    for chain_id, iid in unique_pairs:
        if chain_id in ambiguous:
            new_name = next(gen)
            pair_to_new[(chain_id, iid)] = new_name
            existing.add(new_name)

    renamed = atom_array.copy()
    for (orig_chain_id, iid), new_name in pair_to_new.items():
        mask = (atom_array.chain_id == orig_chain_id) & (atom_array.chain_iid == iid)
        renamed.chain_id[mask] = new_name
        af3_chain_id_to_original_iid[new_name] = iid
    return renamed, af3_chain_id_to_original_iid


def renumber_res_ids_per_chain(atom_array: AtomArray) -> AtomArray:
    """Renumber ``res_id`` to ``1..N`` within each chain.

    NOTE: Chains are identified by ``chain_id`` alone, so callers must
    ensure chains are distinguishable by ``chain_id`` for the renumbering
    to be meaningful.
    """
    # Avoid modifying input AtomArray
    out = atom_array.copy()
    new_res_id = out.res_id.copy()

    # Loop through chains in their first-occurrence order
    _, first_occ = np.unique(out.chain_id, return_index=True)
    for chain_id in out.chain_id[np.sort(first_occ)]:
        in_chain = np.where(out.chain_id == chain_id)[0]
        bounds = struc.get_residue_starts(out[in_chain], add_exclusive_stop=True)
        for new_idx in range(len(bounds) - 1):
            start, stop = int(bounds[new_idx]), int(bounds[new_idx + 1])
            new_res_id[in_chain[start:stop]] = new_idx + 1

    # Update the residue ids in-place
    out.res_id = new_res_id

    return out


@dataclass(slots=True)
class AF3Input:
    """What AF3 is given for one design (the oracle's manifest entry, less its paths and seed)."""

    name: str
    cif: str
    """The design as AF3 reads it: chains disambiguated, covalent modifiers renamed, NaN atoms dropped."""
    user_ccd: str | None
    """Dictionary text defining every non-standard residue and ligand from the design's atoms, or ``None``."""
    template: dict[str, dict] | None
    """Per chain ``{"mmcif", "query_indices", "template_indices"}`` for the templated atoms, or ``None``."""
    atom_array: AtomArray
    """The array the CIF was written from (finite coordinates only)."""
    disambiguated_atom_array: AtomArray
    """The array after chain disambiguation and renaming, before atoms without coordinates were dropped."""
    af3_chain_id_to_original_iid: dict[str, str] | None
    """For mapping AF3's chains back to the design's ``chain_iid``, or ``None`` without that annotation."""


def build_af3_input(
    atom_array: AtomArray,
    name: str,
    *,
    template_query: str | None = None,
    preserve_res_id: bool = True,
) -> AF3Input:
    """The input AF3 folds for ``atom_array``, built as :py:class:`AF3Oracle` builds it.

    Args:
        atom_array: The design, as AtomWorks parses it (with bonds, for covalent modifiers).
        name: The job name, which is the CIF's data block name.
        template_query: AtomWorks query selecting the atoms to template (the oracle's ``template_query``); ``None``
            for no templates.
        preserve_res_id: Write the design's per-chain ``res_id`` verbatim (default), or renumber each chain ``1..N``.

    Raises:
        ValueError: If the design has a residue named ``UNL`` (AF3 reserves it and drops most of its atoms) or the
            template query selects non-protein residues.
    """
    if np.any(atom_array.res_name == "UNL"):
        raise ValueError(
            f"Input structure for {name!r} contains residues named 'UNL'. "
            "AF3 treats 'UNL' as a reserved token and silently drops most ligand atoms. "
            "Rename UNL residues to a different code (e.g. 'L;G') before folding with AF3."
        )
    template_mask = np.zeros(len(atom_array), dtype=bool)
    if template_query:
        from rfproteina.utils.mask_spec import resolve_mask  # (loads atomworks.ml, hence PyTorch, only here)

        template_mask = resolve_mask(atom_array.copy(), template_query)

    # Disambiguate chain IDs so AF3 can distinguish chain_iid variants and keep an id_map for remapping outputs
    disambiguated, af3_chain_id_to_original_iid = disambiguate_chain_ids(atom_array)

    # Force covalent modifiers through the user-CCD path. Built-in CCD entries can contain
    # leaving atoms absent from the design, which AF3 would otherwise add to its output.
    disambiguated = rename_covalent_modifiers_for_af3(disambiguated)

    # Match OF3's tokenization before anything derives from this array; AF3's own
    # weights learned the single-token convention, so gate it.
    if using_of3_weights():
        disambiguated = atomize_ligand_bonded_residues(disambiguated)

    # AF3's CIF parser refuses NaN coordinates; it pads them back and we reconcile later.
    valid_mask = np.all(np.isfinite(disambiguated.coord), axis=-1)
    valid = disambiguated[valid_mask]
    template_mask = template_mask[valid_mask]

    # Biotite's CIF writer requires occupancy + b_factor; designs have no real values.
    valid.set_annotation("occupancy", np.ones(valid.array_length()))
    valid.set_annotation("b_factor", np.zeros(valid.array_length()))

    # Optionally squash all res_id gaps to contiguous 1..N per chain
    if not preserve_res_id:
        valid = renumber_res_ids_per_chain(valid)

    return AF3Input(
        name=name,
        cif=to_cif_string(valid.copy(), config=CIFWriteConfig(id=name)),
        user_ccd=build_user_ccd_string(valid),
        template=build_af3_template_dict(valid, template_mask, template_query),
        atom_array=valid,
        disambiguated_atom_array=disambiguated,
        af3_chain_id_to_original_iid=af3_chain_id_to_original_iid,
    )
