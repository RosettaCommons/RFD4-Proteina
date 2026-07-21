#!/usr/bin/env python3
"""Build an annotated CIF from an RFD3-style design-specification JSON.

``mask=True`` means CONDITIONED; ``C_IDX`` defaults all-on (unindexing inverted);
``ligand`` splits on ``,`` only; only unindexed tokens are subset to fixed atoms;
sequence conditioning defaults True; missing atoms are an error, never added.

Usage::

    build_cif.py <benchmark.json> <example_key> --output out.cif
    build_cif.py <benchmark.json> <example_key> --dump      # summary, no file
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fire
import numpy as np
from atomworks.constants import (
    DNA_BACKBONE_ATOM_NAMES,
    METAL_ELEMENTS,
    PROTEIN_BACKBONE_ATOM_NAMES,
    RNA_BACKBONE_ATOM_NAMES,
    STANDARD_AA_TIP_ATOM_NAMES,
)
from atomworks.enums import ChainType
from atomworks.io.parser import parse
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.io.utils.selection import get_residue_starts
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from atomworks.ml.utils.expandable_segment import insert_expandable_segment
from biotite.structure import Atom, AtomArray, BondList, BondType, connect_via_residue_names
from biotite.structure import array as struc_array

from rfproteina.conditions import C_CATH, C_CRDGRP, C_HB, C_HOT, C_IDXGRP, C_INTCRD, C_RAS, HBondEnum, RasaEnum
from rfproteina.inference_utils.components import (
    ComponentValidationError,
    fetch_mask_from_component,
    fetch_mask_from_idx,
    get_design_pattern_with_constraints,
    get_motif_components_and_breaks,
    get_name_mask,
    unravel_components,
)

# NA-SS grammar (dot-bracket / region / position syntax -> token indices).
from rfproteina.inference_utils.nucleic_ss_input import NucleicSSInputSpec, resolve_nucleic_ss_to_token_payload
from rfproteina.transforms.utils.na_geom_utils import annotate_na_ss_from_specification
from rfproteina.utils.parse_presets import DISTILLATION
from rfproteina.utils.standard_annotations import SA_SRC

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #


#: Diffused residue stubs: backbone atom names from atomworks + a representative atom (CB for
#: protein, C4 for NA). Element is the atom's first character for all of these.
def _stub(names: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((n, n[0]) for n in names)


_PROTEIN_STUB = (*(a for a in PROTEIN_BACKBONE_ATOM_NAMES if a != "OXT"), *STANDARD_AA_TIP_ATOM_NAMES["ALA"])
DIFFUSED_PROTEIN_ATOMS = _stub(_PROTEIN_STUB)
DIFFUSED_PROTEIN_RES_NAME = "ALA"
DIFFUSED_RNA_ATOMS = _stub((*RNA_BACKBONE_ATOM_NAMES, "C4"))
DIFFUSED_RNA_RES_NAME = "A"
DIFFUSED_DNA_ATOMS = _stub((*DNA_BACKBONE_ATOM_NAMES, "C4"))
DIFFUSED_DNA_RES_NAME = "DA"

#: Contig polymer-type suffix -> diffused residue stub.
_DIFFUSED_BY_POLYMER = {
    "P": (DIFFUSED_PROTEIN_RES_NAME, DIFFUSED_PROTEIN_ATOMS),
    "R": (DIFFUSED_RNA_RES_NAME, DIFFUSED_RNA_ATOMS),
    "D": (DIFFUSED_DNA_RES_NAME, DIFFUSED_DNA_ATOMS),
}

#: Annotations carried by hand through assembly; the rest is re-derived on read-back.
_CARRIED = (
    "chain_id",
    "res_id",
    "res_name",
    "atom_name",
    "element",
    "charge",
    "b_factor",
    "occupancy",
    "hetero",
    "ins_code",
)

#: Keys this builder understands; anything else is rejected loudly.
_SUPPORTED_KEYS = {
    "dialect",
    "input",
    "ligand",
    "unindex",
    "length",
    "contig",
    "select_fixed_atoms",
    "select_unfixed_sequence",
    "select_hotspots",
    "select_hbond_donor",
    "select_hbond_acceptor",
    "select_buried",
    "select_partially_buried",
    "select_exposed",
    "internal_coord_groups",
    "nucleic_ss",
    "add_cath_code",
    # Write-only metadata; safe to ignore.
    "extra",
    # Accepted only in its falsy form (see the guard in `build`); False is a no-op.
    "redesign_motif_sidechains",
}

#: Recognised keys we deliberately refuse, each with a reason.
_REJECTED_KEYS = {
    "ori_token": "`ori_token` is not implemented; unexercised by the corpus.",
    "atomwise_hbond": "`atomwise_hbond` is not implemented; unexercised by the corpus.",
    "infer_ori_strategy": "`infer_ori_strategy` is not implemented; unreachable in the corpus.",
}

#: ``select_*`` field -> RASA bin, in apply order. Order is load-bearing: all three write
#: one shared annotation, so last selection wins on a shared atom.
_RASA_SELECTIONS = (
    ("select_buried", RasaEnum.BURIED),
    ("select_partially_buried", RasaEnum.INTERMEDIATE),
    ("select_exposed", RasaEnum.EXPOSED),
)


class BuildError(RuntimeError):
    """Raised when a specification cannot be honoured as written."""


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #


def load_input(path: Path) -> AtomArray:
    """Parse the input structure (DISTILLATION preset, add_missing_atoms=False)."""
    if not path.exists():
        raise BuildError(f"Input structure not found: {path}")

    config = DISTILLATION.replace(
        load_standard_annotations=False,
        remove_waters=False,
        remove_ccds=(),
        hydrogen_policy="remove",
        build_assembly=("1",),
    )
    result = parse(source=str(path), config=config)
    if "assemblies" in result and "1" in result["assemblies"]:
        return result["assemblies"]["1"][0]
    return result["asym_unit"][0]


# --------------------------------------------------------------------------- #
# Selection resolution
# --------------------------------------------------------------------------- #


def resolve_selection(value: str | bool | dict | None, atom_array: AtomArray) -> dict[str, list[str]]:
    """Canonicalise a selection into ``{"<chain><res_id>": [atom names]}``.

    Accepts None, bool (all/no atoms per residue), ``"A39,A219"``, or a per-component dict.
    """
    if value is None:
        return {}

    raw: dict[str, str] = {}
    if isinstance(value, bool):
        starts = get_residue_starts(atom_array, add_exclusive_stop=True)
        for start in starts[:-1]:
            raw[f"{atom_array.chain_id[start]}{atom_array.res_id[start]}"] = "ALL" if value else ""
    elif isinstance(value, str):
        for component in unravel_components(value, atom_array=atom_array, allow_multiple_matches=True):
            if isinstance(component, str) and component[0].isalpha():
                raw[component] = "ALL"
    elif isinstance(value, dict):
        for key, names in value.items():
            for component in unravel_components(key, atom_array=atom_array, allow_multiple_matches=True):
                raw[component] = ",".join(names) if isinstance(names, list) else names
    else:
        raise BuildError(f"Cannot interpret selection of type {type(value).__name__}: {value!r}")

    resolved: dict[str, list[str]] = {}
    for component, names in raw.items():
        token = atom_array[fetch_mask_from_idx(component, atom_array=atom_array)]
        try:
            name_mask = get_name_mask(token.atom_name, names, token.res_name[0])
        except ComponentValidationError as exc:
            raise BuildError(
                f"Component {component!r} ({token.res_name[0]}): cannot select {names!r}. "
                f"Present atoms: {token.atom_name.tolist()}."
            ) from exc
        resolved[component] = token.atom_name[name_mask].tolist()
    return resolved


def selection_mask(selection: dict[str, list[str]], atom_array: AtomArray) -> np.ndarray:
    """Atom-level boolean mask over ``atom_array`` for a resolved selection."""
    mask = np.zeros(atom_array.array_length(), dtype=bool)
    for component, names in selection.items():
        component_mask = fetch_mask_from_idx(component, atom_array=atom_array)
        mask[component_mask] = np.isin(atom_array.atom_name[component_mask], names)
    return mask


def spec_selection_mask(spec: dict, field: str, source: AtomArray) -> np.ndarray:
    """Resolve ``spec[field]`` against ``source`` and return its atom-level mask."""
    return selection_mask(resolve_selection(spec.get(field), source), source)


def rasa_annotation(spec: dict, source: AtomArray) -> np.ndarray | None:
    """Per-atom :py:class:`RasaEnum` over ``source``, or ``None`` if no RASA field is set."""
    if all(spec.get(field) is None for field, _ in _RASA_SELECTIONS):
        return None

    values = np.full(source.array_length(), RasaEnum.UNSPECIFIED, dtype=int)
    for field, rasa_bin in _RASA_SELECTIONS:
        if spec.get(field) is not None:
            values[spec_selection_mask(spec, field, source)] = rasa_bin
    return values


def hbond_annotation(spec: dict, source: AtomArray) -> np.ndarray:
    """Per-atom :py:class:`HBondEnum` over ``source``.

    An atom named by both donor and acceptor fields is `BOTH` (diverges from the oracle).
    """
    donor = spec_selection_mask(spec, "select_hbond_donor", source)
    acceptor = spec_selection_mask(spec, "select_hbond_acceptor", source)

    values = np.full(source.array_length(), HBondEnum.NONE, dtype=int)
    values[donor] = HBondEnum.DONOR
    values[acceptor] = HBondEnum.ACCEPTOR
    values[donor & acceptor] = HBondEnum.BOTH
    return values


def cath_annotation(spec: dict, atom_array: AtomArray) -> np.ndarray:
    """Per-atom CATH label over the **assembled** array.

    ``add_cath_code`` is ``{"1.2.x": <selection>}``: a code plus the region it applies to,
    resolved against the assembled (not input) array. Codes pad/truncate to four levels.
    """
    values = np.full(atom_array.array_length(), C_CATH.NULL_VALUE, dtype=np.dtype("U23"))
    for cath_code, selection in (spec.get("add_cath_code") or {}).items():
        mask = selection_mask(resolve_selection(selection, atom_array), atom_array)
        if not mask.any():
            raise BuildError(
                f"`add_cath_code` entry {cath_code!r} -> {selection!r} selects no atom in the "
                f"assembled array (resolves post-contig, not against the input file)."
            )
        levels = cath_code.split(".")
        levels += ["x"] * (4 - len(levels))
        values[mask] = ".".join(levels[:4])
    return values


def nucleic_ss_annotations(spec: dict, atom_array: AtomArray) -> None:
    """Set ``C_NA_SS_STATE`` / ``C_BP`` on the assembled array from ``nucleic_ss``.

    Resolves against the assembled token layout (diverges from the oracle, which discards it).
    """
    payload = resolve_nucleic_ss_to_token_payload(atom_array, NucleicSSInputSpec(**spec["nucleic_ss"]))
    annotate_na_ss_from_specification(atom_array, {"parsed_nucleic_ss": payload})


def _map_from_source(values: np.ndarray, source_of: np.ndarray, default: float | int | bool) -> np.ndarray:
    """Scatter per-``source``-atom ``values`` onto the assembled array.

    Handles scalars ``[n_source]`` and vectors ``[n_source, 3]``; synthesised atoms take ``default``.
    """
    out = np.full((len(source_of), *values.shape[1:]), default, dtype=values.dtype)
    from_input = source_of >= 0
    out[from_input] = values[source_of[from_input]]
    return out


def internal_coord_groups(spec: dict, source: AtomArray, fixed_coord_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ``internal_coord_groups`` into per-``source``-atom group ids and internal coords.

    Each group withholds its absolute position but keeps intra-group geometry. Returns
    ``(group_id, coords)``: ``group_id`` is 0 outside any group, ``k`` (1-based) inside group
    *k*; ``coords`` is each group centred on its own centroid, NaN elsewhere (emitted unrotated).
    """
    groups = spec["internal_coord_groups"]
    if not isinstance(groups, list):
        raise BuildError(
            f"`internal_coord_groups` must be a list of selections; got {type(groups).__name__}: {groups!r}."
        )

    finite = np.isfinite(source.coord).all(axis=1)
    group_id = np.zeros(source.array_length(), dtype=int)
    coords = np.full((source.array_length(), 3), np.nan, dtype=float)

    for k, selection in enumerate(groups, start=1):
        mask = selection_mask(resolve_selection(selection, source), source) & finite
        if not mask.any():
            raise BuildError(f"Internal coord group {k} ({selection!r}) selects no atom with finite coordinates.")
        # Groups must be mutually exclusive and disjoint from `select_fixed_atoms`.
        clash = np.where(mask & (group_id > 0))[0]
        if len(clash):
            raise BuildError(
                f"Internal coord group {k} ({selection!r}) overlaps group {group_id[clash[0]]} on "
                f"{len(clash)} atom(s), e.g. {source.chain_id[clash[0]]}{source.res_id[clash[0]]}:"
                f"{source.atom_name[clash[0]]}."
            )
        clash = np.where(mask & fixed_coord_mask)[0]
        if len(clash):
            raise BuildError(
                f"Internal coord group {k} ({selection!r}) overlaps `select_fixed_atoms` on "
                f"{len(clash)} atom(s), e.g. {source.chain_id[clash[0]]}{source.res_id[clash[0]]}:"
                f"{source.atom_name[clash[0]]}."
            )

        group_id[mask] = k
        coords[mask] = source.coord[mask] - source.coord[mask].mean(axis=0, keepdims=True)

    return group_id, coords


# --------------------------------------------------------------------------- #
# Piece builders
# --------------------------------------------------------------------------- #


def make_diffused(n: int, polymer_type: str = "P") -> AtomArray:
    """Build ``n`` coordinate-free residue stubs representing a diffused segment.

    ``polymer_type`` is the contig suffix: ``P`` (protein), ``R`` (RNA), ``D`` (DNA).
    """
    if polymer_type not in _DIFFUSED_BY_POLYMER:
        raise BuildError(f"Unknown diffused polymer type {polymer_type!r}; expected {sorted(_DIFFUSED_BY_POLYMER)}.")
    if n <= 0:
        raise BuildError(f"Diffused segment length must be positive, got {n}.")

    res_name, stub = _DIFFUSED_BY_POLYMER[polymer_type]
    atoms = [
        Atom(np.zeros(3, dtype=np.float32), res_name=res_name, res_id=res_id)
        for res_id in range(1, n + 1)
        for _ in stub
    ]
    array = struc_array(atoms)
    array.set_annotation("atom_name", np.array([a for a, _ in stub] * n, dtype="<U6"))
    array.set_annotation("element", np.array([e for _, e in stub] * n, dtype="<U3"))
    array.set_annotation("charge", np.zeros(array.array_length(), dtype=int))
    array.set_annotation("b_factor", np.zeros(array.array_length(), dtype=float))
    array.set_annotation("occupancy", np.ones(array.array_length(), dtype=float))
    array.set_annotation("hetero", np.zeros(array.array_length(), dtype=bool))
    array.set_annotation("ins_code", np.full(array.array_length(), "", dtype="<U1"))
    return array


#: Selection fields that need an input structure; used only for the no-input diagnostic.
_INPUT_DEPENDENT_SELECTIONS = (
    "contig",
    "unindex",
    "select_fixed_atoms",
    "select_unfixed_sequence",
    "select_buried",
    "select_partially_buried",
    "select_exposed",
    "select_hbond_acceptor",
    "select_hbond_donor",
    "select_hotspots",
)


def empty_source() -> AtomArray:
    """A zero-atom stand-in for ``source`` when the spec names no input structure."""
    array = AtomArray(0)
    for name in _CARRIED:
        array.set_annotation(name, _carried_default(name, 0))
    array.bonds = BondList(0)
    return array


def check_input_is_available(spec: dict) -> None:
    """Reject a spec whose selections need an input structure it does not provide."""
    selections = [field for field in _INPUT_DEPENDENT_SELECTIONS if spec.get(field) is not None]
    if not selections:
        return
    if selections == ["contig"]:
        if not any(part.strip()[:1].isalpha() for part in spec["contig"].split(",")):
            return
        raise BuildError(f"Contig {spec['contig']!r} names motif components but the spec has no `input`.")
    raise BuildError(f"Selection(s) {selections} require an input structure, but the spec has no `input`.")


def _normalise(array: AtomArray) -> AtomArray:
    """Reduce an array to exactly the carried annotation set, so pieces concatenate."""
    out = AtomArray(array.array_length())
    out.coord = array.coord.copy()
    for name in _CARRIED:
        if name in array.get_annotation_categories():
            out.set_annotation(name, array.get_annotation(name).copy())
    out.bonds = array.bonds.copy() if array.bonds is not None else BondList(array.array_length())
    return out


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def build(spec: dict, benchmark_root: Path, fix_res_id_gap: bool = False) -> AtomArray:
    """Assemble the design problem described by ``spec`` into an AtomArray."""
    for key, reason in _REJECTED_KEYS.items():
        if key in spec:
            raise BuildError(reason)
    unknown = set(spec) - _SUPPORTED_KEYS
    if unknown:
        raise BuildError(f"Unsupported specification key(s) {sorted(unknown)}. Covered: {sorted(_SUPPORTED_KEYS)}.")
    if int(spec.get("dialect", 2)) < 2:
        raise BuildError("Dialect 1 is not supported; use dialect 2.")
    if spec.get("redesign_motif_sidechains"):
        raise BuildError("redesign_motif_sidechains=True is not implemented (would synthesise atoms).")
    # `select_fixed_atoms` is required only when there is an input; a de-novo spec omits it.
    if spec.get("input") is None:
        check_input_is_available(spec)
        source = empty_source()
    else:
        if "select_fixed_atoms" not in spec:
            raise BuildError("`select_fixed_atoms` must be specified (use `true`/`false` for the trivial cases).")
        source = load_input(benchmark_root / spec["input"])

    # --- Resolve every selection against the input numbering --------------- #
    fixed = resolve_selection(spec.get("select_fixed_atoms"), source)
    unfixed_seq = resolve_selection(spec.get("select_unfixed_sequence"), source)
    unindexed = resolve_selection(spec.get("unindex"), source)

    fixed_coord_mask = selection_mask(fixed, source)
    # Sequence conditioning is ON by default for input-derived atoms; opt out.
    fixed_seq_mask = ~selection_mask(unfixed_seq, source)
    unindexed_mask = selection_mask(unindexed, source)

    # Conditioning that annotates input atoms without shaping the assembly (scattered in step 8).
    rasa_values = rasa_annotation(spec, source)
    hbond_values = hbond_annotation(spec, source)
    hotspot_mask = spec_selection_mask(spec, "select_hotspots", source)

    intcrd_group_id, intcrd_coords = (None, None)
    if spec.get("internal_coord_groups"):
        intcrd_group_id, intcrd_coords = internal_coord_groups(spec, source, fixed_coord_mask)
    # Atoms carried by an internal-coordinate group; needed in step 2.
    intcrd_mask = np.zeros(source.array_length(), dtype=bool) if intcrd_group_id is None else intcrd_group_id > 0

    overlap = selection_mask(resolve_selection(spec.get("contig"), source), source) & unindexed_mask
    if overlap.any():
        raise BuildError("`contig` and `unindex` selections must not overlap.")

    pieces: list[AtomArray] = []
    src_component: list[np.ndarray] = []
    crd: list[np.ndarray] = []
    seq: list[np.ndarray] = []
    unidx: list[np.ndarray] = []
    brk: list[np.ndarray] = []
    lig: list[np.ndarray] = []
    src_index: list[np.ndarray] = []

    def emit(
        piece: AtomArray,
        component: str,
        is_crd: bool | np.ndarray,
        is_seq: bool | np.ndarray,
        is_unindexed: bool | np.ndarray,
        is_break: bool | np.ndarray,
        is_ligand: bool | np.ndarray = False,
        source_idx: np.ndarray | None = None,
    ) -> None:
        """Append a piece plus its per-atom conditioning flags (scalars broadcast).

        ``source_idx`` records each atom's origin in ``source`` (-1 if synthesised).
        """
        n = piece.array_length()
        pieces.append(_normalise(piece))
        src_component.append(np.full(n, component, dtype="<U10"))
        crd.append(np.broadcast_to(is_crd, (n,)).astype(bool))
        seq.append(np.broadcast_to(is_seq, (n,)).astype(bool))
        unidx.append(np.broadcast_to(is_unindexed, (n,)).astype(bool))
        brk.append(np.broadcast_to(is_break, (n,)).astype(bool))
        lig.append(np.broadcast_to(is_ligand, (n,)).astype(bool))
        src_index.append(np.full(n, -1, dtype=int) if source_idx is None else np.asarray(source_idx, dtype=int))

    # --- 1. Diffused / contig components ----------------------------------- #
    # A genuine range is collapsed to a sentinel (resamplable length); a degenerate range is realised.
    contig = spec.get("contig") or spec["length"]
    length = spec.get("length")
    raw_parts = [p for p in contig.split(",") if p.strip()]

    components = get_design_pattern_with_constraints(contig, length=length, atom_array=source)
    n_motif_residues = sum(1 for c in components if str(c) != "/0" and str(c)[0].isalpha())
    plan = _plan_diffused_segments(raw_parts, length, n_motif_residues)

    # Collapsed: record the original contig; realised: record the sampled lengths.
    collapsing = any(bounds is not None for bounds in plan)
    sampled_contig = contig if collapsing else ",".join(str(c) for c in components)

    chain = "A"
    res_id = 1
    diffused_ordinal = 0
    pending_sentinels: list[dict] = []
    for component in components:
        component = str(component)
        if component == "/0":
            chain = chr(ord(chain) + 1)
            res_id = 1
            continue
        if component[0].isalpha():
            # Indexed (contig) motif token: keep the whole residue.
            component_mask = fetch_mask_from_idx(component, atom_array=source)
            token = source[component_mask]
            token_crd = fixed_coord_mask[component_mask]
            token_seq = fixed_seq_mask[component_mask]
            token.res_id = np.full(token.array_length(), res_id)
            token.chain_id = np.full(token.array_length(), chain, dtype=token.chain_id.dtype)
            emit(token, component, token_crd, token_seq, False, False, source_idx=np.where(component_mask)[0])
            res_id += 1
            continue

        bounds = plan[diffused_ordinal]
        diffused_ordinal += 1
        if bounds is not None:
            # Collapsed: emit no atoms; splice the sentinel in after assembly.
            pending_sentinels.append(
                {
                    "chain": chain,
                    "seg_min": bounds[0],
                    "seg_max": bounds[1],
                    "insert_at": sum(p.array_length() for p in pieces),
                    "ordinal": len(pending_sentinels),
                }
            )
            continue

        polymer_type = component[-1] if component[-1] in "PRD" else "P"
        n = int(component[:-1]) if component[-1] in "PRD" else int(component)
        if n == 0:
            continue
        segment = make_diffused(n, polymer_type)
        segment.chain_id = np.full(segment.array_length(), chain, dtype="<U4")
        segment.res_id = segment.res_id + res_id - 1
        # Diffused residues are conditioned on nothing.
        emit(segment, component, False, False, False, False)
        res_id += n
    if diffused_ordinal != len(plan):
        raise BuildError(
            f"Internal error: contig {contig!r} planned {len(plan)} diffused segment(s) but the "
            f"component walk consumed {diffused_ordinal}."
        )

    # --- 2. Unindexed motif components ------------------------------------- #
    # Each comma-separated part starts a break; a range "A579-581" breaks once.
    unindexed_components, breaks = ([], [])
    if spec.get("unindex"):
        unindexed_components, breaks = get_motif_components_and_breaks(spec["unindex"])

    # `unindexed_res_id` continues the contig counter, producing the step-5 double-bump gap.
    unindexed_chain = "A"
    unindexed_res_id = res_id
    for component, is_break in zip(unindexed_components, breaks, strict=True):
        # `/0` is a chain break: the following unindexed motif lands on a new chain.
        if component == "/0":
            unindexed_chain = chr(ord(unindexed_chain) + 1)
            unindexed_res_id = 1
            continue

        component_mask = fetch_mask_from_idx(component, atom_array=source)
        # Only unindexed tokens are subset to their fixed atoms, plus any internal-coord-group atom.
        keep = component_mask & (fixed_coord_mask | intcrd_mask)
        if not keep.any():
            raise BuildError(
                f"Unindexed component {component!r} has no coordinate-conditioned atoms; "
                f"`select_fixed_atoms` (or `internal_coord_groups`) must name at least one atom."
            )
        token = source[keep]
        token.chain_id = np.full(token.array_length(), unindexed_chain, dtype=token.chain_id.dtype)
        token.res_id = np.full(token.array_length(), unindexed_res_id)
        # is_crd is `fixed_coord_mask[keep]`, not True: intcrd-only atoms withhold absolute position.
        emit(
            token,
            component,
            fixed_coord_mask[keep],
            fixed_seq_mask[keep],
            True,
            is_break,
            source_idx=np.where(keep)[0],
        )
        unindexed_res_id += 1

    # --- 3. Ligands --------------------------------------------------------- #
    # `ligand` splits on "," only; ";" and ":" are literal res_name characters.
    if spec.get("ligand"):
        polymer_chains = set(np.unique(np.concatenate([p.chain_id for p in pieces]))) if pieces else set()
        resolved_ligands: list[tuple[str, int, str, np.ndarray]] = []
        for token in spec["ligand"].split(","):
            for name in unravel_components(token, atom_array=source, allow_multiple_matches=True):
                mask = fetch_mask_from_component(name, atom_array=source)
                start = int(np.where(mask)[0][0])
                resolved_ligands.append((str(source.chain_id[start]), start, name, mask))

        # A ligand landing on a polymer's chain is relabelled (mmCIF forbids sharing a chain).
        # New ids continue past the highest polymer chain letter, one per clashing chain, sorted.
        clashing = sorted({chain for chain, *_ in resolved_ligands if chain in polymer_chains})
        if clashing:
            if not all(len(c) == 1 for c in polymer_chains | set(clashing)):
                raise BuildError(
                    f"Cannot repair a polymer/non-polymer chain clash for multi-character chain IDs "
                    f"{sorted(polymer_chains | set(clashing))} (repair is single-letter only)."
                )
            max_existing = max(ord(c) for c in polymer_chains)
            remap = {c: chr(max_existing + i + 1) for i, c in enumerate(clashing)}
            for source_chain, new_chain in remap.items():
                print(
                    f"[chain-clash] Chain ID {source_chain} contains both polymer and non-polymer "
                    f"residues. Assigning non-polymers to new chain {new_chain}",
                    file=sys.stderr,
                )
            resolved_ligands = [
                (remap.get(chain, chain), start, name, mask) for chain, start, name, mask in resolved_ligands
            ]

        # A CIF requires res_id non-decreasing within a chain. Chains keep first-named order;
        # ligands sharing a destination chain are sorted by res_id.
        chain_order: dict[str, int] = {}
        for chain, *_ in resolved_ligands:
            chain_order.setdefault(chain, len(chain_order))

        for chain, _, _, mask in sorted(
            resolved_ligands, key=lambda r: (chain_order[r[0]], int(source.res_id[r[3]][0]))
        ):
            ligand = source[mask]
            ligand.chain_id = np.full(ligand.array_length(), chain, dtype=ligand.chain_id.dtype)
            # Ligands keep every atom and are sequence-conditioned; only select_fixed_atoms get coords.
            emit(
                ligand,
                ligand.res_name[0],
                fixed_coord_mask[mask],
                fixed_seq_mask[mask],
                False,
                False,
                is_ligand=True,
                source_idx=np.where(mask)[0],
            )

    # --- 4. Concatenate ----------------------------------------------------- #
    atom_array = _concatenate(pieces)
    source_of = np.concatenate(src_index)
    _carry_source_bonds(atom_array, source, source_of)
    is_crd = np.concatenate(crd)
    is_seq = np.concatenate(seq)
    is_unindexed = np.concatenate(unidx)
    is_break = np.concatenate(brk)
    is_ligand = np.concatenate(lig)
    SA_SRC.set_annotation(atom_array, np.concatenate(src_component))

    # --- 5. res_id disambiguation for unindexed residues -------------------- #
    # Unindexed residues must not collide with polymer numbering. The oracle double-bumps, leaving
    # a gap (reproduced by default); `--fix_res_id_gap` numbers contiguously instead.
    indexed_polymer = ~is_unindexed & ~is_ligand
    if is_unindexed.any() and not is_unindexed.all() and indexed_polymer.any():
        max_indexed = int(atom_array.res_id[indexed_polymer].max())
        if fix_res_id_gap:
            offset = max_indexed + 1 - int(atom_array.res_id[is_unindexed].min())
            atom_array.res_id[is_unindexed] += offset
        else:
            atom_array.res_id[is_unindexed] += max_indexed + 1

    # --- 6. Centre on the motif, zero the diffused ------------------------- #
    if is_crd.any():
        atom_array.coord -= np.nan_to_num(np.mean(atom_array.coord[is_crd], axis=0))
    else:
        atom_array.coord = np.zeros_like(atom_array.coord)
    atom_array.coord[~is_crd] = 0.0

    # --- 7. Bonds ----------------------------------------------------------- #
    # Carried bonds hold coordination/aromatic types; inferred bonds fill the poly-ALA backbone.
    for i, j, bond_type in connect_via_residue_names(atom_array, inter_residue=True).as_array():
        atom_array.bonds.add_bond(i, j, bond_type)

    # --- 8. Conditions ------------------------------------------------------ #
    atom_array = as_atom_array_plus(atom_array)
    C_CRD.set_mask(atom_array, is_crd)
    C_SEQ.set_mask(atom_array, is_seq)
    # C_IDX defaults all-on: mark indexed, i.e. the inverse of unindexed.
    C_IDX.set_mask(atom_array, ~is_unindexed)
    C_IDXGRP.set_annotation(atom_array, _index_groups(atom_array, is_unindexed, is_break))

    # C_HB set unconditionally (all-NONE == absent); C_RAS / C_HOT only when selected.
    C_HB.set_annotation(atom_array, _map_from_source(hbond_values, source_of, HBondEnum.NONE))
    if rasa_values is not None:
        C_RAS.set_annotation(atom_array, _map_from_source(rasa_values, source_of, RasaEnum.UNSPECIFIED))
    if spec.get("select_hotspots") is not None:
        C_HOT.set_annotation(atom_array, _map_from_source(hotspot_mask, source_of, False))

    # NA-SS addresses tokens by index, so it must run BEFORE sentinel insertion shifts indices.
    if spec.get("nucleic_ss") is not None:
        nucleic_ss_annotations(spec, atom_array)

    # Internal-coord groups, set AFTER C_CRD: `C_CRDGRP.default_annotation` reads C_CRD, then
    # group ids overwrite it.
    if intcrd_group_id is not None:
        group_id = _map_from_source(intcrd_group_id, source_of, 0)
        in_group = group_id > 0

        coord_groups = C_CRDGRP.default_annotation(atom_array)
        coord_groups[in_group] = group_id[in_group]
        C_CRDGRP.set_annotation(atom_array, coord_groups)

        C_INTCRD.set_mask(atom_array, in_group)
        C_INTCRD.set_annotation(atom_array, _map_from_source(intcrd_coords, source_of, np.nan))

    # --- 9. Entity IDs ------------------------------------------------------ #
    # Set `label_entity_id` to pin the oracle's 0-based IDs; else the writer invents a 1-based column.
    atom_array.set_annotation("label_entity_id", atom_array.chain_entity.astype(np.int16))

    # --- 10. Collapse the variable diffused segments ------------------------ #
    # Every mask above is indexed against the pre-sentinel array, so this cannot move earlier.
    if pending_sentinels:
        atom_array = _insert_diffused_sentinels(atom_array, pending_sentinels)

    # --- 11. CATH ----------------------------------------------------------- #
    # AFTER step 10: CATH addresses atoms by selection (not index), so it survives sentinel
    # insertion; the label rides through expansion onto every realised residue.
    if spec.get("add_cath_code") is not None:
        C_CATH.set_annotation(atom_array, cath_annotation(spec, atom_array))

    atom_array.set_annotation("sampled_contig", np.full(atom_array.array_length(), sampled_contig, dtype="<U64"))
    return atom_array


def _insert_diffused_sentinels(atom_array: AtomArray, pending: list[dict]) -> AtomArray:
    """Splice in one collapsed expandable-segment sentinel per variable diffused segment.

    Insertions run in descending ``insert_at`` order (ties: later segment first) so splices
    don't shift pending offsets.
    """
    for seg in sorted(pending, key=lambda s: (s["insert_at"], s["ordinal"]), reverse=True):
        chain, at = seg["chain"], seg["insert_at"]
        n = atom_array.array_length()
        in_chain = np.where(atom_array.chain_id == chain)[0]

        if len(in_chain) == 0:
            # Nothing else on this chain: the segment is the whole chain.
            used = set(np.unique(atom_array.chain_id).tolist())
            chain_id = chain if chain not in used else next(c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used)
            atom_array = insert_expandable_segment(
                atom_array,
                seg_min=seg["seg_min"],
                seg_max=seg["seg_max"],
                new_chain=True,
                new_chain_id=chain_id,
                # Enum member, not name: `ChainType` coerces via mmCIF vocabulary ("polypeptide(L)").
                new_chain_type=ChainType.POLYPEPTIDE_L,
            )
            continue

        following = in_chain[in_chain >= at]
        if len(following) and at < n:
            anchor, before = int(following[0]), True
        else:
            # Chain-final: anchor on the chain's last atom, insert after it.
            anchor, before = int(in_chain[-1]), False
        atom_array = insert_expandable_segment(
            atom_array, seg_min=seg["seg_min"], seg_max=seg["seg_max"], atom_idx=anchor, insert_before=before
        )
    return atom_array


def _parse_range(spec: str) -> tuple[int, int]:
    """``"140-170"`` -> ``(140, 170)``; ``"180"`` -> ``(180, 180)``."""
    if "-" in spec:
        low, high = spec.split("-", 1)
        return int(low), int(high)
    return int(spec), int(spec)


def _plan_diffused_segments(raw_parts: list[str], length: str | None, n_motif_residues: int) -> list:
    """Per-diffused-part ``(seg_min, seg_max)`` to collapse, or ``None`` to realise.

    With no ``length`` each variable part samples over its own range. With ``length`` and one
    variable part, its range is intersected with the budget ``length`` leaves. ``length`` plus
    multiple variable parts couples them (constrains only their sum), which sentinels cannot
    express, so this raises.
    """
    diffused_specs = [b for p in raw_parts if (b := diffused_bounds(p)) is not None]
    variable = [(lo, hi) for lo, hi in diffused_specs if lo != hi]

    if length is None:
        return [None if lo == hi else (lo, hi) for lo, hi in diffused_specs]

    if len(variable) > 1:
        raise BuildError(
            f"Contig {','.join(raw_parts)!r} has {len(variable)} length ranges {variable} AND a "
            f"`length` ({length!r}), which couples them; sentinels can't express a constrained sum. "
            "(Drop `length`, or pin all but one range.)"
        )

    length_min, length_max = _parse_range(length)
    # `length` counts motif + diffused residues; subtract the motif to get the diffused budget.
    budget_min = length_min - n_motif_residues
    budget_max = length_max - n_motif_residues
    # Every other diffused part is degenerate, contributing its own fixed count to the budget.
    degenerate_total = sum(lo for lo, hi in diffused_specs if lo == hi)

    plan: list = []
    for lo, hi in diffused_specs:
        if lo == hi:
            plan.append(None)
            continue
        seg_min = max(lo, budget_min - degenerate_total)
        seg_max = min(hi, budget_max - degenerate_total)
        if seg_min > seg_max:
            raise BuildError(
                f"Contig segment {lo}-{hi} cannot satisfy `length` {length!r}: budget "
                f"[{budget_min - degenerate_total}, {budget_max - degenerate_total}] does not "
                f"intersect [{lo}, {hi}]."
            )
        plan.append((seg_min, seg_max))
    return plan


def diffused_bounds(part: str) -> tuple[int, int] | None:
    """``(min, max)`` residue count for a diffused contig part, else ``None`` (motif or ``/0``)."""
    part = part.strip()
    if not part or part == "/0":
        return None
    if part[-1] in "PRD":
        part = part[:-1]
    if any(c.isalpha() for c in part):
        return None
    if "-" in part:
        low, high = part.split("-", 1)
        return int(low), int(high)
    return int(part), int(part)


# Typical metal-ligand coordination is ~1.9-2.5 A (not atomworks' permissive 3.6 A filter).
_METAL_COORD_MAX_DIST = 2.6
_COORDINATING_ELEMENTS = frozenset({"N", "O", "S", "SE", "CL", "BR"})


def _resolve_atom(atom_array: AtomArray, spec: str) -> int:
    """Resolve a ``CHAIN/RES_NAME/RES_ID/ATOM_NAME`` selector (e.g. ``A/CYS/137/SG``) to one atom index."""
    parts = spec.split("/")
    if len(parts) != 4:
        raise BuildError(f"Bad atom selector {spec!r}: expected CHAIN/RES_NAME/RES_ID/ATOM_NAME")
    chain, res_name, res_id, atom_name = parts
    mask = (
        (atom_array.chain_id == chain)
        & (atom_array.res_name == res_name)
        & (atom_array.res_id == int(res_id))
        & (atom_array.atom_name == atom_name)
    )
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise BuildError(f"Atom selector {spec!r} matched nothing.")
    if len(idx) > 1:
        raise BuildError(f"Atom selector {spec!r} matched {len(idx)} atoms; must be unique.")
    return int(idx[0])


def add_coordination_bonds(atom_array: AtomArray, bond_specs: list[str]) -> list[str]:
    """Add COORDINATION bonds between explicitly named atom pairs (never inferred).

    Args:
        bond_specs: ``"CHAIN/RES/ID/ATOM,CHAIN/RES/ID/ATOM"`` pairs, e.g. ``"C/MG/422/MG,A/ASP/95/OD1"``.
    """
    if not bond_specs:
        return []
    if atom_array.bonds is None:
        atom_array.bonds = BondList(atom_array.array_length())

    added = []
    for spec in bond_specs:
        if spec.count(",") != 1:
            raise BuildError(f"Bad coordination bond {spec!r}: expected two selectors separated by a comma")
        left, right = (s.strip() for s in spec.split(","))
        i, j = _resolve_atom(atom_array, left), _resolve_atom(atom_array, right)
        d = float(np.linalg.norm(atom_array.coord[j] - atom_array.coord[i]))
        # Distance is reported, never a gate.
        atom_array.bonds.add_bond(i, j, BondType.COORDINATION)
        flag = "" if d <= _METAL_COORD_MAX_DIST else "   <-- WARNING: long for a coordination bond"
        added.append(f"{left} -- {right}  ({d:.2f}A){flag}")
    return added


def find_metal_coordination_candidates(atom_array: AtomArray, max_dist: float = _METAL_COORD_MAX_DIST) -> list[str]:
    """Report unbonded metals and their geometric neighbours as candidates. Reporting only."""
    if atom_array.bonds is None:
        bonded: set[int] = set()
    else:
        bonded = {int(x) for i, j, _ in atom_array.bonds.as_array() for x in (i, j)}

    elements = np.array([str(e).upper() for e in atom_array.element])
    out = []
    for i in np.where(np.isin(elements, list(METAL_ELEMENTS)))[0]:
        if int(i) in bonded:
            continue
        near = []
        for j in range(atom_array.array_length()):
            if j == i or elements[j] not in _COORDINATING_ELEMENTS:
                continue
            d = float(np.linalg.norm(atom_array.coord[j] - atom_array.coord[i]))
            if d <= max_dist:
                near.append(
                    f"{atom_array.chain_id[j]}/{atom_array.res_name[j]}/{atom_array.res_id[j]}/"
                    f"{atom_array.atom_name[j]} ({d:.2f}A)"
                )
        metal = f"{atom_array.chain_id[i]}/{atom_array.res_name[i]}/{atom_array.res_id[i]}/{atom_array.atom_name[i]}"
        out.append(f"{metal}: {', '.join(sorted(near)) if near else 'no candidates within range'}")
    return out


def _carry_source_bonds(atom_array: AtomArray, source: AtomArray, source_of: np.ndarray) -> int:
    """Re-attach source bonds whose two endpoints landed in different pieces.

    A bond spanning two pieces (covale, disulfide, metal link) is dropped by concat, so replay
    ``source.bonds`` via each atom's source index. Diverges from the oracle, which drops them;
    the annotated CIF must carry covalent modifications. Returns the number re-attached.
    """
    if source.bonds is None:
        return 0
    new_of_source: dict[int, list[int]] = defaultdict(list)
    for new_idx, s in enumerate(source_of.tolist()):
        if s >= 0:
            new_of_source[s].append(new_idx)

    added = 0
    for i, j, bond_type in source.bonds.as_array():
        # A bond whose endpoint was subset away is correctly absent.
        for new_i in new_of_source.get(int(i), ()):
            for new_j in new_of_source.get(int(j), ()):
                atom_array.bonds.add_bond(new_i, new_j, bond_type)
                added += 1
    return added


#: Per-annotation dtype/default for pieces missing a carried annotation; ``_concatenate`` fills gaps.
_CARRIED_DTYPE = {"res_id": int, "charge": int, "b_factor": float, "occupancy": float, "hetero": bool}
_CARRIED_WIDTH = {"chain_id": "<U4", "res_name": "<U5", "atom_name": "<U6", "element": "<U3", "ins_code": "<U1"}
_CARRIED_DEFAULT = {"occupancy": 1.0}  # everything else -> 0 / False / ""


def _carried_default(name: str, n: int) -> np.ndarray:
    """A default-filled array for a piece missing carried annotation ``name``."""
    dtype = _CARRIED_DTYPE.get(name, _CARRIED_WIDTH.get(name, "<U4"))
    fill = _CARRIED_DEFAULT.get(name, 0 if name in _CARRIED_DTYPE else "")
    return np.full(n, fill, dtype=dtype)


def _concatenate(pieces: list[AtomArray]) -> AtomArray:
    """Concatenate pieces, preserving each piece's bonds."""
    total = sum(p.array_length() for p in pieces)
    out = AtomArray(total)
    out.coord = np.concatenate([p.coord for p in pieces])
    for name in _CARRIED:
        values = [
            p.get_annotation(name)
            if name in p.get_annotation_categories()
            else _carried_default(name, p.array_length())
            for p in pieces
        ]
        out.set_annotation(name, np.concatenate(values))
    bonds = BondList(total)
    offset = 0
    for piece in pieces:
        for i, j, bond_type in piece.bonds.as_array():
            bonds.add_bond(offset + i, offset + j, bond_type)
        offset += piece.array_length()
    out.bonds = bonds
    return out


def _index_groups(atom_array: AtomArray, is_unindexed: np.ndarray, is_break: np.ndarray) -> np.ndarray:
    """Equivalence classes of residues with known index offsets.

    Indexed atoms keep the default grouping; unindexed atoms are grouped by ``cumsum(breaks)``,
    offset past the indexed groups.
    """
    default = C_IDXGRP.default_annotation(atom_array)
    starts = get_residue_starts(atom_array)
    group_per_residue = np.cumsum(is_break[starts], dtype=int)

    indexed_residue = ~is_unindexed[starts]
    if indexed_residue.any():
        group_per_residue[~indexed_residue] += int(default[~is_unindexed].max()) + 1

    groups = np.repeat(group_per_residue, np.diff(np.append(starts, atom_array.array_length())))
    groups[~is_unindexed] = default[~is_unindexed]
    return groups


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def dump_summary(atom_array: AtomArray, label: str) -> None:
    """Print a summary comparable to the oracle dump."""
    n = atom_array.array_length()
    print(f"\n### {label}: {n} atoms")
    if "sampled_contig" in atom_array.get_annotation_categories():
        print(f"    sampled_contig = {atom_array.get_annotation('sampled_contig')[0]!r}")
    print(f"    annotations: {sorted(atom_array.get_annotation_categories())}")

    starts = get_residue_starts(atom_array, add_exclusive_stop=True)
    rows = [
        (str(atom_array.chain_id[s]), int(atom_array.res_id[s]), str(atom_array.res_name[s]), e - s)
        for s, e in itertools.pairwise(starts)
    ]
    print(f"\n### residues: {len(rows)}")
    print("    by chain:", dict(Counter(r[0] for r in rows)))
    print("    resname counts:", dict(Counter(r[2] for r in rows).most_common(8)))
    print("    non-ALA residues (the motif + ligands):")
    for chain, res_id, res_name, n_atoms in rows:
        if res_name != "ALA":
            print(f"      {chain}{res_id:<5} {res_name:<6} {n_atoms:3d} atoms")

    print("\n### condition masks (True = CONDITIONED):")
    for name in sorted(atom_array.get_annotation_categories()):
        if not name.startswith(("mask_", "condition_", "annotation_", "is_motif", "atomize")):
            continue
        values = atom_array.get_annotation(name)
        if values.dtype == bool or set(np.unique(values)) <= {0, 1}:
            print(f"    {name:44s} {int(np.sum(values.astype(bool))):5d}/{len(values)} True")
        else:
            uniq = np.unique(values)
            print(f"    {name:44s} dtype={values.dtype} uniq={uniq[:6]}{'...' if len(uniq) > 6 else ''}")

    n_bonds = len(atom_array.bonds.as_array()) if atom_array.bonds is not None else 0
    print(f"\n### bonds: {n_bonds}")
    if atom_array.bonds is not None:
        print(f"    by type: {dict(Counter(int(b[2]) for b in atom_array.bonds.as_array()))}")
    print(f"### charges: nonzero = {int(np.sum(atom_array.charge != 0))}/{n}  uniq={np.unique(atom_array.charge)}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(
    benchmark: str,
    key: str,
    output: str | None = None,
    dump: bool = False,
    benchmark_root: str | None = None,
    fix_res_id_gap: bool = False,
    coordination_bond: str | list[str] | None = None,
    list_metal_candidates: bool = False,
) -> None:
    """Build the annotated CIF for one benchmark example.

    Args:
      benchmark: Benchmark JSON/YAML path (falls back to data/benchmarks/<benchmark>).
      key: Example key within the benchmark.
      output: Output CIF path; omit to only build.
      dump: Print a summary of the built atom array.
      benchmark_root: Root for the spec's ``input`` paths (default: the benchmark's directory).
      fix_res_id_gap: Number unindexed residues contiguously.
      coordination_bond: ``"C/MG/422/MG,A/ASP/95/OD1"`` (or a list). Add explicit COORDINATION
        bonds; metals are otherwise left unbonded. Use ``list_metal_candidates`` to find them.
      list_metal_candidates: Report unbonded metals' geometric neighbours, then exit.
    """
    benchmark_path = Path(benchmark)
    if not benchmark_path.exists():
        benchmark_path = Path("data/benchmarks") / benchmark
        if not benchmark_path.exists():
            raise SystemExit(f"Benchmark not found: {benchmark}")

    root = Path(benchmark_root) if benchmark_root else benchmark_path.parent
    text = benchmark_path.read_text()
    if benchmark_path.suffix in (".yaml", ".yml"):
        import yaml

        specs = yaml.safe_load(text)
    else:
        specs = json.loads(text)

    global_args = specs.pop("global_args", {}) or {}  # inherited by every example
    if key not in specs:
        raise SystemExit(f"Key {key!r} not in {benchmark_path}. Available: {sorted(specs)[:10]}")

    atom_array = build({**specs[key], **global_args}, root, fix_res_id_gap=fix_res_id_gap)

    if list_metal_candidates:
        cands = find_metal_coordination_candidates(atom_array)
        if cands:
            print(f"unbonded metal(s) in {benchmark_path.name}:{key} (candidates only):")
            for c in cands:
                print(f"    {c}")
            print("Name real ones with coordination_bond='C/MG/422/MG,A/ASP/95/OD1'.")
        else:
            print(f"no unbonded metals in {benchmark_path.name}:{key}")
        return

    if coordination_bond:
        bonds = [coordination_bond] if isinstance(coordination_bond, str) else list(coordination_bond)
        added = add_coordination_bonds(atom_array, bonds)
        print(f"[coordination] added {len(added)} bond(s): {added}", file=sys.stderr)

    if dump:
        dump_summary(atom_array, f"{benchmark_path.name}:{key}")

    if output:
        contig = atom_array.get_annotation("sampled_contig")[0]  # provenance, not per-atom
        atom_array.del_annotation("sampled_contig")
        to_cif_file(
            atom_array,
            output,
            save_standard_annotations=True,
            chain_disambiguation="chain_iid",
            extra_categories={"rfd_design": {"sampled_contig": contig}},
        )
        print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    fire.Fire(main)
