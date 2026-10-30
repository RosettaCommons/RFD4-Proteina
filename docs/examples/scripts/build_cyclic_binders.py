#!/usr/bin/env python3
"""Build cyclic peptide *binder* inputs: a macrocyclic peptide against a PPI benchmark target.

A macrocyclic peptide chain joined to a real PPI benchmark target
chain. The peptide carries no coordinates -- sequence and structure are both designed -- while
the target keeps its experimental coordinates, its full sequence condition and its hotspots.

Why the peptide chain is built explicitly rather than as a length range: the PPI benchmark
specifies binders as a single placeholder row (``80-150``, one UNK atom, nan coordinates).
A closure bond is a ``_struct_conn`` record between two named atoms of two numbered residues,
so there has to be something to point at. A length-range binder cannot carry one.

Usage::

    python docs/examples/scripts/build_cyclic_binders.py \
        --target data/benchmarks/ppi/ppi_bb_only/01_PD1.cif.zst \
        --out-dir scratch/cyclic_binders --length 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from atomworks.io.parser import parse
from atomworks.io.transforms.atom_array import add_iid_annotations
from atomworks.io.utils.atom_array_plus import as_atom_array_plus
from atomworks.io.utils.io_utils import to_cif_file
from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ
from biotite.structure import BondType

import rfproteina.conditions  # noqa: F401 -- registers condition classes; without this
# import the target's hotspot / hbond / index-group conditions are silently dropped on write.
from rfproteina.utils.parse_presets import ANNOTATED_CIF

# chemistry -> (residue at position 1, residue at position L, closure atom pair)
CHEMISTRIES = {
    "head_tail": ("GLY", "GLY", ("N", "C")),
    "disulfide": ("CYS", "CYS", ("SG", "SG")),
    "isopeptide": ("LYS", "ASP", ("NZ", "CG")),
}

ATOMS = {
    "GLY": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")],
    "CYS": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"), ("SG", "S")],
    "LYS": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
            ("CG", "C"), ("CD", "C"), ("CE", "C"), ("NZ", "N")],
    "ASP": [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O"), ("CB", "C"),
            ("CG", "C"), ("OD1", "O"), ("OD2", "O")],
}
ONE_LETTER = {"GLY": "G", "CYS": "C", "LYS": "K", "ASP": "D"}


def residue_names(length: int, chemistry: str) -> list[str]:
    first, last, _ = CHEMISTRIES[chemistry]
    names = ["GLY"] * length
    names[0], names[-1] = first, last
    return names


def peptide_source_cif(length: int, chemistry: str, chain: str = "P") -> str:
    """A coordinate-free CIF holding just the macrocycle and its closure record."""
    names = residue_names(length, chemistry)
    atom_i, atom_j = CHEMISTRIES[chemistry][2]

    rows, atom_id = [], 0
    for res_idx, res_name in enumerate(names, start=1):
        for atom_name, element in ATOMS[res_name]:
            atom_id += 1
            rows.append(
                f"ATOM {atom_id} {element} {atom_name} . {res_name} {chain} 0 {res_idx} ? "
                f"{res_idx} {res_name} {chain} {atom_name} nan nan nan 1 0.0 ? 1"
            )

    seq = "".join(ONE_LETTER[n] for n in names)
    lines = [
        f"data_cyclic_{chemistry}_{length}", "#",
        f"_entry.id cyclic_{chemistry}_{length}", "#",
        "_entity.id 0", "_entity.type polymer", "#",
        "_entity_poly.entity_id 0", "_entity_poly.type polypeptide(L)",
        "_entity_poly.nstd_linkage no", "_entity_poly.nstd_monomer no",
        f"_entity_poly.pdbx_seq_one_letter_code {seq}",
        f"_entity_poly.pdbx_seq_one_letter_code_can {seq}",
        f"_entity_poly.pdbx_strand_id {chain}", "#",
        "loop_", "_entity_poly_seq.entity_id", "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id", "_entity_poly_seq.hetero",
    ]
    lines += [f"0 {i} {r} n" for i, r in enumerate(names, start=1)]
    lines += [
        "#",
        "_struct_conn.id 1",
        # covale, not disulf: the parse preset reads only ("covale", "metalc").
        "_struct_conn.conn_type_id covale",
        "_struct_conn.pdbx_value_order ?",
        f"_struct_conn.ptnr1_label_asym_id {chain}",
        f"_struct_conn.ptnr2_label_asym_id {chain}",
        f"_struct_conn.ptnr1_label_comp_id {names[0]}",
        f"_struct_conn.ptnr2_label_comp_id {names[-1]}",
        "_struct_conn.ptnr1_label_seq_id 1",
        f"_struct_conn.ptnr2_label_seq_id {length}",
        f"_struct_conn.ptnr1_label_atom_id {atom_i}",
        f"_struct_conn.ptnr2_label_atom_id {atom_j}",
        "_struct_conn.pdbx_ptnr1_PDB_ins_code .",
        "_struct_conn.pdbx_ptnr2_PDB_ins_code .",
        "#", "loop_",
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.auth_seq_id", "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id", "_atom_site.auth_atom_id",
        "_atom_site.Cartn_x", "_atom_site.Cartn_y", "_atom_site.Cartn_z",
        "_atom_site.pdbx_PDB_model_num", "_atom_site.occupancy",
        "_atom_site.pdbx_formal_charge", "_atom_site.B_iso_or_equiv",
        *rows, "#",
    ]
    return "\n".join(lines) + "\n"


def reconcile_annotations(a, b) -> None:
    """Give each array any annotation the other has, defaulted, so they can concatenate."""
    for src, dst in ((a, b), (b, a)):
        for cat in src.get_annotation_categories():
            if cat in dst.get_annotation_categories():
                continue
            ref = src.get_annotation(cat)
            # Some conditions are per-atom vectors (coordinates are [n, 3]); keep trailing dims.
            shape = (dst.array_length(),) + ref.shape[1:]
            if ref.dtype.kind in "US":
                fill = np.full(shape, "", dtype=ref.dtype)
            else:
                # Expandable-segment bounds use -1 for "not an expandable segment". Filling
                # them with 0 instead declares a zero-length segment, and
                # RealizeExpandableSegments then deletes the chain -- silently, with the run
                # still exiting 0 and the chain simply missing from the output.
                default = -1 if "expseg" in cat else 0
                fill = np.full(shape, default, dtype=ref.dtype)
            dst.set_annotation(cat, fill)


def load_target(target_path: Path):
    """Parse a PPI benchmark input and drop its length-range placeholder binder chain."""
    aa = as_atom_array_plus(parse(target_path, config=ANNOTATED_CIF)["asym_unit"][0])
    # The placeholder binder is the chain whose residue name is a literal length range.
    is_placeholder = np.array(["-" in str(n) for n in aa.res_name], dtype=bool)
    if is_placeholder.any():
        dropped = sorted(set(aa.chain_id[is_placeholder].tolist()))
        aa = aa[~is_placeholder]
        print(f"  dropped placeholder binder chain(s): {dropped}")
    return aa


def build_case(target_path: Path, chemistry: str, length: int, out_path: Path, scratch: Path) -> dict:
    names = residue_names(length, chemistry)
    atom_i, atom_j = CHEMISTRIES[chemistry][2]

    src = scratch / f"{chemistry}_{length}_peptide.cif"
    src.write_text(peptide_source_cif(length, chemistry))
    pep = as_atom_array_plus(parse(src, config=ANNOTATED_CIF)["asym_unit"][0])
    tgt = load_target(target_path)

    # The target input carries condition annotations the bare peptide has not got.
    # Concatenation requires identical categories, so fill each side's gaps with defaults.
    reconcile_annotations(pep, tgt)

    # Peptide first so its residue numbering is the low range and easy to select later.
    combined = pep + tgt
    if "transformation_id" not in combined.get_annotation_categories():
        combined.set_annotation("transformation_id",
                                np.full(combined.array_length(), "1", dtype=str))
    combined = add_iid_annotations(combined)

    is_pep = np.isin(combined.chain_id, np.unique(pep.chain_id))

    # Both sides were parsed independently, so their entity ids are numbered from zero and
    # collide on write -- the target would end up referencing the peptide's _entity_poly_seq.
    # Renumber explicitly: peptide 0, target 1.
    for cat in ("label_entity_id", "chain_entity", "pn_unit_entity", "molecule_entity"):
        if cat in combined.get_annotation_categories():
            ids = combined.get_annotation(cat).copy()
            ids[is_pep] = 0
            ids[~is_pep] = 1
            combined.set_annotation(cat, ids)

    # Conditions. Peptide: nothing fixed except the identities the chemistry needs.
    # Target: coordinates and sequence both conditioned; hotspots ride along from the input.
    fixed_seq = ~is_pep
    if chemistry != "head_tail":
        fixed_seq = fixed_seq | (is_pep & np.isin(combined.res_id, (1, length)))
    C_CRD.set_annotation_from_ground_truth(combined, ~is_pep)
    C_SEQ.set_annotation_from_ground_truth(combined, fixed_seq)
    # The peptide's residue indices are fixed (it is a ring of known length), its coordinates
    # are not. Index-condition the whole complex.
    idx_cat = next((c for c in combined.get_annotation_categories()
                    if c.startswith("condition_index_")), None)
    if idx_cat is not None:
        combined.set_annotation(idx_cat, np.ones(combined.array_length(), dtype=bool))

    # A designed chain reaches the model as an *expandable segment*, and the pipeline replaces
    # those segments' nan coordinates with zeros before featurising. An explicitly enumerated
    # chain is not an expandable segment, so that substitution never fires and the chain is
    # dropped before generation -- the run still exits 0, with the peptide simply absent from
    # the output. Do the same substitution here.
    coords = combined.coord.copy()
    coords[is_pep] = 0.0
    combined.coord = coords

    # Re-declare the closure on the combined array; concatenation does not carry bonds across.
    def atom_index(res_id: int, atom_name: str) -> int:
        hits = np.flatnonzero(is_pep & (combined.res_id == res_id)
                              & (combined.atom_name == atom_name))
        assert len(hits) == 1, f"{res_id}.{atom_name} matched {len(hits)} atoms"
        return int(hits[0])

    combined.bonds.add_bond(atom_index(1, atom_i), atom_index(length, atom_j), BondType.SINGLE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    to_cif_file(combined, out_path, save_standard_annotations=True,
                chain_disambiguation="chain_iid")

    return {
        "name": out_path.stem,
        "target": target_path.name,
        "chemistry": chemistry,
        "length": length,
        "closure": {
            "ptnr1": {"res_id": 1, "res_name": names[0], "atom": atom_i},
            "ptnr2": {"res_id": length, "res_name": names[-1], "atom": atom_j},
        },
        "n_target_atoms": int((~is_pep).sum()),
        "n_peptide_atoms": int(is_pep.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--length", type=int, default=12)
    ap.add_argument("--scratch", type=Path, default=None)
    args = ap.parse_args()

    import tempfile

    stem = args.target.name.split(".")[0]
    with tempfile.TemporaryDirectory() as tmp:
        scratch = args.scratch or Path(tmp)
        scratch.mkdir(parents=True, exist_ok=True)
        cases = []
        for chemistry in CHEMISTRIES:
            out = args.out_dir / f"{stem}_{chemistry}_{args.length:02d}.cif"
            print(f"building {out.name}")
            cases.append(build_case(args.target, chemistry, args.length, out, scratch))

    manifest = {
        "schema_version": 1,
        "description": f"Cyclic peptide binders against {stem}, three closure chemistries.",
        "target": stem,
        "length": args.length,
        "cases": cases,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(cases)} inputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
