# Existing contig/length JSON/YAML reference

Source of truth: `docs/workflow/inputs.md`,
`skills/rfd4-proteina-inputs/scripts/build_from_rfd3.py`, and the current
input-schema implementation. Read this reference for every supplied legacy
contig/length JSON/YAML artifact. Interpret the file as evidence of intent,
confirm its material semantics, then use the compatibility adapter when the
confirmed intent fits this dialect. Do not generate a JSON intermediary for
prose, raw structures, or annotated-CIF inputs; those belong to
`rfd4-proteina-annotated-cif`.

## The two schemas (dispatch on which is present; they never co-occur)

| Schema | Fields | Meaning |
| --- | --- | --- |
| **A — contig** | `contig`, `ligand`, `select_fixed_atoms` (bool) | `contig` is often a bare length range (`"100-155"`) with no motif |
| **B — length/unindex** | `length`, `unindex`, `ligand`, `select_fixed_atoms` (dict) | scaffold a fixed length around unindexed motif residues |

Richer configs add `select_buried`/`select_exposed` (RASA),
`select_hbond_donor`/`_acceptor`, `select_hotspots`, `select_unfixed_sequence`,
`redesign_motif_sidechains`, `internal_coord_groups`.

## Field semantics (verified; do not re-derive)

- **`ligand` splits on `,` only.** `L;G`, `T:S`, `H:H`, `K:X`, `lpr` are literal
  residue names.
- **`select_fixed_atoms` is required.** Omitting a ligand from it **zeroes that
  ligand's coordinates**, which is why mcsa configs spell out e.g. `"MG": "MG"`.
- Atom keywords: `ALL` = every atom; `BKBN` = `N,CA,C,O` (no CB); `TIP` =
  residue-specific tip atoms (raises on ALA/GLY/PRO/UNK); `""` = no atoms.
- **Sequence conditioning defaults TRUE** on every input-derived atom;
  `select_unfixed_sequence` opts *out*.
- **`C_IDX` defaults ALL-ON**; `unindex` is the marked op. Comma-separated parts
  start independent breaks: `"A39,A219"` -> two independent unindexed groups; a
  range `"A579-581"` -> one rigid group (index-tied). Index group = `cumsum(breaks)`.
- Component resolution order: **resname -> chain_id -> chain+res_id**. A bare `"A"`
  matches RNA adenosine before chain A.
- **Only *unindexed* tokens are subset to their fixed atoms**; contig tokens keep
  whole residues (this is why an unindexed ARG can arrive with 4 atoms).

## Realize vs collapse (the length rule)

A collapsed expandable segment has independent `min`/`max`. Contig is expressible
iff its variable parts are independent — coupled only when `length` constrains
their sum.

| Case | Action |
| --- | --- |
| No genuine range (`"180-180"`, bare `20`) | **Realize** (min == max, exact) |
| 1 variable part | One collapsed sentinel (intersect with `length` if set) |
| 2+ variable parts, no `length` | N independent collapsed sentinels |
| 2+ variable parts **with** `length` | **Inexpressible** — raise |

## Compatibility adapter

```text
build_from_rfd3.py <benchmark> <key> [--output out.cif] [--dump]
                   [--benchmark_root DIR] [--fix_res_id_gap]
                   [--coordination_bond 'C/ZN/296/ZN,A/CYS/365/SG' ...]
                   [--list_metal_candidates]
```

Invoke it through:

```bash
bash skills/rfd4-proteina-inputs/scripts/build_from_rfd3.sh input.json example_key --output out.cif
```

- `benchmark` falls back to `data/benchmarks/<benchmark>` if it is not a path.
- Omitting `--output` builds without writing; `--dump` prints a summary.
- Coordination-bond selectors address the built array as
  `CHAIN/RES_NAME/RES_ID/ATOM_NAME`. A selector matching zero or multiple atoms is
  an error. Use `--list_metal_candidates` to inspect candidates; never infer a
  coordination bond from distance alone.

After conversion, use the sibling
[`rfd4-proteina-annotated-cif`](../../rfd4-proteina-annotated-cif/SKILL.md)
skill for condition semantics, Python inspection, invariant interpretation, and
any direct edits to the resulting CIF.
