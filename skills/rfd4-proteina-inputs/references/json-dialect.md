# Legacy JSON dialect reference

Source of truth: `docs/guides/input.md`, `.claude/commands/json2cif.md`, and
`scripts/json2cif/build_cif.py`. The JSON `InputSpecification` (dialect 2) is
retained for RFD3 compatibility; **annotated CIFs are preferred**. Use this
reference when converting a legacy JSON with `build_cif.sh`.

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
- **`select_fixed_atoms` has NO default — it raises** if omitted (docs/input.md and
  the field docstring are stale on this). Omitting a ligand from it **zeroes that
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

## `build_cif.py` CLI (via `build_cif.sh`)

```
build_cif.py <benchmark> <key> [--output out.cif] [--dump]
             [--benchmark_root DIR] [--fix_res_id_gap]
             [--coordination_bond 'C/ZN/296/ZN,A/CYS/365/SG' ...]
             [--list_metal_candidates]
```

- `benchmark` falls back to `data/benchmarks/<benchmark>` if not a path.
- `--output` omitted => build only (no file). `--dump` prints a summary.
- Coordination-bond selectors are `CHAIN/RES_NAME/RES_ID/ATOM_NAME` against the
  **built** array (the build renumbers residues; use `--list_metal_candidates` to
  get usable selectors). A selector matching zero or multiple atoms is an error.

## `check_annotated_cif.py` CLI (via `check_annotated_cif.sh`)

```
check_annotated_cif.py <file.cif> [--strict] [--skip_pipeline] [--quiet]
```

Part 1: invariant pipeline + missing-atom / res_id-gap checks. Part 2: gotcha scan
(formal charges, ligand bond orders, metal bonds, duplicate res_names,
struct_conn). Exit 1 on any error (`--strict` also on warnings), 2 if the file
cannot load. Severities: ERROR (gates), WARN (gates only under `--strict`), NOTE
(never gates — expected-but-worth-confirming).
