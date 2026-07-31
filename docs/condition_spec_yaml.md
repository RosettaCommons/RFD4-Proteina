# YAML/JSON Condition Specs

A **condition spec** describes one or more design problems as data (YAML or JSON) instead of a
PyMOL session or Python script. It builds the *exact same* Condition-annotated CIF the plugin
builds — annotated CIFs are still the preferred format; this is just a convenient shorthand.

Read `docs/guides/annotated_cif_inputs.md` first — every concept here (Conditions, expandable segments,
the interaction rules) is defined there.

> **Everything you don't mention is not provided.** Setting only `C_HOT` gives the model a hotspot
> and nothing else — no fixed coordinates, no known sequence. The tool prints a summary of what it
> actually set every run; read it.

## File shape

The default format for any `.json`/`.yaml`/`.yml` passed to `generate.py` — no marker needed.
Each top-level key is one example:

```yaml
global_args:                  # optional: shared defaults folded into every example below
  input_as_annotated_cif: true

my_design:
  input: ./inputs/complex.cif
  contig: "A1-120, /0, B1-80"
  conditions: {...}
```

| Field | Default | Meaning |
|---|---|---|
| `input` | required | Starting structure(s). **Must be `.cif`/`.cif.gz`/`.bcif`** (see FAQ). A list is combined into one structure (chain ids must not overlap). Relative paths resolve against the spec file's directory. |
| `input_as_annotated_cif` | `false` | `true`: `input` already carries Condition annotations (a prior output of this tool, or a plugin CIF) — loaded as-is to edit further. `false`: a raw structure — heavy atoms added, MSE→MET, any existing conditioning ignored. |
| `contig` | *keep everything* | Subset the input and insert expandable segments in one pass. See below. |
| `conditions` | *(none)* | Conditions to set. See below. |

`global_args` is an optional reserved top-level key (not an example): any of the fields above
placed there apply to **every** example, and a per-example value overrides the shared default.

**Order per example:** load + combine `input` → apply `contig` → apply `conditions` (resolved
against the assembled structure) → dry-run safeguards + printed summary. If a safeguard fails the
tool stops instead of writing a CIF it can't guarantee.

## Selection syntax

Every `select:` is one atomworks query string. Two interchangeable styles, mixable in one string:

- **Path syntax** — concise, for the common chain/residue/atom case:
  `sel('<chain>/<res_name>/<res_id>/<atom_name>')`. Trailing fields default to `*` (any); `[...]`
  holds a value list or `res_id` range(s), e.g. `[10-20]`, `[10, 20, 130]`, `[1-30, 120-150]`.
- **Query syntax** — general: a Python-boolean expression over *any* annotation (`chain_id`,
  `res_id`, `res_name`, `is_ligand`, `is_protein_backbone`, ...), combinable with `and`/`or`/`not`.

```yaml
select: "sel('A/*/[10-20]')"                   # concise path: chain A, res_id 10-20
select: "sel('*/E1')"                          # an expandable segment, by the name you gave it in `contig`
select: "is_ligand"                            # query syntax: select by any (even computed) annotation
select: "sel('A/*/[10-20]') and not is_ligand" # the two styles composed in one string
```

`res_id` (in either style) resolves against your **original** input numbering — see [`contig`](#contig).

## `contig`

One comma-separated string of tokens, e.g. `"A42-48, E1:2-20, A52, E2:10R"`:

| Token | Meaning |
|---|---|
| `/0` | Chainbreak — end the current output chain, start a new one (id picked automatically). |
| `A42-48`, `AA52` | **Preserved segment** — keep this residue/range from the input, by its *original* `chain_id`/`res_id`. Ranges are low-to-high only (`A42-48`; a reversed `A48-42` is rejected as a typo, not read as a reversed ordering). Output chain id unchanged. |
| `E1:2-20` | **Named expandable segment** — a variable/fixed-length placeholder. `<name>` becomes its `res_name` (unique; not `MASKED`), so `conditions:` can select it. Optional trailing `P`/`R`/`D` sets chain type. |
| `20-30`, `10R` | Unnamed expandable segment — same, without a queryable name. |

- Any input atom **not** covered by a preserved segment is dropped — that's the subsetting.
- **Chain assignment is per-run, not left-to-right.** A run of tokens between `/0`s is one output
  chain. If the run contains a preserved segment, the *whole run* takes that segment's chain id and
  type — even expandable segments earlier in the string. E.g. `A1-10, /0, E1:1-20, B1-10`: `E1`
  and `B1-10` are one run, so `E1` lands on chain `B`. A run with no preserved segment gets a fresh
  chain id (and defaults to type `P`).
- An expandable segment's type defaults to its run's anchor type (RNA anchor → RNA, etc.); an
  explicit suffix that disagrees with the anchor is rejected.
- Expandable segments are inserted **collapsed**; the model samples their length at generation time.
- Output `res_id` is a fresh contiguous per-chain counter starting at 1.

**Referencing original numbering:** `conditions:` `res_id` selections resolve against your
**original** (pre-`contig`) numbering, so `res_id == 45` means "residue 45 in my input file," not
"after renumbering." Expandable segments have no original `res_id` — select them by `res_name`.

> **⚠️ `res_id` means the mmCIF `label_seq_id`, never the author-deposited `auth_seq_id`.** Both
> `contig` tokens and `conditions:` `res_id` selections match against `label_seq_id` — the
> programmatic, per-entity numbering — not `auth_seq_id`, which is what PyMOL may show unless you use the following (recommended)
setting: `set cif_use_auth, off`.
> The two often differ (author numbering can start anywhere, skip values, etc)

## `conditions`

```yaml
conditions:
  <annotation>:        # shorthand alias (C_CRD, C_HOT, ...) or full name (coordinate, hotspot, ...)
    <value>:
      - select: <query>
      - select: <query>   # a value maps to a list, even for one entry
```

```yaml
conditions:
  C_CRD:                 # mask-only (see below): reveals ground-truth coords for the selection
    true:
      - select: "sel('B')"
  C_HOT:
    true:
      - select: "sel('A/*/[45, 46, 82]')"
  C_RAS:
    BURIED:              # enum member name, or its raw int / numeric string ("3") — both accepted
      - select: "sel('A/*/[10-20]')"
  C_CENTROID:            # literal xyz, not derived from the selection
    "12.3,-4.5,0.0":
      - select: "sel('B')"
```

- Most annotations set both a **value** (written to matching atoms) and that atom's mask.
- **`C_CRD`, `C_SEQ`, `C_IDX`, `C_INTCRD` are mask-only** — `<value>` is just `true`/`false` (which mask),
  no separate value: `C_CRD`'s value is the atom's real coordinate, `C_SEQ`'s is the residue's real
  `res_name`, `C_IDX`'s is structural, `C_INTCRD`'s is computed. `C_IDX: false` and
  `C_INTCRD: true` additionally need a `group:` (below).
- **Enum conditions** (`C_RAS`, `C_HB`, ...) take the member name or the int; don't pass `true`/`false`.
- **`C_CENTROID` is per-chain and all-or-nothing.** Its `select:` must cover *every* atom of
  each chain you condition (`sel('B')` / `chain_id == 'B'`, not one residue), and either every
  chain in the example gets a centroid or none does — a partially covered chain leaves the rest
  of it with a NaN centroid the model reads as real. Validation rejects both shapes.
- Overlapping entries for the same annotation resolve **last-write-wins** in declaration order.
- **Not supported:** 2-body conditions like base-pairing (no selection-pair grammar). Everything
  else, including `C_CENTROID`, works — use the plugin/Python API for the rest.

### Groups (`C_IDX: false`, `C_INTCRD: true`)

These two need a `group:` sibling of `select:`:
- For unindexed regions (`C_IDX: false`), these groups 
determine which groups have fixed *relative* indices (e.g., unindexing several adjacent residues as one 
unindexed group ensures that they will be adjacent in the output).
- For internal coordinate regions (`C_INTCRD: true`), these groups determine which regions are treated as a single rigid body (e.g., a framework that should move as one piece). So if you have several motifs that should preserve their internal 
structure but can have arbitrary relative positioning, each one should comprise a separate group.

```yaml
conditions:
  C_IDX:
    false:
      - select: "sel('A/*/[10-12]')"
        group: linker_a
      - select: "sel('B/*/[40-42]')"
        group: linker_a       # same label -> one shared unindexed region
  C_INTCRD:
    true:
      - select: "sel('C')"
        group: framework
```

The label is an arbitrary string/int, used only to say "these selections are the same group" — it's
never written verbatim (it's mapped to a fresh internal id). `C_INTCRD`'s coordinates are computed
for you (centered + randomly rotated). No other value takes a `group:`.

**A `group:` is required here — there is no default.** Omitting it raises rather than falling back to
a per-molecule or per-contiguous-run split. Grouping is the whole point of these two values (which
residues share relative indices, which atoms move as one rigid body), and a silent default is easy
to not notice and hard to debug once the design is already generated. This is stricter than the raw
Python API and the PyMOL plugin, where `C_IDXGRP`/`C_CRDGRP` can be set independently of
`C_IDX`/`C_INTCRD`; here the group ids are always derived from these labels.

## Running the model

Point `inputs=` at the spec file, like any CIF — it's consumed directly, one example per top-level key:

```bash
./src/rfproteina/generate.py inputs=my_designs.yaml num_replicates=10 batch_size=8
```

## Examples

Two complete, runnable specs live beside this file:

- [docs/condition_spec_ppi.yaml](condition_spec_ppi.yaml) — a de-novo PPI binder against PD-L1
  (`rfd3_pdl1` from `data/benchmarks/cleaned_ppi.yaml`): expandable binder chain, backbone-only
  coordinate conditioning on the target, atom-level hotspots.
- [docs/condition_spec_complicated.yaml](condition_spec_complicated.yaml) — a small-molecule pocket
  redesign exercising most of the above (multi-chain contig, an expandable segment with
  secondary-structure conditioning, groups, and both selection styles).

**Fix a binder backbone, redesign its sequence, hotspot the target:**

```yaml
binder_redesign:
  input: complex.cif
  conditions:
    # `is_protein_backbone` is what makes this backbone-only -- `sel('B')` alone would
    # condition on every sidechain atom too, which is not "redesign its sequence".
    C_CRD: {true: [{select: "sel('B') and is_protein_backbone"}]}  # backbone-only coords, no sequence leak
    C_HOT: {true: [{select: "sel('A/*/[45, 46, 82]')"}]}
```

**Scaffold a motif with a variable-length linker to a new binder chain:**

```yaml
motif_scaffold:
  input: motif.cif
  contig: "A1-50, LINK:10-30, /0, BINDER:60-90"   # BINDER's run has no anchor -> fresh chain id
  conditions:
    C_CRD: {true: [{select: "sel('A/*/[1, 50]')"}]}
    C_SEQ: {true: [{select: "chain_id == 'A' and res_name != 'LINK'"}]}   # query syntax: exclude the placeholder
```

Materialize a spec to annotated CIF(s) (for inspection — `generate.py` doesn't need this):

```bash
scripts/write_condition_spec_cifs.py docs/condition_spec_ppi.yaml --out-dir ./cifs
```

## FAQ / gotchas

**Why is my `.pdb` rejected?** PDB doesn't reliably encode bonds or formal charges, which the model
consumes. Convert to CIF with atomworks (`parse` + `to_cif_file`) and audit the bonds/charges first.

**A condition matched 0 atoms.** `conditions:` resolve against the *post-`contig`* structure, except
`res_id` (pre-`contig` numbering for preserved segments). A dropped `res_id`, or a `res_id` used
where you meant `res_name`, matches nothing. Trust the printed summary over your intent.

**My selection caught an expandable placeholder.** Sentinels have no original `res_id`, so any
`res_id ==`/range selection excludes them — but a `chain_id`/`res_name`-only selection, or a negated
`res_id !=`, matches them too. For conditions that need real sequence/coords (`C_SEQ`, `C_CRD`), add
`and res_name != '<segment>'`. Conditions meant for the placeholder itself (`C_HOT`) don't need this.

**Are specs checked before the run?** Not by default — each example is built when the run reaches it.
Pass `prevalidate_inputs=true` to `generate.py` to build every example up front and run the same
safeguards the CIF-writer runs (catches typos across a batch, at the cost of building each example
before the run starts). The CIF-writer script below does this by default; pass `--no-prevalidate`
to skip it. Separately, an unrecognized example field — a typo'd `conditions:`, or a stale
`global_args` key folded onto every example — always raises at build time rather than being
silently ignored, whether or not you prevalidate.

**Do I need to write CIFs to disk?** No — `generate.py` builds each example in memory. To
materialize CIFs for inspection/visualization, run `scripts/write_condition_spec_cifs.py <spec> --out-dir <dir>`.
This verification is highly recommended for complicated inputs! 

**I have an old contig-JSON file.** It still works — any `.json`/`.yaml` with legacy-only fields
(`select_fixed_atoms`, `unindex`, ...) or no `input` key is auto-routed through the legacy dialect.
But it's deprecated (emits a `DeprecationWarning`) and will be removed — migrate when you can.
