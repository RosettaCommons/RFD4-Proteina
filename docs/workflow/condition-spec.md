# YAML and JSON condition specs

A **condition spec** describes one or more design problems as data, in YAML or JSON. It builds
a condition-annotated CIF using the Python API. Annotated CIFs are still the preferred input
format, and a condition spec is shorthand for authoring one.

Read [Preparing inputs](inputs.md) first. It defines every concept used here, including
conditions, expandable segments and the interaction rules.

:::{admonition} On a configured compute cluster?
:class: ipd-cluster
Drop the `python` and call `./src/rfproteina/generate.py` directly, so its shebang runs it inside
the apptainer image. Do this on an interactive GPU. Full form in
[Running the model](running.md#running-inference).
:::

> **Anything you do not mention is not provided.** Set only `C_HOT` and the model gets a hotspot
> and nothing else. No fixed coordinates, no known sequence. The tool prints a summary of what it
> actually set on every run. Read it.

## File shape

This is the default format for any `.json`, `.yaml` or `.yml` you pass to `generate.py`, with no
marker needed. Each top level key is one example.

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
| `input` | required | Starting structures. **Must be `.cif`, `.cif.gz` or `.bcif`**, as the FAQ explains. A list gets combined into one structure, and chain ids must not overlap. Relative paths resolve against the spec file's directory. |
| `input_as_annotated_cif` | `false` | Set `true` when `input` already carries condition annotations, from a prior output or the Python API. It loads as is so you can edit further. Set `false` for a raw structure, which adds heavy atoms, converts MSE to MET, and ignores any existing conditioning. |
| `contig` | *keep everything* | Subset the input and insert expandable segments in one pass. See below. |
| `conditions` | *(none)* | Conditions to set. See below. |

`global_args` is an optional reserved top level key rather than an example. Any of the fields above
placed there apply to **every** example, and a per example value overrides the shared default.

Each example runs in this order. Load and combine `input`, apply `contig`, apply `conditions`
resolved against the assembled structure, then run dry-run safeguards and print a summary. If a
safeguard fails, the tool stops rather than writing a CIF it cannot guarantee.

## Selection syntax

Every `select:` is one atomworks query string. Two styles work, and you can mix them in one string.

- **Path syntax** is concise, for the common chain, residue and atom case:
  `sel('<chain>/<res_name>/<res_id>/<atom_name>')`. Trailing fields default to `*`, meaning any.
  Square brackets hold a value list or `res_id` ranges, like `[10-20]`, `[10, 20, 130]` or
  `[1-30, 120-150]`.
- **Query syntax** is general. Write a Python boolean expression over *any* annotation, including
  `chain_id`, `res_id`, `res_name`, `is_ligand` and `is_protein_backbone`, combined with `and`,
  `or` and `not`.

```yaml
select: "sel('A/*/[10-20]')"                   # concise path: chain A, res_id 10-20
select: "sel('*/E1')"                          # an expandable segment, by the name you gave it in `contig`
select: "is_ligand"                            # query syntax: select by any (even computed) annotation
select: "sel('A/*/[10-20]') and not is_ligand" # the two styles composed in one string
```

In either style, `res_id` resolves against your **original** input numbering. See
[`contig`](#contig).

## `contig`

One comma separated string of tokens, like `"A42-48, E1:2-20, A52, E2:10R"`.

| Token | Meaning |
|---|---|
| `/0` | Chainbreak. Ends the current output chain and starts a new one, with the id picked automatically. |
| `A42-48`, `AA52` | **Preserved segment.** Keeps this residue or range from the input, by its *original* `chain_id` and `res_id`. Ranges run low to high only, so `A42-48` works and a reversed `A48-42` gets rejected as a typo rather than read as reversed ordering. The output chain id stays the same. |
| `E1:2-20` | **Named expandable segment**, a placeholder of variable or fixed length. The name becomes its `res_name`, which must be unique and cannot be `MASKED`, so `conditions:` can select it. An optional trailing `P`, `R` or `D` sets the chain type. |
| `20-30`, `10R` | Unnamed expandable segment. Same thing without a queryable name. |

- Any input atom **not** covered by a preserved segment gets dropped. That is the subsetting.
- **Chain assignment happens per run, not left to right.** A run of tokens between `/0`s is one
  output chain. If that run contains a preserved segment, the *whole run* takes that segment's chain
  id and type, including expandable segments earlier in the string. Take `A1-10, /0, E1:1-20,
  B1-10`. Here `E1` and `B1-10` are one run, so `E1` lands on chain `B`. A run with no preserved
  segment gets a fresh chain id and defaults to type `P`.
- An expandable segment's type defaults to its run's anchor type, so an RNA anchor gives you RNA. An
  explicit suffix that disagrees with the anchor gets rejected.
- Expandable segments get inserted **collapsed**, and the model samples their length at generation
  time.
- Output `res_id` is a fresh contiguous per chain counter starting at 1.

**A note on original numbering.** The `res_id` selections in `conditions:` resolve against your
**original** numbering, before `contig` ran. So `res_id == 45` means residue 45 in your input file,
not residue 45 after renumbering. Expandable segments have no original `res_id`, so select them by
`res_name`.

> **`res_id` means the mmCIF `label_seq_id`, never the author deposited `auth_seq_id`.** Both
> `contig` tokens and `conditions:` `res_id` selections match against `label_seq_id`, the
> programmatic per entity numbering. PyMOL may show you `auth_seq_id` instead unless you set
> `set cif_use_auth, off`, which we recommend. The two often differ, because author numbering can
> start anywhere and skip values.

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
  C_CRD:                 # reveals ground-truth coords for the selection
    true:
      - select: "sel('B')"
  C_HOT:
    true:
      - select: "sel('A/*/[45, 46, 82]')"
  C_RAS:
    BURIED:              # enum member name, or its raw int / numeric string ("3") — both accepted
      - select: "sel('A/*/[10-20]')"
  C_HB:
    BOTH:
      - select: >-
          res_name == 'SAM' and atom_name == "O2'"
  C_PISTACK:
    true:
      - select: "res_name == 'SAM' and atom_name in ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8', 'N9']"
  C_PICATION:
    AROMATIC:
      - select: "res_name == 'SAM' and atom_name in ['N1', 'C2', 'N3', 'C4', 'C5', 'C6', 'N7', 'C8', 'N9']"
  C_CATH:
    "2.60.40":
      - select: "sel('A')"
  C_CENTROID:            # literal xyz, not derived from the selection
    "12.3,-4.5,0.0":
      - select: "sel('B')"
  C_SEQ_PROB_CYS:         # explicit zero is different from leaving CYS unspecified
    0.0:
      - select: "sel('B/*/[10-20]')"
  C_SEQ_PROB_HIS:
    0.0:
      - select: "sel('B/*/[10-20]')"
```

- Conditions store a single **annotation value**; whether an atom is conditioned is derived from that value.
- **`C_CRD` and `C_SEQ` use boolean selections in the spec.** `C_CRD` stores the
  atom's real coordinate and `C_SEQ` stores the residue's real `res_name`.
  Set internal coordinate conditioning through `C_CRDGRP`; direct `C_INTCRD` entries are rejected.
- **Enum conditions** like `C_RAS`, `C_HB`, and `C_PICATION` take the member name or the int. Do not pass `true` or
  `false`.
- **Interaction roles describe the selected atoms.** `C_HB` accepts DONOR, ACCEPTOR, and BOTH;
  BOTH requests donation and acceptance in different bonds. `C_PICATION` accepts AROMATIC,
  CATION, and BOTH. For `C_PISTACK` and the AROMATIC side of `C_PICATION`, select complete
  aromatic rings. Setting both π conditions requests both interaction types.
- **`C_CATH` takes a quoted C.A.T label** (class, architecture, topology), with `x` for
  unspecified trailing levels: `"1.x.x"` requests a mainly-alpha class, `"2.60.40"` an
  immunoglobulin topology, and `"x.x.x"` clears the label and mask. Select whole residues,
  often an entire chain or expandable segment.
- **Sequence probabilities** use one Condition per canonical polymer token in
  `UNIFIED_ATOM37_ENCODING`: 20 amino acids, 4 RNA bases, and 4 DNA bases. They are named
  `C_SEQ_PROB_<token>` (for example `C_SEQ_PROB_CYS`, `C_SEQ_PROB_U`, or `C_SEQ_PROB_DA`). Unknown,
  mask, atomized, and gap tokens are not conditioning options. Only specified identities are
  constrained; omitted identities remain flexible and share the remaining probability mass. Values
  must be in `[0, 1]`, and specified values on a residue may not sum above 1. A value of zero is an
  explicit exclusion, while an omitted value is represented by NaN in the sole value annotation. On a
  `C_SEQ` residue, the pipeline completes these Conditions to the exact one-hot sequence distribution;
  any already specified probability that conflicts with that identity is an error. Sequence-conditioned
  tokens outside the 28-token vocabulary receive explicitly specified zeros for all 28 probabilities.
- **`C_CENTROID` works per chain and all or nothing.** Its `select:` must cover *every* atom of each
  chain you condition, so `sel('B')` or `chain_id == 'B'` rather than one residue. And either every
  chain in the example gets a centroid or none does. A partially covered chain leaves the rest with
  a NaN centroid that the model reads as real. Validation rejects both shapes.
- Overlapping entries for the same annotation resolve **last write wins**, in declaration order.
  Separate DONOR and ACCEPTOR entries do not combine automatically; use BOTH.
- JSON uses the same structure, with quoted object keys, including boolean mask values (`"true"`, `"false"`).
- **Not supported.** Two body conditions like base pairing, because there is no selection pair
  grammar. Everything else works, `C_CENTROID` included. Use the Python API for the rest.

### Index and coordinate groups

`C_IDX: false` requires a `group:` label beside each selection. Selections sharing a
label receive one fresh index-group ID, preserving their relative residue offsets.

For internal coordinate conditioning, assign a **positive integer `C_CRDGRP` ID**
(numeric strings also work). Atoms sharing an ID form one rigid group. Use `null`
to remove membership; reserved IDs `0` and `-1` cannot be assigned directly.
Coordinate-group entries take only `select:`, without a separate `group:` label.

```yaml
conditions:
  C_IDX:
    false:
      - select: "sel('A/*/[10-12]')"
        group: linker_a
      - select: "sel('B/*/[40-42]')"
        group: linker_a
  C_CRDGRP:
    1:
      - select: "sel('C')"
    null:
      - select: "sel('C/*/50')"
```

New internal coordinate targets are left unset until model-input preparation, just
before unindexing. Complete saved `C_INTCRD` targets (for example from a previous
model output when doing partial diffusion) are preserved. Renaming a whole group preserves its targets; changing
its member atoms clears the affected groups' targets, which are rebuilt from the
live structure. Partially populated or nonfinite saved targets raise an error.
Before assigning an internal coord group to atoms with absolute conditioning, clear
their absolute targets with a `C_CRD: false` entry before the `C_CRDGRP` entry.
Setting `C_CRD: true` removes atoms from their internal coord group.

Independent input files receive distinct coordinate-group IDs before concatenation.
Within one `C_CRDGRP` declaration, assignments are applied together, so renaming or
swapping IDs does not change geometry when whole-group membership is unchanged.

## Running the model

Point `inputs=` at the spec file the same way you would at a CIF. It gets consumed directly, one
example per top level key.

```bash
./src/rfproteina/generate.py inputs=my_designs.yaml num_replicates=10 batch_size=8
```

## Examples

Two complete, runnable specs live beside this file.

- [docs/condition_spec_ppi.yaml](../condition_spec_ppi.yaml) builds a de novo PPI binder against
  PD-L1, matching `data/benchmarks/ppi/ppi_bb_only/rfd3_pdl1.cif.zst`. It uses an expandable
  binder chain, backbone only coordinate conditioning on the target, and atom level hotspots.
  Its raw target structure is [5o45_cropped.cif](../examples/inputs/5o45_cropped.cif).
- [docs/condition_spec_complicated.yaml](../condition_spec_complicated.yaml) redesigns a small
  molecule pocket and exercises most of the above, with a multi chain contig, an expandable segment
  carrying secondary structure conditioning, groups, and both selection styles.

**Fix a binder backbone, redesign its sequence, hotspot the target.**

```yaml
binder_redesign:
  input: complex.cif
  conditions:
    # `is_protein_backbone` is what makes this backbone-only -- `sel('B')` alone would
    # condition on every sidechain atom too, which is not "redesign its sequence".
    C_CRD: {true: [{select: "sel('B') and is_protein_backbone"}]}  # backbone-only coords, no sequence leak
    C_HOT: {true: [{select: "sel('A/*/[45, 46, 82]')"}]}
```

**Scaffold a motif with a variable length linker to a new binder chain.**

```yaml
motif_scaffold:
  input: motif.cif
  contig: "A1-50, LINK:10-30, /0, BINDER:60-90"   # BINDER's run has no anchor -> fresh chain id
  conditions:
    C_CRD: {true: [{select: "sel('A/*/[1, 50]')"}]}
    C_SEQ: {true: [{select: "chain_id == 'A' and res_name != 'LINK'"}]}   # query syntax: exclude the placeholder
```

To inspect what a spec produces, materialize it to annotated CIFs. `generate.py` does not need
this step.

```bash
scripts/write_condition_spec_cifs.py docs/condition_spec_ppi.yaml --out-dir ./cifs
```

## FAQ and gotchas

**Why is my `.pdb` rejected?** PDB does not reliably encode bonds or formal charges, and the model
consumes both. Convert to CIF with atomworks using `parse` and `to_cif_file`, then audit the bonds
and charges before you run.

**A condition matched 0 atoms.** Conditions resolve against the structure *after* `contig` runs,
except for `res_id`, which uses pre `contig` numbering for preserved segments. A dropped `res_id`,
or a `res_id` where you meant `res_name`, matches nothing. Trust the printed summary over your
intent.

**My selection caught an expandable placeholder.** Sentinels have no original `res_id`, so any
`res_id ==` or range selection excludes them. But a selection on `chain_id` or `res_name` alone
matches them, and so does a negated `res_id !=`. For conditions that need real sequence or
coordinates, meaning `C_SEQ` and `C_CRD`, add `and res_name != '<segment>'`. Conditions aimed at the
placeholder itself, like `C_HOT`, do not need it.

**Are specs checked before the run?** Not by default. Each example gets built when the run reaches
it. Pass `prevalidate_inputs=true` to `generate.py` to build every example up front and run the same
safeguards the CIF writer runs. That catches typos across a whole batch, at the cost of building
every example before the run starts. The CIF writer script does this by default, and
`--no-prevalidate` turns it off. Expandable segments are realized on a temporary copy for the
pre-save pipeline check; the saved CIF retains their placeholders and length ranges. This
checks one realization, not every combination of allowed lengths.
Separately, an unrecognized example field always raises at build
time rather than getting ignored, whether or not you prevalidate. That covers a typo'd `conditions:`
and a stale `global_args` key folded onto every example.

**Do I need to write CIFs to disk?** No. `generate.py` builds each example in memory. To materialize
CIFs for inspection, run `scripts/write_condition_spec_cifs.py <spec> --out-dir <dir>`. For anything
complicated, do this. It is the cheapest way to catch a mistake before you spend GPU time on it.
