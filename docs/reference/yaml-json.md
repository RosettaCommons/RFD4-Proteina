# YAML/JSON condition-spec parameters

Every field accepted in a YAML/JSON condition specification file. This page is a lookup table; for
the concepts and worked examples see [YAML/JSON condition spec](../how-to/input_prep/yaml-json.md),
and for common errors see the [FAQ](../how-to/faq.md).
For the conditioning model these annotations feed into, see the
[annotated CIF guide](../how-to/input_prep/index.md).

**Contents**

- [Top-level fields](#top-level-fields)
- [`global_args`](#global-args)
- [`contig` tokens](#contig-tokens)
- [Selection syntax](#selection-syntax)
- [`conditions`](#conditions)
  - [Entry shape](#entry-shape)
  - [Condition annotations](#condition-annotations)
  - [Index and coordinate group fields](#index-and-coordinate-group-fields)
  - [Not supported](#not-supported)

## Top-level fields

There can be multiple design tasks in a single YAML/JSON file, they are differentiated by naming the design.
This name acts as a "key" that is linked to the constraints that are specified for it.  
The constraints for each design are either indented from the name in the case of a YAML file (see example below)
or in a set of curly brackets in the case of a JSON file.

```yaml
my_design:            # a name you chose
  input: ./inputs/complex.cif
  contig: "A1-120, /0, B1-80"
  conditions: {...}
```

The one reserved key is `global_args` (see [below](#global-args)) — every other top-level key works as shown above.

| Field | Type | Required/Default | Description |
| --- | --- | --- | --- |
| `input` | path, or list of paths | **Required** | Starting structure(s). Must be `.cif`, `.cif.gz`, or `.bcif`. A list gets combined into one structure; chain ids must not overlap. Relative paths resolve against the spec file's directory. |
| `input_as_annotated_cif` | bool | Optional — `false` | `true` loads `input` as is, preserving existing condition annotations (from a prior output or the Python API) so you can edit further. `false` treats `input` as a raw structure: adds heavy atoms, converts MSE to MET, and ignores any existing conditioning. |
| `contig` | string | Optional — keeps everything | Subsets the input and inserts expandable segments in one pass. See [`contig` tokens](#contig-tokens). |
| `conditions` | mapping | Optional — none | Conditions to set, resolved against the structure after `contig` runs. See [`conditions`](#conditions). |

An unrecognized field under a design's key always raises at build time, whether or not you pass
`prevalidate_inputs=true`.

## `global_args`

A reserved top-level key — it does not name a design. Any of the [top-level fields](#top-level-fields)
placed there become the shared default for **every** design in the file; a value set directly under
a design's own key overrides it.

```yaml
global_args:
  input_as_annotated_cif: true
```

## `contig` tokens

One comma-separated string of tokens, e.g. `"A42-48, E1:2-20, A52, E2:10R"`. `res_id` in both
`contig` and `conditions` resolves against the **original** input numbering (the mmCIF
`label_seq_id`, not the `auth_seq_id`), before `contig` renumbers the output.

```{note}
PyMOL may show you `auth_seq_id` instead of `label_seq_id` unless you set `set cif_use_auth, off`,
which we recommend. The two often differ, because author numbering can start anywhere and skip
values.
```

| Token | Meaning |
| --- | --- |
| `/0` | Chainbreak. Ends the current output chain and starts a new one, with the id picked automatically. |
| `A42-48`, `AA52` | **Preserved segment.** Keeps this residue or range from the input, by its original `chain_id` and `res_id`. Ranges run low to high only. The output chain id stays the same as the input. |
| `E1:2-20` | **Named expandable segment**, a placeholder of variable or fixed length. The name becomes its `res_name` (must be unique, cannot be `MASKED`) so `conditions:` can select it. An optional trailing `P`, `R`, or `D` sets the chain type. |
| `20-30`, `10R` | Unnamed expandable segment. Same as above without a queryable name. |

Any input atom not covered by a preserved segment is dropped. Output `res_id` is a fresh
contiguous per-chain counter starting at 1.

**Chain assignment happens per run, not left to right.** A run of tokens between `/0`s is one
output chain. If that run contains a preserved segment, the *whole run* takes that segment's chain
id and type, including expandable segments earlier in the string. Take `A1-10, /0, E1:1-20,
B1-10`. Here `E1` and `B1-10` are one run, so `E1` lands on chain `B`. A run with no preserved
segment gets a fresh chain id and defaults to type `P`.

An expandable segment's type defaults to its run's anchor type, so an RNA anchor gives you RNA. An
explicit suffix that disagrees with the anchor gets rejected.

Expandable segments get inserted **collapsed**, and the model samples their length at generation
time.

A `contig` must contain at least one preserved segment; one made entirely of expandable segments is
rejected, since `input` exists only to be referenced by a preserved segment. There is no `contig`-only
way to generate a design entirely from scratch — see
[FAQ: fully unconditioned design](../how-to/faq.md#can-i-run-a-fully-unconditioned-design).

## Selection syntax

Every `select:` value is one atomworks query string, in one of two styles (mixable in one string):

| Style | Form | Example |
| --- | --- | --- |
| Path syntax | `sel('<chain>/<res_name>/<res_id>/<atom_name>')`, trailing fields default to `*`. Brackets hold a value list or `res_id` ranges: `[10-20]`, `[10, 20, 130]`, `[1-30, 120-150]`. | `sel('A/*/[10-20]')` |
| Query syntax | A Python boolean expression over any annotation (`chain_id`, `res_id`, `res_name`, `is_ligand`, `is_protein_backbone`, ...), combined with `and`/`or`/`not`. | `is_ligand`, `sel('A/*/[10-20]') and not is_ligand` |

## `conditions`

Conditions store a single **annotation value**; whether an atom is conditioned is derived from
that value.

### Entry shape

```yaml
conditions:
  <annotation>:        # shorthand alias (C_CRD, C_HOT, ...) or full name (coordinate, hotspot, ...)
    <value>:
      - select: <query>
      - select: <query>   # a value maps to a list, even for one entry
```

```{note}
  - Overlapping entries for the same annotation resolve **last write wins**, in declaration order.
  - JSON uses the same structure, with quoted object keys, including boolean mask values (`"true"`,
    `"false"`).
```

### Condition annotations

| Annotation | Full name | Value(s) | `select:` covers | Notes |
| --- | --- | --- | --- | --- |
| `C_CRD` | coordinate | `true` / `false` | atoms | Boolean selection; stores the atom's real coordinate. |
| `C_SEQ` | sequence | `true` / `false` | residues | Boolean selection; stores the residue's real `res_name`. |
| `C_HOT` | hotspot | `true` / `false` | atoms | |
| `C_RAS` | rasa-binned | `BURIED`, `INTERMEDIATE`, `EXPOSED`, `UNSPECIFIED` (name, int, or numeric string) | atoms | Binned RASA value. |
| `C_HB` | hbond | `DONOR`, `ACCEPTOR`, `BOTH` | atoms | Interaction role. `BOTH` requests donation and acceptance in different bonds; separate `DONOR`/`ACCEPTOR` entries do not combine automatically. |
| `C_PISTACK` | pistack | `true` / `false` | atoms | Select complete aromatic rings. |
| `C_PICATION` | pication | `AROMATIC`, `CATION`, `BOTH` | atoms | `AROMATIC` side selects complete aromatic rings; setting both π conditions requests both interaction types. |
| `C_CATH` | cath-label | quoted C.A.T label, e.g. `"2.60.40"`, with `x` for unspecified trailing levels (`"1.x.x"`, `"x.x.x"` clears it) | whole residues, usually a chain or expandable segment | Class, architecture, topology. |
| `C_CENTROID` | ChainCentroid | literal `"x,y,z"` string | every atom of a chain | Per chain and all-or-nothing: `select:` must cover the *entire* chain, and either every chain in the design gets a centroid or none does — a partially covered chain is rejected. |
| `C_SEQ_PROB_<TOKEN>` | sequence-probability | float in `[0, 1]` | residues | One Condition per canonical polymer token in `UNIFIED_ATOM37_ENCODING`: 20 amino acids, 4 RNA bases, 4 DNA bases (e.g. `C_SEQ_PROB_CYS`, `C_SEQ_PROB_U`, `C_SEQ_PROB_DA`). Unknown, mask, atomized, and gap tokens are not conditioning options. `0.0` is an explicit exclusion; an omitted identity remains flexible and shares the remaining probability mass. Specified values on a residue may not sum above 1. On a `C_SEQ` residue these complete to the one-hot sequence distribution; a conflicting already-specified probability is an error. Sequence-conditioned tokens outside the 28-token vocabulary receive explicit zeros for all 28 probabilities. |
| `C_IDX` | index | `false` | atoms, with a `group:` label | See [Index and coordinate group fields](#index-and-coordinate-group-fields). |
| `C_CRDGRP` | CoordinateGroup | positive integer ID (numeric strings work), or `null` | atoms | See [Index and coordinate group fields](#index-and-coordinate-group-fields). |

`C_CRD` and `C_SEQ` are the two conditions that take boolean selections directly in the spec;
everything else above is an enum, string, or float value keyed in `conditions:`. Direct `C_INTCRD`
entries are rejected — set internal coordinate conditioning through `C_CRDGRP` instead.

### Index and coordinate group fields

`C_IDX: false` requires a `group:` label beside each selection (in addition to `select:`).
Selections sharing a label receive one fresh index-group ID, preserving their relative residue
offsets:

```yaml
conditions:
  C_IDX:
    false:
      - select: "sel('A/*/[10-12]')"
        group: linker_a
      - select: "sel('B/*/[40-42]')"
        group: linker_a
```

`C_CRDGRP` entries take only `select:`, without a `group:` label — the mapping key itself is the
group ID:

```yaml
conditions:
  C_CRDGRP:
    1:
      - select: "sel('C')"
    null:
      - select: "sel('C/*/50')"
```

Reserved IDs `0` and `-1` cannot be assigned directly. `null` removes an atom's group membership.

New internal-coordinate targets are left unset until model-input preparation, just before
unindexing. Complete saved `C_INTCRD` targets (for example from a previous model output when doing
partial diffusion) are preserved. Renaming a whole group preserves its targets; changing its member
atoms clears the affected group's targets, which are rebuilt from the live structure. Partially
populated or nonfinite saved targets raise an error.

Before assigning an internal-coord group to atoms with absolute conditioning, clear their absolute
targets with a `C_CRD: false` entry before the `C_CRDGRP` entry. Setting `C_CRD: true` removes
atoms from their internal-coord group.

Independent input files receive distinct coordinate-group IDs before concatenation. Within one
`C_CRDGRP` declaration, assignments are applied together, so renaming or swapping IDs does not
change geometry when whole-group membership is unchanged.

### Not supported

Two-body conditions like base pairing have no selection-pair grammar in the spec format. Use the
Python API for these; everything else, `C_CENTROID` included, works through the spec.
