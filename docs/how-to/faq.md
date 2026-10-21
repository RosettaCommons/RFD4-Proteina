# FAQ

## Can I run a fully unconditioned design?

Yes, two ways. If you have a real input structure and just want nothing fixed, omit `conditions:`
entirely (or leave it `{}`) — nothing gets revealed, so the model designs freely at whatever size
and chain layout `input`/`contig` produced. If you want a *pure* de novo design with no real
structure contributing anything, the spec's `contig` cannot express it (see
[`contig` tokens](../reference/yaml-json.md#contig-tokens)); build a raw annotated CIF with one
wholly expandable chain and no conditions through the Python API instead, the way
`data/benchmarks/monomer/*.cif.zst` do.

## Why is my `.pdb` rejected?

PDB does not reliably encode bonds or formal charges, and the model consumes both. Convert to CIF
with atomworks using `parse` and `to_cif_file`, then audit the bonds and charges before you run.

## A condition matched 0 atoms.

Conditions resolve against the structure *after* `contig` runs, except for `res_id`, which uses
pre `contig` numbering for preserved segments. A dropped `res_id`, or a `res_id` where you meant
`res_name`, matches nothing. Trust the printed summary over your intent.

## My selection caught an expandable placeholder.

Sentinels have no original `res_id`, so any `res_id ==` or range selection excludes them. But a
selection on `chain_id` or `res_name` alone matches them, and so does a negated `res_id !=`. For
conditions that need real sequence or coordinates, meaning `C_SEQ` and `C_CRD`, add
`and res_name != '<segment>'`. Conditions aimed at the placeholder itself, like `C_HOT`, do not
need it.

## Are specs checked before the run?

Not by default. Each design gets built when the run reaches it. Pass `prevalidate_inputs=true` to
`generate.py` to build every design up front and run the same safeguards the CIF writer runs. That
catches typos across a whole batch, at the cost of building every design before the run starts. The
CIF writer script does this by default, and `--no-prevalidate` turns it off. Expandable segments are
realized on a temporary copy for the pre-save pipeline check; the saved CIF retains their
placeholders and length ranges. This checks one realization, not every combination of allowed
lengths. Separately, an unrecognized field under a design's key always raises at build time rather
than getting ignored, whether or not you prevalidate. That covers a typo'd `conditions:` and a stale
`global_args` key folded onto every design.

## Do I need to write CIFs to disk?

No. `generate.py` builds each design in memory. To materialize CIFs for inspection, run
`scripts/write_condition_spec_cifs.py <spec> --out-dir <dir>`. For anything complicated, do this. It
is the cheapest way to catch a mistake before you spend GPU time on it.
