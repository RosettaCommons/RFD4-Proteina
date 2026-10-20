# Output format reference

Every design comes back as a CIF, with per-atom provenance (standard annotations that record
where each atom came from and how the model produced it) recorded in ordinary `_atom_site`
columns that any CIF reader picks up. This page covers the full output picture:
- where a run's files land on disk
- how they're named
- what the metrics CSVs contain
- the provenance annotation schema

For the metrics that consume these files, see the [metrics reference](metrics.md); for the
task-oriented refolding walkthrough, see the [refolding guide](../how-to/refolding.md); for the
*input*-side conditioning annotations (what you told the model to build, rather than what it
built) see the [annotations reference](annotations.md); for using the provenance annotations to find
a region again in a generated structure, see [Reading outputs](../how-to/read-outputs.md).

**Contents**

- [Where things land](#where-things-land)
- [File naming](#file-naming)
- [Compression](#compression)
- [Metrics CSVs and aggregation](#metrics-csvs-and-aggregation)
- [Refold addresses](#refold-addresses)
- [Tags derived from an address](#tags-derived-from-an-address)
- [Comparing two refolds](#comparing-two-refolds)
- [Standard annotations, and why there is no output JSON](#standard-annotations-and-why-there-is-no-output-json)

## Where things land

Every run writes under `out_dir` (`./runs/<run_name_>/` when you leave it `null`).
`rfd4-generate` and `rfd4-evaluate` use different subdirectory names, but the same layout
underneath:

| Subdirectory | Written by | Contains |
| --- | --- | --- |
| `inference_samples/` | `rfd4-generate` | Designed CIFs, one subdirectory per input example |
| `inference_metrics/` | `rfd4-generate`, when `metrics_tags` is set | `per_example_metrics.csv` and friends |
| `evaluation_samples/` | `rfd4-evaluate` | Same per-example layout; the original input structure is symlinked in rather than copied |
| `evaluation_metrics/` | `rfd4-evaluate` | `per_example_metrics.csv` |

Training-time validation (both the flow model and the autoencoder) writes the analogous
`val_samples/` and `val_metrics/` instead.

Depending on what you asked the run to produce, each example's own subdirectory can hold:

- the designed structure
- a copy (or, for `rfd4-evaluate`, a symlink) of the input structure
- a chain-of-interest-only CIF, when `save_chain_of_interest=true`
- a `trajectories/` folder, when `generation.args.save_trajectory_every` is nonzero
- one refolded CIF plus a confidence JSON per oracle call, when refolding metrics are on

FoldSeek diversity clustering additionally stages extracted single-chain CIFs under
`foldseek_input_chains/{dataset_hash}/{task}/`. These are deleted once clustering finishes unless
you pass `--cleanup_chains=false` to `aggregate_metrics` — they're cheap to regenerate from the
chain-of-interest output, so there's rarely a reason to keep them.

RFD4-Proteina does not write a separate run manifest. Hydra's own config snapshot goes to
`./hydra_outputs` (or wherever `hydra.run.dir` points), not under `out_dir`. The closest thing to a
per-design manifest is `per_example_metrics.csv` itself: every row embeds the fully resolved
generation config as `generation.*` columns, alongside `ckpt_path`, `ckpt_name`, and `seed`.

## File naming

`{example_id}` is the input's example key: the filename stem for a single annotated CIF input, or
the key assigned to each entry of a condition-spec file or directory.

| File | Pattern |
| --- | --- |
| Designed structure | `{example_id}_generated.cif` |
| Input sidecar | `{example_id}_input.cif` |
| Chain of interest | `{example_id}_chain.cif` |
| Trajectory, all-atom | `{example_id}_{traj_key}.cif` |
| Trajectory, CA only (default) | `{example_id}_{traj_key}_bb_ca.xtc` |
| Refold | `{example_id}_{save_id}_{oracle_name}_refolded.cif`, or `{example_id}_{oracle_name}_refolded.cif` when no [pre-oracle transform](metrics.md#pre-oracle-transforms) assigned a `save_id` |
| Refold confidences | `{example_id}[_{save_id}]_{oracle_name}_confidences.json` |
| Per-job metrics | `per_example_metrics_job{N}.csv` |

Refolds are cached by this filename: re-running generation or evaluation into the same output
directory reuses whatever a previous run already left under a given `save_id`, instead of calling
the oracle again. That also means repointing a [pre-oracle
transform](../how-to/advanced/pre-oracle-transforms.md) at a
different structure needs a fresh output directory, or the stale refold gets reported under the new
name.

## Compression

`rfd4-generate` and `rfd4-evaluate` write plain `.cif` by default; training-time validation
defaults to `.cif.zst` (Zstandard level 3). `compress_outputs=true`/`false` overrides either
direction, applying uniformly to every structure file above — designed, input, chain-of-interest,
trajectory, and refolded CIFs all get the same suffix. Plain, gzip, and Zstd inputs are all readable
regardless of this setting.

Two exceptions always write plain CIF, regardless of `compress_outputs`: FoldSeek's extracted
chain-of-interest inputs, and any CIF exported for sharing. `skip_existing` and resuming an
interrupted run preserve whatever encoding a file already has, so keep one encoding per output
directory rather than mixing them.

## Metrics CSVs and aggregation

Each row of `per_example_metrics.csv` is one design, with columns prefixed by the metric's config
key — `clash_metrics.frac_atoms_clashing_1p5A`, `oracle_ppi.min_ipae_coi`, `ppi.pass`, and so on.
See the [metrics reference](metrics.md) for every column that ships.

Large runs split across SLURM array jobs each write their own `per_example_metrics_job{N}.csv`.
Concatenate them with:

```bash
python -m rfproteina.inference_utils.aggregate_metrics ./runs/my_run/inference_metrics/
```

This writes `aggregated_metrics.csv` and `per_task_metrics.csv` next to the per-job files, and
moves the originals into `individual_job_results/`. `launch_generate.sh` runs this automatically as
a dependent job.

Pass `--cluster_with_foldseek` to also cluster designs structurally: it adds a
`foldseek_cluster_id` column, reading the `chain_of_interest` chain the pipeline already identified
from conditioning. That needs `save_chain_of_interest=true` at generation time, since clustering
reads the extracted single-chain CIFs rather than the full design. See [Evaluating
results](../how-to/run-the-model.md#evaluating-results) for the full command set, including
`--output_dir`, `--foldseek_threads`, and the other `aggregate_metrics` flags.

## Refold addresses

A design can have more than one refold: each configured oracle produces one, and a
[pre-oracle transform](metrics.md#pre-oracle-transforms) can fan a single oracle call out into several
named outputs (for example, folding the designed chain alone as `apo`, or against each of several
off-targets). Every refolding metric receives the design as `ground_truth_atom_array` and the refold
currently being scored as `predicted_atom_array`, plus access to **every** refold the design has. You
can name any of those refolds two ways:

| Address | Names the refold by | Example |
| --- | --- | --- |
| `refold.{save_id}` or `refold.untransformed` | what it **is**, either one output of a pre-oracle transform or the design folded as is, and only when exactly one refold is of that kind | `refold.apo`, `refold.her2`, `refold.untransformed` |
| `refold.{oracle_config_key}[.{save_id}]` | which **oracle** produced it | `refold.oracle_ppi`, `refold.oracle_apo.apo` |

The by-oracle form is always available, since oracle config keys are unique. The by-kind form is a
convenience alias that only exists when it is unambiguous: if two untransformed oracles run at once,
or the name would shadow an oracle's own config key, there is no `refold.untransformed` at all, and no
`is_refold:` or `exists_refold:` tags for it either. Name the oracle instead, as in `refold.oracle_ppi`.
Running two untransformed oracles together therefore switches off every metric gated on
`is_refold:untransformed`, with a warning, rather than scoring each of them into one column and
reporting whichever pass happened to fire last.

A transform's fan-out outputs get addressed and tagged the same way per output. An oracle with the
**config key** `oracle_off_target` and a two output transform reports `oracle_off_target.her2.*` and
`oracle_off_target.egfr.*` — as everywhere else, the prefix is the oracle's **config key**, not its
`oracle_name` — and you get both `refold.her2` / `is_refold:her2` and `refold.oracle_off_target.her2` /
`is_refold:oracle_off_target.her2`. A metric gated on `is_refold:her2` scores that one output alone;
gate on the oracle's own tag instead of an `is_refold:` address and the metric runs once per output,
repeating itself under every `save_id`. Config keyed by the bare output name, such as a
`frac_under_thresholds` entry or an `aggregation` list, applies to every output of the fan out.

## Tags derived from an address

Each address gets injected as a tag twice over, alongside an `oracle:{oracle_name}` tag. These three
are added automatically at scoring time, based on which refolds/oracles actually ran for that design —
they are never something you add to `metrics_tags` yourself:

- `is_refold:{address}` names *which* refold the current scoring pass is scoring. One pass runs per
  refold, so without this gate a metric written for one kind of refold also runs against every other
  refold the design has, repeating its value once per refold.
- `exists_refold:{address}` says the design **has** that refold at all, regardless of which pass is
  running. You need it wherever a metric's `input_kwarg_names` reads a *different* refold than the one
  it is being scored on.

The [default refolding metrics](metrics.md#refolding-metrics-after-the-oracle) all carry
`required_tags_all: ["is_refold:untransformed"]`, so by default none of them score a transformed
refold — a transform can change the shape of what is being compared. A metric written for a
transformed refold instead requires that refold's own address, as `is_refold:apo`, plus
`exists_refold:{address}` for every other refold it reads.

## Comparing two refolds

Naming two addresses in a metric's `input_kwarg_names` turns any `StructureComparisonMetric` into a
comparison between two *refolds* of the same design, rather than design against refold. That is all
the built in `refolded_apo_holo_rmsd` is — see
[Comparing two refolds](../how-to/advanced/custom-metrics.md#comparing-two-refolds) for the worked example.

## Standard annotations, and why there is no output JSON

Older workflows expected an output JSON next to the generated CIF, carrying the unindex mapping,
which input contig an inserted region came from, and which atoms were coordinate conditioned. **That
JSON no longer exists.** The same information now goes into **standard annotations written directly
into the output CIF**, defined in `rfproteina.utils.standard_annotations`. It round trips with the
structure itself instead of living in a side file that can drift out of sync.

These are distinct from the [input conditioning annotations](annotations.md): those tell the model
what to build, these record what it actually built and where each piece came from.

| Standard annotation | CIF column | What it tells you |
| --- | --- | --- |
| `SA_SRC` | `annotation_src-component_1_atom` | Per atom source component contig label like `A3-10`, naming which input region this atom came from (`.` if unassigned) |
| `SA_CRDSRC` | `annotation_src-coordinate-conditioned-region_1_atom` | Per atom source contig for the coordinate-conditioned region, where `"."` means not conditioned and determines the derived mask |
| `SA_IDXGRPSRC` | `annotation_src-index-group_1_atom` | Original unindexed `IndexGroup` of a mapped motif token, where `-1` means the token was not an unindexed insertion |
| `SA_CWI` | `_annotation_coords-with-insertion_1_atom` loop | The ground truth coordinate to compare against for coordinate-conditioned atoms, which is what `design_motif_allatom_rmsd` and friends align and score against. This one lives in its own loop rather than an `_atom_site` column |
| `SA_UII` | `is_unindexed_insertion` | Per atom boolean, where `True` marks the atom as an unindexed insertion target. That is what `design_unindexed_motif_allatom_rmsd` restricts itself to |

Read any of these off a parsed structure the same way, for example `SA_CRDSRC.mask(atom_array)` or
`SA_SRC.annotation(atom_array)`. They are exactly the fields the [motif and unindexed RMSD
metrics](metrics.md#design-metrics-inline-and-before-refolding) already consume — that is the whole
point of turning on `design_motif_allatom_rmsd` and `design_unindexed_motif_allatom_rmsd` instead of
reading the CIF by hand; the pipeline does the bookkeeping for you. Generated CIFs already carry
saved conditioning targets onto the matched atoms, and chaining a refinement run should preserve
them — rebuilding them from the generated structure instead would change the design problem, since
the saved coordinate targets can differ from the generated coordinates.

For using these annotations to find a conditioned region or ligand again in a generated structure,
including a worked selection-syntax example, see [Reading outputs](../how-to/read-outputs.md).
