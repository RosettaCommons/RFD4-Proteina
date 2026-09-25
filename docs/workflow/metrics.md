# Metrics

You should not have to hand parse output CIFs. You *can* reconstruct the unindexed mapping, or work
out which chain an inserted region came from, or measure how closely a design followed its
coordinate conditioning, all by reading raw CIF fields. Do that only when nothing else works.
Turn on the metric that already computes it and let the pipeline run it. Metrics that run in the
pipeline can also drive [reward guided search](search.md), which finds good designs on far less
compute than generating everything and filtering afterwards.

:::{admonition} On a configured compute cluster?
:class: ipd-cluster
Drop the `python` and call `./src/rfproteina/generate.py` directly, so its shebang runs it inside
the apptainer image. Do this on an interactive GPU. Full form in
[Running the model](running.md#running-inference).
:::

## What each metric is looking at

Metrics do not all measure the same object. Four different things get measured, and knowing which
one a metric looks at tells you what a bad number means.

| Family | Looks at | Answers | Examples |
| --- | --- | --- | --- |
| **Design** | the design alone | Is this a physically sensible structure? | clashes, Ramachandran |
| **Adherence** | design against your input | Did it build what you asked for? | motif RMSD, sequence adherence |
| **Refolding** | the refolded structure alone | Does a folding model believe it? | pLDDT, PAE, pTM |
| **Cross comparison** | input, design and refold together | Does the whole story hold up? | insertion RMSD, chirality |

A bad design metric means the model built something strained. A bad adherence metric means it
ignored part of your spec. Those need different fixes, which is why the split matters.

[Every metric that ships](#every-metric-that-ships) below lists all of them, what each outputs,
and the tag that turns it on.

## Binder and target selections

Binder RMSDs use `is_chain_of_interest`; target alignment uses conditioned atoms outside that chain.
Ambiguous inference requires an explicit `chain_iid` or `chain_id` query.
The backbone-aligned variant aligns target backbone atoms but still measures all binder atoms.

## Metrics can stop a run early

Metrics are not only scored at the end. Three points in a run read them and decide what proceeds,
and that is where most of the compute savings live.

| Gate | Sits | Decides |
| --- | --- | --- |
| **Pre-oracle filters** | before refolding | which designs are worth an oracle call at all |
| **Pass criteria** | after refolding | what counts as a success, judging design, oracle and refold together |
| **Search pruning** | inside the trajectory | which candidates survive to the next checkpoint |

Refolding costs the most, so a pre-oracle filter that drops obviously broken designs is your biggest
lever on cost per success. One exception. When you benchmark per example, turn every pre-oracle
filter off, or you end up measuring the filter instead of the model.

## How many metrics ship enabled

`configs/metrics/all_metrics.yaml` already registers **46 metrics, oracles, pass criteria and
filters**, and
they run automatically. No code, no new config files. Most sit behind one or more **`metrics_tags`**,
so you turn one on by passing a tag on the command line.

```bash
python src/rfproteina/generate.py \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'
```

Here is what is wired in.

| Category | Count | Config |
| --- | --- | --- |
| Design metrics, run inline before refolding | 21 | `configs/metrics/inline_metrics/design_metrics.yaml` and `developability_metrics.yaml` |
| Structure prediction metrics | 1 | `configs/metrics/inline_metrics/structure_prediction_metrics.yaml` |
| Refolding metrics, design against refold or refold against refold | 7 | `configs/metrics/refolding_metrics/all_refolding_metrics.yaml` |
| Refolding oracles, the AF3 variants | 4 | `configs/metrics/refolding_oracles/af3.yaml` |
| Pass criteria | 10 | `configs/metrics/pass_criteria/all_pass_criteria.yaml` |
| Pre-oracle filters, which skip the oracle when a cheap check fails | 6 | `configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml` |

Another **9 metric classes already live in the codebase** without a default config. They cover TMol
interface ddG, H-bonds and electrostatics, an atom pair distance constraint used for catalytic
geometry, a QM ligand interaction energy, MACE ML potential energies, charge complementarity, and a
few others. Turning one on takes **no Python**, just a few lines of YAML, exactly like the [worked
example](custom-metrics.md#worked-example-an-existing-metric-your-own-config). See [Ready-made
but opt-in](#ready-made-but-opt-in).

Write a new `Metric` subclass only when none of those 53 cover what you need. See [Writing your
own](custom-metrics.md#writing-your-own).

## Running metrics offline on a folder of structures

You do not have to re-run generation to get metrics. `evaluate.py` scores a folder of structures you
already have, on the same Hydra config, so `metrics_tags` and any custom metric config work exactly
as above.

```bash
python src/rfproteina/evaluate.py \
    inputs=/path/to/structures \
    metrics_tags='[unconditional_refolding]' \
    out_dir=/path/to/eval_output
```

`evaluate.py` scans `inputs` recursively for CIF-like files. `out_dir` is required, because
`evaluate.py` has no run directory to derive one from, and it gets the usual `evaluation_metrics/`
subdirectory.

Pass `task_csv=<csv with designed_structure_path and task columns>` when different structures need
different tags or pass criteria. Without it, every structure is scored the same way. The [Developer
Guide](../development.md#evaluation-and-metrics) points at this same script for custom metric work,
so you write a metric once and use it both inline during generation and offline over old outputs.

For large batches split across SLURM array jobs (`launch_generate.sh --njobs N`), each job writes its
own `per_example_metrics_job{N}.csv`. Concatenate them afterwards.

```bash
python -m rfproteina.inference_utils.aggregate_metrics \
    ./runs/my_run/inference_metrics/
```

`launch_generate.sh` also runs this for you as a dependent job. See [Evaluating
results](running.md#evaluating-results) for FoldSeek diversity clustering and the other
`aggregate_metrics` options.

## Every metric that ships

Three keys control when a metric runs. `required_tags_any` runs it if you pass **any** of the listed
tags. `required_tags_all` runs it only if you pass **all** of them. `prohibited_tags` skips it if
you pass a listed tag, even when the required tags match. A metric with no tags always runs. The
[custom metrics guide](custom-metrics.md#how-kwargs_to_compute_args-works) has the full rules.

Design tags and refolding tags live in separate namespaces, so pass both when you want both
families: `metrics_tags='[design, unconditional_refolding]'`.

### Design metrics, inline and before refolding

These run directly on the generated structure, before any oracle refolds it.

| Config key | Tag gate | What it measures | Output columns (prefixed `<key>.`) |
| --- | --- | --- | --- |
| `bond_length_metrics` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Fraction of bonds with an unreasonable length, on element dependent thresholds | `frac_unreasonable`, plus a per category breakdown in verbose mode |
| `clash_metrics` | any: `design`, `structure_prediction`<br>not: `small_molecule_only` | Fraction of atoms in a steric clash at a configurable threshold, 1.5 Å by default, split by clash type | `frac_atoms_clashing_1p5A` plus 5 category breakdowns, intra and inter residue by polymer and ligand |
| `hotspot_obedience` | any: `hotspot_conditioning` | Fraction of hotspot atoms with a binder atom within 4.5 Å | `hotspot_obedience` |
| `rasa_adherence` | any: `rasa_conditioning` | How well the generated burial matches your RASA bin conditioning | `buried_conditioning_adherence`, `intermediate_conditioning_adherence`, `exposed_conditioning_adherence` |
| `centroid_conditioning_adherence` | any: `centroid_conditioning` | Mean distance between each conditioned chain centroid and the requested position | `centroid_conditioning_mae` |
| `interchain_interactions` | any: `interchain_interactions`<br>not: `small_molecule_only`, `structure_prediction` | Adherence plus atom and pair counts for H-bond, π-stacking, π-cation, salt bridge, hydrophobic and metal coordination conditioning, interchain only | `*_conditioning_adherence`, `*_atom_count_coi` / `*_residue_count_coi`, `*_atom_pair_count_coi` / `*_residue_pair_count_coi` per interaction type |
| `sequence_adherence` | any: `design` | Fraction of sequence conditioned residues whose generated identity matches what you asked for. NaN when nothing was sequence conditioned, and in the offline re-scoring path, which has no input structure to read the conditioning off | `sequence_adherence`, `n_violations` |
| `not_cysteine_adherence` | any: `design` | Fraction of `C_NCYS`-conditioned residues that are not CYS. Counts violating residues; returns NaN adherence and zero violations when no residues are conditioned | `not_cysteine_adherence`, `n_violations` |
| `sse_distribution` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Helix, sheet and loop fraction, plus the number of contiguous SSE elements, from P-SEA | `helix_fraction`, `sheet_fraction`, `loop_fraction`, `num_ss_elements` |
| `sse_accuracy` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Accuracy of generated secondary structure against your secondary structure conditioning | `overall_accuracy`, `helix_recall`, `sheet_recall`, `loop_recall`, `n_conditioned_atoms` |
| `posebusters_metrics` | any: `small_molecule_only` | Ligand geometry and validity checks from PoseBusters | one pass or fail column per PoseBusters check |
| `chiral_accuracy` | any: `small_molecule_only` | Fraction of stereogenic centers with correct handedness, split tetrahedral against non-tetrahedral | `tetrahedral_percent_correct_chirality`, `non_tetrahedral_percent_correct_chirality` |
| `design_motif_allatom_rmsd` | any: `3d_coordinate_conditioning`, `unindexed`, `internal_coordinate_conditioning` | All atom RMSD of the coordinate conditioned motif region, measuring self consistency within the design itself | `rmsd`, the median over replicates, and per `CoordinateGroup` when there are several motifs |
| `design_unindexed_motif_allatom_rmsd` | all: `unindexed` | All atom RMSD restricted to **unindexed** insertion atoms only | `rmsd` |
| `unindexed_insertion_mapping` | all: `unindexed` | Whether the insertion head applied its unindexed to generated token mapping in full, meaning every unindexed token got inserted, onto a residue of the conditioned identity wherever the unindexed region was sequence conditioned | `n_unmapped_tokens`, `n_sequence_mismatches` |
| `contact_molecular_surface` | any: `cms` | Distance weighted interfacial contact surface area between binder and target, following Lawrence and Colman 1993 | see `rfproteina.metrics.physics.contact_ms.metric` |
| `sequence_similarity_to_input` | any: `partial_diffusion` | Sequence identity against the seed structure, for partial diffusion | `sequence_identity_to_input` |
| `cath_metric` | any: `cath` | CATH superfamily classification of the chain of interest, through Foldseek | `C`, `A`, `T`, `H`, `type`, `evalue`, `tmscore` |
| `sequence_developability` | any: `developability`<br>not: `small_molecule_only`, `structure_prediction` | GRAVY and the Guruprasad instability index on the designed chain. Half-life is deliberately omitted | `gravy`, `instability_index` |
| `antibody_contacts` | any: `antibody` | Fraction of contacting variable-domain residues in Chothia CDRs | `cdr_contact_count`, `fw_contact_count`, `cdr_contact_fraction` |
| `cdr_torsion_naturalness` | any: `antibody` | Chothia CDR phi/psi naturalness against bundled native samples | `H1`–`L3`, corresponding `*_length`, `n_scored_cdrs` |
| `rosetta_developability` | any: `developability`<br>not: `small_molecule_only`, `structure_prediction` | Rosetta SAP and ExposedHydrophobics through a nested PyRosetta apptainer. Set `PYROSETTA_CONTAINER` outside the configured cluster environments | `sap_score`, `hydrophobic_exposure_sasa` |

Antibody metrics use ANARCII and basic Chothia CDRs across all protein chains,
independently of conditioning masks. `annotate_antibody_numbering()` caches
annotations; use `overwrite=True` after sequence edits. Contacts compare numbered
variable domains against chains without antibody domains.

Torsions use VH/VL references when a light domain is present, otherwise VHH. For an
isolated VH, set `cdr_torsion_naturalness.reference_set=vhvl`. Incomplete or unsupported
CDRs return NaN; score multiple domains of the same antibody chain type separately.

### Structure prediction metrics

| Config key | Tag gate | What it measures | Output columns |
| --- | --- | --- | --- |
| `structure_prediction_lddt` | any: `structure_prediction` | All atom lDDT against ground truth, for structure prediction style validation runs rather than design | `lddt` |

### Refolding metrics, after the oracle

These compare the **design**, treated as ground truth, against the **oracle refolded structure**,
treated as the prediction. The `is_refold:` and `exists_refold:` gates below only matter once an
oracle carries a pre-oracle transform, which [pre-oracle transforms](#pre-oracle-transforms)
explains.

| Config key | Tag gate | What it measures | Output columns |
| --- | --- | --- | --- |
| `refolded_backbone_rmsd` | any: `unconditional_refolding`, `ppi_refolding`, `antibody`, `dna_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding`<br>all: `is_refold:untransformed` | Backbone self-consistency RMSD (scRMSD) between design and refold | `rmsd` |
| `refolded_target_aligned_binder_rmsd` | any: `ppi_refolding`, `dna_refolding`, `antibody`<br>all: `is_refold:untransformed` | Binder all-atom RMSD after aligning on the fixed target | `rmsd` |
| `refolded_target_backbone_aligned_binder_rmsd` | any: `ppi_refolding`<br>all: `is_refold:untransformed` | RMSD of atoms without coordinate conditioning after aligning on coordinate-conditioned protein backbone atoms; used for PPI passing | `rmsd` |
| `refolded_motif_allatom_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`<br>all: `is_refold:untransformed` | All atom motif RMSD between design and refold, unindexed regions included | `rmsd` |
| `refolded_motif_ca_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`<br>all: `is_refold:untransformed` | CA only motif RMSD between design and refold, unindexed regions included | `rmsd` |
| `refolded_backbone_aligned_ligand_rmsd` | any: `sm_binder_refolding`<br>all: `is_refold:untransformed` | Ligand RMSD after backbone alignment | `rmsd` |
| `max_ligand_centroid_distance` | any: `disco_refolding`<br>all: `is_refold:untransformed` | Largest ligand centroid displacement after backbone alignment | `ligand_centroid_distance` |
| `refolded_apo_holo_rmsd` | all: `is_refold:apo` (i.e. `apo_refolding`, which is what runs that oracle), `exists_refold:untransformed` | Backbone RMSD of the designed chain between its **apo** refold and its refold in the whole design. Does the binder hold its fold once the target is gone? **Needs a paired untransformed oracle**, as the note below explains | `apo.rmsd` |

Every refolding metric gets the design as `ground_truth_atom_array` and the refold being scored as
`predicted_atom_array`, plus **every** refold of that design. You can address those two ways.

| Address | Names the refold by | Example |
| --- | --- | --- |
| `refold.{save_id}` or `refold.untransformed` | what it **is**, either one output of a [pre-oracle transform](#pre-oracle-transforms) or the design folded as is, and only when exactly one refold is of that kind | `refold.apo`, `refold.her2`, `refold.untransformed` |
| `refold.{oracle_config_key}[.{save_id}]` | which **oracle** produced it | `refold.oracle_ppi`, `refold.oracle_apo.apo` |

Each address gets injected as a tag twice over, alongside the `oracle:{oracle_name}` tag.
`exists_refold:{address}` covers every refold this design has, and `is_refold:{address}` covers the
one being scored right now. The note below explains how to gate on them.

Name two of those in `input_kwarg_names` and any `StructureComparisonMetric` becomes a comparison of
two *refolds* rather than design against refold. That is all `refolded_apo_holo_rmsd` is.

```yaml
input_kwarg_names:
  ground_truth_atom_array: refold.apo             # chain of interest, folded alone
  predicted_atom_array: refold.untransformed      # the same chain, folded in the whole design
```

A by-kind name only exists when it is unambiguous. If two untransformed oracles run at once, or if
the name would shadow an oracle's own config key, you get no `refold.untransformed` at all and no
`is_refold:` or `exists_refold:` tags for it. Name the oracle instead, as in `refold.oracle_ppi`.
So running two untransformed oracles together switches off every metric gated on
`is_refold:untransformed`, with a warning. That beats scoring each of them into one column and
reporting whichever pass happened to fire last.

:::{important}
**Using pre-oracle transforms or more than one OracleMetric? Gate every metric on the relevant
refold specific tags.** `is_refold:{address}` names *which* refold this scoring pass is scoring. One
pass runs per refold, so without that gate a metric written for one kind of refold also runs against
the others and repeats its value once per refold. `exists_refold:{address}` says the design **has**
that refold, and you need it wherever `input_kwarg_names` reads another refold.
`refolded_apo_holo_rmsd` uses one of each.

```yaml
input_kwarg_names:
  ground_truth_atom_array: refold.apo             # scored on the pass for this refold...
  predicted_atom_array: refold.untransformed      # ...against this one, which must exist
required_tags_all: ["is_refold:apo", "exists_refold:untransformed"]
```
:::

### Refolding oracles

An oracle runs the actual structure prediction that the refolding metrics above compare against. It
is heavy and AF3 based. [Evaluating results](running.md#evaluating-results) covers the
apptainer and AF3 setup you need.

| Config key | Tag gate | Notes | Key outputs (prefix `<key>.`) |
| --- | --- | --- | --- |
| `oracle` | any: `af3_refolding`, `unconditional_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding` | Untemplated AF3 refold | `overall_plddt`, `overall_ptm`, `overall_iptm`, `plddt_coi`, `ptm_coi`, `pae_coi`, `ipae_coi`, `min_ipae_coi`, `iptm_coi`, `frac_under_1`/`1.5`/`2` (on `min_ipae_coi`) |
| `oracle_ppi` | any: `ppi_refolding`, `antibody` | Templates protein atoms with ordinary or internal coordinate conditioning | same key set, prefixed `oracle_ppi.` |
| `oracle_dna` | any: `dna_refolding` | DNA templated and fixed, for DNA binder design | same key set, prefixed `oracle_dna.` |

#### Refolding with OpenFold3 instead

The same oracles run on OpenFold3's Apache licensed parameters. Converted OF3 weights are an
ordinary AF3 model directory, so there is no separate runtime. The worker reads the provenance file
the converter leaves beside them and enables the OF3 architecture branches itself.

1. Convert the weights once: `python scripts/setup_of3_weights.py --image "$AF3_IMAGE" --output_dir <dir>`
2. Point `AF3_MODEL_DIR` at `<dir>`.
3. Swap the config group: `metrics/refolding_oracles=of3`.

Step 3 only changes the artifact namespace to `of3`, `of3_ppi` and `of3_dna`, so results do not
collide with AF3 ones. The parameter family follows `AF3_MODEL_DIR` alone. One job uses one family,
so run AF3 and OF3 as separate jobs and compare afterwards.

### Pass criteria

A pass criterion thresholds one or more already computed metric values into a binary flag at
`{config_key}.pass`, evaluated after the design, oracle and refolding metrics. All of them are built
from the generic `PassCriterionMetric`. See [Adding your own pass
criterion](custom-metrics.md#pass-criteria-config).

Separate entries are ANDed together. `min` and `max` are inclusive.
To threshold the best, worst or mean of several keys instead,
say when either of two interfaces being good enough will do, use a `min_of`, `max_of` or `mean_of`
list. See [Thresholding a reduction over several
keys](custom-metrics.md#thresholding-a-reduction-over-several-keys).

| Config key | Tag gate | Thresholds |
| --- | --- | --- |
| `unconditional` | any: `unconditional_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 1.5 |
| `ppi` | any: `ppi_refolding` | `oracle_ppi.ptm_coi` ≥ 80, `oracle_ppi.min_ipae_coi` ≤ 1.5, `refolded_target_backbone_aligned_binder_rmsd.rmsd` ≤ 2.5 |
| `antibody` | any: `antibody` | `oracle_ppi.min_ipae_coi` ≤ 2.0, `refolded_target_aligned_binder_rmsd.rmsd` ≤ 2.0 Å |
| `dna` | any: `dna_refolding` | `oracle_dna.min_ipae_coi` ≤ 1.5, `refolded_target_aligned_binder_rmsd.rmsd` ≤ 8.0 |
| `motif_scaffold` | any: `motif_scaffold_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 2.0, `refolded_motif_ca_rmsd.rmsd` ≤ 1.0 |
| `internal_coords` | any: `internal_coordinate_motif_scaffold_refolding` | Same thresholds as `motif_scaffold` |
| `sm_binder` | any: `sm_binder_refolding` | `oracle.min_ipae_coi` ≤ 1.5, `oracle.iptm_coi` ≥ 80, `refolded_backbone_rmsd.rmsd` ≤ 1.5, `refolded_backbone_aligned_ligand_rmsd.rmsd` ≤ 5.0 |
| `ame` | any: `ame_refolding`<br>not: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5, zero inter residue polymer and ligand clashes |
| `ame_no_ligand` | any: `ame_refolding`<br>all: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5 |
| `disco` | any: `disco_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `max_ligand_centroid_distance.ligand_centroid_distance` ≤ 2.0 |
| `developability` | any: `developability` | `rosetta_developability.sap_score` ≤ 75, `rosetta_developability.hydrophobic_exposure_sasa` ≤ 1500, `sequence_developability.gravy` ≤ 0.1, `sequence_developability.instability_index` ≤ 40 |

### Pre-oracle filters

These are cheap checks that decide whether a design even reaches the **expensive oracle**. They are
tag gated like every other metric in the codebase, and a subset of the built in filters run by
default. See
[`all_pre_oracle_filters.yaml`](../../configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml).

A filter that reads an absent metric key counts as a failure, which happens when the metric's own
tag gate kept it from running. So keep a filter's tag gate at least as strict as its metric's.

> **Benchmarking.** Turn every pre-oracle filter off with `~metrics/pre_oracle_filters` when you
> benchmark **per example**. A filter that discards designs before the oracle inflates the success
> rate among the designs that survive. Filters are fine, and in fact the whole point, when you
> **compute normalize** and measure success per unit of oracle compute, because a filtered design
> still costs its share of the budget.

| Config key | Tag gate | Threshold |
| --- | --- | --- |
| `sequence_adherence_filter` | any: `design` | `sequence_adherence.n_violations` = 0, so the design must reproduce every conditioned residue exactly. Vacuous when nothing was sequence conditioned |
| `not_cysteine_adherence_filter` | any: `design` | `not_cysteine_adherence.n_violations` = 0. Rejects any CYS at a residue conditioned with `C_NCYS`; cysteines elsewhere are allowed. Passes when no no-cysteine conditioning was requested |
| `unindexed_insertion_mapping_filter` | all: `unindexed` | `unindexed_insertion_mapping.n_unmapped_tokens` = 0 and `.n_sequence_mismatches` = 0, so the whole unindexed motif got inserted at the conditioned identities |
| `ligand_clash_filter` | all: `ligand_clash_filter`<br>not: `no_ligand` | Zero inter residue polymer and ligand clashes, read off `clash_metrics` |
| `insertion_rmsd_filter` | all: `unindexed`, `insertion_rmsd_filter` | `unindexed_metrics.insertion_rmsd` ≤ 1.0. You need a metric producing that exact key, such as an aliased copy of `design_unindexed_motif_allatom_rmsd`, or this filter has nothing to threshold |
| `motif_rmsd_filter` | all: `motif_rmsd_filter` | `motif_rmsd.rmsd` ≤ 1.0. Likewise, it needs a metric under the `motif_rmsd` key |

### Pre-oracle transforms

A pre-oracle *filter* decides **whether** a design reaches the oracle. A pre-oracle *transform*
changes **what** gets folded. You attach one to an oracle through its `pre_oracle_transform` key,
and it can fan a single design out into several named outputs. Each output gets refolded separately
and saved into the design's own output directory under a filename carrying its `save_id`, as
`{example_id}_{save_id}_{oracle_name}_refolded.cif`, with `.zst` appended when compression is enabled (see
[output compression](outputs.md)). Its metrics carry that `save_id` between the
metric name and the output key.

So an oracle with the **config key** `oracle_off_target` carrying a two output transform reports
`oracle_off_target.her2.min_ipae_coi` and `oracle_off_target.egfr.min_ipae_coi`, not one
`oracle_off_target.min_ipae_coi`. As everywhere else, the prefix is the config key the oracle is
registered under in `af3.yaml`, not its `oracle_name`. Config keyed by the bare output name, such as
a `frac_under_thresholds` entry or an `aggregation` list, applies to every output of the fan out.

Each output gets addressed and tagged separately, so you get `refold.her2` and `is_refold:her2`
alongside `refold.oracle_off_target.her2` and `is_refold:oracle_off_target.her2`. A metric gated on
`is_refold:her2` is scored on that off-target's pass alone and reports under `.her2.`. Gate on the
oracle instead and the metric runs once per output, repeating itself under every `save_id`.

**The default refolding metrics above do not run against a transformed refold**, because a transform
can change the shape of what is being compared. Each carries `required_tags_all:
["is_refold:untransformed"]`, so it only sees the refold of the design as generated. Override that
gate in your own `custom_metrics` config to score transformed refolds too. A metric written for a
transformed refold usually requires that refold instead, as `is_refold:apo`, plus
`exists_refold:{address}` for every other refold it compares against.

| Class | What it folds | Outputs |
| --- | --- | --- |
| `ApoOnlyTransform` | The chain of interest alone, everything else dropped | `apo` |
| `OffTargetSubstitutionTransform` | The chain of interest, with its binding partner swapped for each off-target loaded from disk | one per entry in `off_target_paths` |

Off-target files get read as annotated CIFs. An annotation only survives into the folded structure
when **both** sides carry it, because concatenation keeps only what its inputs share. Anything that
knows its own default, meaning a registered lazy annotator or a `StandardAnnotation`, gets filled in
automatically on the side that lacks it. So if you need an off-target to carry a *non-default* value
for one, put it in the file.

`OffTargetSubstitutionTransform` flags every substituted atom `is_loaded_target` and every design
atom `False`, which separates the experimental half of the complex from the designed half. You could
use that to template the target through the oracle's
`template_query: "is_loaded_target & is_protein"`. That annotation is the way to template a
substituted off-target. The transform gives it no conditioning it did not already carry on disk, so
a query selecting coordinate-conditioning masks, as used by `oracle_ppi`, will generally
select nothing on its half of the complex.

A transform's refolds are cached by filename, as `{example_id}_{save_id}_{oracle_name}_*`, so
re-running into the same output directory reuses whatever a previous run left under that save_id.
If you repoint `off_target_paths` at a different structure, write to a fresh `output_dir` or delete
the stale refolds. Otherwise the previous structure's numbers get reported under the new one's name.

**Substituting a binding partner is only the first thing this got used for.** A transform can
rewrite the design however it likes before folding, and the fan out exists so one design can be
asked several questions in a single row of the metrics table. In enzyme design, a transform can swap
the bound species for another state along the reaction coordinate and fold that too.

**Check any substitution that cuts into bonded chemistry.** `OffTargetSubstitutionTransform` gets
away with concatenating two structures because it only ever joins protein chains that were never
bonded to each other. Anything that breaks or forms a covalent bond, or touches a residue
coordinating a metal, has to come out consistent in its bonds, atoms, charges, elements and
annotations. A substituted array that wires a bond to the wrong atom, or silently drops one, still
folds. Be careful here.

Both live in
[`pre_oracle_transform.py`](../../src/rfproteina/metrics/oracles/pre_oracle_transform.py), and
[`af3.yaml`](../../configs/metrics/refolding_oracles/af3.yaml) has wired examples. To add your own,
subclass `PreOracleTransform` and implement `forward`. It receives the generated AtomArray and
returns one `PreOracleOutput` per structure to fold.

### Ready-made but opt-in

These classes exist and are tested, but no default config includes them. Add a YAML entry to turn
one on, with no Python required. The [worked
example](custom-metrics.md#worked-example-an-existing-metric-your-own-config) shows how.

| Class | Lives in | What it's for |
| --- | --- | --- |
| `AtomPairDistanceMetric` | `rfproteina.metrics.atom_pair_distance_metric` | Configurable upper bound distance constraints on atom pairs, for catalytic geometry and motif contact checks |
| `TMolBlockPairMetric` | `rfproteina.metrics.physics.tmol_block_pair_metric` | General interface ddG metric between two atom selection regions. Presets for ligand ddG, H-bonds and electrostatics, plus PPI ddG and Coulomb, live in [`tmol_block_pair_metrics.yaml`](../../configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml). GPU only, and deliberately out of the default set |
| `QMInteractionEnergyIPA` | `rfproteina.metrics.ipa_interaction_energy_metric` | QM ligand and protein interaction energy through IPA. See [`ipa_metrics.yaml`](../../configs/metrics/inline_metrics/ipa_metrics.yaml), currently commented out of the defaults pending a digs fix |
| `MACEEnergyMetric` | `rfproteina.metrics.physics.mace_energy_metric` | MACE ML potential energy of a structure |
| `MACEInteractionEnergyMetric` | `rfproteina.metrics.physics.mace_interaction_energy_metric` | MACE-POLAR-1 interaction energy decomposition for protein and ligand complexes |
| `ChargeComplementarityMetric` | `rfproteina.metrics.physics.ddg_salt_metrics` | Salt bridge charge complementarity score across a binder and target interface |
| `InternalResidueBondMetric` | `rfproteina.metrics.bond_length_metric` | Per residue bond lengths against CCD ideal geometry |
| `NucleicSSSimilarityMetrics` | `rfproteina.metrics.nucleic_ss_metrics` | Nucleic acid secondary structure conditioning adherence, meaning base pairing |
| `ContractCAMetric` | `rfproteina.metrics.contract_ca_metric` | Trivial differentiable guidance metric that contracts CA toward the origin, mainly for exercising the search and guidance path |

## Standard annotations, and why there is no output JSON

Older workflows expected an output JSON next to the generated CIF, carrying the unindex mapping,
which input contig an inserted region came from, and which atoms were coordinate conditioned. **That
JSON no longer exists.** The same information now goes into **standard annotations written directly
into the output CIF**, defined in `rfproteina.utils.standard_annotations`. It round trips with the
structure itself instead of living in a side file that can drift out of sync.

| Standard annotation | CIF column | What it tells you |
| --- | --- | --- |
| `SA_SRC` | `annotation_src-component_1_atom` | Per atom source component label like `A3`, naming which input residue this atom came from; expandable segments use range labels like `3-10` |
| `SA_CRDSRC` | `mapped_to_coordinate_conditioned_region` plus `annotation_src-coordinate-conditioned-region_1_atom` | Per atom mask and source contig for the coordinate-conditioned region, where `"."` means not conditioned |
| `SA_CRDGRPSRC` | `annotation_src-coord-group_1_atom` | Integer `CoordinateGroup` id the atom maps to, where `-1` means not mapped. This is how motif RMSD metrics subdivide designs with several motifs |
| `SA_CWI` | `_annotation_coords-with-insertion_1_atom` loop | The ground truth coordinate to compare against for coordinate-conditioned atoms, which is what `design_motif_allatom_rmsd` and friends align and score against |
| `SA_UII` | `is_unindexed_insertion` | Per atom boolean, where `True` marks the atom as an unindexed insertion target. That is what `design_unindexed_motif_allatom_rmsd` restricts itself to |

These are exactly the fields the motif and unindexed RMSD metrics above already consume. That is the
whole point of turning on `design_motif_allatom_rmsd` and `design_unindexed_motif_allatom_rmsd`
instead of reading the CIF by hand. The pipeline does the bookkeeping for you.

To explore the annotation format directly, say for an analysis no built in metric covers, the
runnable **[annotated CIF tutorial notebook](../examples/annotated_cif_tutorial.ipynb)** is your best
reference. It builds up the same annotation system these outputs use, atom by atom. In code, parse
an output CIF with the `STANDARD_ANNOTATIONS_ONLY` preset from `rfproteina.utils.parse_presets` to
load all of the above with minimal fuss.

```{toctree}
:hidden:
:maxdepth: 1

custom-metrics
```
