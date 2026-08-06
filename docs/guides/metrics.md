# Metrics Reference

> **You shouldn't need to hand-parse output CIFs.** It's possible to reconstruct things like the
> unindexed mapping, which chain/residue an inserted region came from, or coordinate-conditioning adherence
> by reading raw CIF fields — but that's the fallback, not the workflow. The preferred path is:
> turn on the **metric(s)** that already compute what you want, and let them run automatically and
> efficiently in the main pipeline (they can also drive [reward-guided search](search.md), which is
> far more sample-efficient than reprocessing outputs and filtering after the fact).

- [How many metrics ship enabled](#how-many-metrics-ship-enabled)
- [Metric reference](#metric-reference)
  - [Design metrics](#design-metrics-inline-pre-refolding)
  - [Structure-prediction metrics](#structure-prediction-metrics)
  - [Refolding metrics](#refolding-metrics-post-oracle)
  - [Refolding oracles](#refolding-oracles)
  - [Pass criteria](#pass-criteria)
  - [Pre-oracle filters](#pre-oracle-filters)
  - [Ready-made but opt-in](#ready-made-but-opt-in)
- [Writing your own](#writing-your-own)
- [Worked example: an existing metric, your own config](#worked-example-an-existing-metric-your-own-config)
- [Standard annotations (not an output JSON)](#standard-annotations-not-an-output-json)
- [Running metrics offline on a folder of structures](#running-metrics-offline-on-a-folder-of-structures)

---

## How many metrics ship enabled

**40 metric / pass-criterion / filter entries are already registered** in
`configs/metrics/all_metrics.yaml` and run automatically — no code, no new config files. Most are
gated behind one or more **`metrics_tags`**, so turning one on is just passing a tag on the command
line:

```bash
python src/rfproteina/generate.py \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'
```

Breakdown of what's already wired in:

| Category | Count | Config |
| --- | --- | --- |
| Design metrics (run inline, pre-refolding) | 16 | `configs/metrics/inline_metrics/design_metrics.yaml` |
| Structure-prediction metrics | 1 | `configs/metrics/inline_metrics/structure_prediction_metrics.yaml` |
| Refolding metrics (design vs. refolded structure, or refold vs. refold) | 7 | `configs/metrics/refolding_metrics/all_refolding_metrics.yaml` |
| Refolding oracles (AF3 variants) | 4 | `configs/metrics/refolding_oracles/af3.yaml` |
| Pass criteria (pass/fail thresholds) | 9 | `configs/metrics/pass_criteria/all_pass_criteria.yaml` |
| Pre-oracle filters (skip the oracle on cheap-check failures) | 5 | `configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml` |

Beyond those, **10 more metric classes already exist in the codebase** but aren't wired into any
default config — they cover TMol interface ddG/H-bonds/electrostatics, an atom-pair distance
constraint metric (used for catalytic geometry), a QM ligand-interaction-energy metric, MACE
ML-potential energies, charge complementarity, and a couple of others. Turning one of these on
needs **no Python** — just a few lines of YAML, exactly like the [worked
example](#worked-example-an-existing-metric-your-own-config) below. See [Ready-made but
opt-in](#ready-made-but-opt-in).

Only write a brand-new `Metric` subclass when none of the ~50 above cover what you need — see
[Writing your own](#writing-your-own).

---

## Metric reference

Tag semantics (see [custom_metrics.md](custom_metrics.md#how-kwargs_to_compute_args-works) for the
full rules): `required_tags_any` — runs if **any** listed tag is passed; `required_tags_all` — runs
only if **all** listed tags are passed; `prohibited_tags` — skipped if a listed tag is passed, even
if the required tags also match. No tags at all = always runs. Design tags and refolding tags are
separate namespaces; pass both if you want both families to run:
`metrics_tags='[design, unconditional_refolding]'`.

### Design metrics (inline, pre-refolding)

Computed directly on the generated structure, before any oracle refolds it.

| Config key | Tag gate | What it measures | Output columns (prefixed `<key>.`) |
| --- | --- | --- | --- |
| `bond_length_metrics` | any: `design`; not: `small_molecule_only`, `structure_prediction` | Fraction of bonds with an unreasonable length (element-dependent thresholds) | `frac_unreasonable` (+ per-category breakdown in verbose mode) |
| `clash_metrics` | any: `design`, `structure_prediction`; not: `small_molecule_only` | Fraction of atoms in a steric clash at a configurable threshold (default 1.5 Å), by clash-type | `frac_atoms_clashing_1p5A` + 5 category breakdowns (intra/inter-residue × polymer/ligand) |
| `hotspot_obedience` | any: `hotspot_conditioning` | Fraction of hotspot atoms with a binder atom within 4.5 Å | `hotspot_obedience` |
| `rasa_adherence` | any: `rasa_conditioning` | Adherence of generated burial to RASA-bin conditioning | `buried_conditioning_adherence`, `intermediate_conditioning_adherence`, `exposed_conditioning_adherence` |
| `interchain_interactions` | any: `interchain_interactions`; not: `small_molecule_only`, `structure_prediction` | Adherence + atom/pair counts for H-bond, π-stacking, π-cation, salt-bridge, hydrophobic, and metal-coordination conditioning, interchain only | `*_conditioning_adherence`, `*_atom_count_coi` / `*_residue_count_coi`, `*_atom_pair_count_coi` / `*_residue_pair_count_coi` per interaction type |
| `sequence_adherence` | any: `design` | Fraction of sequence-conditioned residues whose generated identity matches what was asked for; NaN when nothing was sequence-conditioned, and in the offline re-scoring path (which has no input structure to read the conditioning off) | `sequence_adherence`, `n_violations` |
| `sse_distribution` | any: `design`; not: `small_molecule_only`, `structure_prediction` | Helix/sheet/loop fraction and number of contiguous SSE elements (P-SEA) | `helix_fraction`, `sheet_fraction`, `loop_fraction`, `num_ss_elements` |
| `sse_accuracy` | any: `design`; not: `small_molecule_only`, `structure_prediction` | Accuracy of generated secondary structure vs. secondary-structure conditioning | `overall_accuracy`, `helix_recall`, `sheet_recall`, `loop_recall`, `n_conditioned_atoms` |
| `posebusters_metrics` | any: `small_molecule_only` | Ligand geometry/validity checks (PoseBusters) | one pass/fail column per PoseBusters check |
| `chiral_accuracy` | any: `small_molecule_only` | Fraction of stereogenic centers with correct handedness, split tetrahedral vs. non-tetrahedral | `tetrahedral_percent_correct_chirality`, `non_tetrahedral_percent_correct_chirality` |
| `design_motif_allatom_rmsd` | any: `3d_coordinate_conditioning`, `unindexed`, `internal_coordinate_conditioning` | All-atom RMSD of the coordinate-conditioned motif region, self-consistency within the design itself | `rmsd` (median over replicates; per-`CoordinateGroup` when multiple motifs) |
| `design_unindexed_motif_allatom_rmsd` | all: `unindexed` | All-atom RMSD restricted to **unindexed** insertion atoms only — this is RFD3's "insertion RMSD" | `rmsd` |
| `unindexed_insertion_mapping` | all: `unindexed` | Whether the insertion head's unindexed → generated token mapping was applied in full: every unindexed token inserted, onto a residue of the conditioned identity where the unindexed region was sequence-conditioned | `n_unmapped_tokens`, `n_sequence_mismatches` |
| `contact_molecular_surface` | any: `cms` | Distance-weighted interfacial contact surface area between binder and target (Lawrence & Colman 1993) | see `rfproteina.metrics.physics.contact_ms.metric` |
| `sequence_similarity_to_input` | any: `partial_diffusion` | Sequence identity vs. the seed structure, for partial diffusion | `sequence_identity_to_input` |
| `cath_metric` | any: `cath` | CATH superfamily classification of the chain of interest via Foldseek | `C`, `A`, `T`, `H`, `type`, `evalue`, `tmscore` |

### Structure-prediction metrics

| Config key | Tag gate | What it measures | Output columns |
| --- | --- | --- | --- |
| `structure_prediction_lddt` | any: `structure_prediction` | All-atom lDDT vs. ground truth (for structure-prediction-style validation runs, not design) | `lddt` |

### Refolding metrics (post-oracle)

Compare the **design** (ground truth) against the **oracle-refolded structure** (predicted). See
[pre-oracle transforms](#pre-oracle-transforms) for what the `is_refold:` / `exists_refold:` gates
below are for — they only matter once an oracle carries a pre-oracle transform.

| Config key | Tag gate | What it measures | Output columns |
| --- | --- | --- | --- |
| `refolded_backbone_rmsd` | any: `unconditional_refolding`, `ppi_refolding`, `dna_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding`; all: `is_refold:untransformed` | Backbone self-consistency RMSD (scRMSD) between design and refold | `rmsd` |
| `refolded_target_aligned_binder_rmsd` | any: `ppi_refolding`, `dna_refolding`; all: `is_refold:untransformed` | Binder RMSD after aligning on the fixed target | `rmsd` |
| `refolded_motif_allatom_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`; all: `is_refold:untransformed` | All-atom motif RMSD between design and refold (includes unindexed regions) | `rmsd` |
| `refolded_motif_ca_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`; all: `is_refold:untransformed` | CA-only motif RMSD between design and refold (includes unindexed regions) | `rmsd` |
| `refolded_backbone_aligned_ligand_rmsd` | any: `sm_binder_refolding`; all: `is_refold:untransformed` | Ligand RMSD after backbone alignment | `rmsd` |
| `max_ligand_centroid_distance` | any: `disco_refolding`; all: `is_refold:untransformed` | Max ligand-centroid displacement after backbone alignment | `ligand_centroid_distance` |
| `refolded_apo_holo_rmsd` | all: `is_refold:apo` (i.e. `apo_refolding`, which is what runs that oracle), `exists_refold:untransformed` | Backbone RMSD of the designed chain between its **apo** refold and its refold in the whole design — does a binder hold its fold once the target is gone? **Requires a paired untransformed oracle** — see the warning below | `apo.rmsd` |

Every refolding metric is handed the design as `ground_truth_atom_array` and the refold being
scored as `predicted_atom_array` — plus **every** refold of that design, addressable two ways:

| Address | Names the refold by | Example |
| --- | --- | --- |
| `refold.{save_id}` / `refold.untransformed` | what it **is** — one output of a [pre-oracle transform](#pre-oracle-transforms), or the design folded as-is (only when exactly one refold is of that kind) | `refold.apo`, `refold.her2`, `refold.untransformed` |
| `refold.{oracle_config_key}[.{save_id}]` | which **oracle** produced it | `refold.oracle_ppi`, `refold.oracle_apo.apo` |

Each address is injected as a tag twice over, alongside the `oracle:{oracle_name}` tag —
`exists_refold:{address}` for every refold this design has, and `is_refold:{address}` for the one
being scored right now. See the note below on gating.

Naming two of those in `input_kwarg_names` turns any `StructureComparisonMetric` into a comparison
of two *refolds* instead of design vs. refold; that is all `refolded_apo_holo_rmsd` is:

```yaml
input_kwarg_names:
  ground_truth_atom_array: refold.apo             # chain of interest, folded alone
  predicted_atom_array: refold.untransformed      # the same chain, folded in the whole design
```

A by-kind name is only defined when it is unambiguous — if two untransformed oracles run at once,
or if the name would shadow an oracle's own config key, `refold.untransformed` is not offered at
all, nor are its `is_refold:` / `exists_refold:` tags; name the oracle (`refold.oracle_ppi`) to pick
one. So running two untransformed oracles at once switches every metric gated on
`is_refold:untransformed` off (with a warning) rather than scoring each of them twice into one
column and reporting whichever pass happened to fire last.

> [!IMPORTANT]
> **If using pre-oracle transforms and/or multiple OracleMetric instances, gate every metric on
> the relevant refold-specific tags.** `is_refold:{address}` says *which* refold this scoring pass is scoring — one
> pass runs per refold, so without it a metric written for one kind of refold also runs against the
> others, repeating its value once per refold. `exists_refold:{address}` says the design **has**
> that refold — required wherever `input_kwarg_names` reads another refold.
> `refolded_apo_holo_rmsd` uses one of each:
>
> ```yaml
> input_kwarg_names:
>   ground_truth_atom_array: refold.apo             # scored on the pass for this refold...
>   predicted_atom_array: refold.untransformed      # ...against this one, which must exist
> required_tags_all: ["is_refold:apo", "exists_refold:untransformed"]
> ```

### Refolding oracles

An oracle actually runs the (heavy, AF3-based) structure prediction that the refolding metrics
above then compare against. See [Evaluating Results](inference.md#evaluating-results) for the
apptainer/AF3 setup requirement.

| Config key | Tag gate | Notes | Key outputs (prefix `<key>.`) |
| --- | --- | --- | --- |
| `oracle` | any: `af3_refolding`, `unconditional_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding` | Untemplated AF3 refold | `overall_plddt`, `overall_ptm`, `overall_iptm`, `plddt_coi`, `ptm_coi`, `pae_coi`, `ipae_coi`, `min_ipae_coi`, `iptm_coi`, `frac_under_1`/`1.5`/`2` (on `min_ipae_coi`) |
| `oracle_ppi` | any: `ppi_refolding` | Target chain templated (fixed) for protein-binder design | same key set, prefixed `oracle_ppi.` |
| `oracle_dna` | any: `dna_refolding` | DNA templated (fixed) for DNA-binder design | same key set, prefixed `oracle_dna.` |

#### Refolding with OpenFold3 instead

The same oracles run on OpenFold3's Apache-licensed parameters. Converted OF3 weights are an
ordinary AF3 model directory, so there is no separate runtime — the worker reads the provenance
file the converter leaves beside them and enables the OF3 architecture branches itself.

1. Convert the weights once: `python scripts/setup_of3_weights.py --image "$AF3_IMAGE" --output_dir <dir>`
2. Point `AF3_MODEL_DIR` at `<dir>`.
3. Swap the config group: `metrics/refolding_oracles=of3`.

Step 3 only changes the artifact namespace (`of3`, `of3_ppi`, `of3_dna`) so results do not collide
with AF3 ones; the parameter family follows `AF3_MODEL_DIR` alone. One job uses one family —
run AF3 and OF3 as separate jobs and compare afterwards.

### Pass criteria

Threshold one or more already-computed metric values into a binary pass/fail flag
(`{config_key}.pass`), evaluated after design + oracle + refolding metrics. All are built from the
generic `PassCriterionMetric` — see [Adding your own pass criterion](custom_metrics.md#pass-criteria-config).
Separate entries are ANDed; to threshold the best/worst/mean of several keys instead (e.g. either of
two interfaces being good enough), use a `min_of` / `max_of` / `mean_of` list — see
[Thresholding a reduction over several keys](custom_metrics.md#thresholding-a-reduction-over-several-keys).

| Config key | Tag gate | Thresholds |
| --- | --- | --- |
| `unconditional` | any: `unconditional_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 1.5 |
| `ppi` | any: `ppi_refolding` | `oracle_ppi.ptm_coi` ≥ 80, `oracle_ppi.min_ipae_coi` ≤ 1.5, `refolded_target_aligned_binder_rmsd.rmsd` ≤ 2.5 |
| `dna` | any: `dna_refolding` | `oracle_dna.min_ipae_coi` ≤ 1.5, `refolded_target_aligned_binder_rmsd.rmsd` ≤ 8.0 |
| `motif_scaffold` | any: `motif_scaffold_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 2.0, `refolded_motif_ca_rmsd.rmsd` ≤ 1.0 |
| `internal_coords` | any: `internal_coordinate_motif_scaffold_refolding` | Same thresholds as `motif_scaffold` |
| `sm_binder` | any: `sm_binder_refolding` | `oracle.min_ipae_coi` ≤ 1.5, `oracle.iptm_coi` ≥ 80, `refolded_backbone_rmsd.rmsd` ≤ 1.5, `refolded_backbone_aligned_ligand_rmsd.rmsd` ≤ 5.0 |
| `ame` | any: `ame_refolding`; not: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5, zero inter-residue polymer-ligand clashes |
| `ame_no_ligand` | any: `ame_refolding`; all: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5 |
| `disco` | any: `disco_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `max_ligand_centroid_distance.ligand_centroid_distance` ≤ 2.0 |

### Pre-oracle filters

Cheap checks that decide whether a design is even **forwarded to the (expensive) oracle**.
These are tag-gated like other Metrics in the codebase. A subset of the built-in filters run by default: see 
[`all_pre_oracle_filters.yaml`](../../configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml).
A filter reading a metric key that is absent (e.g. because the metric's own tag gate kept it from
running) counts as a failure, so keep a filter's tag gate at least as strict as its metric's.

> **Benchmarking:** turn every pre-oracle filter off (`~metrics/pre_oracle_filters`) when you are
> benchmarking **per example** — a filter that discards designs before the oracle inflates the
> success rate among the designs that survive. They are fine, and are the point, when you are
> **compute-normalizing** (success per unit of oracle compute), since a filtered design still
> costs its share of the budget.

| Config key | Tag gate | Threshold |
| --- | --- | --- |
| `sequence_adherence_filter` | any: `design` | `sequence_adherence.n_violations` = 0, i.e. the design must reproduce every conditioned residue exactly. Vacuous when nothing was sequence-conditioned |
| `unindexed_insertion_mapping_filter` | all: `unindexed` | `unindexed_insertion_mapping.n_unmapped_tokens` = 0 and `.n_sequence_mismatches` = 0, i.e. the whole unindexed motif was inserted, at the conditioned identities |
| `ligand_clash_filter` | all: `ligand_clash_filter`; not: `no_ligand` | Zero inter-residue polymer-ligand clashes (from `clash_metrics`) |
| `insertion_rmsd_filter` | all: `unindexed`, `insertion_rmsd_filter` | `unindexed_metrics.insertion_rmsd` ≤ 1.0 — you'll need a metric producing that exact key (e.g. an aliased copy of `design_unindexed_motif_allatom_rmsd`) for this filter to have something to threshold |
| `motif_rmsd_filter` | all: `motif_rmsd_filter` | `motif_rmsd.rmsd` ≤ 1.0 — likewise, needs a metric under the `motif_rmsd` key |

### Pre-oracle transforms

Where a pre-oracle *filter* decides **whether** a design reaches the oracle, a pre-oracle
*transform* changes **what** gets folded. A transform is attached to an oracle via its
`pre_oracle_transform` key and may fan one design out into several named outputs; each is
refolded separately and saved into the design's own output directory under a filename carrying
its `save_id` (`{example_id}_{save_id}_{oracle_name}_refolded.cif`), and its metrics carry that
`save_id` between the metric name and the output key. So an oracle whose **config key** is
`oracle_off_target` and which carries a two-output transform reports
`oracle_off_target.her2.min_ipae_coi` and `oracle_off_target.egfr.min_ipae_coi` rather than one
`oracle_off_target.min_ipae_coi`. (As everywhere else, the prefix is the config key the oracle is
registered under in `af3.yaml`, not its `oracle_name`.) Config keyed by the bare output name — a
`frac_under_thresholds` entry, an `aggregation` list — applies to every output of the fan-out.

Each output is also addressed and tagged separately — `refold.her2` / `is_refold:her2` alongside
`refold.oracle_off_target.her2` / `is_refold:oracle_off_target.her2` — so a metric gated on
`is_refold:her2` is scored on that off-target's pass alone and reports under `.her2.`. Gate on the
oracle instead and the metric runs once per output, repeating itself under every `save_id`.

**The default refolding metrics above do not run against a transformed refold**, since a transform can
change the shape of what is being compared. Each carries `required_tags_all:
["is_refold:untransformed"]`, so it runs only the refold of the design as generated —
override that gate in your own `custom_metrics` config to score transformed refolds too.
A metric written for a transformed refold typically requires that refold instead (`is_refold:apo`), plus
`exists_refold:{address}` for every other refold it compares against.

| Class | What it folds | Outputs |
| --- | --- | --- |
| `ApoOnlyTransform` | The chain of interest alone, everything else dropped | `apo` |
| `OffTargetSubstitutionTransform` | The chain of interest, with its binding partner swapped for each off-target loaded from disk | one per entry in `off_target_paths` |

Off-target files are read as annotated CIFs, and an annotation is only guaranteed to survive into the folded
structure if **both** sides have it — concatenation keeps only what its inputs share. Anything
that knows its own default (a registered lazy annotator or a `StandardAnnotation`) is filled in
automatically on the side that lacks it, so if you need an off-target to carry a *non-default*
value for one, put it in the file.

`OffTargetSubstitutionTransform` flags every substituted atom `is_loaded_target` (and the design's
atoms `False`), separating the experimental half of the complex from the designed half. It could for
example be used to template the target via the oracle's
`template_query: "is_loaded_target & is_protein"`. That annotation is the way to template a
substituted off-target: the transform gives it no conditioning it did not already carry on disk, so
a mask-based query (`oracle_ppi`'s `template_query: "mask_coordinate_1_atom"`) will generally select
nothing on its half of the complex.

A transform's refolds are cached by filename (`{example_id}_{save_id}_{oracle_name}_*`), so a re-run
into the same output directory reuses whatever a previous run left under that save_id. Repointing
`off_target_paths` at a different structure means writing to a fresh `output_dir` (or deleting the
stale refolds), or the previous structure's numbers will be reported under the new one's name.

**Substituting a binding partner is only the first thing this was used for.** A transform may
rewrite the design however it likes before folding, and the fan-out is there so one design can be
asked several questions in a single row of the metrics table. In enzyme design, for instance, a
transform can swap the bound species for another state along the reaction coordinate and fold that as well.

**Substitutions that cut into bonded chemistry need checking.** `OffTargetSubstitutionTransform`
can get away with concatenating two structures because it only ever joins protein chains that were
never bonded to each other. Anything that breaks or forms a covalent bond, or
touches a residue coordinating a metal, has to come out consistent in its bonds, leaving atoms,
charges, elements and annotations. A substituted array that wires a bond to the wrong atom, or
silently drops one, will still fold, so it's important to be careful with this.

Both live in
[`pre_oracle_transform.py`](../../src/rfproteina/metrics/oracles/pre_oracle_transform.py); see
[`af3.yaml`](../../configs/metrics/refolding_oracles/af3.yaml) for wired examples. To add your
own, subclass `PreOracleTransform` and implement `forward` — it receives the generated AtomArray
and returns one `PreOracleOutput` per structure to fold.

### Ready-made but opt-in

These classes exist and are tested, but aren't included in any default config — add a YAML entry
(see the [worked example](#worked-example-an-existing-metric-your-own-config)) to turn one on, no
Python required.

| Class | Lives in | What it's for |
| --- | --- | --- |
| `AtomPairDistanceMetric` | `rfproteina.metrics.atom_pair_distance_metric` | Configurable atom-pair upper-bound distance constraints — catalytic geometry, motif-contact checks |
| `TMolBlockPairMetric` | `rfproteina.metrics.physics.tmol_block_pair_metric` | General interface-ddG metric between two atom-selection regions; presets for ligand ddG/H-bonds/electrostatics and PPI ddG/Coulomb live in [`tmol_block_pair_metrics.yaml`](../../configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml) (GPU, intentionally not in the default set) |
| `QMInteractionEnergyIPA` | `rfproteina.metrics.ipa_interaction_energy_metric` | QM ligand-protein interaction energy via IPA; see [`ipa_metrics.yaml`](../../configs/metrics/inline_metrics/ipa_metrics.yaml) (commented out of defaults pending a digs fix) |
| `MACEEnergyMetric` | `rfproteina.metrics.physics.mace_energy_metric` | MACE ML-potential energy of a structure |
| `MACEInteractionEnergyMetric` | `rfproteina.metrics.physics.mace_interaction_energy_metric` | MACE-POLAR-1 interaction energy decomposition for protein-ligand complexes |
| `ChargeComplementarityMetric` | `rfproteina.metrics.physics.ddg_salt_metrics` | Salt-bridge charge-complementarity score across a binder-target interface |
| `InternalResidueBondMetric` | `rfproteina.metrics.bond_length_metric` | Per-residue bond lengths vs. CCD ideal geometry |
| `NucleicSSSimilarityMetrics` | `rfproteina.metrics.nucleic_ss_metrics` | Nucleic-acid secondary-structure (base-pairing) conditioning adherence |
| `NoCysteineAdherenceMetric` | `rfproteina.metrics.no_cysteine_metric` | Fraction of not-cysteine-conditioned residues that are actually not `CYS` |
| `ContractCAMetric` | `rfproteina.metrics.contract_ca_metric` | Trivial differentiable guidance metric (contract CA toward the origin); mainly for exercising the search/guidance path |

---

## Writing your own

If none of the ~47 metrics above cover what you need, **[custom_metrics.md](custom_metrics.md)**
walks through writing a brand-new `Metric` subclass from scratch: the three metric categories
(design / refolding / pass-criterion), the `kwargs_to_compute_args` routing, registering it in a
git-ignored config so nothing you write is ever committed by accident, and — if it's generally
useful — contributing it back.

---

## Worked example: an existing metric, your own config

Often you don't need new Python at all — an existing metric just needs **your** parameters. Say
you're doing enzyme design and want to check catalytic-triad geometry: `AtomPairDistanceMetric`
(see [Ready-made but opt-in](#ready-made-but-opt-in)) already does exactly this — evaluate one or
more atom-pair upper-bound distance constraints — you just need to tell it *which* atoms and
*what* distances.

**Step 1 — write the config.** Custom metrics live in the git-ignored
`configs/metrics/custom_metrics/` directory (scaffolded by `setup.sh` on first run — see
[custom_metrics.md](custom_metrics.md)):

```yaml
# configs/metrics/custom_metrics/all_custom_metrics.yaml
# @package metrics

inline_metrics:
  catalytic_geometry:
    _target_: rfproteina.metrics.atom_pair_distance_metric.AtomPairDistanceMetric
    # No required_tags_* set -> always enabled, on every generated design.
    constraints:
      - name: triad_his_asp
        max_distance_angstrom: 3.2
        atom_a: "res_id == 31 and res_name == 'HIS' and atom_name == 'NE2'"
        atom_b: "res_id == 14 and res_name == 'ASP' and atom_name == 'OD1'"
      - name: oxyanion_cys
        max_distance_angstrom: 3.0
        atom_a: "chain_id == 'B' and res_name == 'lpr' and atom_name == 'O29'"
        atom_b: "chain_id == 'A' and res_name == 'CYS' and atom_name == 'N'"
```

Each `atom_a`/`atom_b` is an AtomWorks query string (see
[`resolve_mask`](../../src/rfproteina/utils/mask_spec.py), the one shared string-to-mask
evaluator used everywhere in this codebase) that must resolve to exactly one atom; give a
list of query strings instead of a single one to check the minimum distance over several
alternative atoms. Alternatives that resolve to zero or several atoms are skipped
individually, so one inapplicable option doesn't discard the rest of the list. For the full
query and path-selection grammar, see AtomWorks' [Atom Selection
Syntax](../../lib/atomworks/docs/io/utils/selection_syntax.rst) guide.
Omitting `required_tags_any`/`required_tags_all` means the
metric has no gate and runs on every design — appropriate here since a design either has a catalytic
triad or it doesn't (the metric reports `inf`/violated if the named atoms are missing, rather than
erroring).

**Step 2 — run.** Hydra dict-merges `configs/metrics/custom_metrics/all_custom_metrics.yaml` into
the built-in metrics config automatically (it's already a default in
[`all_metrics.yaml`](../../configs/metrics/all_metrics.yaml), listed last so it merges on top) — new
keys are **added**, and a key that matches an existing built-in name gets **deep-merged over it**
(e.g. redeclaring `clash_metrics:` with just a new `threshold:` overrides that one field and inherits
everything else). So the normal run command picks it up with no extra flags:

```bash
python src/rfproteina/generate.py \
    inputs=enzyme_design.cif num_replicates=50
```

Results land in `per_example_metrics.csv` as `catalytic_geometry.triad_his_asp_distance`,
`catalytic_geometry.triad_his_asp_satisfied`, `catalytic_geometry.n_satisfied`,
`catalytic_geometry.all_satisfied`, etc. — see the
[`AtomPairDistanceMetric` docstring](../../src/rfproteina/metrics/atom_pair_distance_metric.py) for the
full output set (per-constraint distance/violation/satisfied/soft-score, plus aggregate
`soft_min_score`, `total_violation`, `worst_violation`).

---

## Standard annotations (not an output JSON)

Older workflows expected an explicit output JSON alongside the generated CIF — carrying things like
the unindex mapping, which input contig an inserted region came from, or which atoms were
coordinate-conditioned. **That JSON no longer exists.** The same information is now written as
**standard annotations directly into the output CIF** (`rfproteina.utils.standard_annotations`), so
it round-trips with the structure itself instead of living in a side-channel file that can drift out
of sync:

| Standard annotation | CIF column | What it tells you |
| --- | --- | --- |
| `SA_SRC` | `src-component` | Per-atom source-component contig label (e.g. `A3-10`) — which input region/component this atom came from |
| `SA_CRDSRC` | `src-coordinate-conditioned-region` | Per-atom source contig for the coordinate-conditioned region specifically (`"."` = not conditioned) |
| `SA_CRDGRPSRC` | `src-coord-group` | Integer `CoordinateGroup` id the atom maps to (`-1` = not mapped) — how motif RMSD metrics subdivide multi-motif designs |
| `SA_CWI` | `coords-with-insertion` | The ground-truth coordinate to compare against for coordinate-conditioned atoms (what `design_motif_allatom_rmsd` etc. align/score against) |
| `SA_UII` | `is-unindexed-insertion` (alias `is_unindexed_insertion`) | Per-atom boolean — `True` marks the atom as an unindexed-insertion target (what `design_unindexed_motif_allatom_rmsd` restricts to) |

These are exactly the fields the motif/unindexed RMSD metrics above already consume — the whole
point of turning on `design_motif_allatom_rmsd` / `design_unindexed_motif_allatom_rmsd` instead of
reading the CIF by hand is that the pipeline does this bookkeeping for you.

If you do want to explore the annotation format directly (e.g. for a bespoke analysis a built-in
metric doesn't cover), the runnable **[annotated CIF tutorial
notebook](../examples/annotated_cif_tutorial.ipynb)** is the best reference — it builds up the same
annotation system these outputs use, atom by atom.
Programmatically, parse an output CIF with the `STANDARD_ANNOTATIONS_ONLY` preset
(`rfproteina.utils.parse_presets`) to load all of the above with minimal intervention.

---

## Running metrics offline on a folder of structures

Metrics don't require re-running generation — score any folder of already-generated structures with
`evaluate.py` (same Hydra config, so `metrics_tags` and any custom-metric config apply exactly as
above):

```bash
python src/rfproteina/evaluate.py \
    inputs=/path/to/structures \
    metrics_tags='[unconditional_refolding]' \
    out_dir=/path/to/eval_output
```

`inputs` is scanned recursively for CIF-like files. `out_dir` is mandatory (`evaluate.py` has no
run directory to derive one from) and receives the usual `evaluation_metrics/` subdirectory.

Pass `task_csv=<csv with designed_structure_path,
task columns>` if different structures need different `metrics_tags`/pass-criteria treatment;
otherwise every structure is scored the same way. This is the same script the [Developer
Guide](development.md#evaluation-and-metrics) points to for custom-metric development — write a
metric once and use it both inline during generation and offline over existing outputs.

For large batches split across SLURM array jobs (`launch_infer.sh --njobs N`), each job writes its
own `per_example_metrics_job{N}.csv`; concatenate them afterward with:

```bash
python -m rfproteina.inference_utils.aggregate_metrics ./inference/my_run/metrics/
```

This also runs automatically as a dependent job when using `launch_infer.sh`. See [Evaluating
Results](inference.md#evaluating-results) for FoldSeek diversity clustering and other
`aggregate_metrics` options.
