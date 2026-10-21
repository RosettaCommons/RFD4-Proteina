# Metrics reference

This page lists very metric, oracle, pass criterion, and pre-oracle filter that ships with RFD4-Proteina, what each
one measures, and the tag you pass throught the CLI to have it run. 

See the [metrics guide](../how-to/evaluate-with-metrics.md) for where these pieces fit into the RFD4-proteina workflow and guidance on how to use them, and the [refolding guide](../how-to/refolding.md) specifically for the refolding family: oracle setup, task tags, and running a design through more than one refold.

**Contents**

- [Every metric, criteria, and filter that ships](#every-metric-criteria-and-filter-that-ships)
  - [Design metrics, inline and before refolding](#design-metrics-inline-and-before-refolding)
    - [Extra information for antibodies](#extra-information-for-antibodies)
  - [Structure prediction metrics](#structure-prediction-metrics)
  - [Refolding metrics, after the oracle](#refolding-metrics-after-the-oracle)
  - [Refolding oracles](#refolding-oracles)
  - [Pass criteria](#pass-criteria)
  - [Pre-oracle filters](#pre-oracle-filters)
  - [Pre-oracle transforms](#pre-oracle-transforms)
  - [Ready-made but opt-in](#ready-made-but-opt-in)

## Every metric, criteria, and filter that ships

Each row's **Tag gate** says which `metrics_tags` turn that metric on. You specify metric tags via the CLI when you run RFD4-proteina.

There are three categories of tags: 
- An `any` entry runs the
  metric/filter/criteria if you pass **at least one** of the listed tags. 
- An `all` entry runs the metric/filter/criteria only if you pass
  **every one** of the tags listed after "all". 
- A `not` entry skips the metric/filter/criteria if you pass that tag, even when its `any` or `all` tags match. A row with no gate listed always runs.

```{note}
Design metrics, structure prediction metrics, pass criteria, and pre-oracle filters are gated purely
on the tags you pass via `metrics_tags`. [Refolding metrics](#refolding-metrics-after-the-oracle) are
the one exception: their gates are checked against your `metrics_tags` plus `is_refold:`,
`exists_refold:`, and `oracle:` tags that the framework injects automatically per scoring pass, you
never type those on the CLI yourself.
```

For example, the `bond_length_metrics` will run for any RFD4-proteina run whose `metrics_tags` includes `design` but will not run for a run that specifies `small_molecule_only` **or** `structure_prediction`. And metrics like `insertion_rmsd_filter` will only run if `metrics_tags` has at least both `unindexed` and `insertion_rmsd_filter`.

See the [custom metrics
guide](../how-to/advanced/custom-metrics.md#gating-with-tags) for how to write these gates yourself.

Design tags and refolding tags live in separate namespaces, so pass both when you want both
families: `metrics_tags='[design, unconditional_refolding]'`.

### Design metrics, inline and before refolding

These run directly on the generated structure, before any oracle refolds it.

| Config key | Tag gate | What it measures | Output columns (prefixed `<key>.`) |
| --- | --- | --- | --- |
| `amino_acid_composition` | Always | Generated UNK count and ALA, VAL and GLY fractions among standard amino acids; sequence-conditioned UNK is exempt from the generated UNK count | `n_generated_unknown_residues`, `n_standard_residues`, `ala_fraction`, `val_fraction`, `gly_fraction` |
| `bond_length_metrics` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Fraction of bonds with an unreasonable length, on element dependent thresholds | `frac_unreasonable`, plus a per category breakdown in verbose mode |
| `clash_metrics` | any: `design`, `structure_prediction`<br>not: `small_molecule_only` | Fraction of atoms in a steric clash at a configurable threshold, 1.5 Å by default, split by clash type | `frac_atoms_clashing_1p5A` plus 5 category breakdowns, intra and inter residue by polymer and ligand |
| `hotspot_obedience` | any: `hotspot_conditioning` | Fraction of hotspot atoms with a binder atom within 4.5 Å | `hotspot_obedience` |
| `rasa_adherence` | any: `rasa_conditioning` | How well the generated burial matches your RASA bin conditioning | `buried_conditioning_adherence`, `intermediate_conditioning_adherence`, `exposed_conditioning_adherence` |
| `centroid_conditioning_adherence` | any: `centroid_conditioning` | Mean distance between each conditioned chain centroid and the requested position | `centroid_conditioning_mae` |
| `interchain_interactions` | any: `interchain_interactions`<br>not: `small_molecule_only`, `structure_prediction` | Adherence plus atom and pair counts for H-bond, π-stacking, π-cation, salt bridge, hydrophobic and metal coordination conditioning, interchain only | `*_conditioning_adherence`, `*_atom_count_coi` / `*_residue_count_coi`, `*_atom_pair_count_coi` / `*_residue_pair_count_coi` per interaction type |
| `sequence_adherence` | any: `design` | Fraction of sequence conditioned residues whose generated identity matches what you asked for. NaN when nothing was sequence conditioned, and in the offline re-scoring path, which has no input structure to read the conditioning off | `sequence_adherence`, `n_violations` |
| `sequence_probability_adherence` | any: `design` | Adherence to explicitly specified probability-zero exclusions and probability-one identities. Soft probability errors are pooled across designs during aggregation. No default pre-oracle filter is applied | `zero_adherence`, `one_adherence`, `n_zero_conditions`, `n_zero_violations`, `n_one_conditions`, `n_one_violations` |
| `sse_distribution` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Helix, sheet and loop fraction, plus the number of contiguous SSE elements, from P-SEA | `helix_fraction`, `sheet_fraction`, `loop_fraction`, `num_ss_elements` |
| `sse_accuracy` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Accuracy of generated secondary structure against your secondary structure conditioning | `overall_accuracy`, `helix_recall`, `sheet_recall`, `loop_recall`, `n_conditioned_atoms` |
| `posebusters_metrics` | any: `small_molecule_only` | Ligand geometry and validity checks from PoseBusters | one pass or fail column per PoseBusters check |
| `chiral_accuracy` | any: `small_molecule_only` | Fraction of reference stereoisomers recovered, split by center geometry and E/Z double bonds | `tetrahedral_percent_correct_chirality`, `non_tetrahedral_percent_correct_chirality`, `double_bond_percent_correct_stereochemistry` |
| `design_motif_allatom_rmsd` | any: `3d_coordinate_conditioning`, `unindexed`, `internal_coordinate_conditioning` | All atom RMSD of the coordinate conditioned motif region, measuring self consistency within the design itself | `rmsd`, the median over replicates, and per `CoordinateGroup` when there are several motifs |
| `design_unindexed_motif_allatom_rmsd` | all: `unindexed` | All atom RMSD restricted to **unindexed** insertion atoms only | `rmsd` |
| `unindexed_insertion_mapping` | all: `unindexed` | Whether the insertion head applied its unindexed to generated token mapping in full, meaning every unindexed token got inserted, onto a residue of the conditioned identity wherever the unindexed region was sequence conditioned | `n_unmapped_tokens`, `n_sequence_mismatches` |
| `contact_molecular_surface` | any: `cms` | Distance weighted interfacial contact surface area between binder and target, following [Lawrence and Colman 1993](https://www.sciencedirect.com/science/article/pii/S0022283683716487?via%3Dihub) | see `rfproteina.metrics.physics.contact_ms.metric` |
| `sequence_similarity_to_input` | any: `partial_diffusion` | Sequence identity against the seed structure, for partial diffusion | `sequence_identity_to_input` |
| `ca_rmsd_to_input` | any: `partial_diffusion` | Protein CA RMSD to the input after CA alignment, measuring how much of the seed backbone was retained; omitted if CA counts differ | `rmsd` |
| `cath_metric` | any: `cath` | CATH superfamily classification of the chain of interest, through Foldseek | `C`, `A`, `T`, `H`, `type`, `evalue`, `tmscore` |
| `sequence_developability` | any: `developability`<br>not: `small_molecule_only`, `structure_prediction` | GRAVY and the Guruprasad instability index on the designed chain. Half-life is deliberately omitted | `gravy`, `instability_index` |
| `antibody_contacts` | any: `antibody` | Fraction of contacting variable-domain residues in Chothia CDRs | `cdr_contact_count`, `fw_contact_count`, `cdr_contact_fraction` |
| `cdr_torsion_naturalness` | any: `antibody` | Chothia CDR phi/psi naturalness against native samples. Opt in with `metrics/inline_metrics=all_inline_metrics_with_cdr_torsion`; the reference data is downloaded from Hugging Face on first use | `H1`–`L3`, corresponding `*_length`, `n_scored_cdrs` |
| `rosetta_developability` | any: `developability`<br>not: `small_molecule_only`, `structure_prediction` | Rosetta SAP and ExposedHydrophobics through a nested PyRosetta apptainer. Set `PYROSETTA_CONTAINER` outside the configured cluster environments | `sap_score`, `hydrophobic_exposure_sasa` |

`design_motif_allatom_rmsd` and `design_unindexed_motif_allatom_rmsd` read their ground truth off
[standard annotations](output-format.md#standard-annotations-and-why-there-is-no-output-json) written
directly into the input and output CIFs, rather than a side file.

#### Extra information for antibodies
Antibody metrics use [ANARCII](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarcii/) and basic [Chothia CDRs](https://www.sciencedirect.com/science/article/pii/S0022283610011496?via%3Dihub) across all protein chains,
independently of conditioning masks. `annotate_antibody_numbering()` caches
annotations; use `overwrite=True` after sequence edits. Contacts compare numbered
variable domains against chains without antibody domains.

Torsions use VH/VL references when a light domain is present, otherwise VHH. For an
isolated VH, set `cdr_torsion_naturalness.reference_set=vhvl`. Incomplete or unsupported
CDRs return NaN; score multiple domains of the same antibody chain type separately.

### Structure prediction metrics

These score the generated structure against a **known ground truth**, for validation runs that test
the model as a structure predictor rather than as a generative designer.

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
| `refolded_apo_holo_rmsd` | all: `is_refold:apo` (i.e. `apo_refolding`, which is what runs that oracle), `exists_refold:untransformed` | Backbone RMSD of the designed chain between its **apo** refold and its refold in the whole design. Does the binder hold its fold once the target is gone? **Needs a paired untransformed oracle**, as explained in the [output format reference](output-format.md) | `apo.rmsd` |

Every refolding metric gets the design as `ground_truth_atom_array` and the refold being scored as
`predicted_atom_array`, plus **every** refold of that design, each individually addressable and
tagged. See the [output format reference](output-format.md#refold-addresses) for how a refold gets
addressed, how the `is_refold:`/`exists_refold:`/`oracle:` tags above are derived from that address,
and the [custom metrics guide](../how-to/advanced/custom-metrics.md#comparing-two-refolds) for comparing two
refolds in your own metric, as `refolded_apo_holo_rmsd` does.

### Refolding oracles

An oracle runs the structure prediction that the refolding metrics above compare against. AF3 is
the default; select OpenFold3 or AlphaFold 2 through the corresponding `metrics/refolding_oracles`
group. See [Refolding](../how-to/refolding.md) for runtime setup and supported molecular inputs.
The table below describes the default AF3 group. An oracle runs when its tag gates match
`metrics_tags`.

| Config key | Tag gate | Notes | Key outputs (prefix `<key>.`) |
| --- | --- | --- | --- |
| `oracle` | any: `af3_refolding`, `unconditional_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding` | Untemplated AF3 refold | `overall_plddt`, `overall_ptm`, `overall_iptm`, `plddt_coi`, `ptm_coi`, `pae_coi`, `ipae_coi`, `min_ipae_coi`, `iptm_coi`, `frac_under_1`/`1.5`/`2` (on `min_ipae_coi`) |
| `oracle_ppi` | any: `ppi_refolding`, `antibody` | Templates protein atoms with ordinary or internal coordinate conditioning | same key set, prefixed `oracle_ppi.` |
| `oracle_dna` | any: `dna_refolding` | DNA templated and fixed, for DNA binder design | same key set, prefixed `oracle_dna.` |
| `oracle_apo` | any: `apo_refolding` | Refolds the designed chain alone through its apo pre-oracle transform; pair with an untransformed oracle for apo/holo comparison | `oracle_apo.apo.*` |

### Pass criteria

A pass criterion thresholds one or more already computed metric values into a binary flag at
`{config_key}.pass`, evaluated after the design, oracle and refolding metrics. All of them are built
from the generic `PassCriterionMetric`. See [Adding your own pass
criterion](../how-to/advanced/custom-metrics.md#pass-criteria-config).

```{important}
- Separate entries are ANDed together: a design only passes the criterion if every entry's threshold is satisfied. 
- `min` and `max` are inclusive.
- To threshold the best, worst or mean of several keys instead,
say when either of two interfaces being good enough will do, use a `min_of`, `max_of` or `mean_of`
list. 
See [Thresholding a reduction over several
keys](../how-to/advanced/custom-metrics.md#thresholding-a-reduction-over-several-keys).
```

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
[`all_pre_oracle_filters.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml).

A filter that reads an absent metric key counts as a failure, which happens when the metric's own
tag gate kept it from running. So keep a filter's tag gate at least as strict as its metric's.

Rejected designs skip the oracle but remain in per-example summaries as zero-pass attempts,
including ordinary and clustered pass-rate denominators. `pre_oracle_pass` records the filter
decision. Disable filters with `~metrics/pre_oracle_filters` only when you want to refold all designs.

| Config key | Tag gate | Threshold |
| --- | --- | --- |
| `unknown_residue_filter` | All | `amino_acid_composition.n_generated_unknown_residues` = 0; input sequence-conditioned UNK is exempt |
| `sequence_adherence_filter` | any: `design` | `sequence_adherence.n_violations` = 0, so the design must reproduce every conditioned residue exactly. Vacuous when nothing was sequence conditioned |
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
[output compression](output-format.md#compression)). Its metrics carry that `save_id` between the
metric name and the output key. See the [output format reference](output-format.md#refold-addresses)
for how a fan-out's outputs get addressed and tagged, and which gates the
[default refolding metrics](#refolding-metrics-after-the-oracle) use to skip them.

| Class | What it folds | Outputs |
| --- | --- | --- |
| `ApoOnlyTransform` | The chain of interest alone, everything else dropped | `apo` |
| `OffTargetSubstitutionTransform` | The chain of interest, with its binding partner swapped for each off-target loaded from disk | one per entry in `off_target_paths` |

Both live in
[`pre_oracle_transform.py`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/src/rfproteina/metrics/oracles/pre_oracle_transform.py), and
[`af3.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/metrics/refolding_oracles/af3.yaml) has wired examples. See the
[pre-oracle transforms guide](../how-to/advanced/pre-oracle-transforms.md)
for using `OffTargetSubstitutionTransform` safely and writing your own transform class.

### Ready-made but opt-in

These classes exist and are tested, but no default config includes them. Add a YAML entry to turn
one on, with no Python required. The [worked
example](../how-to/advanced/custom-metrics.md#worked-example-an-existing-metric-your-own-config) shows how.

| Class | Lives in | What it's for |
| --- | --- | --- |
| `AtomPairDistanceMetric` | `rfproteina.metrics.atom_pair_distance_metric` | Configurable upper bound distance constraints on atom pairs, for catalytic geometry and motif contact checks |
| `TMolBlockPairMetric` | `rfproteina.metrics.physics.tmol_block_pair_metric` | General interface ddG metric between two atom selection regions. Presets for ligand ddG, H-bonds and electrostatics, plus PPI ddG and Coulomb, live in [`tmol_block_pair_metrics.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml). GPU only, and deliberately out of the default set |
| `QMInteractionEnergyIPA` | `rfproteina.metrics.ipa_interaction_energy_metric` | QM ligand and protein interaction energy through IPA. See [`ipa_metrics.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/metrics/inline_metrics/ipa_metrics.yaml), currently commented out pending a backend compatibility fix |
| `ChargeComplementarityMetric` | `rfproteina.metrics.physics.ddg_salt_metrics` | Salt bridge charge complementarity score across a binder and target interface |
| `InternalResidueBondMetric` | `rfproteina.metrics.bond_length_metric` | Per residue bond lengths against CCD ideal geometry |
| `NucleicSSSimilarityMetrics` | `rfproteina.metrics.nucleic_ss_metrics` | Nucleic acid secondary structure conditioning adherence, meaning base pairing |
| `ContractCAMetric` | `rfproteina.metrics.contract_ca_metric` | Trivial differentiable guidance metric that contracts CA toward the origin, mainly for exercising the search and guidance path |
