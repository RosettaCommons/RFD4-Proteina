# Metrics reference

Metric tags, output columns and passing thresholds. For commands, see
[Evaluate with metrics](../how-to/evaluate-with-metrics.md); for prediction model
setup, see [Refolding](../how-to/refolding.md).

**Contents**

- [Reading the tables](#reading-the-tables)
- [Design metrics](#design-metrics)
  - [Antibody metrics](antibody-metrics.md)
- [Structure prediction metrics](#structure-prediction-metrics)
- [Refolding metrics](#refolding-metrics)
- [Refolding oracles](#refolding-oracles)
- [Pass criteria](#pass-criteria)
- [Pre-oracle filters](#pre-oracle-filters)
- [Pre-oracle transforms](#pre-oracle-transforms)
- [Optional metrics](#optional-metrics)

(every-metric-criteria-and-filter-that-ships)=
## Reading the tables

The **Tags** column lists which `metrics_tags` enable each check:

- `any`: include at least one listed tag.
- `all`: include every listed tag.
- `not`: exclude all listed tags.
- `Always`: no tag restriction.

All restrictions in a row apply. To run both design and refolding checks, use
`metrics_tags='[design,unconditional_refolding]'`.

Output columns start with the config key. For example, the `rmsd` output of
`refolded_backbone_rmsd` becomes `refolded_backbone_rmsd.rmsd`.

:::{admonition} Refold tags
:class: dropdown

Refolding metrics also use `is_refold:`, `exists_refold:` and `oracle:` tags.
These are added automatically; do not pass them on the command line.
They select which refold to score and which other refolds must be available.

See [refold addresses](output-format.md#refold-addresses) for the naming rules
and [custom metrics](../how-to/advanced/custom-metrics.md#gating-with-tags)
for configuring tag requirements.
:::

(design-metrics-inline-and-before-refolding)=
## Design metrics

These score the generated structure without refolding.

:::{admonition} Structure quality
:class: dropdown

| Config key | Tags | Measures | Output suffixes |
| --- | --- | --- | --- |
| `amino_acid_composition` | Always | Generated UNK count and ALA, VAL and GLY fractions among standard amino acids; sequence-conditioned UNK is exempt from the generated UNK count | `n_generated_unknown_residues`, `n_standard_residues`, `ala_fraction`, `val_fraction`, `gly_fraction` |
| `bond_length_metrics` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Fraction of unreasonable bond lengths, using element-specific thresholds | `frac_unreasonable`; per-category counts in verbose mode |
| `clash_metrics` | any: `design`, `structure_prediction`<br>not: `small_molecule_only` | Fraction of atoms clashing at a configurable distance (default 1.5 Å) | `frac_atoms_clashing_1p5A` plus 5 category breakdowns, intra and inter residue by polymer and ligand |
| `sse_distribution` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Helix, sheet and loop fraction, plus the number of contiguous SSE elements, from P-SEA | `helix_fraction`, `sheet_fraction`, `loop_fraction`, `num_ss_elements` |
| `protruding_helix` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Binary flag for a long helix with a consecutive stretch lacking protein packing | `present` (0 or 1) |
| `posebusters_metrics` | any: `small_molecule_only` | Ligand geometry and validity checks from PoseBusters | one pass or fail column per PoseBusters check |
| `chiral_accuracy` | any: `small_molecule_only` | Fraction of reference stereoisomers recovered, split by center geometry and E/Z double bonds | `tetrahedral_percent_correct_chirality`, `non_tetrahedral_percent_correct_chirality`, `double_bond_percent_correct_stereochemistry` |

`protruding_helix.present` is `1` when any P-SEA alpha-helical segment has at least
30 consecutive residues, a first-to-last C-alpha span of at least 35 Å, and at least
15 consecutive residues without a protein C-alpha contact within 10 Å outside that
helix. Contacts to same-chain neighbors within five residue numbers are excluded.
Residue-number gaps split segments. All protein chains, including conditioned context,
are inspected; ligands do not count as packing partners. This is a geometric screen,
so naturally extended helices can also be flagged. The cutoffs are configurable under
`metrics.inline_metrics.protruding_helix`. The aggregate mean is the fraction of designs
flagged: `0.025` means 2.5%.
:::

:::{admonition} Conditioning
:class: dropdown

| Config key | Tags | Measures | Output suffixes |
| --- | --- | --- | --- |
| `hotspot_obedience` | any: `hotspot_conditioning` | Fraction of hotspot atoms with a binder atom within 4.5 Å | `hotspot_obedience` |
| `rasa_adherence` | any: `rasa_conditioning` | Agreement with conditioned solvent-exposure bins | `buried_conditioning_adherence`, `intermediate_conditioning_adherence`, `exposed_conditioning_adherence` |
| `centroid_conditioning_adherence` | any: `centroid_conditioning` | Mean distance between each conditioned chain centroid and the requested position | `centroid_conditioning_mae` |
| `interchain_interactions` | any: `interchain_interactions`<br>not: `small_molecule_only`, `structure_prediction` | Adherence plus atom and pair counts for H-bond, π-stacking, π-cation, salt bridge, hydrophobic and metal coordination conditioning, interchain only | `*_conditioning_adherence`, `*_atom_count_coi` / `*_residue_count_coi`, `*_atom_pair_count_coi` / `*_residue_pair_count_coi` per interaction type |
| `sequence_adherence` | any: `design` | Fraction matching conditioned residue identities; NaN without sequence conditioning or when offline scoring lacks the input | `sequence_adherence`, `n_violations` |
| `sequence_probability_adherence` | any: `design` | Adherence to explicitly specified probability-zero exclusions and probability-one identities. Soft probability errors are pooled across designs during aggregation. No default pre-oracle filter is applied | `zero_adherence`, `one_adherence`, `n_zero_conditions`, `n_zero_violations`, `n_one_conditions`, `n_one_violations` |
| `sse_accuracy` | any: `design`<br>not: `small_molecule_only`, `structure_prediction` | Accuracy of generated secondary structure against your secondary structure conditioning | `overall_accuracy`, `helix_recall`, `sheet_recall`, `loop_recall`, `n_conditioned_atoms` |
| `design_motif_allatom_rmsd` | any: `3d_coordinate_conditioning`, `unindexed`, `internal_coordinate_conditioning` | All-atom motif RMSD against conditioned coordinates | `rmsd`, the median over replicates, and per `CoordinateGroup` when there are several motifs |
| `design_unindexed_motif_allatom_rmsd` | all: `unindexed` | All-atom RMSD of unindexed insertion atoms | `rmsd` |
| `unindexed_insertion_mapping` | all: `unindexed` | Missing unindexed tokens and mismatches to conditioned residue identities | `n_unmapped_tokens`, `n_sequence_mismatches` |

Motif RMSDs read reference coordinates from the
[standard annotations](output-format.md#standard-annotations-and-why-there-is-no-output-json)
in the CIFs.
:::

:::{admonition} Interfaces, similarity and classification
:class: dropdown

| Config key | Tags | Measures | Output suffixes |
| --- | --- | --- | --- |
| `contact_molecular_surface` | any: `cms` | Distance weighted interfacial contact surface area between binder and target, following [Lawrence and Colman 1993](https://www.sciencedirect.com/science/article/pii/S0022283683716487?via%3Dihub) | see `rfproteina.metrics.physics.contact_ms.metric` |
| `sequence_similarity_to_input` | any: `partial_diffusion` | Sequence identity against the seed structure, for partial diffusion | `sequence_identity_to_input` |
| `ca_rmsd_to_input` | any: `partial_diffusion` | Cα RMSD to the input after Cα alignment; omitted if Cα counts differ | `rmsd` |
| `cath_metric` | any: `cath` | CATH superfamily classification of the chain of interest, through Foldseek | `C`, `A`, `T`, `H`, `type`, `evalue`, `tmscore` |
:::

:::{admonition} Developability and antibody metrics
:class: dropdown

| Config key | Tags | Measures | Output suffixes |
| --- | --- | --- | --- |
| `sequence_developability` | any: `developability`<br>not: `small_molecule_only`, `structure_prediction` | GRAVY and Guruprasad instability index on the designed chain | `gravy`, `instability_index` |
| `atomworks_developability` | any: `developability`<br>not: `small_molecule_only`, `structure_prediction` | SAP and exposed hydrophobic surface on the designed chain; calibrated against PyRosetta | `sap_score`, `hydrophobic_exposure_sasa` |
| `antibody_contacts` | any: `antibody_refolding` | Fraction of contacting variable-domain residues in North CDRs | `cdr_contact_count`, `fw_contact_count`, `cdr_contact_fraction` |
| `antibody_developability_scores` | any: `antibody_developability` | Native SAP and sequence checks on the designed VHH, with North H3 selections | `sap_score`, `cdrh3_sap_score`, `net_charge`, `cdrh3_net_charge`, `unpaired_cysteines` |
| `cdr_torsion_naturalness` | any: `antibody_refolding` | Chothia CDR phi/psi naturalness against native samples. Opt in with `metrics/inline_metrics=all_inline_metrics_with_cdr_torsion`; the reference data is downloaded from Hugging Face on first use | `H1`–`L3`, corresponding `*_length`, `n_scored_cdrs` |
:::

(sap-calculation)=
:::{admonition} SAP calculation
:class: dropdown

SAP uses Rosetta hydrophobicity and maximum sidechain SASA tables, a 1.1 Å
probe and implicit-hydrogen neighborhood multiplicities. Its scale was fitted
on 28 structures and evaluated on 13 held-out structures (Pearson r = 0.998;
MAE = 5.8 SAP units).

The exposed-hydrophobic-surface calculation assumes complete N, CA, C, O and CB
coordinates. It does not reconstruct missing atoms as PyRosetta import does.
:::

(extra-information-for-antibodies)=
(antibody-definitions)=
(vhh-developability-details)=
### Antibody metrics

See [Antibody metrics](antibody-metrics.md) for definitions, thresholds, output
columns and interpretation.

## Structure prediction metrics

For validation against a known structure, rather than design:

| Config key | Tags | Measures | Output suffixes |
| --- | --- | --- | --- |
| `structure_prediction_lddt` | any: `structure_prediction` | All-atom lDDT against the reference structure | `lddt` |

(refolding-metrics-after-the-oracle)=
## Refolding metrics

These compare the design with its refold, or compare two refolds.

:::{admonition} Design and apo/holo comparisons
:class: dropdown

| Config key | Tags | Measures | Output suffixes |
| --- | --- | --- | --- |
| `refolded_backbone_rmsd` | any: `unconditional_refolding`, `ppi_refolding`, `antibody_refolding`, `dna_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding`<br>all: `is_refold:untransformed` | Backbone self-consistency RMSD (scRMSD) between design and refold | `rmsd` |
| `refolded_target_aligned_binder_rmsd` | any: `ppi_refolding`, `dna_refolding`, `antibody_refolding`<br>all: `is_refold:untransformed` | Binder all-atom RMSD after aligning on the fixed target | `rmsd` |
| `refolded_target_backbone_aligned_binder_rmsd` | any: `ppi_refolding`<br>all: `is_refold:untransformed` | Binder all-atom RMSD after alignment on coordinate-conditioned target backbone atoms | `rmsd` |
| `refolded_motif_allatom_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`<br>all: `is_refold:untransformed` | All atom motif RMSD between design and refold, unindexed regions included | `rmsd` |
| `refolded_motif_ca_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`<br>all: `is_refold:untransformed` | CA only motif RMSD between design and refold, unindexed regions included | `rmsd` |
| `refolded_backbone_aligned_ligand_rmsd` | any: `sm_binder_refolding`<br>all: `is_refold:untransformed` | Ligand RMSD after backbone alignment | `rmsd` |
| `max_ligand_centroid_distance` | any: `disco_refolding`<br>all: `is_refold:untransformed` | Largest ligand centroid displacement after backbone alignment | `ligand_centroid_distance` |
| `refolded_apo_holo_rmsd` | all: `is_refold:apo`, `exists_refold:untransformed` | Designed-chain backbone RMSD between apo and holo refolds; requires both refolds | `apo.rmsd` |
:::

:::{admonition} Antibody comparisons and confidence
:class: dropdown

| Config key | Tags | Measures | Output suffixes |
| --- | --- | --- | --- |
| `antibody_target_aligned_cdr_rmsd` | any: `antibody_refolding`<br>all: `is_refold:untransformed` | North CDR backbone RMSD after target backbone alignment | `rmsd` |
| `antibody_target_aligned_backbone_rmsd` | any: `antibody_refolding`<br>all: `is_refold:untransformed` | Whole-antibody backbone RMSD after target backbone alignment | `rmsd` |
| `antibody_holo_cdr_confidence` | any: `antibody_refolding`<br>all: `is_refold:oracle_ppi` | Mean North CDR Cα pLDDT on the designed antibody in its holo refold | `plddt` |
| `antibody_apo_cdr_confidence` | any: `antibody_developability`<br>all: `is_refold:oracle_apo.apo` | Mean apo CDR Cα pLDDT | `apo.plddt` |
| `antibody_apo_holo_cdr_rmsd` | any: `antibody_developability`<br>all: `is_refold:oracle_apo.apo`, `exists_refold:oracle_ppi` | VHH CDR backbone RMSD after framework backbone alignment | `apo.rmsd` |
| `antibody_apo_holo_rmsd` | any: `antibody_developability`<br>all: `is_refold:oracle_apo.apo`, `exists_refold:oracle_ppi` | Whole-VHH backbone RMSD after whole-VHH backbone alignment | `apo.rmsd` |
:::

For custom comparisons, see
[comparing two refolds](../how-to/advanced/custom-metrics.md#comparing-two-refolds).

## Refolding oracles

An oracle predicts a structure for refolding checks. AF3 is the default;
OpenFold3 and AlphaFold 2 are also available. See
[Refolding](../how-to/refolding.md) for setup and supported inputs.

:::{admonition} Default oracle configurations
:class: dropdown

These entries belong to the default AF3 configuration.

| Config key | Tags | Input | Outputs |
| --- | --- | --- | --- |
| `oracle` | any: `af3_refolding`, `unconditional_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding` | Untemplated AF3 refold | `overall_plddt`, `overall_ptm`, `overall_iptm`, `plddt_coi`, `ptm_coi`, `pae_coi`, `ipae_coi`, `min_ipae_coi`, `iptm_coi`, `frac_under_1`/`1.5`/`2` (on `min_ipae_coi`) |
| `oracle_ppi` | any: `ppi_refolding`, `antibody_refolding`, `antibody_developability` | Templates protein atoms with ordinary or internal coordinate conditioning | same key set, prefixed `oracle_ppi.` |
| `oracle_dna` | any: `dna_refolding` | DNA templated and fixed, for DNA binder design | same key set, prefixed `oracle_dna.` |
| `oracle_apo` | any: `apo_refolding`, `antibody_developability` | Refolds the designed chain alone through its apo pre-oracle transform; pair with an untransformed oracle for apo/holo comparison | `oracle_apo.apo.*` |
:::

## Pass criteria

Each criterion reports `<config_key>.pass`: 1 if every threshold passes,
otherwise 0. `min` and `max` are inclusive; `min_exclusive` and
`max_exclusive` are strict.

| Config key | Tags | Thresholds |
| --- | --- | --- |
| `unconditional` | any: `unconditional_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 1.5 |
| `ppi` | any: `ppi_refolding` | `oracle_ppi.ptm_coi` ≥ 80, `oracle_ppi.min_ipae_coi` ≤ 1.5, `refolded_target_backbone_aligned_binder_rmsd.rmsd` ≤ 2.5 |
| `antibody` | any: `antibody_refolding` | See [antibody binding thresholds](antibody-metrics.md#binding) |
| `dna` | any: `dna_refolding` | `oracle_dna.min_ipae_coi` ≤ 1.5, `refolded_target_aligned_binder_rmsd.rmsd` ≤ 8.0 |
| `motif_scaffold` | any: `motif_scaffold_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 2.0, `refolded_motif_ca_rmsd.rmsd` ≤ 1.0 |
| `internal_coords` | any: `internal_coordinate_motif_scaffold_refolding` | Same thresholds as `motif_scaffold` |
| `sm_binder` | any: `sm_binder_refolding` | `oracle.min_ipae_coi` ≤ 1.5, `oracle.iptm_coi` ≥ 80, `refolded_backbone_rmsd.rmsd` ≤ 1.5, `refolded_backbone_aligned_ligand_rmsd.rmsd` ≤ 5.0 |
| `ame` | any: `ame_refolding`<br>not: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5, zero inter residue polymer and ligand clashes |
| `ame_no_ligand` | any: `ame_refolding`<br>all: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5 |
| `disco` | any: `disco_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `max_ligand_centroid_distance.ligand_centroid_distance` ≤ 2.0 |
| `developability` | any: `developability` | `atomworks_developability.sap_score` ≤ 50, `atomworks_developability.hydrophobic_exposure_sasa` ≤ 1500, `sequence_developability.gravy` ≤ 0.1, `sequence_developability.instability_index` ≤ 40 |
| `antibody_developability` | any: `antibody_developability` | All eight [VHH developability checks](antibody-metrics.md#vhh-developability) must pass |

To change thresholds or combine scores with `min_of`, `max_of` or `mean_of`,
see [custom pass criteria](../how-to/advanced/custom-metrics.md#pass-criteria-config).

## Pre-oracle filters

Filters reject designs before refolding. Rejected designs still count as failed
attempts in pass rates, including clustered pass rates.
`pre_oracle_pass` records the decision.

A missing metric counts as a failure, so enable the metrics a filter needs.
Use `~metrics/pre_oracle_filters` to disable filtering and refold all designs.

:::{admonition} Filter thresholds
:class: dropdown

| Config key | Tags | Threshold |
| --- | --- | --- |
| `unknown_residue_filter` | Always | `amino_acid_composition.n_generated_unknown_residues` = 0; input sequence-conditioned UNK is exempt |
| `sequence_adherence_filter` | any: `design` | `sequence_adherence.n_violations` = 0; passes if no sequence is conditioned |
| `unindexed_insertion_mapping_filter` | all: `unindexed` | `unindexed_insertion_mapping.n_unmapped_tokens` = 0 and `.n_sequence_mismatches` = 0 |
| `ligand_clash_filter` | all: `ligand_clash_filter`<br>not: `no_ligand` | Zero inter residue polymer and ligand clashes, read off `clash_metrics` |
| `insertion_rmsd_filter` | all: `unindexed`, `insertion_rmsd_filter` | `unindexed_metrics.insertion_rmsd` ≤ 1.0; requires a metric configured to produce this key |
| `motif_rmsd_filter` | all: `motif_rmsd_filter` | `motif_rmsd.rmsd` ≤ 1.0; requires a metric configured to produce this key |
:::

## Pre-oracle transforms

Transforms change the input to a refold, such as removing the target or
replacing it with an off-target. Set one through an oracle's
`pre_oracle_transform` option. Each resulting structure is refolded separately.

| Class | Input | Outputs |
| --- | --- | --- |
| `ApoOnlyTransform` | The chain of interest alone, everything else dropped | `apo` |
| `OffTargetSubstitutionTransform` | The chain of interest, with its binding partner swapped for each off-target loaded from disk | one per entry in `off_target_paths` |

See [pre-oracle transforms](../how-to/advanced/pre-oracle-transforms.md)
for configuration examples and [output format](output-format.md#refold-addresses)
for filenames and metric prefixes.

(ready-made-but-opt-in)=
## Optional metrics

These are not included in the default configuration. To enable one, follow the
[custom metric example](../how-to/advanced/custom-metrics.md#worked-example-an-existing-metric-your-own-config).

:::{admonition} Available metric classes
:class: dropdown

| Class | Module | Purpose |
| --- | --- | --- |
| `AtomPairDistanceMetric` | `rfproteina.metrics.atom_pair_distance_metric` | Configurable upper bound distance constraints on atom pairs, for catalytic geometry and motif contact checks |
| `TMolBlockPairMetric` | `rfproteina.metrics.physics.tmol_block_pair_metric` | General interface ddG metric between two atom selection regions. Presets for ligand ddG, H-bonds and electrostatics, plus PPI ddG and Coulomb, live in [`tmol_block_pair_metrics.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml). GPU only, and deliberately out of the default set |
| `QMInteractionEnergyIPA` | `rfproteina.metrics.ipa_interaction_energy_metric` | QM ligand and protein interaction energy through IPA. See [`ipa_metrics.yaml`](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/configs/metrics/inline_metrics/ipa_metrics.yaml), currently commented out pending a backend compatibility fix |
| `ChargeComplementarityMetric` | `rfproteina.metrics.physics.ddg_salt_metrics` | Salt bridge charge complementarity score across a binder and target interface |
| `InternalResidueBondMetric` | `rfproteina.metrics.bond_length_metric` | Per residue bond lengths against CCD ideal geometry |
| `NucleicSSSimilarityMetrics` | `rfproteina.metrics.nucleic_ss_metrics` | Nucleic acid secondary structure conditioning adherence, meaning base pairing |
| `ContractCAMetric` | `rfproteina.metrics.contract_ca_metric` | Contracts Cα atoms toward the origin; used to test search and guidance |
:::
