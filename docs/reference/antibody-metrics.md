# Antibody metrics

Definitions, thresholds and output columns for antibody binding and VHH
developability. For commands, see [Evaluate with metrics](../how-to/evaluate-with-metrics.md#vhh-developability).
For campaign strategy, see the
[antibody design guide](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/examples/antibody_design/README.md).

- [Binding](#binding)
- [VHH developability](#vhh-developability)
- [Score definitions](#score-definitions)
- [Optional CDR torsion scores](#optional-cdr-torsion-scores)

## Binding

The `antibody_refolding` tag enables antibody contacts, target-bound (holo)
refolding, confidence and RMSD metrics. Both thresholds below must pass for
`antibody.pass` to be 1:

| Output column | Requirement |
| --- | --- |
| `oracle_ppi.min_ipae_coi` | ≤ 2.0 |
| `refolded_target_aligned_binder_rmsd.rmsd` | ≤ 2.0 Å |

Passing is a cutoff, not a ranking: lower `oracle_ppi.min_ipae_coi` is better.
For campaign selection, prefer values < 1.0 among otherwise comparable designs
for the same target. This is a ranking preference, not an additional pass threshold
or a substitute for the RMSD and developability checks.

The second check measures binder **all-atom** RMSD after target alignment.
The following are also reported, but do not have default pass thresholds:

| Output column | Measures |
| --- | --- |
| `antibody_target_aligned_cdr_rmsd.rmsd` | North CDR backbone RMSD after target-backbone alignment |
| `antibody_target_aligned_backbone_rmsd.rmsd` | Whole-antibody backbone RMSD after target-backbone alignment |
| `antibody_holo_cdr_confidence.plddt` | Mean North CDR Cα pLDDT in the holo refold |
| `antibody_contacts.cdr_contact_count` | Number of contacting CDR residues |
| `antibody_contacts.fw_contact_count` | Number of contacting framework residues |
| `antibody_contacts.cdr_contact_fraction` | CDR contacts divided by all antibody contacts |

## VHH developability

The `antibody_developability` tag checks the designed VHH's surface and sequence,
and compares its refolds alone (apo) and with the target (holo). These checks use
the default AF3 configuration and require [refolding setup](../how-to/refolding.md).

These scores cover a single chain of interest and their thresholds are calibrated
for single-domain VHHs, so they are not meaningful for a paired VH/VL Fv. Leave
the tag off for paired designs and judge them on binding and the RMSDs above.

All eight checks must pass for `antibody_developability.pass` to be 1. This is
separate from `antibody.pass`.

| Check | Requirement | Output column |
| --- | --- | --- |
| Mean apo CDR Cα pLDDT | > 85 | `antibody_apo_cdr_confidence.apo.plddt` |
| Apo/holo CDR backbone RMSD, aligned on the framework | < 1.5 Å | `antibody_apo_holo_cdr_rmsd.apo.rmsd` |
| Apo/holo whole-VHH backbone RMSD, aligned on the whole VHH | < 1.5 Å | `antibody_apo_holo_rmsd.apo.rmsd` |
| CDRH3 SAP | < 10 | `antibody_developability_scores.cdrh3_sap_score` |
| Whole-VHH SAP | < 55 | `antibody_developability_scores.sap_score` |
| Whole-VHH net charge | ≤ −3 | `antibody_developability_scores.net_charge` |
| CDRH3 net charge | < 0 | `antibody_developability_scores.cdrh3_net_charge` |
| Cysteine count | Even (score = 0) | `antibody_developability_scores.unpaired_cysteines` |

Missing or NaN scores cannot pass. Missing inputs also set
`antibody_developability.all_metrics_present` to zero. The comparisons use
`oracle_apo.apo` and `oracle_ppi`; additional refolds do not change which pair
is scored.

## Score definitions

- **CDRs:** ANARCII assigns Chothia numbering from the sequence. Contacts,
  developability, confidence and RMSD use [North CDR boundaries](https://pmc.ncbi.nlm.nih.gov/articles/PMC3065967/).
  For VHHs these are H1 23–35, H2 50–58 and H3 93–102, inclusive of
  insertion-coded residues. CDR4 is excluded.
- **Contacts:** antibody variable-domain residues are counted when a target Cα
  is within 7 Å of their Cα. Each residue counts once. Targets are protein chains
  without a recognized antibody domain; conditioning masks do not determine
  this selection. The contact fraction is NaN when there are no contacts.
- **CDR confidence:** mean Cα `b_factor` from the refolded CIF, on a 0–100 pLDDT
  scale. Chain-of-interest pLDDT instead comes from the prediction model's
  confidence JSON. Experimental B-factors are not pLDDT.
- **SAP:** spatial aggregation propensity on the generated VHH, excluding the
  target. CDRH3 SAP scores only H3 while retaining the whole VHH for solvent
  occlusion and neighboring hydrophobicity. See the shared {ref}`SAP calculation
  notes <sap-calculation>` for calibration and missing-atom assumptions.
- **Net charge:** `K + R − D − E`, calculated on the VHH or H3 sequence.
  Histidine and termini are neutral.
- **Cysteines:** the score is the cysteine count modulo 2, not a count of
  structurally unpaired cysteines. An even count does not establish disulfide pairing.

## Optional CDR torsion scores

Add `metrics/inline_metrics=all_inline_metrics_with_cdr_torsion` alongside
`antibody_refolding` to enable `cdr_torsion_naturalness`. Reference data is
downloaded from Hugging Face on first use.

- Torsion scores use basic Chothia CDR boundaries, rather than North boundaries,
  to match their reference distributions.
- VH/VL references are used when a light domain is present; otherwise VHH
  references are used. For an isolated VH, set
  `cdr_torsion_naturalness.reference_set=vhvl`.
- Outputs use the `cdr_torsion_naturalness.` prefix: `H1`–`L3`, corresponding
  `*_length` columns and `n_scored_cdrs`. Incomplete or unsupported CDRs return
  NaN. Score multiple domains of the same chain type separately.

:::{admonition} Using antibody annotations in code
:class: dropdown

Selection queries create `is_north_cdr`, `is_north_cdrh3` and `is_chothia_cdr`
masks automatically. In Python, use AtomWorks' `ensure_annotations` to request
them. They share cached ANARCII results; ANARCII loads only when needed.

After editing sequences or chains, call
`remove_annotations(array, *ANTIBODY_ANNOTATIONS)` before requesting masks again.
`ANTIBODY_ANNOTATIONS` is exported by `rfproteina.utils.antibody`.
:::
