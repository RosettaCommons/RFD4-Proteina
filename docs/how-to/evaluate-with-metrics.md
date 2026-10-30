# Evaluate with metrics

Use the built-in metrics instead of parsing output CIFs by hand. They can check
whether a motif was inserted correctly, measure agreement with conditioning,
and compare a design with a predicted structure.

- **Design metrics** check the generated structure directly, including clashes,
  sequence constraints, hotspots and motif coordinates.
- **Refolding metrics** compare designs with predictions from a separate folding
  model and report confidence and structural agreement.
- **Pass criteria** combine metric thresholds into a pass/fail result.
- **Pre-refolding filters** skip costly predictions when a design fails an earlier
  check. Rejected designs still count as failed attempts in pass rates.

Metrics can also guide [reward-guided search](../concepts/search.md), rather than
only filtering finished designs. See [Metrics](../concepts/metrics.md) for how
these checks fit together and the [metrics reference](../reference/metrics.md)
for all tags, output columns and thresholds.

## Running metrics inline

:::{admonition} Refolding setup
:class: dropdown

Refolding uses the configured oracle: AF3 by default, or OpenFold3/AlphaFold 2 through
`metrics/refolding_oracles=of3` or `metrics/refolding_oracles=af2`. Each needs its own runtime
and weights; see [oracle setup](refolding.md). Search only needs an oracle when a configured reward
or continuation criterion uses refolding. [Structure-only search](../concepts/search.md)
can score designs without a refolding runtime.
:::

Pass `metrics_tags` during inference and metrics get computed right after each design. Results land
under `<out_dir>/inference_metrics/` in `per_example_metrics.csv` (or a per-job CSV for
SLURM arrays), while structures go under `<out_dir>/inference_samples/`.

Pass `metrics_tags` as a bracketed, quoted list. Only checks matching the supplied
tags run; including a metric in the configuration does not enable it by itself.

```bash
# Single tag
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'

# Multiple tags
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

:::{admonition} Common metric tags
:class: dropdown

| Tag | Activates |
| --- | --- |
| `unconditional_refolding` | Refolding plus pLDDT and scRMSD for unconditional generation |
| `ppi_refolding` | Refolding, target templating and interface metrics for protein binder design |
| `dna_refolding` | Refolding and interface metrics for DNA binder design |
| `antibody_refolding` | Antibody contacts, templated refolding, holo North-CDR pLDDT and antibody pass criteria; CDR torsion scores are opt-in |
| `antibody_developability` | VHH SAP/charge/cysteine checks, apo and holo refolds, CDR confidence and apo/holo RMSDs, plus a combined pass criterion |
| `motif_scaffold_refolding` | Refolding plus motif RMSD for motif scaffolding and enzyme design |
| `sm_binder_refolding` | Refolding plus binding metrics for small molecule binder design |
| `ame_refolding` | Refolding for enzyme (MCSA) active site benchmarks |
| `hotspot_conditioning` | Hotspot obedience metrics, needs hotspot input |
| `3d_coordinate_conditioning` | Motif coordinate RMSD metrics |
| `rasa_conditioning` | RASA adherence metrics |
| `interchain_interactions` | Interchain interaction adherence and counts, covering H-bonds, π-stacking, π-cation, salt bridges, hydrophobic contacts and metal coordination |

This is a shortlist. The [metrics reference](../reference/metrics.md) also covers
optional metrics and how to enable CDR torsion scores.
:::

## Running metrics offline on a folder of structures

To score structures you already have:

```bash
rfd4-evaluate \
    inputs=./runs/designs/inference_samples \
    metrics_tags='[ppi_refolding,hotspot_conditioning]' \
    out_dir=./runs/evaluation
```

Results go under `<out_dir>/evaluation_metrics/`. See
[offline evaluation](advanced/offline-metrics.md) for assigning tasks with
`task_csv`, and [multi-job runs](advanced/multi-job-runs.md) for combining results.

## VHH developability

To check both antibody binding and VHH developability, use:

```bash
metrics_tags='[antibody_refolding,antibody_developability]'
```

- **`antibody_refolding`** checks binding, target-aligned CDR and whole-binder RMSDs, and
  CDR confidence in the target-bound (holo) refold. Binding passes are reported
  as `antibody.pass`; holo CDR confidence is reported without a pass threshold.
- **`antibody_developability`** checks surface and sequence properties and
  compares the VHH refolded alone (apo) with its holo refold. It runs both refolds
  and reports `antibody_developability.pass` separately from binding.
- These developability checks use the default AF3 configuration and require
  [refolding setup](refolding.md). Missing or NaN scores cannot pass.

See [Antibody metrics](../reference/antibody-metrics.md) for the passing
thresholds, score definitions and output columns.

For campaign advice, see the
[antibody design recommendations](https://github.com/RosettaCommons/RFD4-Proteina/blob/main/examples/antibody_design/README.md).

## Custom metrics and thresholds

Start with an [existing metric](../reference/metrics.md#optional-metrics) before
writing a new one. Optional checks include interface energies and atom-pair
distance constraints. The [custom metrics guide](advanced/custom-metrics.md)
shows how to enable them, change pass thresholds, combine scores and add a new
metric when needed.

```{toctree}
:hidden:
:maxdepth: 1

advanced/custom-metrics
advanced/offline-metrics
```
