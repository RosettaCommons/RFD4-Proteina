# Evaluate with metrics

You should not have to hand parse output CIFs. You *can* reconstruct the unindexed mapping, or work
out which chain an inserted region came from, or measure how closely a design followed its
coordinate conditioning, all by reading raw CIF fields. Do that only when nothing else works.
Turn on the metric that already computes it and let the pipeline run it. Metrics that run in the
pipeline can also drive [reward guided search](../concepts/search.md), which finds good designs on far less
compute than generating everything and filtering afterwards.

For what each metric family actually measures, how binder and target selections resolve, and where
in a run a metric can stop further work, see [Metrics](../concepts/metrics.md). The
[metrics reference](../reference/metrics.md#every-metric-criteria-and-filter-that-ships) lists all
of them, what each outputs, and the tag that turns it on. For the refolding family specifically — the
oracle, the AF3/OF3 setup it needs, and how a design can carry more than one refold — see the
[refolding guide](refolding.md).

## What the default metrics configuration registers

With its default AF3 oracle group, `configs/metrics/all_metrics.yaml` registers **53 metric,
oracle, pass-criterion and filter entries**. Registered entries do not all run on every design:
most are gated by `metrics_tags`, and refolding metrics also depend on which refolds are available.
Select tags on the command line:

```bash
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'
```

| Category | Count | Config |
| --- | --- | --- |
| Design and developability metrics, run inline before refolding | 22 | `configs/metrics/inline_metrics/design_metrics.yaml` and `developability_metrics.yaml` |
| Antibody contact metrics | 1 | `configs/metrics/inline_metrics/antibody_metrics.yaml` |
| Structure prediction metrics | 1 | `configs/metrics/inline_metrics/structure_prediction_metrics.yaml` |
| Refolding metrics, design against refold or refold against refold | 8 | `configs/metrics/refolding_metrics/all_refolding_metrics.yaml` |
| Refolding oracles, the default AF3 variants | 4 | `configs/metrics/refolding_oracles/af3.yaml` |
| Pass criteria | 11 | `configs/metrics/pass_criteria/all_pass_criteria.yaml` |
| Pre-oracle filters, which skip the oracle when a cheap check fails | 6 | `configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml` |

Selecting another oracle group or adding custom metrics changes this inventory. CDR torsion
naturalness is opt-in through `inline_metrics/all_inline_metrics_with_cdr_torsion.yaml`; it is not
part of the default count.

Additional metrics are available in the codebase and opt-in config groups, including TMol interface
energies and atom-pair distance constraints. Enable an existing metric with a YAML entry, as in the
[worked example](advanced/custom-metrics.md#worked-example-an-existing-metric-your-own-config).
See [Ready-made but opt-in](../reference/metrics.md#ready-made-but-opt-in).

Write a new `Metric` subclass only when none of the existing metrics cover what you need.
See [Writing your own](advanced/custom-metrics.md#writing-your-own).

## Running metrics inline

:::{note}
Refolding uses the configured oracle: AF3 by default, or OpenFold3/AlphaFold 2 through
`metrics/refolding_oracles=of3` or `metrics/refolding_oracles=af2`. Each needs its own runtime
and weights; see [oracle setup](refolding.md). Search only needs an oracle when a configured reward
or continuation criterion uses refolding. [Structure-only search](../concepts/search.md)
can score designs without a refolding runtime.
:::

Pass `metrics_tags` during inference and metrics get computed right after each design. Results land
under `<out_dir>/inference_metrics/` in `per_example_metrics.csv` (or a per-job CSV for
SLURM arrays), while structures go under `<out_dir>/inference_samples/`.

The [metrics reference](../reference/metrics.md) covers every metric that ships enabled, what it
outputs, and which tag turns it on. See the [custom metrics guide](advanced/custom-metrics.md) for
how to layer your own config on top of an existing metric.

`metrics_tags` takes a Hydra style list, so pass it bracketed and quoted.

```bash
# Single tag
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'

# Multiple tags
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

| Tag | Activates |
| --- | --- |
| `unconditional_refolding` | AF3 refolding plus pLDDT and scRMSD for unconditional generation |
| `ppi_refolding` | AF3 refolding, target templating and interface metrics for protein binder design |
| `antibody` | Antibody contacts, templated refolding and antibody pass criteria; CDR torsions require the opt-in config above |
| `motif_scaffold_refolding` | AF3 refolding plus motif RMSD for motif scaffolding and enzyme design |
| `sm_binder_refolding` | AF3 refolding plus binding metrics for small molecule binder design |
| `ame_refolding` | AF3 refolding for enzyme (MCSA) active site benchmarks |
| `hotspot_conditioning` | Hotspot obedience metrics, needs hotspot input |
| `3d_coordinate_conditioning` | Motif coordinate RMSD metrics |
| `rasa_conditioning` | RASA adherence metrics |
| `interchain_interactions` | Interchain interaction adherence and counts, covering H-bonds, π-stacking, π-cation, salt bridges, hydrophobic contacts and metal coordination |

A metric that needs a tag gets skipped silently when the tag is absent. The table above is a
shortlist. The [metrics reference](../reference/metrics.md) covers everything that ships enabled
plus every metric class that is ready made but opt in.

## Running metrics offline on a folder of structures

See [Running metrics offline on a folder of structures](advanced/offline-metrics.md) for scoring
structures you already have with `evaluate.py`, `task_csv`, and concatenating multi-job results.

```{toctree}
:hidden:
:maxdepth: 1

advanced/custom-metrics
advanced/offline-metrics
```
