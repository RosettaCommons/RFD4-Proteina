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

## How many metrics ship enabled

`configs/metrics/all_metrics.yaml` already registers **46 metrics, oracles, pass criteria and
filters**, and
they run automatically. No code, no new config files. Most sit behind one or more **`metrics_tags`**,
so you turn one on by passing a tag on the command line.

```bash
rfd4-generate \
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

Another **7 metric classes already live in the codebase** without a default config. They cover TMol
interface ddG, H-bonds and electrostatics, an atom pair distance constraint used for catalytic
geometry, a QM ligand interaction energy, charge complementarity, and a
few others. Turning one on takes **no Python**, just a few lines of YAML, exactly like the [worked
example](advanced/custom-metrics.md#worked-example-an-existing-metric-your-own-config). See [Ready-made
but opt-in](../reference/metrics.md#ready-made-but-opt-in).

Write a new `Metric` subclass only when none of the existing metrics cover what you need. See [Writing your
own](advanced/custom-metrics.md#writing-your-own).

## Running metrics inline

:::{note}
Refolding metrics and reward guided search both use the **AF3 oracle**, which runs only inside an
apptainer container or a native install, configured through the cluster env files. See the
[Developer Guide](../contributing/developer-guide.md). Plain generation needs none of it. For oracle setup, the
task specific tags below, and how a design can carry more than one refold, see the [refolding
guide](refolding.md).
:::

Pass `metrics_tags` during inference and metrics get computed right after each design. Results land
in `per_example_metrics.csv` next to the CIF files.

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
| `antibody` | Antibody contacts and CDR torsions, templated refolding, and antibody pass criteria |
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
