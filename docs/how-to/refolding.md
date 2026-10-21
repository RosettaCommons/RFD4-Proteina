# Refolding

Refolding re-predicts a design with an independent structure prediction model and checks whether it
agrees with what RFD4-Proteina built. It backs self-consistency RMSD, confidence metrics like pLDDT
and pAE, several pass criteria, and the reward used by [reward guided search](../concepts/search.md). This page
is the practical guide to turning it on, what it costs, and how it is wired together. For why
refolding is its own metric family and how a design can carry more than one refold, see the
[refolding concept guide](../concepts/refolding.md). For the full metric and tag listing, see the
[metrics reference](../reference/metrics.md); for how individual refolds get named, see the
[output format reference](../reference/output-format.md).

## Set up the oracle

Refolding needs a structure prediction **oracle**, which runs in a separate runtime from
the model environment. AF3 and OpenFold3 use the AF3 Apptainer runtime; AF2 also supports
Docker and Pyxis through `AF2_RUNTIME_BACKEND`.

- **AF3** is the default. Set `AF3_IMAGE`, `AF3_MODEL_DIR`, and `AF3_JAX_CACHE_DIR` in your
  environment file, following the paths the download script prints. See
  [installation](../installation.rst).
- **OpenFold3** runs the same oracles on Apache licensed parameters, with no separate runtime beyond
  AF3's own apptainer. Converted OF3 weights are an ordinary AF3 model directory, so the worker just
  reads the provenance file the converter leaves beside them and enables the OF3 architecture
  branches itself. Two ways to get the weights:
  - **Easiest:** rerun `./scripts/download_assets.sh <DATA_PATH>` without `--skip-of3` on a
    host with Apptainer. It fetches the prebuilt OF3 image and weights and prints the `AF3_IMAGE`,
    `AF3_MODEL_DIR`, and `AF3_JAX_CACHE_DIR` values to export.
  - **Manual:** point `AF3_MODEL_DIR` at a directory of converted OpenFold3 weights. The model repository on
    Hugging Face (`nvidia/RFD4-Proteina`) ships them with their provenance file under `openfold3/weights/`.

  Either way, swap the config group with `metrics/refolding_oracles=of3`. That only changes the
  artifact namespace to `of3`, `of3_ppi` and `of3_dna`, so results do not collide with AF3 ones. The
  parameter family follows `AF3_MODEL_DIR` alone, so one job uses one family — run AF3 and OF3 as
  separate jobs and compare afterwards.

- **AlphaFold 2** supports protein refolding through `metrics/refolding_oracles=af2`.
  Set `AF2_IMAGE`, `AF2_MODEL_DIR` and `AF2_JAX_CACHE_DIR` for its separate runtime;
  `bash setup.sh --native` offers AF2 setup. The AF2 group provides untemplated and protein-binder
  oracles; it does not replace the AF3/OF3 DNA or small-molecule oracles.

Plain generation with no refolding tags and structure-only search need none of this.

## Turn it on

Refolding is driven by the same `metrics_tags` mechanism as every other metric. Pass a task specific
tag and the matching oracle, refolding metrics, and pass criteria all turn on together.

```bash
# Unconditional generation: AF3 refolding plus pLDDT and scRMSD
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'

# Protein binder design: templated refolding and interface metrics
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

| Tag | Activates |
| --- | --- |
| `unconditional_refolding` | AF3 refolding plus pLDDT and scRMSD for unconditional generation |
| `ppi_refolding` | AF3 refolding, target templating and interface metrics for protein binder design |
| `antibody` | Antibody contacts, templated refolding and antibody pass criteria; CDR torsions require the opt-in config described in [Running metrics inline](evaluate-with-metrics.md) |
| `dna_refolding` | AF3 refolding, DNA templating and interface metrics for DNA binder design |
| `motif_scaffold_refolding` | AF3 refolding plus motif RMSD for motif scaffolding and enzyme design |
| `internal_coordinate_motif_scaffold_refolding` | Same as above, for internal coordinate conditioned motifs |
| `sm_binder_refolding` | AF3 refolding plus binding metrics for small molecule binder design |
| `ame_refolding` / `ame_no_ligand_refolding` | AF3 refolding for enzyme (MCSA) active site benchmarks |
| `disco_refolding` | AF3 refolding plus ligand centroid displacement, for the DISCO benchmark |
| `apo_refolding` | Folds the designed chain alone; pair it with another refolding tag to also get the holo (whole complex) refold |

A metric or oracle that needs a tag you did not pass is skipped silently, with no error. The same
applies offline, through `rfd4-evaluate inputs=<dir> metrics_tags='[...]' out_dir=<dir>`. See [Running
metrics offline](advanced/offline-metrics.md).

## What runs underneath a refolding tag

See [What runs underneath a refolding tag](../concepts/refolding.md#what-runs-underneath-a-refolding-tag)
for how the oracle, refolding metrics, and pass criteria fit together, and the
[metrics reference](../reference/metrics.md) for every oracle, metric, and pass criterion by name.

In practice, leave pre-oracle filters on — they are your biggest lever on refolding cost — and
disable them only when benchmarking per example, with `~metrics/pre_oracle_filters`.

## A design can have more than one refold

See [A design can have more than one refold](../concepts/refolding.md#a-design-can-have-more-than-one-refold)
for why a design can carry several refolds and how they get addressed.

In practice: pair `apo_refolding` with another refolding tag (`ppi_refolding`,
`unconditional_refolding`, ...) to supply that second refold for a comparison like
`refolded_apo_holo_rmsd`, or the comparison silently drops. See the
[output format reference](../reference/output-format.md) for the full addressing scheme, and
[Comparing two refolds](advanced/custom-metrics.md#comparing-two-refolds) for writing a custom metric that
reads more than one refold.

## Pre-oracle transforms

See [Pre-oracle transforms](advanced/pre-oracle-transforms.md) for substituting off-targets with
`OffTargetSubstitutionTransform` and writing your own transform.

## Refolding in reward guided search

[Reward guided search](../concepts/search.md) can use an oracle's or refolding metric's output as a reward that
prunes candidates mid-trajectory, instead of only scoring finished designs. List both the oracle and
the comparison metric explicitly in `metrics: [...]` — search does not infer the dependency between
them — and the same `oracle:`/`is_refold:`/`exists_refold:` routing constraints from above still
apply, so a reward compares the structures you intend. See [Metric
selection](search.md#metric-selection) for the full configuration.

## Troubleshooting

| Issue | Solution |
| --- | --- |
| AF3 or refolding errors | Check the selected oracle runtime, weights and environment settings in [Set up the oracle](#set-up-the-oracle); structure-only search needs no oracle |
| A refolding metric or pass criterion is silently missing from the CSV | Check the tag that gates it was actually passed in `metrics_tags`; a missing tag skips the metric with no error |
| A refold comparison metric (like `refolded_apo_holo_rmsd`) is missing | You likely need a second refolding tag to supply the other refold it compares against — see [A design can have more than one refold](#a-design-can-have-more-than-one-refold) |
| Running two untransformed oracles together drops `refold.untransformed` | That by-kind address is only unambiguous with one untransformed oracle; address the refold by oracle name instead, per the [output format reference](../reference/output-format.md#refold-addresses) |
| Every design is dropped before the oracle, with no error | A [pre-oracle filter](../reference/metrics.md#pre-oracle-filters) is reading a metric key that never got computed, because the filter's tag gate is looser than the metric's. A filter reading an absent key counts it as a failure, not a skip |

```{toctree}
:hidden:
:maxdepth: 1

advanced/pre-oracle-transforms
```
