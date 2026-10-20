# Troubleshooting

This page is an index, not a duplicate: every FAQ and troubleshooting table in the docs lives on
its task page, next to the workflow it covers. Use the links below to jump straight to the
relevant section instead of searching each guide.

- [Setup and training](#setup-and-training)
- [Generation and inference](#generation-and-inference)
- [Refolding and metrics](#refolding-and-metrics)
- [Metric guidance](#metric-guidance)
- [Input preparation](#input-preparation)

## Setup and training

[Development guide: Troubleshooting](../contributing/developer-guide.md#troubleshooting) covers environment and
training issues: missing autoencoder checkpoints, `DATA_PATH`/parquet resolution, training not
resuming, out-of-memory, the rdkit/posebusters segfault workaround, and SLURM launcher failures.

## Generation and inference

[Run the model: Troubleshooting](../how-to/run-the-model.md#troubleshooting) covers
`rfd4-generate`/CLI issues: missing `DATA_PATH` or checkpoints, `atomworks`/`hydra` import errors,
Hydra override syntax mistakes, out-of-memory, the rdkit/posebusters segfault workaround, AF3/
refolding errors, and silent pre-oracle filter failures.

## Refolding and metrics

[Refolding guide: Troubleshooting](../how-to/refolding.md#troubleshooting) covers AF3/refolding
setup errors, metrics silently missing from the output CSV, missing refold comparison metrics,
ambiguous untransformed-oracle addressing, and designs silently dropped before the oracle.

## Metric guidance

[Metric guidance how-to: Troubleshooting](../how-to/advanced/metric-guidance.md#troubleshooting)
covers differentiable-guidance issues: Hydra not finding the `guidance` group, non-differentiable
metrics, guidance terms not found in metric outputs, no visible effect, sample collapse, and
out-of-memory or slowdowns.

See also the [metric guidance concept page](../concepts/metric-guidance.md), which links back to
the same section.

## Input preparation

[Input preparation overview: FAQ and gotchas](../how-to/input_prep/index.md#faq-and-gotchas)
covers annotated CIFs vs. YAML/JSON inputs, opening annotated CIFs in PyMOL and other tools,
fixed sidechains implicitly fixing sequence, conditioning expandable segments, and ligand/PTM
`IndexGroup` membership.

[FAQ: condition specs](../how-to/faq.md) answers narrower questions on YAML/JSON condition specs:
running a fully unconditioned design, rejected `.pdb` inputs, conditions matching 0 atoms,
selections catching expandable placeholders, whether specs are checked before the run, and
whether CIFs must be written to disk. The [YAML/JSON condition spec
reference](yaml-json.md) also links back to this FAQ for common errors.
