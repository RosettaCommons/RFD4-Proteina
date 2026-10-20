# Refolding

Refolding re-predicts a design with an independent structure prediction model and checks whether it
agrees with what RFD4-Proteina built. This page covers what refolding is for, what it produces, and
how a design can end up with more than one refold. For turning it on, setting up the AF3/OF3 oracle,
and the full tag listing, see the [refolding guide](../how-to/refolding.md); for the complete metric
and tag reference, see the [metrics reference](../reference/metrics.md).

## Why refold a design

The [metrics concept guide](metrics.md#what-each-metric-is-looking-at) splits metrics into four
families by what they look at. Refolding sits apart from the other three:

| Family | Looks at | Answers |
| --- | --- | --- |
| Design | the design alone | Is this a physically sensible structure? |
| Adherence | design against your input | Did it build what you asked for? |
| **Refolding** | **the refolded structure alone** | **Does a folding model believe it?** |
| Cross comparison | input, design and refold together | Does the whole story hold up? |

A refold is the output of handing the design's sequence (and, for templated oracles, part of its
structure) to a structure prediction model and asking it to fold that sequence from scratch. If the
refold's confidence is low, or its backbone lands far from the design's own backbone, that is
evidence the sequence does not actually encode the structure RFD4-Proteina drew — independent of
whether the design looks reasonable on its own.

## What runs underneath a refolding tag

Turning on refolding activates three things together, in order:

1. **An oracle** ([reference](../reference/metrics.md#refolding-oracles)) — `oracle`, `oracle_ppi`, or
   `oracle_dna` — which actually calls AF3/OF3 and reports its own confidence outputs
   (`overall_plddt`, `ptm_coi`, `min_ipae_coi`, and friends).
2. **Refolding metrics** ([reference](../reference/metrics.md#refolding-metrics-after-the-oracle))
   that compare the design against that refold, such as `refolded_backbone_rmsd` (self-consistency
   RMSD) or `refolded_motif_allatom_rmsd`.
3. **Pass criteria** ([reference](../reference/metrics.md#pass-criteria)) that threshold those oracle
   and refolding outputs into a single `{criterion}.pass` flag, like `ppi.pass` or `sm_binder.pass`.

Refolding is also the most expensive step in the pipeline. A **pre-oracle filter**
([reference](../reference/metrics.md#pre-oracle-filters)) can reject an obviously broken design
before it ever reaches the oracle — your biggest lever on cost per success, except when benchmarking
per example, where filters should be turned off entirely so you measure the model rather than the
filter.

## A design can have more than one refold

A [pre-oracle transform](../reference/metrics.md#pre-oracle-transforms) can fan a single oracle call
out into several refolds of the same design — for example, folding the designed chain on its own,
separately from the normal "untransformed" refold of the whole complex. When a design has more than
one refold, every refolding metric and pass criterion needs to say *which* refold it reads, so
RFD4-Proteina gives each one an address (`refold.apo`, `refold.oracle_ppi`, ...) and derives
`is_refold:`/`exists_refold:`/`oracle:` tags from it automatically — you never pass those on the CLI
yourself.

This is what lets a metric like `refolded_apo_holo_rmsd` ask "does the binder hold its fold once the
target is gone?" by comparing an apo refold against the whole-complex one. Supplying that second
refold, and pairing the right tags so the comparison doesn't silently drop, is covered in the
[refolding guide](../how-to/refolding.md#a-design-can-have-more-than-one-refold).

See the [output format reference](../reference/output-format.md) for the full addressing scheme, and
[Comparing two refolds](../how-to/advanced/custom-metrics.md#comparing-two-refolds) for writing a custom
metric that reads more than one refold.
