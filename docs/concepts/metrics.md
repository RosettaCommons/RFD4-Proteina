# Metrics

Metrics do not all measure the same object, and they do not only run at the end of generation.
This page covers how to read the metrics system as a whole: what a bad number means depending on
which family produced it, how binder and target selections resolve, and where in a run a metric can
stop further work. 

The [metrics guide](../how-to/evaluate-with-metrics.md) covers turning metrics on and running
them, inline or offline; the [metrics reference](../reference/metrics.md#every-metric-criteria-and-filter-that-ships)
lists every metric, oracle, pass criterion, and filter that ships, with what each one outputs and the
tag that turns it on.

## What each metric is looking at

Metrics do not all measure the same object. Four different things get measured, and knowing which
one a metric looks at tells you what a bad number means.

| Family | Looks at | Answers | Examples |
| --- | --- | --- | --- |
| **Design** | the design alone | Is this a physically sensible structure? | clashes, Ramachandran |
| **Adherence** | design against your input | Did it build what you asked for? | motif RMSD, sequence adherence |
| **Refolding** | the refolded structure alone | Does a folding model believe it? | pLDDT, PAE, pTM |
| **Cross comparison** | input, design and refold together | Does the whole story hold up? | insertion RMSD, chirality |

A bad design metric means the model built something strained. A bad adherence metric means it
ignored part of your specification. Those need different fixes, which is why the split matters.

For the refolding family specifically - the oracle, the AF3/OF3 setup it needs, and how a design can
carry more than one refold — see the [refolding guide](../how-to/refolding.md).

## Binder and target selections

Binder RMSDs use `is_chain_of_interest`; target alignment uses conditioned atoms outside that chain.
Ambiguous inference requires an explicit `chain_iid` or `chain_id` query.
The backbone-aligned variant aligns target backbone atoms but still measures all binder atoms.

## Metrics can stop a run early

Metrics are not only scored at the end. Three points in a run read them and decide what proceeds,
and that is where most of the compute savings live.

| Gate | Sits | Decides |
| --- | --- | --- |
| **Pre-oracle filters** | before refolding | which designs are worth an oracle call at all |
| **Pass criteria** | after refolding | what counts as a success, judging design, oracle and refold together |
| **Search pruning** | inside the trajectory | which candidates survive to the next checkpoint |

Pre-oracle filters save refolding compute by skipping designs that already fail a criterion.
Filtered designs remain in ordinary and clustered pass-rate denominators, so filters can stay
enabled for benchmarks. Report the filters used; disable them only when you want to refold every design.
