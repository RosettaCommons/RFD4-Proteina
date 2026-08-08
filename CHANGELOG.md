# RFD4-Proteína v0.0.1

---

Version 0.0.1 is the first tagged release of RFD4-Proteína. It broadens what you can score a design
against — an AF2 refolding oracle, OpenFold3 parameters, and pre-oracle transforms that refold each
design apo, holo, or against its off-targets — adds conditioning-adherence checks that catch a design
before it reaches the oracle, and replaces the hash-sharded output tree with one readable directory
per design.

Documentation links point at the [docs site](https://rosettacommons.github.io/RFD4-Proteina/latest/).

### Change log

Below is a selection of the user-facing changes since `4d92f51` (2026-07-23). Training-only work,
CI, and internal refactors are omitted.

### New features

* A database-free AF2 refolding oracle with ColabFold-compatible `initial_guess` and `pct_seq_mask`
  knobs, configured in `configs/metrics/refolding_oracles/af2.yaml`. It defaults to no initial guess
  and no sequence masking; set `pct_seq_mask: 0.15` for Superfold parity
  ([#186](https://github.com/RosettaCommons/RFD4-Proteina/pull/186),
  [#204](https://github.com/RosettaCommons/RFD4-Proteina/pull/204)) — see the
  [metrics guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/metrics.html).
* Refolding with OpenFold3's Apache-licensed parameters, selected with
  `metrics/refolding_oracles=of3`; results land under their own `of3` keys so they do not collide
  with AF3 ones. It runs through the same speed-optimised AF3 runtime and warm daemon, so there is no
  second install to maintain ([#125](https://github.com/RosettaCommons/RFD4-Proteina/pull/125)).
* Pre-oracle transforms refold *variants* of each design — apo/holo, or against off-target receptors
  — through a general transform applied before the oracle runs, supported by both the AF3 and AF2
  oracles ([#145](https://github.com/RosettaCommons/RFD4-Proteina/pull/145),
  [#216](https://github.com/RosettaCommons/RFD4-Proteina/pull/216)).
* Two new metrics report whether the model honoured what you asked for — `sequence_adherence` (the
  fraction of sequence-conditioned residues whose identity matches) and `unindexed_insertion_mapping`
  (whether every unindexed motif token was placed, on a residue of the right identity) — each backed
  by a default-on pre-oracle filter that drops violating designs before the expensive fold. Disable
  pre-oracle filters when benchmarking per-example, since dropping designs early inflates the success
  rate among survivors. Thanks [@angiechen918](https://github.com/angiechen918)
  ([#182](https://github.com/RosettaCommons/RFD4-Proteina/pull/182)).
* Oracles emit `min_ipae_<chain_i>_<chain_j>` and per-pair iPTM alongside the aggregate interface
  numbers, so on a complex with more than two chains the tightest interface no longer hides the rest.
  Pass criteria gain `min_of` / `max_of` / `mean_of` for "either of these interfaces is good enough"
  thresholds ([#195](https://github.com/RosettaCommons/RFD4-Proteina/pull/195)) — see the
  [custom metrics guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/custom_metrics.html).
* RMSD metrics resolve indistinguishable atoms (e.g. tyrosine ring flips, carboxylate and phosphate
  oxygens, guanidinium nitrogens) before scoring, so designs are no longer penalised for how the
  ground truth happens to be labelled
  ([#148](https://github.com/RosettaCommons/RFD4-Proteina/pull/148)).
* `checkpoint_overrides` lets a beam search prune on a cheap reward early on and switch to an
  expensive one — refolding, say — at a later checkpoint, with each override inheriting anything it
  does not name from the base reward
  ([#190](https://github.com/RosettaCommons/RFD4-Proteina/pull/190)) — see the
  [search guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/search.html).
* `generation.dataset.sampling_transforms` runs a transform for every sample, so each replicate of a
  CIF input can draw its own conditioning instead of reusing one fixed guess
  ([#196](https://github.com/RosettaCommons/RFD4-Proteina/pull/196)) — see the
  [inference guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/inference.html).
* `/json2cif` becomes `/design2cif` and now accepts either an RFD3-style JSON *or* a paragraph
  describing what you want; the prose route drafts the equivalent spec, flags every value it
  inferred, and only builds after you sign off
  ([#150](https://github.com/RosettaCommons/RFD4-Proteina/pull/150)).

### Usability enhancements

* Outputs now land in a single, human-readable directory per design, with the dataset hash as a short
  `_{hash}` suffix instead of three levels of hash-sharded subdirectories. Anything that globs with
  `rglob`/`**` or follows the absolute paths in the metrics CSV keeps working; only scripts that
  hardcode the old shard depth need updating
  ([#181](https://github.com/RosettaCommons/RFD4-Proteina/pull/181)).
* Design conditions can be written as a readable YAML spec and built into an annotated CIF with
  `scripts/write_condition_spec_cifs.py`, as an alternative to editing annotations by hand
  ([#143](https://github.com/RosettaCommons/RFD4-Proteina/pull/143)) — see
  [annotated CIF inputs](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/annotated_cif_inputs.html).
* The AF3 and AF2 oracles now copy the predicted per-atom confidence into the refolded structure's
  B-factor column, where it was previously left at zero, so a metric can restrict pLDDT to the atoms
  you care about — catalytic residues, a ligand, a single chain — instead of relying on a
  whole-structure mean. Thanks [@yakipnis](https://github.com/yakipnis)
  ([#222](https://github.com/RosettaCommons/RFD4-Proteina/pull/222)).
* `subset_to_keys` keeps only the CIFs you name (by file stem) out of a large input directory, and
  raises on a key that matches nothing rather than silently yielding a smaller dataset
  ([#197](https://github.com/RosettaCommons/RFD4-Proteina/pull/197)).
* Extra generation jobs now shard the example set once replicates are exhausted, instead of going
  unused, so high `gen_njobs` counts stay useful on runs with few replicates per input
  ([#212](https://github.com/RosettaCommons/RFD4-Proteina/pull/212)).
* Six agent skills under `skills/` drive the workflow end to end — setup, inputs, generate,
  reward-guided search, evaluate, and autoresearch — each encoding the repo's real commands, config
  keys, and gotchas ([#104](https://github.com/RosettaCommons/RFD4-Proteina/pull/104)).

### Bug fixes

* Beam search no longer fails at its final checkpoint. The last checkpoint's zero-step rollout
  dropped the unindexed-motif annotations, so every candidate scored `-inf` and the run aborted
  rather than pruning; separately, the final beam was folded through the oracle a second time, which
  could emit one survivor as two designs. Final designs now keep the `_bsc<checkpoint>_<candidate>`
  id they were scored under instead of `_bsfinal_<rank>`. Thanks
  [@aruder2](https://github.com/aruder2) for the diagnosis
  ([#210](https://github.com/RosettaCommons/RFD4-Proteina/issues/210), fixed in
  [#217](https://github.com/RosettaCommons/RFD4-Proteina/pull/217)).
* AF3's attention backend now defaults to cuDNN with the tokamax GLU and layer-norm ops pinned to
  XLA, fixing `RESOURCE_EXHAUSTED: Shared memory size limit exceeded` failures that killed entire
  generation jobs. This affected any card with less than ~108 KB of shared memory per block —
  workstation Ampere, Ada, and Blackwell parts — but not datacenter GPUs such as H100/H200
  ([#191](https://github.com/RosettaCommons/RFD4-Proteina/pull/191)).
* Concurrent SLURM array tasks creating the shared output tree on NFS could die with
  `FileExistsError` despite `exist_ok=True`, silently dropping tasks (8 of 64 in one observed
  inference run). Thanks [@alexshida](https://github.com/alexshida) for the detailed report
  ([#90](https://github.com/RosettaCommons/RFD4-Proteina/issues/90), fixed in
  [#191](https://github.com/RosettaCommons/RFD4-Proteina/pull/191) and
  [#212](https://github.com/RosettaCommons/RFD4-Proteina/pull/212)).
* Partial diffusion now starts at the noise level the state actually holds. The integrator was
  previously told the state was far noisier than it was at small
  `partial_trajectory_fraction_steps`, deforming even absolutely coordinate-conditioned regions while
  the outputs still looked plausible; at fraction 0.05, internal RMSD to the parent drops from 9.6 Å
  to 0.15 Å. A `partial_t` at or above 1 now raises instead of being silently clamped
  ([#209](https://github.com/RosettaCommons/RFD4-Proteina/pull/209)).
* `launch_infer.sh` and `launch_evaluate.sh` overwrote the CPU count already resolved from your
  cluster user info with a hardcoded 8, which blocked nodes and left GPUs idling on digs; all three
  launchers also gain a `--cpus-per-task` flag. Thanks
  [@Jonas-Wilhelm](https://github.com/Jonas-Wilhelm)
  ([#208](https://github.com/RosettaCommons/RFD4-Proteina/pull/208)).

### Documentation

* A new end-to-end tutorial on designing binders that discriminate between close homologs, using
  pre-oracle transforms to fold each design against its off-targets. It ships with executed outputs
  plus an interactive complex browser and a specificity chart per result
  ([#192](https://github.com/RosettaCommons/RFD4-Proteina/pull/192),
  [#201](https://github.com/RosettaCommons/RFD4-Proteina/pull/201)) —
  [read it](https://rosettacommons.github.io/RFD4-Proteina/latest/examples/ppi_specificity_tutorial.html).
* The enzyme design guide is no longer a placeholder: it covers how a theozyme maps onto the `C_CRD`
  / `C_SEQ` / `C_IDX` conditions, the commands for scaffold generation and its inverse-folding,
  partial-diffusion, and search follow-ups, and the output annotations
  ([#120](https://github.com/RosettaCommons/RFD4-Proteina/pull/120)) — see
  [enzyme design](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/enzyme_design.html).
* Both tutorials moved into an Examples gallery, rendering as cards with their saved outputs, and the
  annotated-CIF tutorial gained an interactive selection viewer with one button per thing it fixes
  ([#152](https://github.com/RosettaCommons/RFD4-Proteina/pull/152)) —
  [Examples](https://rosettacommons.github.io/RFD4-Proteina/latest/examples.html).

### Contributors

[@RafiBrent](https://github.com/RafiBrent), [@nscorley](https://github.com/nscorley),
[@maxkazman](https://github.com/maxkazman), [@kierandidi](https://github.com/kierandidi),
[@meneshail](https://github.com/meneshail), [@Jonas-Wilhelm](https://github.com/Jonas-Wilhelm)

With thanks to [@alexshida](https://github.com/alexshida), [@aruder2](https://github.com/aruder2),
[@yakipnis](https://github.com/yakipnis), and [@angiechen918](https://github.com/angiechen918) for
reports and diagnoses that became fixes in this release.

**Full Changelog**: [`4d92f51...v0.0.1`](https://github.com/RosettaCommons/RFD4-Proteina/compare/4d92f5155976d3148ae274e7e876d3a7f4fea67c...v0.0.1)
