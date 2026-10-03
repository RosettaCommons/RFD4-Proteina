# RFD4-Proteína v0.0.2

---

Version 0.0.2 adds fold and interaction conditioning, cysteine avoidance, and a new default model.
It improves motif scaffolding and reuse of generated designs, fixes inference with unindexed regions,
and makes interrupted generation and evaluation runs easier to resume.

Below are the user-facing inference and evaluation changes since `v0.0.1`.

### New conditions

* **Choose a fold with CAT conditioning (`C_CATH`).** Request a class, architecture, or topology
  from the CATH hierarchy for a chain or domain, from a broad choice such as `"1.x.x"` to a
  specific topology such as `"2.60.40"`. YAML/JSON condition specs accept
  these labels and validate them before generation
  (PR #488).
* **Request aromatic interactions.** Use `C_PISTACK` for π-stacking or `C_PICATION` for the
  aromatic, cation, or both roles in π-cation interactions. Select complete aromatic rings when
  requesting an aromatic role. These conditions can be supplied through annotated CIFs, the
  Python API, or condition specs
  (PR #488,
  PR #261).
* **Ask selected residues to avoid cysteine (`C_NCYS`).** Apply the condition to residues whose
  sequence is being designed. A new adherence metric reports violations, and the default design
  filter rejects violating designs before refolding. As with all pre-oracle filters, disable
  it for benchmarks that must score every generated sample
  (PR #488).

See [Preparing inputs](https://rosettacommons.github.io/RFD4-Proteina/latest/workflow/inputs.html)
for condition definitions and examples.

### Generation and evaluation

* **A new default checkpoint and sampling recipe.** Generation now uses the September 15 model
  and September 4 autoencoder, with a 60-step sampling schedule and configurable recycling.
  Recycling lets you spend additional model computation at selected stages of generation
  (PR #483,
  PR #417).
* **Refold binders against a cropped target.** Keep the entire designed binder while limiting the
  target to its interface neighborhood, reducing refolding work on large complexes. Cropped
  refolds can be compared against the matching region of the full design
  (PR #260).
* **Classify folds even when Foldseek finds no match.** The CATH metric now falls back to a
  classifier for class, architecture, and topology. You can also select either method alone;
  the classifier requires its additional checkpoint and label files
  (PR #478).
* **Screen developability.** Opt-in metrics report sequence hydrophobicity and instability, plus
  aggregation propensity (SAP) and exposed hydrophobic surface with PyRosetta. Enable them with
  `metrics_tags='[developability]'`; the structure-based scores require a configured PyRosetta
  container (PR #294).
* **Score sensitivity to point mutations.** TMol can estimate changes in interface binding energy
  for specified substitutions, including scans over multiple sites and amino acids
  (PR #255).
* **Save compressed structures.** Set `compress_outputs=true` to write `.cif.zst` designs,
  refolds, and CIF trajectories. Generation and evaluation accept plain, gzip, and Zstandard
  structures; resuming and Foldseek clustering work with compressed outputs. Plain CIF remains
  the default (PR #490).

### Inference bug fixes

* **Unindexed motifs and residue-number gaps are interpreted correctly.** Connected parts of a
  motif retain their connectivity, while separate index groups remain independent. Removing an
  unindexed region no longer leaves incorrect connections in the remaining chain, and inputs
  without an explicit `IndexGroup` annotation receive a default instead of failing
  (PR #244).
* **Backbone-only unindexed motifs can change sequence.** A valid placement is no longer rejected
  because the generated residue has fewer sidechain atoms than the input residue. Thanks
  [@donghyo1225](https://github.com/donghyo1225) for the detailed report
  (issue #248, fixed in
  PR #266).
* **Generated designs retain their original motif constraints.** Reusing a design for partial
  diffusion preserves its requested coordinates and residue identities. Motif RMSDs respect
  independently positioned coordinate groups, and output residue labels preserve the link back
  to the input motif
  (PR #236,
  PR #410,
  PR #452,
  PR #406).
* **Chain-centroid requests stay aligned with the input.** Centering a structure no longer moves
  it away from the supplied centroid condition
  (PR #242).
* **Binder metrics identify the designed chain more reliably**, including co-folding and
  backbone-conditioned targets. Target-aligned binder RMSD uses the target backbone, so changes
  to target sidechains do not distort the alignment
  (PR #317,
  PR #493).
* **More ligand-containing designs can be refolded and scored.** AF3/OF3 refolding handles covalent
  modifiers with leaving atoms, and AF3 preserves standalone glycan ligands. Ligand RASA
  adherence no longer becomes undefined because protein-only atom radii were used
  (PR #479,
  PR #487,
  PR #253). Thanks
  [@Jonas-Wilhelm](https://github.com/Jonas-Wilhelm) for reporting and diagnosing the RASA issue
  (issue #189).
* **Resuming finishes evaluation of existing designs.** With `skip_existing=true`, saved designs
  are evaluated instead of disappearing from the run's results, and cached refolds are reused.
  Empty output files are regenerated, interrupted writes no longer look like completed designs,
  and parallel jobs more reliably share an output directory
  (PR #337,
  PR #492).
* **AF3 refolding works when invoking `generate.py` directly on configured Apptainer systems**, and
  startup failures report their cause. Refolds from different inputs with the same design name
  no longer overwrite one another
  (PR #241,
  PR #354).

### Upgrading from v0.0.1

* **Migrate legacy contig JSON inputs.** Use annotated CIFs or the current YAML/JSON condition-spec
  format. Legacy fields such as `select_fixed_atoms` and `unindex` are no longer accepted;
  JSON itself remains supported through condition specs
  (PR #382). See the
  [condition-spec guide](https://rosettacommons.github.io/RFD4-Proteina/latest/workflow/condition-spec.html).
* **Use `scripts/launch_generate.sh` for batch generation.** The old `launch_infer.sh` alias and
  separate legacy `external_eval.py` command have been removed;
  use the supported `evaluate.py` workflow for standalone evaluation
  (PR #241,
  PR #355).

**Full Changelog**: `v0.0.1...v0.0.2`

---

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
  (PR #186,
  PR #204) — see the
  [metrics guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/metrics.html).
* Refolding with OpenFold3's Apache-licensed parameters, selected with
  `metrics/refolding_oracles=of3`; results land under their own `of3` keys so they do not collide
  with AF3 ones. It runs through the same speed-optimised AF3 runtime and warm daemon, so there is no
  second install to maintain (PR #125).
* Pre-oracle transforms refold *variants* of each design — apo/holo, or against off-target receptors
  — through a general transform applied before the oracle runs, supported by both the AF3 and AF2
  oracles (PR #145,
  PR #216).
* Two new metrics report whether the model honoured what you asked for — `sequence_adherence` (the
  fraction of sequence-conditioned residues whose identity matches) and `unindexed_insertion_mapping`
  (whether every unindexed motif token was placed, on a residue of the right identity) — each backed
  by a default-on pre-oracle filter that drops violating designs before the expensive fold. Disable
  pre-oracle filters when benchmarking per-example, since dropping designs early inflates the success
  rate among survivors. Thanks [@angiechen918](https://github.com/angiechen918)
  (PR #182).
* Oracles emit `min_ipae_<chain_i>_<chain_j>` and per-pair iPTM alongside the aggregate interface
  numbers, so on a complex with more than two chains the tightest interface no longer hides the rest.
  Pass criteria gain `min_of` / `max_of` / `mean_of` for "either of these interfaces is good enough"
  thresholds (PR #195) — see the
  [custom metrics guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/custom_metrics.html).
* RMSD metrics resolve indistinguishable atoms (e.g. tyrosine ring flips, carboxylate and phosphate
  oxygens, guanidinium nitrogens) before scoring, so designs are no longer penalised for how the
  ground truth happens to be labelled
  (PR #148).
* `checkpoint_overrides` lets a beam search prune on a cheap reward early on and switch to an
  expensive one — refolding, say — at a later checkpoint, with each override inheriting anything it
  does not name from the base reward
  (PR #190) — see the
  [search guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/search.html).
* `generation.dataset.sampling_transforms` runs a transform for every sample, so each replicate of a
  CIF input can draw its own conditioning instead of reusing one fixed guess
  (PR #196) — see the
  [inference guide](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/inference.html).
* `/design2cif` accepts existing design artifacts or a plain-language description, interviews for
  unresolved intent, and builds or edits the annotated CIF directly with the Python API
  (PR #150).

### Usability enhancements

* Outputs now land in a single, human-readable directory per design, with the dataset hash as a short
  `_{hash}` suffix instead of three levels of hash-sharded subdirectories. Anything that globs with
  `rglob`/`**` or follows the absolute paths in the metrics CSV keeps working; only scripts that
  hardcode the old shard depth need updating
  (PR #181).
* Design conditions can be written as a readable YAML spec and built into an annotated CIF with
  `scripts/write_condition_spec_cifs.py`, as an alternative to editing annotations by hand
  (PR #143) — see
  [annotated CIF inputs](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/annotated_cif_inputs.html).
* The AF3 and AF2 oracles now copy the predicted per-atom confidence into the refolded structure's
  B-factor column, where it was previously left at zero, so a metric can restrict pLDDT to the atoms
  you care about — catalytic residues, a ligand, a single chain — instead of relying on a
  whole-structure mean. Thanks [@yakipnis](https://github.com/yakipnis)
  (PR #222).
* `subset_to_keys` keeps only the CIFs you name (by file stem) out of a large input directory, and
  raises on a key that matches nothing rather than silently yielding a smaller dataset
  (PR #197).
* Extra generation jobs now shard the example set once replicates are exhausted, instead of going
  unused, so high `gen_njobs` counts stay useful on runs with few replicates per input
  (PR #212).
* Six agent skills under `skills/` drive the workflow end to end — setup, inputs, generate,
  reward-guided search, evaluate, and autoresearch — each encoding the repo's real commands, config
  keys, and gotchas (PR #104).

### Bug fixes

* Beam search no longer fails at its final checkpoint. The last checkpoint's zero-step rollout
  dropped the unindexed-motif annotations, so every candidate scored `-inf` and the run aborted
  rather than pruning; separately, the final beam was folded through the oracle a second time, which
  could emit one survivor as two designs. Final designs now keep the `_bsc<checkpoint>_<candidate>`
  id they were scored under instead of `_bsfinal_<rank>`. Thanks
  [@aruder2](https://github.com/aruder2) for the diagnosis
  (issue #210, fixed in
  PR #217).
* AF3's attention backend now defaults to cuDNN with the tokamax GLU and layer-norm ops pinned to
  XLA, fixing `RESOURCE_EXHAUSTED: Shared memory size limit exceeded` failures that killed entire
  generation jobs. This affected any card with less than ~108 KB of shared memory per block —
  workstation Ampere, Ada, and Blackwell parts — but not datacenter GPUs such as H100/H200
  (PR #191).
* Concurrent SLURM array tasks creating the shared output tree on NFS could die with
  `FileExistsError` despite `exist_ok=True`, silently dropping tasks (8 of 64 in one observed
  inference run). Thanks [@alexshida](https://github.com/alexshida) for the detailed report
  (issue #90, fixed in
  PR #191 and
  PR #212).
* Partial diffusion now starts at the noise level the state actually holds. The integrator was
  previously told the state was far noisier than it was at small
  `partial_trajectory_fraction_steps`, deforming even absolutely coordinate-conditioned regions while
  the outputs still looked plausible; at fraction 0.05, internal RMSD to the parent drops from 9.6 Å
  to 0.15 Å. A `partial_t` at or above 1 now raises instead of being silently clamped
  (PR #209).
* `launch_infer.sh` and `launch_evaluate.sh` overwrote the CPU count already resolved from your
  cluster user info with a hardcoded 8, which blocked nodes and left GPUs idling; all three
  launchers also gain a `--cpus-per-task` flag. Thanks
  [@Jonas-Wilhelm](https://github.com/Jonas-Wilhelm)
  (PR #208).

### Documentation

* A new end-to-end tutorial on designing binders that discriminate between close homologs, using
  pre-oracle transforms to fold each design against its off-targets. It ships with executed outputs
  plus an interactive complex browser and a specificity chart per result
  (PR #192,
  PR #201) —
  [read it](https://rosettacommons.github.io/RFD4-Proteina/latest/examples/ppi_specificity_tutorial.html).
* The enzyme design guide is no longer a placeholder: it covers how a theozyme maps onto the `C_CRD`
  / `C_SEQ` / `C_IDX` conditions, the commands for scaffold generation and its inverse-folding,
  partial-diffusion, and search follow-ups, and the output annotations
  (PR #120) — see
  [enzyme design](https://rosettacommons.github.io/RFD4-Proteina/latest/guides/enzyme_design.html).
* Both tutorials moved into an Examples gallery, rendering as cards with their saved outputs, and the
  annotated-CIF tutorial gained an interactive selection viewer with one button per thing it fixes
  (PR #152) —
  [Examples](https://rosettacommons.github.io/RFD4-Proteina/latest/examples.html).

### Contributors

[@RafiBrent](https://github.com/RafiBrent), [@nscorley](https://github.com/nscorley),
[@maxkazman](https://github.com/maxkazman), [@kierandidi](https://github.com/kierandidi),
[@meneshail](https://github.com/meneshail), [@Jonas-Wilhelm](https://github.com/Jonas-Wilhelm)

With thanks to [@alexshida](https://github.com/alexshida), [@aruder2](https://github.com/aruder2),
[@yakipnis](https://github.com/yakipnis), and [@angiechen918](https://github.com/angiechen918) for
reports and diagnoses that became fixes in this release.

**Full Changelog**: `4d92f51...v0.0.1`
