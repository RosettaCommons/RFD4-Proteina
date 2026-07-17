# Metrics Reference

> **You shouldn't need to hand-parse output CIFs.** It's possible to reconstruct things like the
> unindexed mapping, which chain/residue an inserted region came from, or coordinate-conditioning adherence
> by reading raw CIF fields — but that's the fallback, not the workflow. The preferred path is:
> turn on the **metric(s)** that already compute what you want, and let them run automatically and
> efficiently in the main pipeline (they can also drive [reward-guided search](search.md), which is
> far more sample-efficient than reprocessing outputs and filtering after the fact).

- [How many metrics ship enabled](#how-many-metrics-ship-enabled)
- [Metric reference](#metric-reference)
  - [Design metrics](#design-metrics-inline-pre-refolding)
  - [Structure-prediction metrics](#structure-prediction-metrics)
  - [Refolding metrics](#refolding-metrics-post-oracle)
  - [Refolding oracles](#refolding-oracles)
  - [Pass criteria](#pass-criteria)
  - [Pre-oracle filters](#pre-oracle-filters)
  - [Ready-made but opt-in](#ready-made-but-opt-in)
- [Writing your own](#writing-your-own)
- [Worked example: an existing metric, your own config](#worked-example-an-existing-metric-your-own-config)
- [Standard annotations (not an output JSON)](#standard-annotations-not-an-output-json)
- [Running metrics offline on a folder of structures](#running-metrics-offline-on-a-folder-of-structures)

---

## How many metrics ship enabled

**36 metric / pass-criterion / filter entries are already registered** in
`configs/metrics/all_metrics.yaml` and run automatically — no code, no new config files. Most are
gated behind one or more **`metrics_tags`**, so turning one on is just passing a tag on the command
line:

```bash
python src/rfproteina/generate.py \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'
```

Breakdown of what's already wired in:

| Category | Count | Config |
| --- | --- | --- |
| Design metrics (run inline, pre-refolding) | 14 | `configs/metrics/inline_metrics/design_metrics.yaml` |
| Structure-prediction metrics | 1 | `configs/metrics/inline_metrics/structure_prediction_metrics.yaml` |
| Refolding metrics (design vs. refolded structure) | 6 | `configs/metrics/refolding_metrics/all_refolding_metrics.yaml` |
| Refolding oracles (AF3 variants) | 3 | `configs/metrics/refolding_oracles/af3.yaml` |
| Pass criteria (pass/fail thresholds) | 9 | `configs/metrics/pass_criteria/all_pass_criteria.yaml` |
| Pre-oracle filters (skip the oracle on cheap-check failures) | 3 | `configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml` |

Beyond those, **10 more metric classes already exist in the codebase** but aren't wired into any
default config — they cover TMol interface ddG/H-bonds/electrostatics, an atom-pair distance
constraint metric (used for catalytic geometry), a QM ligand-interaction-energy metric, MACE
ML-potential energies, charge complementarity, and a couple of others. Turning one of these on
needs **no Python** — just a few lines of YAML, exactly like the [worked
example](#worked-example-an-existing-metric-your-own-config) below. See [Ready-made but
opt-in](#ready-made-but-opt-in).

Only write a brand-new `Metric` subclass when none of the ~46 above cover what you need — see
[Writing your own](#writing-your-own).

---

## Metric reference

Tag semantics (see [custom_metrics.md](custom_metrics.md#how-kwargs_to_compute_args-works) for the
full rules): `required_tags_any` — runs if **any** listed tag is passed; `required_tags_all` — runs
only if **all** listed tags are passed; `prohibited_tags` — skipped if a listed tag is passed, even
if the required tags also match. No tags at all = always runs. Design tags and refolding tags are
separate namespaces; pass both if you want both families to run:
`metrics_tags='[design, unconditional_refolding]'`.

### Design metrics (inline, pre-refolding)

Computed directly on the generated structure, before any oracle refolds it.

| Config key | Tag gate | What it measures | Output columns (prefixed `<key>.`) |
| --- | --- | --- | --- |
| `bond_length_metrics` | any: `design`; not: `small_molecule_only`, `structure_prediction` | Fraction of bonds with an unreasonable length (element-dependent thresholds) | `frac_unreasonable` (+ per-category breakdown in verbose mode) |
| `clash_metrics` | any: `design`, `structure_prediction`; not: `small_molecule_only` | Fraction of atoms in a steric clash at a configurable threshold (default 1.5 Å), by clash-type | `frac_atoms_clashing_1p5A` + 5 category breakdowns (intra/inter-residue × polymer/ligand) |
| `hotspot_obedience` | any: `hotspot_conditioning` | Fraction of hotspot atoms with a binder atom within 4.5 Å | `hotspot_obedience` |
| `rasa_adherence` | any: `rasa_conditioning` | Adherence of generated burial to RASA-bin conditioning | `buried_conditioning_adherence`, `intermediate_conditioning_adherence`, `exposed_conditioning_adherence` |
| `interchain_interactions` | any: `interchain_interactions`; not: `small_molecule_only`, `structure_prediction` | Adherence + atom/pair counts for H-bond, π-stacking, π-cation, salt-bridge, hydrophobic, and metal-coordination conditioning, interchain only | `*_conditioning_adherence`, `*_atom_count_coi` / `*_residue_count_coi`, `*_atom_pair_count_coi` / `*_residue_pair_count_coi` per interaction type |
| `sse_distribution` | any: `design`; not: `small_molecule_only`, `structure_prediction` | Helix/sheet/loop fraction and number of contiguous SSE elements (P-SEA) | `helix_fraction`, `sheet_fraction`, `loop_fraction`, `num_ss_elements` |
| `sse_accuracy` | any: `design`; not: `small_molecule_only`, `structure_prediction` | Accuracy of generated secondary structure vs. secondary-structure conditioning | `overall_accuracy`, `helix_recall`, `sheet_recall`, `loop_recall`, `n_conditioned_atoms` |
| `posebusters_metrics` | any: `small_molecule_only` | Ligand geometry/validity checks (PoseBusters) | one pass/fail column per PoseBusters check |
| `chiral_accuracy` | any: `small_molecule_only` | Fraction of stereogenic centers with correct handedness, split tetrahedral vs. non-tetrahedral | `tetrahedral_percent_correct_chirality`, `non_tetrahedral_percent_correct_chirality` |
| `design_motif_allatom_rmsd` | any: `3d_coordinate_conditioning`, `unindexed`, `internal_coordinate_conditioning` | All-atom RMSD of the coordinate-conditioned motif region, self-consistency within the design itself | `rmsd` (median over replicates; per-`CoordinateGroup` when multiple motifs) |
| `design_unindexed_motif_allatom_rmsd` | all: `unindexed` | All-atom RMSD restricted to **unindexed** insertion atoms only — this is RFD3's "insertion RMSD" | `rmsd` |
| `contact_molecular_surface` | any: `cms` | Distance-weighted interfacial contact surface area between binder and target (Lawrence & Colman 1993) | see `rfproteina.metrics.physics.contact_ms.metric` |
| `sequence_similarity_to_input` | any: `partial_diffusion` | Sequence identity vs. the seed structure, for partial diffusion | `sequence_identity_to_input` |
| `cath_metric` | any: `cath` | CATH superfamily classification of the chain of interest via Foldseek | `C`, `A`, `T`, `H`, `type`, `evalue`, `tmscore` |

### Structure-prediction metrics

| Config key | Tag gate | What it measures | Output columns |
| --- | --- | --- | --- |
| `structure_prediction_lddt` | any: `structure_prediction` | All-atom lDDT vs. ground truth (for structure-prediction-style validation runs, not design) | `lddt` |

### Refolding metrics (post-oracle)

Compare the **design** (ground truth) against the **oracle-refolded structure** (predicted).

| Config key | Tag gate | What it measures | Output columns |
| --- | --- | --- | --- |
| `refolded_backbone_rmsd` | any: `unconditional_refolding`, `ppi_refolding`, `dna_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding` | Backbone self-consistency RMSD (scRMSD) between design and refold | `rmsd` |
| `refolded_target_aligned_binder_rmsd` | any: `ppi_refolding`, `dna_refolding` | Binder RMSD after aligning on the fixed target | `rmsd` |
| `refolded_motif_allatom_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding` | All-atom motif RMSD between design and refold (includes unindexed regions) | `rmsd` |
| `refolded_motif_ca_rmsd` | any: `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding` | CA-only motif RMSD between design and refold (includes unindexed regions) | `rmsd` |
| `refolded_backbone_aligned_ligand_rmsd` | any: `sm_binder_refolding` | Ligand RMSD after backbone alignment | `rmsd` |
| `max_ligand_centroid_distance` | any: `disco_refolding` | Max ligand-centroid displacement after backbone alignment | `ligand_centroid_distance` |

### Refolding oracles

An oracle actually runs the (heavy, AF3-based) structure prediction that the refolding metrics
above then compare against. See [Evaluating Results](inference.md#evaluating-results) for the
apptainer/AF3 setup requirement.

| Config key | Tag gate | Notes | Key outputs (prefix `<key>.`) |
| --- | --- | --- | --- |
| `oracle` | any: `af3_refolding`, `unconditional_refolding`, `motif_scaffold_refolding`, `internal_coordinate_motif_scaffold_refolding`, `sm_binder_refolding`, `ame_refolding`, `ame_no_ligand_refolding`, `ame_original_benchmark_refolding`, `disco_refolding` | Untemplated AF3 refold | `overall_plddt`, `overall_ptm`, `overall_iptm`, `plddt_coi`, `ptm_coi`, `pae_coi`, `ipae_coi`, `min_ipae_coi`, `iptm_coi`, `frac_under_1`/`1.5`/`2` (on `min_ipae_coi`) |
| `oracle_ppi` | any: `ppi_refolding` | Target chain templated (fixed) for protein-binder design | same key set, prefixed `oracle_ppi.` |
| `oracle_dna` | any: `dna_refolding` | DNA templated (fixed) for DNA-binder design | same key set, prefixed `oracle_dna.` |

### Pass criteria

Threshold one or more already-computed metric values into a binary pass/fail flag
(`{config_key}.pass`), evaluated after design + oracle + refolding metrics. All are built from the
generic `PassCriterionMetric` — see [Adding your own pass criterion](custom_metrics.md#pass-criteria-config).

| Config key | Tag gate | Thresholds |
| --- | --- | --- |
| `unconditional` | any: `unconditional_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 1.5 |
| `ppi` | any: `ppi_refolding` | `oracle_ppi.ptm_coi` ≥ 80, `oracle_ppi.min_ipae_coi` ≤ 1.5, `refolded_target_aligned_binder_rmsd.rmsd` ≤ 2.5 |
| `dna` | any: `dna_refolding` | `oracle_dna.min_ipae_coi` ≤ 1.5, `refolded_target_aligned_binder_rmsd.rmsd` ≤ 8.0 |
| `motif_scaffold` | any: `motif_scaffold_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 2.0, `refolded_motif_ca_rmsd.rmsd` ≤ 1.0 |
| `internal_coords` | any: `internal_coordinate_motif_scaffold_refolding` | Same thresholds as `motif_scaffold` |
| `sm_binder` | any: `sm_binder_refolding` | `oracle.min_ipae_coi` ≤ 1.5, `oracle.iptm_coi` ≥ 80, `refolded_backbone_rmsd.rmsd` ≤ 1.5, `refolded_backbone_aligned_ligand_rmsd.rmsd` ≤ 5.0 |
| `ame` | any: `ame_refolding`; not: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5, zero inter-residue polymer-ligand clashes |
| `ame_no_ligand` | any: `ame_refolding`; all: `no_ligand` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `refolded_motif_allatom_rmsd.rmsd` ≤ 1.5 |
| `disco` | any: `disco_refolding` | `refolded_backbone_rmsd.rmsd` ≤ 2.0, `max_ligand_centroid_distance.ligand_centroid_distance` ≤ 2.0 |

### Pre-oracle filters

Cheap checks that decide whether a design is even **forwarded to the (expensive) oracle**. All
require an explicit filter tag in addition to their conditioning tag, so none run unless you opt
in — see [`all_pre_oracle_filters.yaml`](../../configs/metrics/pre_oracle_filters/all_pre_oracle_filters.yaml).

| Config key | Tag gate | Threshold |
| --- | --- | --- |
| `ligand_clash_filter` | all: `ligand_clash_filter`; not: `no_ligand` | Zero inter-residue polymer-ligand clashes (from `clash_metrics`) |
| `insertion_rmsd_filter` | all: `unindexed`, `insertion_rmsd_filter` | `unindexed_metrics.insertion_rmsd` ≤ 1.0 — you'll need a metric producing that exact key (e.g. an aliased copy of `design_unindexed_motif_allatom_rmsd`) for this filter to have something to threshold |
| `motif_rmsd_filter` | all: `motif_rmsd_filter` | `motif_rmsd.rmsd` ≤ 1.0 — likewise, needs a metric under the `motif_rmsd` key |

### Ready-made but opt-in

These classes exist and are tested, but aren't included in any default config — add a YAML entry
(see the [worked example](#worked-example-an-existing-metric-your-own-config)) to turn one on, no
Python required.

| Class | Lives in | What it's for |
| --- | --- | --- |
| `AtomPairDistanceMetric` | `rfproteina.metrics.atom_pair_distance_metric` | Configurable atom-pair upper-bound distance constraints — catalytic geometry, motif-contact checks |
| `TMolBlockPairMetric` | `rfproteina.metrics.physics.tmol_block_pair_metric` | General interface-ddG metric between two atom-selection regions; presets for ligand ddG/H-bonds/electrostatics and PPI ddG/Coulomb live in [`tmol_block_pair_metrics.yaml`](../../configs/metrics/inline_metrics/tmol_block_pair_metrics.yaml) (GPU, intentionally not in the default set) |
| `QMInteractionEnergyIPA` | `rfproteina.metrics.ipa_interaction_energy_metric` | QM ligand-protein interaction energy via IPA; see [`ipa_metrics.yaml`](../../configs/metrics/inline_metrics/ipa_metrics.yaml) (commented out of defaults pending a digs fix) |
| `MACEEnergyMetric` | `rfproteina.metrics.physics.mace_energy_metric` | MACE ML-potential energy of a structure |
| `MACEInteractionEnergyMetric` | `rfproteina.metrics.physics.mace_interaction_energy_metric` | MACE-POLAR-1 interaction energy decomposition for protein-ligand complexes |
| `ChargeComplementarityMetric` | `rfproteina.metrics.physics.ddg_salt_metrics` | Salt-bridge charge-complementarity score across a binder-target interface |
| `InternalResidueBondMetric` | `rfproteina.metrics.bond_length_metric` | Per-residue bond lengths vs. CCD ideal geometry |
| `NucleicSSSimilarityMetrics` | `rfproteina.metrics.nucleic_ss_metrics` | Nucleic-acid secondary-structure (base-pairing) conditioning adherence |
| `NoCysteineAdherenceMetric` | `rfproteina.metrics.no_cysteine_metric` | Fraction of not-cysteine-conditioned residues that are actually not `CYS` |
| `ContractCAMetric` | `rfproteina.metrics.contract_ca_metric` | Trivial differentiable guidance metric (contract CA toward the origin); mainly for exercising the search/guidance path |

---

## Writing your own

If none of the ~46 metrics above cover what you need, **[custom_metrics.md](custom_metrics.md)**
walks through writing a brand-new `Metric` subclass from scratch: the three metric categories
(design / refolding / pass-criterion), the `kwargs_to_compute_args` routing, registering it in a
git-ignored config so nothing you write is ever committed by accident, and — if it's generally
useful — contributing it back.

---

## Worked example: an existing metric, your own config

Often you don't need new Python at all — an existing metric just needs **your** parameters. Say
you're doing enzyme design and want to check catalytic-triad geometry: `AtomPairDistanceMetric`
(see [Ready-made but opt-in](#ready-made-but-opt-in)) already does exactly this — evaluate one or
more atom-pair upper-bound distance constraints — you just need to tell it *which* atoms and
*what* distances.

**Step 1 — write the config.** Custom metrics live in the git-ignored
`configs/metrics/custom_metrics/` directory (scaffolded by `setup.sh` on first run — see
[custom_metrics.md](custom_metrics.md)):

```yaml
# configs/metrics/custom_metrics/all_custom_metrics.yaml
# @package metrics

inline_metrics:
  catalytic_geometry:
    _target_: rfproteina.metrics.atom_pair_distance_metric.AtomPairDistanceMetric
    # No required_tags_* set -> always enabled, on every generated design.
    constraints:
      - name: triad_his_asp
        max_distance_angstrom: 3.2
        atom_a: { res_id: 31, res_name: HIS, atom_name: NE2 }
        atom_b: { res_id: 14, res_name: ASP, atom_name: OD1 }
      - name: oxyanion_cys
        max_distance_angstrom: 3.0
        atom_a: { chain_id: B, res_name: lpr, atom_name: O29 }
        atom_b: { chain_id: A, res_name: CYS, atom_name: N }
```

Each `atom_a`/`atom_b` is an `atom_selection_from_spec` query (`chain_id`, `res_name`, `res_id`,
`atom_name`, or a raw biotite `query` string); give a list instead of a dict to check the minimum
distance over several alternative atoms. Omitting `required_tags_any`/`required_tags_all` means the
metric has no gate and runs on every design — appropriate here since a design either has a catalytic
triad or it doesn't (the metric reports `inf`/violated if the named atoms are missing, rather than
erroring).

**Step 2 — run.** Hydra dict-merges `configs/metrics/custom_metrics/all_custom_metrics.yaml` into
the built-in metrics config automatically (it's already a default in
[`all_metrics.yaml`](../../configs/metrics/all_metrics.yaml), listed last so it merges on top) — new
keys are **added**, and a key that matches an existing built-in name gets **deep-merged over it**
(e.g. redeclaring `clash_metrics:` with just a new `threshold:` overrides that one field and inherits
everything else). So the normal run command picks it up with no extra flags:

```bash
python src/rfproteina/generate.py \
    inputs=enzyme_design.cif num_replicates=50
```

Results land in `per_example_metrics.csv` as `catalytic_geometry.triad_his_asp_distance`,
`catalytic_geometry.triad_his_asp_satisfied`, `catalytic_geometry.n_satisfied`,
`catalytic_geometry.all_satisfied`, etc. — see the
[`AtomPairDistanceMetric` docstring](../../src/rfproteina/metrics/atom_pair_distance_metric.py) for the
full output set (per-constraint distance/violation/satisfied/soft-score, plus aggregate
`soft_min_score`, `total_violation`, `worst_violation`).

---

## Standard annotations (not an output JSON)

Older workflows expected an explicit output JSON alongside the generated CIF — carrying things like
the unindex mapping, which input contig an inserted region came from, or which atoms were
coordinate-conditioned. **That JSON no longer exists.** The same information is now written as
**standard annotations directly into the output CIF** (`rfproteina.utils.standard_annotations`), so
it round-trips with the structure itself instead of living in a side-channel file that can drift out
of sync:

| Standard annotation | CIF column | What it tells you |
| --- | --- | --- |
| `SA_SRC` | `src-component` | Per-atom source-component contig label (e.g. `A3-10`) — which input region/component this atom came from |
| `SA_CRDSRC` | `src-coordinate-conditioned-region` | Per-atom source contig for the coordinate-conditioned region specifically (`"."` = not conditioned) |
| `SA_CRDGRPSRC` | `src-coord-group` | Integer `CoordinateGroup` id the atom maps to (`-1` = not mapped) — how motif RMSD metrics subdivide multi-motif designs |
| `SA_CWI` | `coords-with-insertion` | The ground-truth coordinate to compare against for coordinate-conditioned atoms (what `design_motif_allatom_rmsd` etc. align/score against) |
| `SA_UII` | `is-unindexed-insertion` (alias `is_unindexed_insertion`) | Per-atom boolean — `True` marks the atom as an unindexed-insertion target (what `design_unindexed_motif_allatom_rmsd` restricts to) |

These are exactly the fields the motif/unindexed RMSD metrics above already consume — the whole
point of turning on `design_motif_allatom_rmsd` / `design_unindexed_motif_allatom_rmsd` instead of
reading the CIF by hand is that the pipeline does this bookkeeping for you.

If you do want to explore the annotation format directly (e.g. for a bespoke analysis a built-in
metric doesn't cover), the runnable **[annotated CIF tutorial
notebook](tutorials/annotated_cif_tutorial.ipynb)** is the best reference — it builds up the same
annotation system these outputs use, atom by atom.
Programmatically, parse an output CIF with the `STANDARD_ANNOTATIONS_ONLY` preset
(`rfproteina.utils.parse_presets`) to load all of the above with minimal intervention.

---

## Running metrics offline on a folder of structures

Metrics don't require re-running generation — score any folder of already-generated structures with
`evaluate.py` (same Hydra config, so `metrics_tags` and any custom-metric config apply exactly as
above):

```bash
python src/rfproteina/evaluate.py \
    inputs=/path/to/structures \
    metrics_tags='[unconditional_refolding]'
```

`inputs` is scanned recursively for CIF-like files. Pass `task_csv=<csv with designed_structure_path,
task columns>` if different structures need different `metrics_tags`/pass-criteria treatment;
otherwise every structure is scored the same way. This is the same script the [Developer
Guide](development.md#evaluation-and-metrics) points to for custom-metric development — write a
metric once and use it both inline during generation and offline over existing outputs.

For large batches split across SLURM array jobs (`launch_infer.sh --njobs N`), each job writes its
own `per_example_metrics_job{N}.csv`; concatenate them afterward with:

```bash
python -m rfproteina.inference_utils.aggregate_metrics ./inference/my_run/metrics/
```

This also runs automatically as a dependent job when using `launch_infer.sh`. See [Evaluating
Results](inference.md#evaluating-results) for FoldSeek diversity clustering and other
`aggregate_metrics` options.
