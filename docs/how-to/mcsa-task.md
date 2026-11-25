# M-CSA: take a motif, build a protein, and evaluate it

The M-CSA task asks whether a model can build a protein that holds an enzyme's
catalytic residues in a specified three-dimensional arrangement. M-CSA means the
[Mechanism and Catalytic Site Atlas](https://www.ebi.ac.uk/thornton-srv/m-csa/):
it annotates catalytic residues, chemical roles, cofactors, and mechanisms. It
supplies reference biology; a benchmark must additionally define the inputs,
sampling budget, and success criteria.

```text
Experimental enzyme → catalytic motif → generated structure and sequence
                     → independent structure prediction → geometry and confidence
```

RFD4-Proteina generates sequence and all-atom structure jointly. A separate
inverse-folding step is not required for the commands below. If comparing a
backbone-only method, declare its sequence-design budget separately and fix
the required catalytic identities during inverse folding.

## 1. Specify the motif

Record the M-CSA entry, PDB snapshot and assembly, chain/residue numbering,
conditioned atom names, fixed amino acid identities, ligand chemistry, and
total designed protein length. Preserve the reference-to-design mapping in the
output annotations. Never infer correspondence by matching output residue
numbers to native numbers.

Distinguish **catalytic residues** from **supporting motif residues**. A
32-residue motif can include two catalytic residues and 30 residues defining
their local structural environment. Here, 8–32 describes the motif, not the
length of the whole designed protein.

Backbone scaffolding fixes backbone segments. Atomic active-site scaffolding
can instead specify functional-group atoms and leave their backbone placement
and sequence indices free. These are different tasks. The
[RFdiffusion2 AME benchmark](https://www.nature.com/articles/s41592-025-02975-x)
uses M-CSA-derived atomic motifs. This repository's existing inputs preserve
37 of its 41 targets; see the [input notes](../../data/benchmarks/mcsa/README.md)
for exclusions and chemistry.

The three existing variants are:

| Input directory under `data/benchmarks/mcsa/` | Task |
| --- | --- |
| `mcsa_37` | Catalytic motif plus ligand, with the authored ligand conditioning |
| `mcsa_37_fixed_ligand` | Same target with ligand pose fixed |
| `mcsa_37_no_ligand` | Protein catalytic motif without ligand |

Select the variant explicitly. A ligand-free result cannot establish that a
substrate fits or retains the required pose.

## 2. Generate a protein around an existing M-CSA motif

Complete [installation](../installation.rst), then run the setup preflight:

```bash
bash skills/rfd4-proteina-setup/scripts/preflight.sh
```

Smoke-test one target at batch size 1, using a fresh output directory:

```bash
rfd4-generate \
  inputs=data/benchmarks/mcsa/mcsa_37/M0664_2dhn.cif.zst \
  num_replicates=1 batch_size=1 seed=0 \
  recycle_strategy=medium/midpeak \
  metrics_tags='[design,ame_refolding]' \
  save_chain_of_interest=true out_dir=runs/mcsa-smoke
```

Refolding requires the configured AF3 runtime and weights; see
[refolding setup](refolding.md#set-up-the-oracle). Inspect the output structures
and metric columns before increasing the replicate count. A completed process
without oracle columns is not a successful refolding evaluation.

## 3. What computational refolding tests

The oracle predicts a structure from the generated sequence. The ordinary AF3
oracle is untemplated: it does not receive the motif or design as a protein
structural template. Ligand-containing inputs still provide molecular context.
Templated, holo, and apo predictions must be reported as distinct protocols.

Computational refolding tests sequence–structure consistency. It does not
simulate the physical folding process or measure experimental stability.
Soluble expression, folding, substrate binding, and catalytic activity require
experimental validation; native active-site geometry alone does not establish
the required protonation states, dynamics, or reaction energetics.

For already generated structures:

```bash
rfd4-evaluate inputs=runs/mcsa-smoke/inference_samples \
  metrics_tags='[design,ame_refolding]' out_dir=runs/mcsa-evaluation
```

## 4. Interpret the metrics

| Measurement | Meaning |
| --- | --- |
| Scaffold RMSD | Does the sequence recover the generated overall fold? |
| Motif backbone RMSD | Does the original backbone motif survive prediction? |
| Catalytic-atom RMSD | Are the catalytic functional groups recovered? |
| Worst segment / largest atom displacement | Does an average hide a failed part? |
| Catalytic distances and angles | Are specific mechanistic relationships preserved? |
| Clashes and confidence | Is the arrangement plausible and confidently predicted? |
| Distinct solutions | Did generation find multiple scaffold families? |

Always declare the atom selection and alignment. Fit disconnected motif
segments **together with one rigid transformation**. Fitting each segment
independently would hide incorrect relative placement.

The existing `ame.pass` combines backbone RMSD ≤2 Å, conditioned protein motif
all-atom RMSD ≤1.5 Å, oracle minimum interface PAE <1.5, and zero polymer–ligand
clash fraction at the configured 1.5 Å distance cutoff. These are this
repository's configured criteria; inspect
`configs/metrics/pass_criteria/all_pass_criteria.yaml` for the exact keys.
The no-ligand criterion omits the ligand checks and interface confidence.

The curriculum below uses separate `curriculum_backbone.pass` and
`curriculum_site_geometry.pass` labels. Its scaffold measurement is Cα-only;
the existing `refolded_backbone_rmsd` uses protein backbone atoms. Do not mix
them in a single reported success rate.

For comparison, [MotifBench](https://arxiv.org/abs/2502.12479) evaluates scaffold
Cα RMSD ≤2 Å and motif N/Cα/C RMSD ≤1 Å after independent sequence design and
prediction. Its default eight-sequence-per-backbone protocol differs from
RFD4's direct sequence generation and AF3 evaluation here.

## 5. Run the 8–32-residue curriculum

The versioned [task manifest](../../data/benchmarks/mcsa/curriculum/README.md)
contains real residue selections from experimental 2DHN, the reference for
M-CSA example M0664. It includes seven fragmented tasks and seven contiguous
controls. No structure download is needed.

| Level | Motif residues | Fragment sizes | Intended additional challenge |
| --- | --- | --- | --- |
| 1 | 8 | 8 | Hold one local structural element around Glu22 |
| 2 | 12 | 6 + 6 | Assemble two anchors, including Lys100 |
| 3 | 16 | 6 + 5 + 5 | Position three coupled fragments |
| 4 | 20 | 5 + 5 + 5 + 5 | Accommodate a broader local environment |
| 5 | 24 | 6 × 4 | Connect six short structural islands |
| 6 | 28 | 7 × 4 | Preserve seven islands in a common frame |
| 7 | 32 | 8 × 4 | Assemble eight islands while maintaining the anchors |

The native coordinates define the input geometry. Only selected protein
fragments enter generation. No product, waters, other subunits, or native
scaffold are retained. Glu22 is present throughout; Lys100 is present in
fragmented levels 2–7. Supporting residues have fixed backbone coordinates but
redesignable identities; included catalytic anchors also have fixed observed
side-chain coordinates and identities. All segments remain indexed and in
native sequence order. All use one absolute coordinate frame.

Build the tasks without starting inference:

```bash
python -m script_utils.mcsa_curriculum build \
  --manifest data/benchmarks/mcsa/curriculum/tasks.json \
  --out-dir runs/mcsa-curriculum --length 200 --replicates 100 --seed 0
```

The installed alias is `rfd4-mcsa`. The builder verifies the reference checksum
and writes:

- `inputs.json`: all 14 runnable condition specs, with fixed-length linkers.
- `run_manifest.json`: task definitions, source checksum, seed, budget, and every
  planned attempt, including attempts that may never produce an output.
- `generate.sh`: the exact generation command for review and execution.

Every protein has 200 residues, including the motif. Remaining residues are
distributed as evenly as possible across linkers and both termini. Protein
length, recycle schedule, sampling budget, and evaluation gates are constant
across levels. These placements are a reproducible initial protocol, not an
optimized search over linker lengths.

First build a separate smoke-test directory with `--replicates 1`, then inspect
and execute its generated command after preflight. For the full run:

```bash
bash runs/mcsa-curriculum/generate.sh
```

This requests 1,400 designs and their refolding evaluations. The command uses
batch size 1 and the medium/midpeak recycle schedule. It retains the default
pre-oracle filters; filtered designs remain failed attempts in reporting.
The generated ledger uses condition-spec example IDs and a single generation
job. Do not change the number of replicates, add job suffixes, or replace the
spec with a CIF directory while using that ledger.

## 6. Curriculum scoring and reporting

The opt-in metric groups leave the existing AME defaults unchanged:

```text
metrics/refolding_metrics=mcsa_curriculum
metrics/pass_criteria=mcsa_curriculum
metrics_tags=[design,motif_scaffold_refolding,mcsa_curriculum]
```

`curriculum_geometry` compares each refold to the original conditioned
coordinates, after fitting all motif N/Cα/C atoms together. It reports:

| Output suffix | Definition / gate |
| --- | --- |
| `scaffold_ca_rmsd` | Generated versus refolded Cα coordinates, separately fitted; ≤2 Å |
| `motif_bb_rmsd` | Original versus refolded motif N/Cα/C; ≤1 Å |
| `worst_segment_bb_rmsd` | Largest fragment RMSD in the shared motif frame; ≤1.5 Å |
| `catalytic_rmsd` | Conditioned catalytic side-chain heavy atoms in that frame; ≤1.5 Å |
| `max_catalytic_displacement` | Largest catalytic-atom error in that frame; ≤2 Å |
| `max_catalytic_distance_error` | Largest absolute error over catalytic-atom pair distances; ≤1 Å |

Backbone success uses the first two gates. Site-geometry success uses all six
and requires zero sequence-adherence violations. These latter thresholds are
declared curriculum screens, not the published AME definition. Atom names are
matched literally, so equivalent side-chain atom swaps can cause conservative
failures. When an oracle supplies a structure stack, diagnostics are averaged
over its members, consistent with `StructureComparisonMetric`; fix the oracle
sampling settings when comparing runs.

Pairwise distance recovery screens relative geometry but does not evaluate
mechanism-specific angles, protonation, substrate accessibility, or ligand
clashes in these apo tasks. For a mechanistic benchmark, specify those
additional relationships and their tolerances before running designs. Keep
confidence and the existing clash diagnostics alongside the curriculum gates.

Summarize results against the planned denominator:

```bash
python -m script_utils.mcsa_curriculum summarize \
  --run-manifest runs/mcsa-curriculum/run_manifest.json \
  --metrics runs/mcsa-curriculum/generation/inference_metrics/per_example_metrics.csv \
  --output runs/mcsa-curriculum/summary.csv
```

Multiple metric CSVs can follow `--metrics`. Duplicate or unexpected example
IDs are rejected. Missing rows and nonfinite required metrics cannot pass.
Every task appears even when all its attempts fail. The report separates
planned attempts, available metric rows, evaluated structures, backbone
successes, and site-geometry successes. Individual measurements remain in the
input per-example CSVs. If supplied with FoldSeek-clustered metrics, it also
counts distinct successful clusters; an empty count means unavailable. See
[clustering](advanced/foldseek.md) for the clustering protocol and threshold.

Retain the effective Hydra configuration, code/checkpoint revision, oracle
version and sampling settings, and hardware/cost alongside the run manifest.
Do not tune the gates on the held-out results.

## 7. Establish the actual difficulty ordering

Larger motifs are not automatically harder: extra context can make scaffolding
easier. This ladder intentionally increases fragmentation as well as length;
it is not a nested sequence of constraint sets. Its contiguous controls isolate
some of the effect of length, but contain only one catalytic anchor.

Pilot the full series with fixed baseline methods and equal budgets. Compare
success rates and distinct solutions, retain failed attempts, and repeat with
additional seeds. Reclassify tasks that consistently violate the intended
ordering. Add independently curated enzymes to each level before treating the
series as a general benchmark. This initial one-enzyme curriculum ships input
validation and scoring infrastructure, not measured difficulty or activity.
