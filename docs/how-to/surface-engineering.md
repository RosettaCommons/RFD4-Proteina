# Scoped surface sequence engineering

These experimental controls retain a reference residue layout and change identities
only at an explicitly selected surface mask. Define that mask once from the reference,
subtract functional residues, ligand contacts, lids and disulfides, and freeze all
remaining identities with `C_SEQ`. Use `C_CRD` for functional geometry and ligand atoms.
Generated coordinates still require auditing: coordinate conditioning is not an exact
coordinate-copy guarantee.

## Probability conditioning

`rfproteina.surface_engineering` validates probability rows and annotates complete
residues with native `C_SEQ_PROB_*` conditions. Omitted entries remain unspecified;
explicit zero specifies zero target probability. Learned conditioning does not
hard-mask emitted residues. A complete distribution must sum to one. Partial rows
may leave probability mass for native completion. Exponential tilting preserves exact
zeros and has an identity operation at strength zero.

`scripts/prepare_surface_experiment.py` prepares the lipase pilot from an audited
manifest. It checks an actual annotated-CIF roundtrip, custom ligand chemistry and
all sequence/coordinate conditions. Formal ligand charges must survive in atom-site annotations and actual featurizer tensors; CCD-template charge alone is insufficient. `uv run python scripts/audit_surface_input_features.py --prepared PREPARED --output feature_audit.json` runs the actual native CPU featurizer for both lipases and fails on a token mapping or ligand-charge mismatch. The pilot checks retained ligand charge, elements, bond graph and covalent attachment at output, plus the exact protected protein heavy-atom/element multiset. Canonicalize Amber HIP aliases before PyRosetta loading: CCD HIP is phosphonohistidine, and renaming its output afterward does not remove added phosphate atoms. This pilot preparation rejects insertion-coded
inputs because the current AtomWorks writer clears insertion codes; remap those inputs
explicitly before using it. Do not reuse its ligand-specific preparation for arbitrary
ligands.

## Flow-trunk logit biases

Bias the sequence logits before the flow-trunk argmax used by the AE decoder:

```bash
rfd4-generate inputs=reference.cif out_dir=biased ~metrics \
  'generation.args.sequence_logit_bias={ASP:0.5,GLU:0.25,ASN:-0.5,GLN:-0.5,GLY:-0.5,SER:-0.5}' \
  'generation.args.sequence_bias_token_indices=[4,9,12]'
```

Indices are **zero-based native model tokens**, not residue numbers. Check the real
featurized mapping before applying an override: atomized ligands also occupy tokens.
Fixed, padded, unselected and illegal tokens are preserved. Hydra dictionary keys in
this CLI override must be unquoted; JSON-quoted keys are not valid Hydra syntax.

## Differentiable guidance

Guidance uses the AE sequence distribution on a separate full-generation path:

```bash
rfd4-generate inputs=reference.cif out_dir=guided ~metrics \
  generation.args.condition_decoder_on_sequence_prediction=false \
  'generation.args.sequence_guidance_token_indices=[4,9,12]' \
  +guidance=surface_preference guidance.weight=-0.05
```

Use an otherwise identical AE baseline with guidance absent. The default objective
`xtr` favours D/E and discourages N/Q/G/S. `guidance.objective=sidechain_entropy`
minimizes a published side-chain entropy-loss proxy instead. This kcal/mol proxy is
relative to Ala at 20°C; it excludes backbone entropy and does not distinguish G→A.
It is not sequence Shannon entropy. Negative guidance weights minimize either cost.
Unselected and fixed state leaves are detached **before** decoding to prevent gradients
from leaking through attention. This path is not supported with trunk-sequence
conditioning enabled.

## Beam selection and auditing

`inference_experiment=surface_search` uses retained-layout partial diffusion (0.35),
two branches and two beams, with checkpoints at steps 10 and 21 of its actual 21-step
trajectory. Provide `surface_selected_keys` as `[chain, residue_number, insertion_code]`
triples and `surface_reference_file`. Compare surface+geometry selection against both
geometry-only (`surface_cost_weight=0`) and random selection (cost and geometry zero,
`surface_random_weight=1`) at the same branch budget. Four terminal generated proposals and four intermediate score evaluations are distinguished in the compute ledger. Protected identities are checked before ranking.

`scripts/run_surface_experiment.py` runs matched inverse/partial arms with an absolute
deadline, per-job commands/source hashes, process-group timeouts and resumable completed
jobs. Its audit rejects identity/mapping violations and nonfinite coordinates. Geometry
gates and surface chemistry are reported separately. `scripts/run_advanced_surface.py`
waits for the basic screen to finish before using the GPU for AE and beam controls. Run module-based orchestrators from the repository root using `uv run python -m scripts.run_advanced_surface` or `uv run python -m scripts.run_surface_confirmation`. Guided jobs use `scripts/surface_guidance_probe.py` to audit raw finite, nonzero gradients and zero force outside the selected mutable scope.

Composition shifts, side-chain entropy proxies and refolding confidence are screening
measurements. Claims about stability, solvent tolerance or activity require experiments.
