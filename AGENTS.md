# AGENTS.md

Quick guide for agents running RFD4-Proteina inference, search, and evaluation.
Use the workflow docs and task skills for details; do not duplicate model internals here.

## Work safely

- Run from the repository root (`.project-root` and `pyproject.toml`).
- Check `git status --short` before edits and preserve user changes. Do not discard, stage,
  commit, stash, reset, or switch branches without explicit instruction.
- Keep generated runs, checkpoints, datasets, caches, credentials, and local environment
  files out of source changes. Do not publish packages or public releases without authorization.
- Preserve existing outputs. Record inputs, effective config, code/checkpoint identity,
  seed, output directory, and job IDs without recording secrets.

## Choose the workflow

Read the relevant skill under `skills/` before acting:

- `rfd4-proteina-setup`: environment, checkpoint availability, and preflight.
- `rfd4-proteina-annotated-cif`: creating or editing conditioned structure inputs.
- `rfd4-proteina-generate`: generation, partial diffusion, and inverse folding.
- `rfd4-proteina-search`: reward-guided branching and selection.
- `rfd4-proteina-evaluate`: refolding, metrics, aggregation, and structural clustering.

Start with `docs/quickstart.rst` and `docs/installation.rst`; task semantics are in
`docs/how-to/input_prep/index.md`, `docs/how-to/run-the-model.md`, `docs/how-to/search.md`, and
`docs/how-to/evaluate-with-metrics.md`.

## Environment and smoke test

Use the existing configured environment. For a new installation, follow the setup skill;
`make setup` can use the internal cluster installer, while `bash setup.sh --native`
explicitly selects a native `.venv`. Do not reinstall a working environment unnecessarily.

Local runs read `.env.local` (fallback `.env`); named clusters read `.env.<cluster>`.
Files are not merged; exported settings take precedence. `DATA_PATH` holds native
checkpoints and optional assets; HF bundles use `HF_HOME`.

Before a large run, pass the setup preflight and generate one representative example:
```bash
source .venv/bin/activate
bash skills/rfd4-proteina-setup/scripts/preflight.sh
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst num_replicates=1 out_dir=./runs/smoke
```
Use a fresh output directory. For the actual task, test its longest input at batch size 1
and verify required oracle metrics before scaling.

## Configuration and inputs

- `rfd4-generate` and `rfd4-evaluate` use Hydra `key=value` arguments, not `--key value`.
  Their roots are `configs/inference.yaml` and `configs/evaluate.yaml`; search also uses generation.
- `configs/inference.yaml` selects the release checkpoint pair on the Hugging Face Hub. A non-null `hf_ckpt_path`
  overrides native checkpoint paths (`hf_ckpt_path=null` selects them); preserve the matching AE and pin HF revisions.
- Use `configs/inference_experiment/TEMPLATE.yaml` for reusable inference configs.
  Preserve defaults ordering and `_self_` precedence; verify composition after edits.
- Prefer benchmark examples or validated annotated CIFs. Preserve bonds, annotations,
  charges, and intended conditioning; do not silently repair incomplete residues or atomize inputs.
  Check designed/fixed regions and chain of interest. Fixing sidechains also fixes identity;
  use backbone-only conditioning when redesigning sequence.
- For search, explicitly select metric providers needed by rewards and pass criteria.
- Keep requested sampling, recycle, conditioning, and metric settings explicit.
  Do not substitute another preset just to make a run succeed.
- Select `recycle_strategy` from the [task schedule table](docs/how-to/run-the-model.md#recycling-schedules).
  Use that task's medium preset unless another budget was requested: front-loaded for PPI/DNA, midpeak for MCSA/small molecules, flat for inverse folding. `thinking_*` alias front-loaded presets, not task-aware defaults.

## Scheduler submissions

- Read the launcher and `scripts/.user_info_<cluster>` before submitting. Use
  `scripts/launch_generate.sh` or `scripts/launch_evaluate.sh` for batch inference/evaluation.
- Preserve `PARTITION`, `CPU_PARTITION`, `ACCOUNT`, `GRES`, `RUNTIME`, `MEM`,
  `CPUS_PER_TASK`, and the exact `SLURM_EXTRA_OPTIONS` unless explicitly overridden.
  Explicit supported user launcher flags take precedence; never silently change QoS or priority.
- State cluster, partition, account, GPU allocation, time limit, and extra options before
  submission. Inspect generated SBATCH directives when available before submitting.

## Verify results

- Outputs are under `inference_samples/` and `inference_metrics/` (or `evaluation_samples/`
  and `evaluation_metrics/`). Use `verbose=true` for diagnostics; use fresh output directories
  when changing settings so existing structures are not silently reused.
- Check generated structures and required metrics, not just job exit status. Report missing
  outputs, failed designs, and skipped oracle checks; retain failures in pass-rate denominators.
- Refolding needs the selected oracle's environment and weights. For structural clustering,
  enable `save_chain_of_interest=true` and use the evaluation skill.
- Keep input/target identity when aggregating results. Distinguish ordinary pass rates from
  clustered pass rates and record the criteria used.
- `compress_outputs=true/false` controls CIF compression. Use supported structure IO;
  never rename compressed bytes to `.cif`.
- For code/config edits, run focused tests and config-composition checks. Never weaken
  regression assertions or regenerate baselines without explicit user approval.
