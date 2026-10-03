# AGENTS.md

This is the working guide for AI coding agents in RFD4-Proteína. Keep it concise and update it when the repository's commands, layout, or invariants change. The README and `docs/` are the user-facing source of truth; this file explains how to work safely in the codebase.

## Scope and repository boundaries

- The repository is a standalone Python project with a `src/` layout. Run commands from the repository root (the directory containing `.project-root` and `pyproject.toml`).
- Preserve existing user changes. Check `git status --short` before editing and do not discard, stage, commit, stash, reset, or switch branches unless the user explicitly asks.
- Generated runs, checkpoints, caches, downloaded datasets, and local environment files do not belong in source changes.

## Setup and environment

Distribution is by cloning and local pip installation. Do not publish this project to PyPI.

RFD4-Proteína uses Python 3.12. Install from the repository root:

```bash
make setup
source .venv/bin/activate
rfd4-generate inputs=data/benchmarks/monomer out_dir=./runs/smoke
```

`make setup` installs a native `.venv` from `pyproject.toml` (Linux x86_64,
CUDA 13), including AtomWorks and the matching metrics extras. Setup creates
`.env.local` from its example only if neither `.env.local` nor `.env` exists;
it preserves existing settings and does not edit shell startup files. Setup downloads
the released checkpoint pair to `DATA_PATH`, defaulting to `./data_path` and
adding that default to the active environment file when no path is configured.
Existing checkpoints are reused. `bash setup.sh --skip-weights` installs only
the environment; `bash setup.sh --venv` adds development tools.

Runtime configuration:

- Installed commands read `.env.local` (falling back to `.env`) for unset/local `CLUSTER`, or `.env.<cluster>` for named clusters. Files are not merged.
- Exported shell settings take precedence over environment-file values.
- `DATA_PATH` is the root for released checkpoints and example/model data.
- Per-user SLURM settings live in ignored `scripts/.user_info_<cluster>` files; `scripts/user_info.example` is the template.
- Never commit secrets, credentials, or machine-specific path overrides.

Use `rfd4-generate`, `rfd4-evaluate`, `rfd4-train`, and `rfd4-train-ae` from the
activated environment, or by an explicit path such as `.venv/bin/rfd4-generate`.
Batch jobs use the launchers under `scripts/`. Their cluster argument defaults to
`$CLUSTER` or `local`; a local scheduler profile is optional. Unconfigured partition
and account directives are omitted.

## Launching jobs and preserving scheduler intent

Scheduler choices belong to the user. Never silently replace or “improve” a requested partition, account, QoS, priority, reservation, constraint, GPU type, time limit, or other SLURM option.

- Read the selected `scripts/.user_info_<cluster>` and the relevant launcher before constructing a command. Treat profile values as intentional.
- Preserve `PARTITION`, `CPU_PARTITION`, `ACCOUNT`, `GRES`, `RUNTIME`, `MEM`, `CPUS_PER_TASK`, and `SLURM_EXTRA_OPTIONS` unless the user explicitly supplies an override.
- An explicit user-provided launcher flag has precedence over the profile. Do not add `--partition`, `--gres`, `--time`, `--mem`, or `--cpus-per-task` merely because another value seems faster or more available.
- Priority/QoS-like settings have no common dedicated launcher flag; profiles pass them through `SLURM_EXTRA_OPTIONS`. Preserve that string exactly and do not append a competing directive.
- `launch_train_latents.sh` and `launch_train_ae.sh` take scheduler resources from the profile. `launch_generate.sh`, `launch_evaluate.sh`, and `launch_sweep.sh` support explicit resource flags after loading the profile. Read other launchers individually; do not assume identical precedence.
- Before a real submission, state the resolved cluster, partition, account, GPU allocation, time limit, and extra SLURM options. If the generated SBATCH file is available before submission, inspect its directives.

## Validation commands

Prefer the smallest relevant validation first. CPU, GPU, regression, cluster, and very-slow tests have different environment requirements.

```bash
# Targeted tests
pytest tests/path/to/test_file.py -q
pytest tests/path/to/test_file.py::test_name -q

# Local suite through the repository's libstdc++ workaround
make test

# CI-equivalent CPU marker selection (CI runs this inside the configured image)
pytest -m "not benchmark and not very_slow and not gpu and not regression" tests

# Check formatting and lint without modifying files
ruff format --check .
ruff check .

# Apply the repository formatter/linter fixes
make format
```

`make format` changes files; review its diff and avoid formatting unrelated user work.

GPU/regression tests require the appropriate image, GPU, data paths, and often `SOTA_CKPT_PATH`; use the markers and helpers under `.github/ci/`. Do not weaken regression assertions or regenerate baselines merely to obtain a passing run. First determine whether behavior, expectations, or the environment is wrong; baseline changes require explicit user intent.

Documentation is Sphinx/MyST:

```bash
uv pip install -r docs/docs_requirements.txt
make -C docs html
```

Treat Sphinx warnings, broken references, and failed autodoc imports as failures to investigate.

## Main entry points and configuration

Most public workflows are Hydra applications. Their CLI arguments use Hydra override syntax (`key=value`, `group=choice`, `+new_key=value` when appropriate). External evaluation is an `argparse` CLI.

| Workflow | Python entry point | Base config | Cluster launcher |
| --- | --- | --- | --- |
| Latent model training | `src/rfproteina/train.py` | `configs/training_proteina.yaml` | `scripts/launch_train_latents.sh` |
| Autoencoder training | `src/rfproteina/partial_autoencoder/train.py` | `configs/training_ae.yaml` | `scripts/launch_train_ae.sh` |
| Generation/search | `src/rfproteina/generate.py` | `configs/inference.yaml` | `scripts/launch_generate.sh` |
| Standalone evaluation | `src/rfproteina/evaluate.py` | `configs/evaluate.yaml` | `scripts/launch_evaluate.sh` |

Other useful launchers include `launch_infer_ae.sh` and `launch_sweep.sh`. Read a launcher's header before changing or invoking it; launchers resolve Hydra configs on the host, then submit container-aware SLURM jobs.

Hydra conventions:

- `configs/training_proteina.yaml` and `configs/training_ae.yaml` are composition roots, not experiment snapshots.
- Only release aliases and maintained tutorials live under `configs/experiment/` and `configs/experiment_ae/`; dated release recipes live under
  `configs/releases/flow/` and `configs/releases/ae/`.
- User-facing inference composition starts at `configs/inference.yaml`, usually via `configs/inference_experiment/`.
- Reusable groups live under `configs/nn*`, `datasets`, `dataloaders`, `design_tasks`, `generation`, `guidance`, `metrics`, and `callbacks`.
- Blob-backed sources share `configs/datasets/sources/backend_blob.yaml`; keep storage/loader defaults there and source-specific paths, filters, columns, and sampling policies in their respective configs.
- PDB source configs inherit named filters and additive column groups from `configs/datasets/sources/pdb_common.yaml`; filters override by key and use `null` to disable an inherited rule.
- Preserve defaults-list ordering. `_self_` placement controls precedence and is intentional.
- Required values use `???`; do not replace them with machine-specific defaults.
- Use OmegaConf/Hydra resolvers and environment interpolation instead of hard-coded cluster paths.
- When adding a config option, trace every composition path that needs it and add a config-composition test when precedence or compatibility matters.

## Architecture map

### Data and conditioning

`src/rfproteina/datapipes/pipeline.py` assembles AtomWorks transforms into training and inference pipelines. AtomWorks parses structures into Biotite `AtomArray` objects; transforms annotate, crop, atomize, condition, encode, and finally convert samples to tensors. RFD4-Proteína-specific transforms live in `src/rfproteina/transforms/`.

Important invariants:

- A token may represent a polymer residue or one atomized atom; atom and token axes are not interchangeable.
- Atom37 tensors, token annotations, bond features, masks, and loss weights must stay aligned after cropping, atomization, and unindexing.
- Conditioning constants and enums are centralized in `src/rfproteina/conditions.py`; condition resolution and input parsing live in `utils/condition_resolution.py` and `inference_utils/`.
- `ValidateBatchInvariants` and pipeline tests encode cross-feature assumptions. Update them deliberately when the schema changes.
- New coordinate groups are materialized just before unindexing. `C_CRDGRP.set_annotation` preserves complete saved `C_INTCRD` or raises. Training edits use the default; reserve `on_internal_coord_mismatch="clear"` for inference and input preparation that intentionally edit saved targets. Modify `C_CRD` before changing group 0, and remap independent inputs before concatenation.
- Structure inputs and outputs are primarily mmCIF. Preserve annotations and bonds when transforming `AtomArray` objects.
- Use `utils/structure_files.py` for codec-independent discovery, IDs, and existing-path lookup. Reject ambiguous sibling encodings and preserve dots in IDs.
- Generation and standalone evaluation default to plain CIF; training validation defaults to Zstd level 3. Override with `compress_outputs=true/false`. Use AtomWorks IO directly; never rename compressed bytes to `.cif`.

### Models and training

`src/rfproteina/proteina.py` defines the Lightning `Proteina` module. It combines:

- the partial autoencoder loaded from `src/rfproteina/partial_autoencoder/`;
- `ProductSpaceFlowMatcher` from `flow_matching/`;
- `LocalLatentsTransformer` from `nn/`;
- metric, search, reward, and differentiable-guidance integration used during validation/inference.

`src/rfproteina/train.py` constructs datasets/loaders, the model, callbacks, weight-loading/freezing policies, and the Lightning trainer. Optimization is manual inside `Proteina`; changes to accumulation, clipping, Muon/Adam stepping, EMA, or checkpoint resume behavior require focused tests.

The autoencoder has its own training and inference paths under `partial_autoencoder/`. Keep latent dimensionality and checkpoint compatibility explicit: the latent model derives dimensions from the loaded autoencoder and supports deliberate weight-loading policies in `utils/weights.py`.

### Generation, search, and evaluation

`verbose` controls routine warnings and diagnostic dumps: default false for generation, true for training, training-time validation, and standalone evaluation. Propagate it to workers. Shared numerical guards should handle undefined metrics explicitly.

`generate.py` composes inputs, loads checkpoints, configures `Proteina` for inference, and writes samples/metadata. `inference_utils/` owns input/condition parsing, insertion mapping, symmetry, datasets, aggregation, and solve-rate helpers. `search/` implements search strategies; `rewards/` scores trajectories; `guidance/` applies differentiable metric guidance during sampling.

Metrics implement the interfaces in `src/rfproteina/metrics/metric.py` and are composed through `configs/metrics/`. Some metrics require external models, executables, GPUs, or daemon processes (AF2/AF3); make their requirements and skip/failure behavior explicit. Generated samples and metrics live in separate subdirectories under the configured run root.

## Coding conventions

- Name branches with a standard change-type prefix and a concise kebab-case description, such as `feat/add-new-metric`, `fix/checkpoint-loading`, `test/config-composition`, `chore/update-docs`, `refactor/metric-manager`, or `hotfix/broken-release`.
- Format commit subjects as `<type>: <imperative summary>` using the same change types, for example `feat: add confidence metric` or `fix: preserve checkpoint dtype`.
- Follow existing local patterns before introducing a new abstraction. Keep changes focused and avoid speculative framework code.
- Use type annotations on production code. Use `jaxtyping` where named tensor axes materially clarify the contract.
- Prefer clear module-level functions for stateless operations; use classes where state, Lightning/Hydra interfaces, or transform composition require them.
- Public APIs need concise Google-style docstrings. Explain tensor shapes, masks, coordinate frames, units, and mutation when they are not obvious.
- Comments and docstrings describe only current behavior, current invariants, and current interfaces. Do not record old bugs, previous implementations, migration history, or superseded reasoning; write as though the current design has always been the design.
- Keep consecutive `#` comment blocks to at most two lines. Put longer API or behavioral explanation in a short docstring; put broad design material in `docs/`. Prefer no comment when clear names and structure already explain the code.
- Avoid lazy imports unless they are required for optional dependencies, import cycles, or startup cost and the reason is documented.
- Preserve device, dtype, and batch semantics. Avoid implicit CPU tensors or in-place mutation of shared batch/`AtomArray` state.
- Raise actionable errors for invalid configuration or scientific inputs. Do not silently fall back when that could change a scientific result.
- Do not hard-code checkpoint versions in multiple places. `configs/inference.yaml` is the current source of truth for the released latent checkpoint.
- Ruff is configured in `pyproject.toml` (120 columns, double quotes, import sorting, annotations). Do not apply the root Ruff config to the AtomWorks submodule.

## Testing expectations

- Add a regression test with every bug fix when practical.
- Test transforms both in isolation and in a composed pipeline when alignment or annotations can change.
- Use parameterization for meaningful cases; do not duplicate near-identical tests.
- Assert scientific invariants and observable behavior, not private implementation details.
- Keep ordinary CPU tests independent of cluster storage, checkpoints, network, and GPUs. Mark tests accurately when those dependencies are unavoidable.
- For Hydra changes, test composition and at least one relevant entry-point resolution path.
- For checkpoint/schema changes, test strict loading and any explicitly supported compatibility policy.
- Report exactly what was run and what was skipped or unavailable.

## Where to look first

- User setup and first run: `README.md`, `docs/quickstart.rst`, `docs/installation.rst`
- Workflow and input semantics: `docs/workflow/` and `docs/examples.rst`
- Contributor expectations: `docs/contributor_guide.rst`
- Config behavior: the relevant YAML root and Hydra composition through its workflow entry point
- Launch behavior: launcher header, `scripts/common.sh`, `scripts/resolve_config.py`, and `tests/e2e/test_slurm_launchers.py`
- Generation regression flow: `tests/e2e/` and the CI helper under `.github/ci/`

When code and this file disagree, trust executable code and tests, verify the intended behavior in user-facing documentation, and update this guide as part of the same change.
