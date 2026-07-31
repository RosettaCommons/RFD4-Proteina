# Parameter Reference

Every generation parameter, grouped by purpose. All arguments are `key=value` (Hydra style),
never `--key value`. Back to the [Inference Guide](inference.md).

## Essential

The parameters you almost always want to set — their defaults are rarely what you want for a real run.

| Parameter | Description | Default |
| --- | --- | --- |
| `out_dir` | Output directory. Set this to a real path for anything but a quick test. | `./runs/<run_name>` |
| `num_replicates` | Designs generated per input example. `1` is only useful for a smoke test. | 1 |
| `batch_size` | Designs processed in parallel on GPU. Start at 8; confirm your longest input fits at 1 first. | 1 |
| `metrics_tags` | Evaluation metrics to compute inline (Hydra list) — see [Evaluating Results](inference.md#evaluating-results). Empty = none. | `[]` |

## Inputs & checkpoints

| Parameter | Description | Default |
| --- | --- | --- |
| `inputs` | Input file (annotated CIF / JSON / YAML / CIF / PDB) or a directory of them | Required |
| `subset_to_examples` | Run only specific input examples by key (comma-separated) | null |
| `skip_existing` | Skip examples whose output already exists | true |
| `seed` | Random seed (`null` = different each run) | null |
| `ckpt_path` / `ckpt_name` | Override the default checkpoint (dir / filename). Read from `$DATA_PATH` otherwise. | from `$DATA_PATH` |

## Sampling

| Parameter | Description | Default |
| --- | --- | --- |
| `inference_experiment` | Config group controlling generation (nsteps, noise, search, metrics) | `inference_on_user_inputs` |
| `generation.args.nsteps` | Diffusion steps — 100 is plenty with the default `vf_tsr` sampler | 100 |
| `generation.args.self_cond` | Self-conditioning — leave on for best quality | true |
| `generation.args.save_trajectory_every` | Save a trajectory snapshot every N steps (0 = off) | 0 |

## Diversity (stochasticity)

Higher values increase diversity but may reduce designability — see [Inference Noise Parameters](inference.md#inference-noise-parameters).

| Parameter | Description | Default |
| --- | --- | --- |
| `generation.model.bb_ca.simulation_step_params.sc_scale_noise` | Backbone stochasticity | 0.3 |
| `generation.model.local_latents.simulation_step_params.sc_scale_noise` | Sidechain/sequence stochasticity | 0.1 |

## Partial diffusion

Refine an existing structure instead of generating from scratch — see [Partial Diffusion](inference.md#partial-diffusion).

| Parameter | Description | Default |
| --- | --- | --- |
| `generation.args.partial_trajectory_fraction_steps` | Simple interface: run this fraction of `nsteps` (e.g. 0.3 = 30%) | null |
| `generation.args.partial_t_backbone` / `partial_t_latent` | Advanced: per-modality starting corruption level, float in (0, 1) | null |
| `generation.args.couple_partial_t` | Derive one modality's `partial_t` from the other via the schedule | false |

## Inverse folding

Hold the input backbone fixed and resample only the sequence and side chains — see [Inverse Folding](inference.md#inverse-folding). The `generation=inverse_folding` preset sets both keys at once.

| Parameter | Description | Default |
| --- | --- | --- |
| `generation.args.inverse_folding` | Fix the backbone and resample sequence/side chains only | false |
| `generation.args.inverse_folding_method` | Sampling method; `resampled_forward_marginal` is the only supported value | `resampled_forward_marginal` |
