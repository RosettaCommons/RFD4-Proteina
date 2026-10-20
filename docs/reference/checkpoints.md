# Checkpoints reference

This page lists the model-weight checkpoints RFD4-Proteina loads, what each one is for, and the
released files available as of the last update of this documentation resource. For installing and downloading them, see
[Installation](../installation.rst). For the CLI flags that select
them at generation time, see [Inputs and checkpoints](cli-options.md#inputs-and-checkpoints) in the
CLI options reference.

```{note}
"Checkpoint" has a second, unrelated meaning in this codebase: a **search checkpoint**
(`search.checkpoints`) is a point in the generation schedule where beam search (see the
[glossary](glossary.rst)) branches and scores candidates. It has nothing to do with the model
weight files on this page.
```

**Contents**

- [Types of checkpoints](#types-of-checkpoints)
- [Released checkpoints](#released-checkpoints)
- [Downloading](#downloading)
- [Overriding at the CLI](#overriding-at-the-cli)
- [Training-time checkpoint roles](#training-time-checkpoint-roles)

## Types of checkpoints

RFD4-Proteina always loads **two** checkpoints together to generate: a denoiser and the
autoencoder it was trained against. Two more are optional, depending on the workflow.

| Role | Config key(s) | Default source | Used for |
| --- | --- | --- | --- |
| Denoiser (flow-matching latent diffusion model) | `hf_ckpt_path`; native: `ckpt_name`, `ckpt_path` | Hugging Face bundle; native files in `$DATA_PATH/flow_checkpoints/` | Generation, search, and standalone evaluation; always required |
| Autoencoder (frozen local-latents encoder/decoder) | `hf_ckpt_path`; native: `autoencoder_ckpt_path` | Same bundle; native files in `$DATA_PATH/ae_checkpoints/` | Loaded alongside the denoiser for generation; also loaded when training a new denoiser, since it fixes the latent space the denoiser is trained in |
| PEFT adapter (optional) | `adapter_path` | user-supplied directory | Inference only; a local fine-tuning adapter, loaded onto the exact base denoiser/autoencoder pair it was fine-tuned against |
| OpenFold3 refolding oracle (optional) | `AF3_IMAGE`, `AF3_MODEL_DIR`, `AF3_JAX_CACHE_DIR` | `$DATA_PATH/openfold3/` | Refolding/designability metrics only (`metrics/refolding_oracles=of3`); not a generation checkpoint |

`configs/inference.yaml` is the **runtime source of truth** for "the SOTA model": `generate.py`
loads the Hub repository `hf_ckpt_path` at the pinned commit `hf_ckpt_revision` from it by default
(the bundle holds the denoiser and the autoencoder), and the GPU regression tests read it too. The
native `ckpt_name` and `autoencoder_ckpt_path` keys name the same release as local files and apply
when you set `hf_ckpt_path=null`. Do not hard-code checkpoint filenames or revisions anywhere else;
update them there when advancing the release.

## Released checkpoints

The current release pair, as pinned in `configs/inference.yaml`:

| File | Role | Training recipe | What it's for |
| --- | --- | --- | --- |
| `rfd4-proteina-flow-release-2026-10-19.ckpt` | Denoiser | `configs/releases/flow/2026_10_19_hero_pretrain.yaml`, fine-tuned with `configs/releases/flow/2026_10_19_hero_lora_c384.yaml` (the 60k EMA checkpoint) | Current SOTA denoiser. Supports unconditional generation, motif scaffolding, enzyme design, small molecule/protein/DNA binder design, folding, and inverse folding. Accepts generalized sequence-probability conditioning and conditions the autoencoder on its own sequence readout by default during inference (`generation.args.condition_decoder_on_sequence_prediction`). |
| `rfd4-proteina-ae-release-2026-10-19.ckpt` | Autoencoder | `configs/releases/ae/2026_10_15_unified_biomolecule.yaml` | Unified protein, RNA, DNA and modified-residue autoencoder with 32 latent channels per residue, trained to 400k steps. Fixes the latent space the released denoiser was trained in; use this pair together. |

Both files hold exponential-moving-average weights, not raw optimizer checkpoints. The
`configs/releases/flow/` and `configs/releases/ae/` recipes are training configs, kept for
reproducibility: they document how a released checkpoint was produced, not alternative downloads.
Only the pair in `configs/inference.yaml` is published.

## Downloading

`make setup` caches the released model from the Hugging Face repository and revision named in
`configs/inference.yaml` (`nvidia/RFD4-Proteina` by default) in the normal Hugging Face cache, where
`HF_HOME` controls its location. Rerunning setup reuses files already in the cache; a failed
download stops setup with an error.

To download manually (after a pip install or `bash setup.sh --skip-weights`):

```bash
./scripts/download_assets.sh ./data_path --skip-of3
```

Pass your configured `DATA_PATH` if you changed it from the default. Details:

- Everything comes from the Hugging Face model repository: the model bundle (`config.json` and
  `model.safetensors`), the `test_dataset` tarball, the OpenFold3 runtime image and the OpenFold3
  weights. For private repositories, provide `HF_TOKEN` or log in with `hf auth login`.
- Add `--with-test-dataset` to also fetch a tiny training tutorial dataset.
- OpenFold3 is *not* fetched by default: pass neither `--skip-of3` nor an already-configured
  `AF3_IMAGE`, and run on a host with Apptainer installed. See
  [Installation](../installation.rst) for the follow-up
  `AF3_IMAGE`/`AF3_MODEL_DIR`/`AF3_JAX_CACHE_DIR` environment settings.
- Select another Hub repository or a local exported bundle with `hf_ckpt_path`, and pin it with
  `hf_ckpt_revision`.

## Overriding at the CLI

Every generation argument is `key=value`, Hydra style:

```bash
rfd4-generate ckpt_path=<dir> ckpt_name=<file.ckpt> ...
```

See [Inputs and checkpoints](cli-options.md#inputs-and-checkpoints) in the CLI options reference
for `ckpt_path`, `ckpt_name`, `autoencoder_ckpt_path`, and `adapter_path`, with their defaults and
an example of each.

## Training-time checkpoint roles

Training (`rfd4-train`, `rfd4-train-ae`) uses two mutually exclusive checkpoint-loading paths:

| Config key | Role |
| --- | --- |
| `pretrain_ckpt_path` | Start a new run from saved weights only — resets optimizer, scheduler, step, epoch, and RNG state. |
| `resume_ckpt_path` | Resume an interrupted run — restores full Lightning trainer state. |

Two policies, honored only on the `pretrain_ckpt_path` path, control how those weights are applied
when the architecture has changed since the checkpoint was produced:

- `weight_loading` (`rfproteina.utils.weights.WeightLoadingConfig`) — when unset, the checkpoint
  loads with `strict=True` and any key or shape mismatch is a hard error. Set a `default_policy`
  (for example `copy`), a `fallback_policy` (for example `reinit`), and per-parameter
  `param_policies` (`zero_init`, `copy_and_zero_pad`, ...) to deliberately load across an
  architecture change.
- `parameter_freezing` (`rfproteina.utils.weights.ParameterFreezingConfig`) — freezes selected
  parameters by name pattern after loading, for example to freeze a pretrained backbone during
  fine-tuning.
