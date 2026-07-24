# Checkpoints reference

RFProteína needs **two** checkpoints, both read from `$DATA_PATH` by default:

| Role | Default path (relative to `$DATA_PATH`) |
| --- | --- |
| Denoiser (latent diffusion, LDM) | `latent_checkpoints/ldm_checkpoints/<denoiser>.ckpt` |
| Autoencoder (frozen sidechain latents) | `latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt` |

Config wiring (`configs/inference.yaml`):

```yaml
ckpt_name: 2026_07_22_release_280k-ema.ckpt
ckpt_path: ${oc.env:DATA_PATH}/latent_checkpoints/ldm_checkpoints
autoencoder_ckpt_path: ${oc.env:DATA_PATH}/latent_checkpoints/ae_checkpoints/AE_ucond_512.ckpt
```

`configs/inference.yaml` is the **single source of truth** for "the SOTA model":
`generate.py` loads `ckpt_name` from it by default, and the GPU regression tests
read it too.

## The checkpoint-name mismatch (read this)

`download_example_data.sh` fetches:

- denoiser: `arch_sweep_2026_07_03_pair_trimult_every2_720k-ema.ckpt`
- autoencoder: `AE_ucond_512.ckpt`

but `configs/inference.yaml` defaults `ckpt_name` to the **release** checkpoint
`2026_07_22_release_280k-ema.ckpt`. If you only ran the example download, the
default name will not resolve. Two options:

1. Obtain the release checkpoint and place it in `ldm_checkpoints/`, or
2. Override on the CLI to use the example checkpoint you downloaded:

```bash
python src/rfproteina/generate.py \
    inputs=data/benchmarks/monomer.json out_dir=./runs/smoke \
    ckpt_name=arch_sweep_2026_07_03_pair_trimult_every2_720k-ema.ckpt
```

You can also point at an entirely different directory with
`ckpt_path=<dir> ckpt_name=<file.ckpt>`.

## Sizes / download

- ~4 GB each (~8 GB total), public HTTPS, no credentials.
- Host override: `RFPROTEINA_DATA_URL` (default `https://YOUR_DATASET_STORE`).
- Add `--with-test-dataset` to also fetch a tiny training set:
  `./scripts/download_example_data.sh "$DATA_PATH" --with-test-dataset`.

## IPD/DIGS

On DIGS the shared data dir + checkpoints + apptainer SIF are wired in
`.ipd/.env.digs`; choose `1) digs` during `setup.sh` and skip the manual
`DATA_PATH` / download steps.
