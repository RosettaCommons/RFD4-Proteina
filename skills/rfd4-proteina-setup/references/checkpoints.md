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

`download_example_data.sh` fetches exactly these two names, so a default
`generate.py` run resolves with no CLI override. To use a different checkpoint,
point at it with `ckpt_path=<dir> ckpt_name=<file.ckpt>`.

## Sizes / download

- ~4 GB each (~8 GB total), public HTTPS, no credentials.
- **Public bucket: `YOUR_DATASET_STORE`** — everything published for
  external use (both checkpoints, the `test_dataset` tarball, the CCD mirror)
  lives there and is served anonymously at
  `https://YOUR_DATASET_STORE`. The internal `YOUR_DATASET_STORE`
  and `YOUR_DATASET_STORE` buckets return `403` without private credentials — never
  point external users at them.
- Host override: `RFPROTEINA_DATA_URL` (default
  `https://YOUR_DATASET_STORE`).
- Add `--with-test-dataset` to also fetch a tiny training set:
  `./scripts/download_example_data.sh "$DATA_PATH" --with-test-dataset`.

## IPD/DIGS

On DIGS the shared data dir + checkpoints + apptainer SIF are wired in
`.ipd/.env.digs`; choose `1) digs` during `setup.sh` and skip the manual
`DATA_PATH` / download steps.
