# Checkpoints reference

RFProteína needs **two** checkpoints, both read from `$DATA_PATH` by default:

| Role | Default path (relative to `$DATA_PATH`) |
| --- | --- |
| Denoiser (latent diffusion, LDM) | `flow_checkpoints/<denoiser>.ckpt` |
| Autoencoder (frozen local latents) | `ae_checkpoints/<autoencoder>.ckpt` |

Internal S3 copies use the same relative keys. Each release file has a `.sha256` sidecar.

Read the active filenames from `configs/inference.yaml`:

```yaml
ckpt_name: <current release denoiser>
ckpt_path: ${oc.env:DATA_PATH}/flow_checkpoints
autoencoder_ckpt_path: ${oc.env:DATA_PATH}/ae_checkpoints/<current release autoencoder>
```

`configs/inference.yaml` is the **runtime source of truth** for "the SOTA model":
`generate.py` loads `ckpt_name` from it by default, and the GPU regression tests
read it too.

`download_example_data.sh` fetches exactly these two names, so a default
`generate.py` run resolves with no CLI override. To use a different checkpoint,
point at it with `ckpt_path=<dir> ckpt_name=<file.ckpt>`.

## Sizes / download

- Public HTTPS, no credentials.
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
