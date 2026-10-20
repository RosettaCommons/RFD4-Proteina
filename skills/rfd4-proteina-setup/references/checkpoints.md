# Model files

`configs/inference.yaml` selects the native denoiser and matching AE by default.
Setup downloads them into `DATA_PATH/flow_checkpoints` and `DATA_PATH/ae_checkpoints`.
If `hf_ckpt_path` selects a Hub repository instead, setup caches `config.json` and
`model.safetensors` at `hf_ckpt_revision` in the normal HF cache (`HF_HOME`).
Private Hub access uses `HF_TOKEN` or `hf auth login`.

Select another Hub repository or local bundle with
`hf_ckpt_path=<repo-or-directory>` and `hf_ckpt_revision=<commit>`.
For native checkpoints, set `hf_ckpt_path=null`, `ckpt_path`,
`ckpt_name`, and `autoencoder_ckpt_path` explicitly. Preserve the matching flow/AE pair.
Older namespace checkpoints need the migration described in `docs/packaging.md`.

Optional tutorial data goes into `DATA_PATH`:
`./scripts/download_assets.sh "$DATA_PATH" --with-test-dataset --skip-of3`.
