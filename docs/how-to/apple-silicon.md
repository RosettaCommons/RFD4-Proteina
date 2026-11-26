# Running on Apple Silicon (MPS)

RFD4-Proteina runs on Apple Silicon through PyTorch's MPS backend. This is an inference path: generation works; training and the BioIR backend are untested on MPS. It is slower than a CUDA GPU and needs
unified memory (about 16 GB for ~550 tokens at batch size 1).

## Install

```bash
bash setup.sh --native      # on an arm64 Mac this installs the `macos` extra (tmol, posebusters, ...)
source .venv/bin/activate
bash skills/rfd4-proteina-setup/scripts/preflight.sh
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst num_replicates=1 out_dir=./runs/smoke
```

The device follows `hardware.accelerator`: the default `gpu` takes CUDA, then MPS, and stops with an error when neither
exists rather than running on the CPU. `auto` falls back to the CPU, and `cpu`, `cuda` or `mps` select that device.
`RFP_DEVICE=cuda|mps|cpu` overrides the config.

## What differs from CUDA

| Area | On MPS |
|---|---|
| Precision | `bf16-mixed` runs as `32-true` (`RFP_MPS_ALLOW_BF16=1` keeps bf16). |
| cuEquivariance | Not available. Attention uses `scaled_dot_product_attention`; triangle multiplication uses the plain PyTorch path. |
| `torch.compile` | Used up to 512 tokens. Longer inputs run eagerly and pad to a multiple of 16 instead of the next bucket (`RFP_MPS_EAGER_ABOVE=<tokens>`, `-1` never forces eager). |
| Unsupported ops | `PYTORCH_ENABLE_MPS_FALLBACK=1` is set so they run on the CPU. |

Measured on one machine, 38_TNFalpha (556 tokens), `medium/front_loaded`, batch size 1: about 440 s and 16 GB.
Run-to-run timing varies by roughly 10%.

## Check a result against the CPU

```bash
python scripts/device_parity.py --seed 0 --out-root runs/parity -- \
    inputs=data/benchmarks/monomer subset_to_examples=monomer-short num_replicates=1 \
    generation.model.bb_ca.gt.mode=ode generation.model.local_latents.gt.mode=ode compile_lengths=null
```

It runs both devices with identical CPU-drawn noise and compares the superimposed CA RMSD. `--devices mps mps
--tokenwise-noise --ref-extra compile_lengths=null` compares eager against compiled on the same device.
