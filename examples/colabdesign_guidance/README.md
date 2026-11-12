# Experimental AlphaFold sequence-gradient guidance

This branch connects the existing native RFD4 guidance hook to ColabDesign's
differentiable AF2-multimer binder protocol. It is an integration example, not a
validated binder-design protocol. The normal file-based AF2/AF3 oracle cannot
supply these gradients.

The path is:

```text
clean RFD4 local latents
  -> AE decoder sequence logits (before argmax)
  -> canonical 20-amino-acid logits in AlphaFold order
  -> ColabDesign soft-sequence AF2 loss and d(loss)/d(logits)
  -> explicit first-order PyTorch backward
  -> AE decoder Jacobian -> local-latent sampling update
```

ColabDesign runs `set_seq` followed by `run(backprop=True)`, not `step`: RFD4 owns
the update. The bridge copies logits and gradients through CPU memory; it needs
no cross-framework shared GPU buffers. It preserves the actual scalar loss and
its raw gradient, with no hidden normalization. Nonfinite values fail explicitly.
Negative RFD4 guidance weights minimize the loss. The example uses the
ColabDesign `plddt` loss (1 minus confidence) and its normalized `i_pae` loss;
these are not the reported RFD4 oracle columns or Angstrom pass thresholds.

## Requirements

Use an RFD4 environment set up according to `docs/installation.rst`. Add
[ColabDesign](https://github.com/sokrypton/ColabDesign) and its AF2-multimer-v3
parameters following its installation instructions. This adapter was inspected
against upstream commit `8fd167defd1b0f98e29f00584ff4f3987e6935dc`; pin that revision
when trying the example. The adapter uses `aux['grad']['seq']` and prepared-target
metadata, which are version-sensitive. No new dependency is added to ordinary
RFD4 installations, and no weights are bundled or downloaded by this example.

JAX and PyTorch must coexist in the same Python process. Use a dedicated
environment and check their CUDA compatibility. Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` **before importing JAX** so that JAX does not
reserve most GPU memory before RFD4 can use it. Both networks and their relevant
backward activations must fit; this example does not establish a VRAM bound.

## Matched PD-L1 example

The preparation script uses the existing cropped PD-L1 input. It writes matched
CIF (RFD4) and PDB (ColabDesign) files from the same atoms, plus a spec for a
60-residue binder. The CIF retains bond/charge annotations. Export omits and reports
unresolved non-backbone placeholders, without rebuilding atoms; all 115 target
residues and their resolved backbone coordinates are retained. It refuses to overwrite
existing inputs. Run from the repository root with its configured Python:

```bash
export CLUSTER=local
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python examples/colabdesign_guidance/prepare_example.py \
    --out-dir runs/colabdesign-inputs

rfd4-generate \
    inputs="$PWD/runs/colabdesign-inputs/binder.yaml" \
    out_dir=./runs/colabdesign-guided \
    inference_backend=native batch_size=1 num_replicates=1 \
    recycle_strategy=medium/front_loaded compile_lengths=null seed=42 \
    +guidance=colabdesign_binder \
    guidance.metrics.colabdesign.objective.target_pdb="$PWD/runs/colabdesign-inputs/target.pdb" \
    guidance.metrics.colabdesign.objective.data_dir=/absolute/path/to/af2-parameters
```

Follow the repository generation preflight before this model run. The target
sequence/order is checked against ColabDesign before every loss evaluation.
The binder is selected from `C_SEQ` and chain IDs, not an assumed chain order.
Only one completely sequence-free standard-protein binder and one completely
sequence-fixed standard-protein target are supported. Padding is allowed;
multiple examples, extra chains, mixed polymers, atomization, unindexed motifs,
and partially fixed binder chains are rejected. Use a fixed length matching
`binder_length`; do not use the variable-length default PPI spec unchanged.

The default window is late and short (`t=0.8–0.85`); its weight `-0.001` is an
untuned starting point. Each active sampling step adds an AF2 forward/backward
pass plus an AE decode. Zero weights/out-of-window terms do not load AF2.
For a same-seed control, repeat the command in a fresh output directory without
the guidance group and its two required path overrides. Afterwards evaluate
both outputs with the same independent refolding/filter settings.

## What this does and does not optimize

This is **sequence-mediated guidance**, like the soft-sequence part of
[BindCraft](https://github.com/martinpacesa/BindCraft). ColabDesign folds the
decoded sequence against the fixed target. The current RFD4 backbone is not
passed as an AF2 binder template, and no agreement loss ties the two predicted
poses together. A better AF2 confidence loss therefore does not demonstrate a
better RFD4 pose, experimental binding, or higher pass rate. A pose-consistency
objective would require additional differentiable coordinate/template inputs
and careful alignment, masking and unit handling.

Only `local_latents` receives a direct update in this config. Decoder attention
can couple tokens, so fixed-target latent slots can receive indirect gradients;
normal sequence/coordinate conditioning and flow emit masks still apply. These
conditions are not a promise of rigid preservation of every output coordinate.
The flow trunk itself remains outside the gradient tape: this is the existing
clean-prediction guidance approximation, not differentiation through the entire
denoising trajectory. Discrete amino-acid sidechain choices inside AF2 also use
ColabDesign's existing approximation. Higher-order gradients are unsupported.

BioIR rejects differentiable guidance. Use native vanilla sampling; beam/FK
search is not qualified by this example. The ordinary refolding oracles remain
unchanged. The example is not the complete BindCraft loss/scheduling pipeline.

## Verification

Run the focused CPU tests with the repository environment:

```bash
CLUSTER=local python -m pytest tests/test_colabdesign_guidance.py tests/test_model_masking.py -q
```

The tests check the external-gradient chain rule and descent direction, amino-acid
ordering, masks/unsupported inputs, nonfinite failures, lazy window behavior,
decoder readout wiring, and the ColabDesign API contract using a test double.
They require no AlphaFold weights. They do **not** certify a real AF2 GPU run or
biological improvement; those need the configured ColabDesign environment and
same-seed guided/unguided sampling followed by independent validation.
