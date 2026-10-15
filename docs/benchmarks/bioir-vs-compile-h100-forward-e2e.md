# Full-workflow BioIR versus Torch compile, forward and end to end

Unconditional generation with the released final checkpoints on one NVIDIA
H100 (96 GB, driver 615.71.09), native `thinking_medium`: 60 diffusion steps and
185 trunk passes. This part is not the 80 GB SXM H100 -- it measures about
2.2 TB/s of copy bandwidth rather than the 3.35 TB/s of HBM3 spec, so
memory-bound kernels read slower here than on an H100-80GB-HBM3. Both backends
used the same inputs, seed 11, no oracle scoring, and `CUTEDSL_FORCE_CUBIN=1`.
Each cell ran a discarded cache-warming process, then a measured process of
three batches; every figure below is the median of batches 2+.

Times are **per batch** -- one `predict_step` over all its rows. Divide by the
batch size for per-design cost.

| N | Batch | BioIR forward | BioIR e2e | Compile forward | Compile e2e | Forward | e2e |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 0.826 s | 1.113 s | 3.628 s | 3.952 s | 4.39x | 3.55x |
| 128 | 2 | 1.077 s | 1.651 s | 3.464 s | 3.972 s | 3.22x | 2.41x |
| 128 | 4 | 1.766 s | 2.793 s | 3.728 s | 4.787 s | 2.11x | 1.71x |
| 400 | 1 | 3.187 s | 4.042 s | 6.074 s | 6.989 s | 1.91x | 1.73x |
| 400 | 2 | 5.737 s | 7.352 s | 11.205 s | 12.788 s | 1.95x | 1.74x |
| 400 | 4 | 11.191 s | 14.285 s | 21.357 s | 24.532 s | 1.91x | 1.72x |

Median 2.11x on forward and 1.74x end to end, over ranges of 1.91-4.39x and
1.71-3.55x.

## What the two columns separate

`e2e` is `Proteina.predict_step`, the seam both backends cross -- BioIR reaches
it through `predict_full_bioir` -- carrying featurisation, the model, and
formatting into atom arrays. `forward` is the model call inside it, scoped to
match: BioIR runs trunk, sampler and decoder in one call, while the native path
integrates the flow and decodes separately, so `full_simulation` and
`sample_formatting` are accumulated. Only that scoping makes one divisible by
the other; timing `full_simulation` alone omits the native decode and overstates
BioIR.

The difference between the columns is 0.29-3.09 s and grows with length and
batch. It is featurisation and formatting, which both backends pay and neither
backend's kernels touch, so it dilutes every ratio: 1.91x forward becomes 1.72x
end to end at 400/4.

## Reading the two lengths differently

At 128 residues the work is launch-bound, and BioIR's lead falls from 4.39x to
2.11x as the batch grows -- batching is what the compiled arm needed to stop
being launch-bound. At 400 residues the lead is flat at about 1.9x across all
three batches, where there is enough arithmetic that the gap is kernel speed.

**400 is off-bucket for the compiled arm.** `compile_lengths` pads it to 448
while BioIR runs 400 natively, so part of every 400-residue row is padding waste
rather than kernel speed. The 128-residue rows are on-bucket for both.

## Limits

Design 1 of each cell is excluded from the medians and reported separately by
the driver. It carries first-touch in the model plus that shape's graph capture
or compilation: 1.09-1.91x of steady for BioIR and 1.45-3.59x for compile. No
runtime warmup removes it, because it depends on the shape.

The driver warms the CUDA context and the BLAS/DNN handles before the timed
region, which is worth 0.42 s of the trunk's first graph warmup on this part and
is paid once per process rather than once per shape. Steady figures are
unaffected by it either way.

Three batches per cell is a thin variance estimate. One process per length and
batch: generation collates across inputs, so pooling lengths into one process
pads a small shape up to a large one at batch > 1 and silently reports a design
at a length it never ran. The driver is
[`benchmark_backends.py`](../../src/script_utils/benchmark_backends.py).
