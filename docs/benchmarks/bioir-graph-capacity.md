# BioIR graph cache capacity

Full-workflow inference (`inference_backend=bioir`) defaults
`execution.bioir.max_graphs` to `null`: reserve each distinct recycle depth for
a full batch and a tail, giving 10 slots with `thinking_medium`. Explicit values
remain actual slot limits and must fit the selected schedule. More lengths or
batch layouts can cause eviction; retained graphs consume GPU memory.

## Historical block-level measurements

The results below predate full-workflow inference and were measured on a
per-trunk block path that no longer exists. Their default of 1 is not the
automatic full-workflow budget, and they are not full-workflow speed claims.

`execution.bioir.max_graphs` bounded exact-shape CUDA graphs retained by each
converted trunk, defaulting to **1**. An explicit value of **3** could retain
B1, B2 and B4 shapes for one token length, avoiding repeated capture when a
request ended with a singleton tail. Additional lengths still caused LRU
eviction, and more retained graphs consumed GPU memory.

A matched A100-SXM4-80GB comparison used fresh processes on one allocation,
identical source and weights, seed 20260921, a 120-token unconditional input,
five replicas and 60 steps/185 trunk passes. All 180 designs passed structural,
identity and independent-row RNG checks. Warm medians use three uninstrumented
requests; separate diagnostic requests measured graph activity.

| Maximum batch | Groups | Capacity 1 | Capacity 3 | Speedup |
|---|---|---:|---:|---:|
| 1 | 1+1+1+1+1 | 15.796 s | 15.833 s | 0.998x |
| 2 | 2+2+1 | 13.930 s | 13.641 s | 1.021x |
| 4 | 4+1 | 12.659 s | 12.359 s | 1.024x |

At B2 and B4, capacity 1 recaptured the flow graph twice and evicted two states
per trunk in each post-warm request. Capacity 3 had zero captures, evictions or
fallbacks; successful flow/encoder/decoder replays were 555/3/3 at B2 and 370/2/2
at B4. After B4, allocated/reserved memory was 2.873/3.582 GB at capacity 1 and
2.922/4.618 GB at capacity 3 (decimal GB). These are allocator endpoints, not
whole-request peaks: tracker warmup resets PyTorch peak counters internally.

The small observed gain does not justify changing the default for every workload.
Sequences were stable for repeated runs within each fixed batch layout and matched
between capacities for that layout; final sequences can differ across layouts.
This single-input shared-triangle experiment does not qualify KF, other lengths,
scientific quality or general deployment memory limits. CPU validation passed
187 tests and 7 subtests; GPU qualification job 36040888 completed successfully.
