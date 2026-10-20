# Inference backends

`rfd4-generate` runs the native model by default (`inference_backend=native`, compiled or eager
according to `compile_lengths`). `inference_backend=bioir` runs the whole workflow through BioIR
instead, with the same input and sampling options:

```bash
rfd4-generate inputs=data/benchmarks/monomer out_dir=./runs/bioir inference_backend=bioir
```

BioIR is built for CUDA 13. Install it with `pip install "rfd4-proteina[metrics-cuda13,bioir]"`; the cu13
Docker image ships it, and the cu12 and cpu images stay native. It runs the released checkpoints,
including the 147-class autoencoder, wide recycle conditioning and the checkpoint's LoRA adapters (folded
into the base weights when the model loads).

BioIR runs feature representations, sampling and the autoencoder; input parsing, structure
formatting, metrics and reward adapters stay native. It needs `bionemo-ir` 0.2.0rc0 or later, which provides
the RFProteína model (`graph_status`, `reset_graphs`, `configure_graphs`); older runtimes are rejected before
anything is converted. It is available from `rfd4-generate`, not from training-time
validation. These options are rejected when BioIR is selected, instead of being ignored: gradient or
classifier-free guidance, trajectory export, LoRA adapters passed through `adapter_path` and native `nn.*` overrides.

Native beam search (`search.checkpoints`) runs through BioIR's own beam search, scored with the
native metric and oracle managers and the per-checkpoint `metrics` and `rewards` you configure. BioIR
supports one branching width, rollout scoring, a last checkpoint at the end of the schedule and a scalar
minimum for saved intermediates; `use_clean_prediction`, `continuation_criteria`, per-checkpoint
`n_branch`, `save_states` and `resume_from` are rejected. By default BioIR draws its own noise,
so results are not bitwise identical to the native backend; `bioir.match_reference_rng=true` draws in
native's order, which is what a one-to-one comparison needs. Under ODE sampling from identical noise, on
the released checkpoints in bf16 (144 PPI designs), the median aligned CA RMSD between the backends is
0.17 A after one step, 2.2 A after four and 0.8 A after sixty: bf16 kernel differences move trajectories
between basins after the first few steps and then neither grow nor heal. Compare designs by distribution,
not one to one. The first velocity prediction from identical noise is the exception, and `test_bioir_first_step_velocity_matches_native` pins it.

BioIR takes no padding buckets: `compile_lengths` is ignored, and each request runs at its own
length. CUDA graphs are captured per `(batch, tokens)` shape. The settings live under `bioir:` in
`configs/inference.yaml`:

| Key | Meaning | Default |
| --- | --- | --- |
| `bioir.cuda_graph` | Capture CUDA graphs for the velocity trunk | `true` |
| `bioir.cuda_graph_max_tokens` | Requests above this many tokens run the trunk eagerly (graph buffers grow with the square of the length); `null` keeps BioIR's limit | `511` |
| `bioir.max_graphs` | Graph slots; `null` reserves two (a full batch and its tail), an explicit value below two is rejected | `null` |
| `bioir.attention_backend` | BioIR attention kernel; `null` selects its default | `null` |
| `bioir.match_reference_rng` | Draw noise and feature permutations in the native backend's order, so one seed gives both backends the same noise; for accuracy comparison, costs an extra draw per design | `false` |

## What BioIR refuses rather than approximates

The bridge raises instead of silently running a different algorithm. It rejects:

- an installed BioIR that lacks what the workflow needs, checked before anything is converted: the model
  API (`graph_status`, `reset_graphs`, `configure_graphs`), the config fields `tail_steps`,
  `tail_step_multiplier`, decoder sequence-conditioning and `match_reference_rng`, and the random-stream
  hooks (`normal_draws`, `permutations`); and `RFProteinaVFStepConfig.tsr_scaled_score_drift`, which stochastic VF-TSR sampling uses;
- a checkpoint whose residue vocabulary or recycle conditioning the installed BioIR cannot convert. The
  released 147-class AE and wide recycle conditioning convert on current BioIR, and are refused up front,
  by name, only on a runtime that predates them;
- guidance, trajectory export, non-beam search and `search.continuation_policy` other than `stop`.

