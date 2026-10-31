# Refolding

Refolding re-predicts a design with an independent structure prediction model and checks whether it
agrees with what RFD4-Proteina built. It backs self-consistency RMSD, confidence metrics like pLDDT
and pAE, several pass criteria, and the reward used by [reward guided search](../concepts/search.md). This page
is the practical guide to turning it on, what it costs, and how it is wired together. For why
refolding is its own metric family and how a design can carry more than one refold, see the
[refolding concept guide](../concepts/refolding.md). For the full metric and tag listing, see the
[metrics reference](../reference/metrics.md); for how individual refolds get named, see the
[output format reference](../reference/output-format.md).

## Set up the oracle

Refolding needs a structure prediction **oracle**, which runs in a separate runtime from
the model environment. AF3 and OpenFold3 use the AF3 Apptainer runtime; AF2 also supports
Docker and Pyxis through `AF2_RUNTIME_BACKEND`.

- **AF3** is the default. Set `AF3_IMAGE`, `AF3_MODEL_DIR`, and `AF3_JAX_CACHE_DIR` in your
  environment file, following the paths the download script prints. See
  [installation](../installation.rst).
- **OpenFold3** runs the same oracles on Apache licensed parameters, with no separate runtime beyond
  AF3's own apptainer. Converted OF3 weights are an ordinary AF3 model directory, so the worker just
  reads the provenance file the converter leaves beside them and enables the OF3 architecture
  branches itself. Two ways to get the weights:
  - **Easiest:** rerun `./scripts/download_assets.sh <DATA_PATH>` without `--skip-of3` on a
    host with Apptainer. It fetches the prebuilt OF3 image and weights and prints the `AF3_IMAGE`,
    `AF3_MODEL_DIR`, and `AF3_JAX_CACHE_DIR` values to export.
  - **Manual:** point `AF3_MODEL_DIR` at a directory of converted OpenFold3 weights. The model repository on
    Hugging Face (`nvidia/RFD4-Proteina`) ships them with their provenance file under `openfold3/weights/`.
  - **No container:** set `AF3_RUNTIME_BACKEND=native` and run the worker from a second venv
    instead of the image — see [Running the oracle without a container](#running-the-oracle-without-a-container).

  Either way, swap the config group with `metrics/refolding_oracles=of3`. That only changes the
  artifact namespace to `of3`, `of3_ppi` and `of3_dna`, so results do not collide with AF3 ones. The
  parameter family follows `AF3_MODEL_DIR` alone, so one job uses one family — run AF3 and OF3 as
  separate jobs and compare afterwards.

- **AlphaFold 2** supports protein refolding through `metrics/refolding_oracles=af2`.
  Set `AF2_IMAGE`, `AF2_MODEL_DIR` and `AF2_JAX_CACHE_DIR` for its separate runtime;
  `bash setup.sh --native` offers AF2 setup. The AF2 group provides untemplated and protein-binder
  oracles; it does not replace the AF3/OF3 DNA or small-molecule oracles.

Plain generation with no refolding tags and structure-only search need none of this.

### Running the oracle without a container

The image never supplied the fold logic — the CIF-to-input adapter and the batch script are checked
into this repository and bind-mounted in. All it carries is the pinned `alphafold3` package, its
compiled extensions, and `jax`. So on a host with no Apptainer you can run the same worker under a
second virtualenv, which makes the whole path pip-installable end to end:

1. `uv pip install "rfd4-proteina[all-cuda12]"` — the model and the metrics stack
   (see [Installation](../installation.rst)); use `all-cuda13` to match a CUDA 13 driver.
2. `./scripts/download_assets.sh "$DATA_PATH" --native-of3` — the oracle weights, no image.
3. A second venv holding the fork, below.

Steps 2 and 3 are the only oracle-specific ones, and neither needs Docker, Apptainer or root:

```bash
uv venv .venv-af3 --python 3.12
uv pip install --python .venv-af3 \
  'alphafold3-open @ https://github.com/kierandidi/alphafold3/releases/download/v3.0.5.post1/alphafold3_open-3.0.5.post1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'
.venv-af3/bin/build_data        # generates the CCD tables the package loads at import
./scripts/download_assets.sh "$DATA_PATH" --native-of3   # weights only, ~1.4 GB, no image
```

`jax[cuda12]` is pinned as a dependency of `alphafold3-open`, so it arrives with the
wheel; do not install it separately. `build_data` is a console script the package
installs, and it must run once per venv — the wheel ships the CCD generators, not the
generated tables. It parses the bundled CCD and writes `ccd.msgpack`,
`chemical_component_sets.msgpack` and `ccd_codes.txt` (51,537 components) next to the
installed package; it takes about 30 seconds. Skipping it leaves the first fold to fail
with `FileNotFoundError` on `chemical_component_sets.msgpack`.

```bash
export AF3_RUNTIME_BACKEND=native
export AF3_PYTHON="$PWD/.venv-af3/bin/python"
export AF3_MODEL_DIR="$DATA_PATH/openfold3/weights"
export AF3_JAX_CACHE_DIR="$DATA_PATH/openfold3/jax_cache"
```

Everything downstream is unchanged: same warm-worker daemon, same manifest protocol, same
`--num_recycles` and attention-backend flags, same provenance-driven selection of the OF3
architecture branches. `AF3_IMAGE` is not read on this route.

Two things to know:

- **It needs the OF3-capable fork, not upstream AlphaFold3.** The adapter sets
  `global_config.of3_weights`, which DeepMind's package has no field for — converting the parameters
  is not enough, because most AF3/OF3 differences have no on-disk form. The flag gates OF3 branches
  across the evoformer, diffusion head, diffusion transformer and atom cross-attention, so the fork
  (`kierandidi/alphafold3`, distributed as `alphafold3-open`) is not substitutable by upstream.
- **It still wants its own venv.** The oracle needs `jax` plus CUDA while the model environment needs
  `torch` plus CUDA; sharing one environment makes the two fight over CUDA runtime libraries. The
  native route removes the container, not the environment split.

#### Which weights

The route is not OpenFold3-only. `AF3_MODEL_DIR` takes either weight family, and the adapter picks
the architecture branches from the directory itself rather than from a flag you have to remember:
it looks for the converter's `of3_ported_weights.provenance.json`, sets `global_config.of3_weights`
when it finds one, and leaves the stock AlphaFold3 branches in place when it does not. So pointing
`AF3_MODEL_DIR` at a directory holding DeepMind's `af3.bin` runs the AF3 path under the same fork,
the same worker and the same manifest protocol.

What differs is how you get them. OpenFold3's weights are Apache-licensed, so
`download_assets.sh --native-of3` fetches them for you. AlphaFold3's are licence-gated: request them
from DeepMind, stage them yourself, and point `AF3_MODEL_DIR` at the directory you staged them in —
no script here can download them on your behalf. That makes OpenFold3 the path of least resistance,
not the only one.

#### Reproducibility

Both routes run the same computation. On an H200, folding one input at a fixed seed through the
container and through the native venv gives **byte-identical coordinates** — same atoms, same
values, same pLDDT — for OpenFold3 and AlphaFold3 weights alike. Swapping the container out does not
change the answer.

That holds once XLA's kernel autotuning is pinned. By default XLA picks kernels by *timing* them, so
the same input, seed, image and node fold to slightly different coordinates on consecutive runs:
around 0.03–0.5 Å of maximum atomwise deviation, with pLDDT stable to about 0.001. That is ordinary
run-to-run variation on either route, not a property of the native backend. To make a run
reproducible:

```bash
export AF3_XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"
```

`AF3_XLA_FLAGS` sets the flags for both backends at once, so the comparison stays honest. Keep
`--xla_gpu_enable_triton_gemm=false`: it is what the image has always set, and dropping it changes
the kernels rather than just how they are chosen. Pinned kernels cost some throughput, so this is
for reproducing a result, not for production throughput.

Two caveats. Determinism is per machine: identical bytes are reproducible on the same GPU model, not
across different ones. And a run under an older image will not match — the image carries the pinned
`alphafold3` commit, so two images are two code versions. Folding the same input under
`af3-open-2026-08-19.sif` and the v3.0.5 image differs by 0.089 Å (OF3) and 0.317 Å (AF3) even with
kernels pinned, reproducibly, because the code differs.

## Turn it on

Refolding is driven by the same `metrics_tags` mechanism as every other metric. Pass a task specific
tag and the matching oracle, refolding metrics, and pass criteria all turn on together.

```bash
# Unconditional generation: AF3 refolding plus pLDDT and scRMSD
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[unconditional_refolding]'

# Protein binder design: templated refolding and interface metrics
rfd4-generate \
    inputs=design.json num_replicates=50 metrics_tags='[ppi_refolding,hotspot_conditioning]'
```

| Tag | Activates |
| --- | --- |
| `unconditional_refolding` | AF3 refolding plus pLDDT and scRMSD for unconditional generation |
| `ppi_refolding` | AF3 refolding, target templating and interface metrics for protein binder design |
| `antibody` | Antibody contacts, templated refolding and antibody pass criteria; CDR torsions require the opt-in config described in [Running metrics inline](evaluate-with-metrics.md) |
| `dna_refolding` | AF3 refolding, DNA templating and interface metrics for DNA binder design |
| `motif_scaffold_refolding` | AF3 refolding plus motif RMSD for motif scaffolding and enzyme design |
| `internal_coordinate_motif_scaffold_refolding` | Same as above, for internal coordinate conditioned motifs |
| `sm_binder_refolding` | AF3 refolding plus binding metrics for small molecule binder design |
| `ame_refolding` / `ame_no_ligand_refolding` | AF3 refolding for enzyme (MCSA) active site benchmarks |
| `disco_refolding` | AF3 refolding plus ligand centroid displacement, for the DISCO benchmark |
| `apo_refolding` | Folds the designed chain alone; pair it with another refolding tag to also get the holo (whole complex) refold |

A metric or oracle that needs a tag you did not pass is skipped silently, with no error. The same
applies offline, through `rfd4-evaluate inputs=<dir> metrics_tags='[...]' out_dir=<dir>`. See [Running
metrics offline](advanced/offline-metrics.md).

## What runs underneath a refolding tag

See [What runs underneath a refolding tag](../concepts/refolding.md#what-runs-underneath-a-refolding-tag)
for how the oracle, refolding metrics, and pass criteria fit together, and the
[metrics reference](../reference/metrics.md) for every oracle, metric, and pass criterion by name.

In practice, leave pre-oracle filters on — they are your biggest lever on refolding cost — and
disable them only when benchmarking per example, with `~metrics/pre_oracle_filters`.

## A design can have more than one refold

See [A design can have more than one refold](../concepts/refolding.md#a-design-can-have-more-than-one-refold)
for why a design can carry several refolds and how they get addressed.

In practice: pair `apo_refolding` with another refolding tag (`ppi_refolding`,
`unconditional_refolding`, ...) to supply that second refold for a comparison like
`refolded_apo_holo_rmsd`, or the comparison silently drops. See the
[output format reference](../reference/output-format.md) for the full addressing scheme, and
[Comparing two refolds](advanced/custom-metrics.md#comparing-two-refolds) for writing a custom metric that
reads more than one refold.

## Pre-oracle transforms

See [Pre-oracle transforms](advanced/pre-oracle-transforms.md) for substituting off-targets with
`OffTargetSubstitutionTransform` and writing your own transform.

## Refolding in reward guided search

[Reward guided search](../concepts/search.md) can use an oracle's or refolding metric's output as a reward that
prunes candidates mid-trajectory, instead of only scoring finished designs. List both the oracle and
the comparison metric explicitly in `metrics: [...]` — search does not infer the dependency between
them — and the same `oracle:`/`is_refold:`/`exists_refold:` routing constraints from above still
apply, so a reward compares the structures you intend. See [Metric
selection](search.md#metric-selection) for the full configuration.

## Troubleshooting

| Issue | Solution |
| --- | --- |
| AF3 or refolding errors | Check the selected oracle runtime, weights and environment settings in [Set up the oracle](#set-up-the-oracle); structure-only search needs no oracle |
| `AttributeError` on `of3_weights` under `AF3_RUNTIME_BACKEND=native` | `AF3_PYTHON`'s venv has upstream AlphaFold3 rather than the OF3-capable fork; reinstall from the fork's release wheel |
| `AF3_PYTHON does not point at an interpreter` | The path must be the venv's `bin/python` itself, not the venv directory |
| The same seed gives different coordinates on consecutive runs | Expected: XLA picks kernels by timing them. Pin them with `AF3_XLA_FLAGS` — see [Reproducibility](#reproducibility) |
| Container and native disagree for the same seed | Set `AF3_XLA_FLAGS` so both share one XLA configuration; an image built before this flag became overridable ignores a host `XLA_FLAGS` entirely |
| A refolding metric or pass criterion is silently missing from the CSV | Check the tag that gates it was actually passed in `metrics_tags`; a missing tag skips the metric with no error |
| A refold comparison metric (like `refolded_apo_holo_rmsd`) is missing | You likely need a second refolding tag to supply the other refold it compares against — see [A design can have more than one refold](#a-design-can-have-more-than-one-refold) |
| Running two untransformed oracles together drops `refold.untransformed` | That by-kind address is only unambiguous with one untransformed oracle; address the refold by oracle name instead, per the [output format reference](../reference/output-format.md#refold-addresses) |
| Every design is dropped before the oracle, with no error | A [pre-oracle filter](../reference/metrics.md#pre-oracle-filters) is reading a metric key that never got computed, because the filter's tag gate is looser than the metric's. A filter reading an absent key counts it as a failure, not a skip |

```{toctree}
:hidden:
:maxdepth: 1

advanced/pre-oracle-transforms
```
