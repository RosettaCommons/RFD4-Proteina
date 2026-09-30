# User-managed clusters

The public release does not ship private cluster profiles or images. On a single
GPU machine, leave CLUSTER unset and follow the installation guide. On SLURM,
create your own scripts/.user_info_<cluster> and optional .ipd/.env.<cluster>
profile before using a launcher. Preserve all user-specified scheduler settings;
do not silently replace partitions, accounts, priorities, resources, or time limits.

Plain generation uses the activated Python environment. If running in a container,
invoke Python explicitly inside that container. Refolding is optional and needs
its own configured oracle runtime.

## OpenFold3: a free, ungated alternative to AF3

AlphaFold3's own weights are gated by Google DeepMind's license. **OpenFold3** runs through the
exact same AF3 runtime and container on Apache-licensed weights — no license request needed —
so it's the default refolding oracle on non-configured clusters: `scripts/download_example_data.sh`
fetches it automatically (skip with `--skip-of3` if you'd rather configure AlphaFold3's licensed
weights instead). Set AF3_IMAGE to use an existing configured oracle image.

To fetch it manually (e.g. to redo it, or on a cluster the script doesn't auto-detect as
new), the public checkpoint bucket hosts a prebuilt container image and pre-converted
weights, so no building or conversion step is required:

```bash
BASE=https://YOUR_DATASET_STORE
mkdir -p "$DATA_PATH/af3/apptainer_images" "$DATA_PATH/af3/af3_weights/of3" "$DATA_PATH/af3/jax_cache"
curl -fL -o "$DATA_PATH/af3/apptainer_images/af3-open-36fad16.sif" \
    "$BASE/af3/apptainer_images/af3-open-36fad16.sif"
curl -fL -o "$DATA_PATH/af3/af3_weights/of3/of3_ported_weights.bin.zst" \
    "$BASE/af3/af3_weights/of3/of3_ported_weights.bin.zst"
curl -fL -o "$DATA_PATH/af3/af3_weights/of3/of3_ported_weights.provenance.json" \
    "$BASE/af3/af3_weights/of3/of3_ported_weights.provenance.json"

export AF3_IMAGE="$DATA_PATH/af3/apptainer_images/af3-open-36fad16.sif"
export AF3_MODEL_DIR="$DATA_PATH/af3/af3_weights/of3"
export AF3_JAX_CACHE_DIR="$DATA_PATH/af3/jax_cache"
```

Then pass `metrics/refolding_oracles=of3` as a Hydra override (instead of leaving the AF3
default) — the same `*_refolding` metric tags work unchanged; only the artifact namespace
changes (`of3`/`of3_ppi`/`of3_dna` instead of `af3`/`af3_ppi`/`af3_dna`). Requires `apptainer`
to be installed (`apt`/release `.deb` from the [apptainer releases
page](https://github.com/apptainer/apptainer/releases) — not preinstalled on most cloud GPU
images). See `docs/workflow/metrics.md` for the full oracle reference.
