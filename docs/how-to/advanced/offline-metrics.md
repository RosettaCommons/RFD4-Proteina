# Running metrics offline on a folder of structures

You do not have to re-run generation to get metrics. `evaluate.py` scores a folder of structures you
already have, on the same Hydra config, so `metrics_tags` and any custom metric config work exactly
as above.

```bash
rfd4-evaluate \
    inputs=/path/to/structures \
    metrics_tags='[unconditional_refolding]' \
    out_dir=/path/to/eval_output
```

`evaluate.py` scans `inputs` recursively for CIF-like files. `out_dir` is required, because
`evaluate.py` has no run directory to derive one from, and it gets the usual `evaluation_metrics/`
subdirectory.

Pass `task_csv=<csv with designed_structure_path and task columns>` when different structures need
different tags or pass criteria. Without it, every structure is scored the same way. The [Developer
Guide](../../contributing/developer-guide.md#evaluation-and-metrics) points at this same script for custom metric work,
so you write a metric once and use it both inline during generation and offline over old outputs.

For large batches split across SLURM array jobs (`launch_generate.sh --njobs N`), each job writes its
own `per_example_metrics_job{N}.csv`. Concatenate them afterwards.

```bash
python -m rfproteina.inference_utils.aggregate_metrics \
    ./runs/my_run/inference_metrics/
```

`launch_generate.sh` also runs this for you as a dependent job. See [Evaluating
results](../run-the-model.md#evaluating-results) for FoldSeek diversity clustering and the other
`aggregate_metrics` options, and the [output format reference](../../reference/output-format.md#metrics-csvs-and-aggregation)
for the exact file layout these commands produce.
