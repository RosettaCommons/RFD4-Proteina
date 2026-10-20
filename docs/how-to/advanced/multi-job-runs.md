# Multi-job runs

When you run large evaluations across several SLURM jobs, each job writes its own
`per_example_metrics_job{N}.csv`. Once they all finish, concatenate them by passing the metrics
directory as the first positional argument.

```bash
python -m rfproteina.inference_utils.aggregate_metrics ./inference/my_run/metrics/
```

That concatenates the per example job CSVs, writes `aggregated_metrics.csv` and
`per_task_metrics.csv`, and moves the individual job files into `individual_job_results/`. Pass
`--output_dir <dir>` to write elsewhere. `launch_generate.sh` runs this for you as a dependent SLURM
job.

Pass `--cluster_with_foldseek` to also run structural diversity clustering, described in [FoldSeek
diversity clustering](../run-the-model.md#foldseek-diversity-clustering).
