# FoldSeek diversity clustering

Pass `--cluster_with_foldseek` and structures get clustered by structural similarity using
[FoldSeek](https://github.com/steineggerlab/foldseek), which adds a `foldseek_cluster_id` column so
you can count structurally distinct designs per task.

**Clustering only looks at the designed chain.** In a multimer output such as binder plus target,
only the most freely designed chain gets extracted and clustered. The pipeline identifies that
chain from the conditioning annotations and stores it as `chain_of_interest` in the metrics CSV.
Clustering reads the extracted per design chain CIFs, which only get written when you generate with
`save_chain_of_interest=true`, so set that if you plan to cluster.

Point `FOLDSEEK_EXEC` in `.env.local` at the FoldSeek executable. Clustering runs `foldseek easy-cluster`
on the extracted single chain with a TM-score threshold of 0.5.
