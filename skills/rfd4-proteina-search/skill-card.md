# Skill Card

## Description

`rfd4-proteina-search` runs reward-guided **beam search** for RFProteína (RFD4-Proteína): instead of sampling once, it explores multiple generation trajectories, scores candidates with a reward (by default an AF3 refolding oracle), and keeps the highest-scoring designs. It runs through the same `generate.py` entry point as generation, selected by a search `inference_experiment` config that adds one `search:` block (a `strategy` and inline `rewards`).

This skill is for research and development only.

## Owner

RFProteína maintainers — Baker Lab x NVIDIA GenAI collaboration.

## License/Terms of Use

See the RFProteína repository. Skill text and documentation follow the NVIDIA agent-skills documentation convention (CC-BY-4.0); helper scripts follow the repository's source license.

## Use Case

Researchers who want the best designs per candidate — optimizing monomers or binders by AF3 confidence (pLDDT / ipTM / ipAE) or by interface interaction counts — and are willing to spend extra compute for it. Requires `rfd4-proteina-setup` complete, with an AF3 apptainer/docker container configured for oracle-backed rewards. For cheap unguided sampling use `rfd4-proteina-generate`; for post-hoc scoring use `rfd4-proteina-evaluate`.

## Deployment Geography for Use

Global, subject to the repository license and export/compliance constraints on the model weights.

### Requirements / Dependencies

Requires API Key or External Credential: No

Credential Type(s): None (checkpoints download over public HTTPS; the AF3 oracle runs from a locally configured container/SIF, no credentials).

Hardware/software: Linux with an NVIDIA GPU (>=40 GB VRAM recommended); `rfd4-proteina-setup` complete; for oracle-backed rewards, an AF3 apptainer/docker container configured via the per-cluster env file (`.ipd/.env.$CLUSTER`). High compute cost — many AF3 oracle calls per run.

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: Search is compute-intensive (many AF3 oracle calls, roughly `beam_width x n_branch x checkpoints` per input) — it can consume large GPU budgets.
Mitigation: The skill documents the cost model and urges starting small (low `beam_width`/`n_branch`, short `step_checkpoints`, small `nsteps`) and using plain `rfd4-proteina-generate` for cheap baselines. See references/cost.md.

Risk: Misconfigured `step_checkpoints` (not starting at 0, not strictly increasing, or not ending at `nsteps`) will error and waste a submission.
Mitigation: The skill states the constraint prominently and the `generation.args.nsteps == last step_checkpoint` rule, with a troubleshooting entry.

Risk: The reward optimizes a proxy (AF3 confidence / interface metrics), not experimental success — high-reward designs are not guaranteed to work in the lab.
Mitigation: The skill frames reward as a proxy and advises experimental validation of any design.

## References

- `docs/guides/search.md`, `docs/guides/inference.md` ("Search Strategies", "Evaluating Results")
- `configs/inference_experiment/inference_beam_example.yaml`, `configs/inference_experiment/sm_binder_hbond_search.yaml`
- `src/rfproteina/search/beam_search.py`, `src/rfproteina/metrics/oracles/pass_criterion_metric.py`, `src/rfproteina/metrics/oracles/oracle_metric.py`
- `scripts/launch_search.sh`
- rfd4-proteina-setup skill (`references/clusters.md` for the AF3 container)
- RFProteína website: https://rosettacommons.github.io/RFD4-Proteina/
- NVIDIA agent skills: https://docs.nvidia.com/skills

## Skill Output

Output type(s): files — annotated `.cif` designs (the final beam, plus any intermediate candidates saved by `intermediate_sample_save_criterion`) and a `per_example_metrics.csv`.

Output format: CIF structures with per-atom annotations; CSV metrics table; each output scored by the reward.

Output parameters: `out_dir/` (e.g. `outputs/beam_example/`), containing the scored design CIFs and `per_example_metrics.csv`.

Other properties: search runs through `generate.py`; oracle results are cached during scoring and reused by the epoch-end metrics loop. Oracle-backed rewards require the container.

## Skill Version

0.1.0 (initial authoring, 2026-07).

## Ethical Considerations

RFProteína is a de novo protein/complex design model with dual-use potential. Reward-guided search actively optimizes designs toward an objective, which sharpens both benefit and misuse potential; use is governed by the repository's usage terms. Do not use to design hazardous biomolecules. Reward is a computational proxy — validate designs experimentally and follow applicable biosafety and export/compliance requirements.
