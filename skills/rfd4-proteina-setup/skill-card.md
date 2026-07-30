# Skill Card

## Description

`rfd4-proteina-setup` prepares a Linux + NVIDIA-GPU host to run RFProteína (RFD4-Proteína) by installing the uv/Python-3.12 environment and `atomworks` submodule, wiring `DATA_PATH` and the per-cluster env file, downloading the model checkpoints, and running a GPU/checkpoint preflight.

This skill is for research and development only.

## Owner

RFProteína maintainers — Baker Lab x NVIDIA GenAI collaboration.

## License/Terms of Use

See the RFProteína repository. Skill text and documentation follow the NVIDIA agent-skills documentation convention (CC-BY-4.0); helper scripts follow the repository's source license.

## Use Case

Anyone standing up RFProteína on a new machine or cluster (DIGS/ORD/GB200/CoreWeave/local) before generating, evaluating, or searching. Run this first; it is a prerequisite for `rfd4-proteina-generate`, `rfd4-proteina-search`, and `rfd4-proteina-evaluate`.

## Deployment Geography for Use

Global, subject to the repository license and export/compliance constraints on the model weights.

### Requirements / Dependencies

Requires API Key or External Credential: No

Credential Type(s): None (checkpoints download over public HTTPS; no credentials).

Hardware/software: Linux with an NVIDIA GPU (>=40 GB VRAM recommended), `git`, `curl`, Python 3.12, ~10 GB disk for checkpoints. Optional apptainer/docker for AF3-backed refolding metrics and reward-guided search.

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate.

## Known Risks and Mitigations

Risk: The checkpoint name in `configs/inference.yaml` must match what `download_example_data.sh` fetches, or a default run fails to resolve the checkpoint.
Mitigation: Both name `2026_07_22_release_280k-ema.ckpt`, served from the public bucket `YOUR_DATASET_STORE`; preflight reports which `.ckpt` files are actually present, and `ckpt_path=`/`ckpt_name=` override either.

Risk: Running the model before the `atomworks` submodule/venv is ready produces confusing import errors.
Mitigation: Preflight checks imports and submodule presence and exits non-zero before any run.

Risk: `setup.sh` edits the user's shell rc file.
Mitigation: It only appends a clearly-marked block, is idempotent (skips if the marker exists), and offers an "other / set manually" choice.

## References

- `docs/guides/inference.md`, `docs/guides/annotated_cif_inputs.md`
- `setup.sh`, `scripts/download_example_data.sh`, `configs/inference.yaml`
- RFProteína website: https://rosettacommons.github.io/RFD4-Proteina/
- NVIDIA agent skills: https://docs.nvidia.com/skills

## Skill Output

Output type(s): environment changes + files (venv, submodule, downloaded checkpoints, env file), and a preflight report.

Output format: console PASS/FAIL report; created files on disk.

Output parameters: `.venv/`, `lib/atomworks/`, `$DATA_PATH/latent_checkpoints/...`, `.ipd/.env.$CLUSTER`.

Other properties: `preflight.sh` is read-only (no writes); exits non-zero on any failed check. `setup.sh` and the download step modify the filesystem and shell rc.

## Skill Version

0.1.0 (initial authoring, 2026-07).

## Ethical Considerations

RFProteína is a de novo protein/complex design model with dual-use potential. Setup does not itself design anything, but it enables generation; downstream use is governed by the design-task skills' ethical notes and the repository's usage terms. Do not use to design hazardous biomolecules.
