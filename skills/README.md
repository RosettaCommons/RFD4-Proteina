# RFProteína Agent Skills

NVIDIA-compliant agent **skills** for RFProteína (RFD4-Proteína), the Baker Lab x
NVIDIA GenAI flow-matching model for de novo protein and complex design. Each
skill is a portable directory containing a `SKILL.md` (plus optional
`references/` and `scripts/`) that an agent reads on demand to drive one stage of
the RFProteína workflow — from standing up the environment, to preparing inputs,
generating designs, searching for better ones, and evaluating the results. They
encode the repository's real commands, config keys, and gotchas so an agent can
operate the model correctly without re-deriving them.

## The six skills

| Skill | What it does | Order |
| --- | --- | --- |
| [`rfd4-proteina-setup`](rfd4-proteina-setup/) | Install the uv/Python-3.12 venv + `atomworks`, wire `DATA_PATH` and the per-cluster env file, download checkpoints, and preflight GPU + weights. | 1 |
| [`rfd4-proteina-inputs`](rfd4-proteina-inputs/) | Turn a design intent (or legacy JSON/YAML) into a correct, validated **annotated CIF** — the recommended lossless input format. | 2 |
| [`rfd4-proteina-generate`](rfd4-proteina-generate/) | Run `generate.py` to produce designs; choose task, replicates, batch size, and inline `metrics_tags`. | 3 |
| [`rfd4-proteina-search`](rfd4-proteina-search/) | Reward-guided search (beam/branching) that keeps the candidates an AF3 reward scores highest — more sample-efficient than generate-then-filter. | 4 (optional) |
| [`rfd4-proteina-evaluate`](rfd4-proteina-evaluate/) | Post-hoc metrics on generated structures + aggregation across parallel jobs (scRMSD/pLDDT/ipAE, designability, FoldSeek diversity, custom metrics). | 5 |
| [`rfd4-proteina-autoresearch`](rfd4-proteina-autoresearch/) | Strategy layer: run an inspectable design -> score -> iterate **campaign** over many waves, composing the skills above with explicit composite-reward engineering and freeze/promote/allocate branch rules. | 6 (optional) |

Typical order of use: **setup -> inputs -> generate -> [search] -> evaluate**, with
**autoresearch** wrapping the whole loop when you want a multi-wave optimization
campaign.

## Install

Skills install as portable directories, each with a `SKILL.md` at its root. Add
one (or all) from the catalog with the `skills` CLI:

```bash
# Add a single skill
npx skills add RosettaCommons/RFD4-Proteina --skill rfd4-proteina-generate

# Add several
npx skills add RosettaCommons/RFD4-Proteina --skill rfd4-proteina-setup
npx skills add RosettaCommons/RFD4-Proteina --skill rfd4-proteina-evaluate
```

The exact `<owner>/<repo>` slug is fixed at catalog-onboarding time; the repo
here is `RosettaCommons/RFD4-Proteina`. Because each skill is just a directory
with a `SKILL.md`, you can also vendor one directly (copy the `skills/<name>/`
folder into your project) without the CLI.

## Compliance and structure

Every skill directory carries, by hand:

- `SKILL.md` — the agent-facing instructions (with `references/` and `scripts/`
  as needed),
- `skill-card.md` — the human-readable skill card (owner, license, risks,
  outputs, ethics),
- `evals/evals.json` — a Tier-3 eval dataset with at least one bounded, checkable
  case.

Per NVIDIA's skills pipeline ([docs.nvidia.com/skills](https://docs.nvidia.com/skills)),
**two more artifacts are added at onboarding time and are NOT committed by hand**:

- `skill.oms.sig` — a detached OMS signature, verifiable against
  `nv-agent-root-cert.pem`,
- `BENCHMARK.md` — a generated benchmark / uplift report.

These are produced by the sync/scan/sign pipeline (SkillSpector scan -> skill
card -> OMS signing). The Tier-3 eval dataset (`evals/evals.json`) is a required
input to that pipeline — the benchmark report is generated from it.

## License

Following NVIDIA's dual-license convention, skill **text and documentation**
follow **CC-BY-4.0**, while **source and helper scripts** follow **Apache-2.0**.
See the repository's top-level license for the authoritative terms.

## Verify a signed skill

Once a skill has been signed, verify its detached OMS signature and certificate
chain with [`model-signing`](https://github.com/sigstore/model-transparency):

```bash
model_signing verify certificate rfd4-proteina-evaluate/ \
    --signature rfd4-proteina-evaluate/skill.oms.sig \
    --certificate_chain nv-agent-root-cert.pem \
    --ignore_unsigned_files
```
