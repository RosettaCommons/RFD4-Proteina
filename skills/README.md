# RFD4-Proteína Agent Skills

Portable agent **skills** for RFD4-Proteína (RFD4-Proteína), the Baker Lab x
NVIDIA GenAI flow-matching model for de novo protein and complex design. Each
skill is a portable directory containing a `SKILL.md` (plus optional
`references/` and `scripts/`) that an agent reads on demand to drive one stage of
the RFD4-Proteína workflow — from standing up the environment, to preparing inputs,
generating designs, searching for better ones, and evaluating the results. They
encode the repository's real commands, config keys, and gotchas so an agent can
operate the model correctly without re-deriving them.

## The ten skills

| Skill | What it does | Order |
| --- | --- | --- |
| [`rfd4-proteina-setup`](rfd4-proteina-setup/) | Install the uv/Python-3.12 venv + `atomworks`, wire `DATA_PATH` and optional runtime settings, download checkpoints, and preflight GPU + weights. | 1 |
| [`rfd4-proteina-annotated-cif`](rfd4-proteina-annotated-cif/) | Create, inspect, modify, and validate annotated CIFs with Python-first AtomWorks workflows, including conditions, derived masks, expandable segments, chemistry, and pre-save invariants. | 2 |
| [`rfd4-proteina-generate`](rfd4-proteina-generate/) | Run `generate.py` to produce designs; choose task, replicates, batch size, and inline `metrics_tags`. | 3 |
| [`rfd4-proteina-search`](rfd4-proteina-search/) | Reward-guided search (beam/branching) that keeps candidates selected by a configured refolding or structure-only reward. | 4 (optional) |
| [`rfd4-proteina-evaluate`](rfd4-proteina-evaluate/) | Post-hoc metrics on generated structures + aggregation across parallel jobs (scRMSD/pLDDT/ipAE, designability, FoldSeek diversity, custom metrics). | 5 |
| [`rfd4-proteina-autoresearch`](rfd4-proteina-autoresearch/) | Wave strategy: iterate over branches with explicit reward engineering and freeze/promote/allocate rules. | 6 (optional) |
| [`rfd4-proteina-commitment-curves`](rfd4-proteina-commitment-curves/) | Partially re-diffuse designs to locate where along the trajectory a property is decided. | 7 (optional) |
| [`rfd4-proteina-campaign`](rfd4-proteina-campaign/) | End-to-end campaign orchestration with a SQLite ledger linking every design, input, config, oracle refold, metric, and passer decision. | 8 (orchestrator) |
| [`rfd4-proteina-package`](rfd4-proteina-package/) | Build a database-free TAR of selected designs, actual metrics, exact waterfall, and aligned on/off-target refolds grouped by annotated-CIF input. | 9 (delivery) |

Typical order of use: **setup -> annotated-cif (or inputs for legacy JSON) ->
generate -> [search] -> evaluate**, with **campaign** owning an end-to-end
program, **autoresearch** running its multi-wave exploration,
**commitment-curves** locating useful reward timing, and **package** creating the
recipient-facing delivery.

## Install and discovery

The `skills/` directory is canonical. Repository discovery uses two thin adapters that point to it:

- `.agents/skills` for Codex;
- `.claude/skills` for Claude Code.

Both agents therefore load the same `SKILL.md`, references, and helper scripts. Do not edit an adapter or copy instructions into a model-specific command. The Claude `/design2cif` command routes to the annotated-CIF authoring skill or the current condition-spec documentation.

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

Every skill directory carries:

- `SKILL.md` — the agent-facing instructions (with `references/` and `scripts/`
  as needed),
- `skill-card.md` — the human-readable skill card (owner, license, risks,
  outputs, ethics),
- `evals/evals.json` — a Tier-3 eval dataset with at least one bounded, checkable
  case.

For NVIDIA catalog publication ([docs.nvidia.com/skills](https://docs.nvidia.com/skills)),
**two more artifacts are added at onboarding time and are NOT committed by hand**:

- `skill.oms.sig` — a detached OMS signature, verifiable against
  `nv-agent-root-cert.pem`,
- `BENCHMARK.md` — a generated benchmark / uplift report.

These are produced by the sync/scan/sign pipeline (SkillSpector scan -> skill
card -> OMS signing). The Tier-3 eval dataset (`evals/evals.json`) is a required
input to that pipeline — the benchmark report is generated from it.

## License

Skill **text and documentation** follow **CC-BY-4.0**, while **source and helper
scripts** follow **BSD-3-Clause**, except where third-party license notices apply.
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
