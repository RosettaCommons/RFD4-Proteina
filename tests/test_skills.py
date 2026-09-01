"""Guard that agent skills stay discoverable and correctly wired for Claude and Codex."""

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SKILLS_DIR = _REPO / "skills"
_SKILL_DIRS = sorted(p for p in _SKILLS_DIR.iterdir() if p.is_dir())


def _read_frontmatter(skill_md: Path) -> dict:
    text = skill_md.read_text()
    assert text.startswith("---"), f"{skill_md} is missing YAML frontmatter"
    _, frontmatter, _ = text.split("---", 2)
    return yaml.safe_load(frontmatter) or {}


def test_skills_directory_is_non_empty():
    # Guards against the parametrized tests below silently vacuously passing.
    assert _SKILL_DIRS, "no skill directories found under skills/"


@pytest.mark.parametrize("adapter", [".claude/skills", ".agents/skills"])
def test_skill_adapter_is_symlink_to_canonical_dir(adapter):
    """Claude (.claude/skills) and Codex (.agents/skills) must both point at skills/,
    so both tools load the same skills rather than a drifting copy."""
    link = _REPO / adapter
    assert link.is_symlink(), f"{adapter} must be a symlink to skills/, not a copy"
    assert link.resolve() == _SKILLS_DIR.resolve(), f"{adapter} must resolve to skills/"


def test_claude_md_includes_agents_md():
    """Claude reads the shared instructions via the one-line CLAUDE.md include."""
    assert (_REPO / "AGENTS.md").is_file(), "AGENTS.md is missing"
    assert (_REPO / "CLAUDE.md").read_text().strip() == "@AGENTS.md"


@pytest.mark.parametrize("skill_dir", _SKILL_DIRS, ids=lambda p: p.name)
def test_skill_is_discoverable_and_complete(skill_dir: Path):
    """Each skill loads: SKILL.md with a name that matches its directory and a
    non-empty description, plus the skill-card and eval dataset the catalog requires."""
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.is_file(), f"{skill_dir.name} is missing SKILL.md"

    frontmatter = _read_frontmatter(skill_md)
    assert (
        frontmatter.get("name") == skill_dir.name
    ), f"{skill_dir.name}: SKILL.md name {frontmatter.get('name')!r} must match the directory"
    assert (frontmatter.get("description") or "").strip(), f"{skill_dir.name}: empty SKILL.md description"

    assert (skill_dir / "skill-card.md").is_file(), f"{skill_dir.name} is missing skill-card.md"
    assert (skill_dir / "evals" / "evals.json").is_file(), f"{skill_dir.name} is missing evals/evals.json"
