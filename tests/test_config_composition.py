"""Compose every selectable training recipe and check the invariants it relies on.

CPU-only, seconds to run. What it catches is silent: a design-task preset that merged
instead of replacing still composes, still trains, just on the wrong mix.
"""

import os

import pytest

# The pipeline reads this at import time; set before hydra pulls the config in.
os.environ.setdefault("CONTAINER_TYPE", "docker")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from tests.helpers import CONFIGS_DIR, PROJECT_ROOT  # noqa: E402

# Backend bases are fragments, not datasets; they set `name: ???` on purpose.
_BACKENDS = {"pipeline_defaults", "backend_pandas", "backend_lmdb"}


def _options(group: str) -> list[str]:
    """Option names under `configs/<group>/`, as hydra would address them."""
    root = PROJECT_ROOT / "configs" / group
    return sorted(str(p.relative_to(root).with_suffix("")) for p in root.rglob("*.yaml") if not p.name.startswith("_"))


def _compose(config_name: str, overrides: list[str]) -> dict:
    """Compose a recipe as a plain container."""
    with initialize_config_dir(config_dir=str(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    # resolve=False: `resolve_import` is registered only inside the app, and raises here.
    return OmegaConf.to_container(cfg, resolve=False)


def _recipes() -> list[tuple[str, str, list[str], str | None]]:
    """(id, entry point, overrides, invariant) for every recipe selectable from the CLI.

    Mixes are selectable without an experiment, so one naming a deleted dataset stays broken
    indefinitely -- nothing else composes it. `datasets/none` is an empty base, so excluded;
    only training_ae declares `datasets` as a group default. An AE mix may be conditional.
    """
    out = [("training_proteina", "training_proteina", [], "tasks"), ("training_ae", "training_ae", [], "uncond")]
    out += [(f"experiment={e}", "training_proteina", [f"experiment={e}"], "tasks") for e in _options("experiment")]
    out += [(f"experiment_ae={e}", "training_ae", [f"experiment_ae={e}"], "uncond") for e in _options("experiment_ae")]
    out += [
        (f"releases={r}", "training_ae", ["~experiment_ae", f"+releases={r}"], "uncond")
        for r in _options("releases")
        if r.startswith("ae/")
    ]
    out += [
        (f"releases={r}", "training_proteina", ["~experiment", f"+releases={r}"], "tasks")
        for r in _options("releases")
        if r.startswith("flow/")
    ]
    for mix in sorted(p.stem for p in (PROJECT_ROOT / "configs" / "datasets").glob("*.yaml") if p.stem != "none"):
        if mix.startswith("ae_"):
            out.append((f"datasets={mix}", "training_ae", ["~experiment_ae", f"datasets={mix}"], None))
        else:
            out.append((f"datasets={mix}", "training_proteina", ["~experiment", f"+datasets={mix}"], "tasks"))
    return out


def _conditional(tasks: dict) -> dict:
    """Task -> frequency for each non-unconditional task that can actually be sampled."""
    active = {t: b.get("frequency") for t, b in tasks.items() if isinstance(b, dict) and t != "unconditional"}
    return {t: f for t, f in active.items() if (f or 0) > 0}


def _leaves() -> list[tuple[str, str]]:
    """(group, option) for every dataset leaf under `configs/datasets/train/`."""
    root = PROJECT_ROOT / "configs" / "datasets" / "train"
    return sorted(
        (f"train/{g}" if (g := p.parent.relative_to(root).as_posix()) != "." else "train", p.stem)
        for p in root.rglob("*.yaml")
        if p.stem not in _BACKENDS
    )


_RECIPES = _recipes()


@pytest.mark.parametrize("entry,overrides,invariant", [r[1:] for r in _RECIPES], ids=[r[0] for r in _RECIPES])
def test_recipe_composes_with_the_right_design_tasks(entry: str, overrides: list[str], invariant: str | None):
    """Every recipe composes, supplies a train mixture, and holds its own task invariant.

    `tasks`: a flow dataset with no design tasks skips task conditioning entirely. `uncond`:
    attaching a design-task group over one that already selected a preset UNIONS the two, so
    an AE recipe inheriting a flow config silently keeps its task frequencies.
    """
    datasets = (_compose(entry, overrides).get("datasets") or {}).get("train") or {}
    assert datasets, "composed with no datasets.train"
    offenders = {}
    for name, cfg in datasets.items():
        if not isinstance(cfg, dict) or not cfg.get("probability"):
            continue  # a dataset sampled with probability 0 cannot contribute
        tasks = ((cfg.get("dataset") or {}).get("transform") or {}).get("design_tasks")
        if invariant == "tasks" and not tasks:
            offenders[name] = "no design_tasks"
        # A non-dict is interpolated wholesale; nothing to inspect without resolving.
        elif invariant == "uncond" and isinstance(tasks, dict) and (bad := _conditional(tasks)):
            offenders[name] = bad
    assert not offenders, f"design-task invariant violated: {offenders}"


@pytest.mark.parametrize("group,option", _leaves())
def test_dataset_leaf_is_fully_formed(group: str, option: str):
    """Every dataset leaf composes into an instantiable dataset.

    Guards a silent Hydra trap: an absolute defaults entry mounts at `<pkg>.datasets.train.
    backend_lmdb`, not the including config's own package, which needs `@_here_`. The wrong
    form still COMPOSES -- `_target_` just stays unset.
    """
    probe = _compose("training_ae", ["~experiment_ae", f"+datasets/{group}@datasets.train.probe={option}"])
    ds = (probe["datasets"]["train"]["probe"] or {}).get("dataset")
    assert isinstance(ds, dict), "no `dataset` node"
    assert ds.get("_target_"), "no _target_ -- a backend base is mounted at the wrong package"
    assert ds.get("name") not in (None, "???"), f"name is {ds.get('name')!r}"
    for stray in ("datasets", *_BACKENDS):
        assert stray not in ds, f"`dataset.{stray}` -- defaults entry needs @_here_"
