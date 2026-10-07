"""Validation for checkpoint-local search configuration."""

from collections.abc import Mapping
from typing import Any


def parse_checkpoints(checkpoints: Mapping[int, Any]) -> dict[int, dict[str, Any]]:
    """Validate and sort self-contained scoring checkpoints; step zero is implicit."""
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError("search.checkpoints must be a nonempty mapping of positive integer steps.")
    parsed = {}
    for step, config in checkpoints.items():
        if isinstance(step, str) and step.isascii() and step.isdigit():
            step = int(step)
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError(f"Checkpoint steps must be positive integers; got {step!r}.")
        if step in parsed:
            raise ValueError(f"Duplicate checkpoint {step}; quote YAML step keys before applying dotted CLI overrides.")
        if not isinstance(config, Mapping) or set(config) - {
            "metrics",
            "rewards",
            "use_clean_prediction",
            "continuation_criteria",
            "n_branch",
        }:
            raise ValueError(
                f"Checkpoint {step} accepts metrics, rewards, use_clean_prediction, continuation_criteria, n_branch."
            )
        if not isinstance(config.get("use_clean_prediction", False), bool):
            raise ValueError(f"Checkpoint {step} use_clean_prediction must be a bool.")
        if "n_branch" in config:
            count = config["n_branch"]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(f"Checkpoint {step} n_branch must be a positive integer.")
        names = config.get("metrics", [])
        if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
            raise ValueError(f"Checkpoint {step} metrics must be a list of metric names.")
        parsed[step] = dict(config)
    return dict(sorted(parsed.items()))
