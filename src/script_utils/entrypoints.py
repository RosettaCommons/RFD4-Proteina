"""Console startup that captures warnings before importing the model package."""

import warnings
from typing import Any


def generate() -> None:
    """Run generation, deferring import warnings until verbosity is configured."""
    with warnings.catch_warnings(record=True) as caught:
        # Import inside the capture so package initialization follows the same warning policy.
        from rfproteina.generate import main_hydra
        from rfproteina.utils.verbosity import queue_startup_warnings

    queue_startup_warnings(caught)
    main_hydra()


def initialize_metrics_worker(mm_cfgs: dict[str, Any], val_dataset_tags: Any, verbose: bool = True) -> None:
    """Capture spawned-worker imports before applying the requested verbosity."""
    with warnings.catch_warnings(record=True) as caught:
        # Keep the spawn initializer importable without initializing rfproteina first.
        from rfproteina.metrics.utils.metrics_utils import _metrics_worker_init
        from rfproteina.utils.verbosity import queue_startup_warnings

    queue_startup_warnings(caught)
    _metrics_worker_init(mm_cfgs, val_dataset_tags, verbose)
