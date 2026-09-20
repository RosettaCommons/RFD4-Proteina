"""Explicit verbosity for entry points and their workers, without changing log destinations."""

import logging
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any

VERBOSE: ContextVar[bool] = ContextVar("verbose", default=True)
_STARTUP_WARNINGS: list[warnings.WarningMessage] = []
_FILTERED_LOGGERS = (
    "atomworks.ml",
    "atomworks.io.utils.standardize",
    "lightning.pytorch.utilities.rank_zero",
    "lightning.fabric.utilities.rank_zero",
    "lightning_fabric.utilities.rank_zero",
)


class CachedDataFilter(logging.Filter):
    """Filter to suppress atomworks cached data logging messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out "Cached data not found" messages
        return "Cached data not found" not in record.getMessage()


class HydrogenRemovalFilter(logging.Filter):
    """Filter atomworks' per-hydrogen "removing non-heavy atom" standardization warnings.

    Parsing with ``add_missing_atoms=True`` logs one WARNING per hydrogen whose name
    doesn't match the CCD template (they're stripped and rebuilt); on protonated inputs
    that's thousands of lines of noise. Heavy-atom warnings are left untouched.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "removing non-heavy atom" not in record.getMessage()


class LightningTipFilter(logging.Filter):
    """Hide optional cloud-service promotions while retaining Lightning diagnostics."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.levelno == logging.INFO
            and message.startswith("💡 Tip: For seamless cloud")
            and ("[litlogger]" in message or "[litmodels]" in message)
        )


def _filter_routine_warnings(action: str) -> None:
    """Choose visibility for routine notices while retaining actionable warnings."""
    warnings.filterwarnings(action, category=DeprecationWarning)
    warnings.filterwarnings(action, message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*")
    warnings.filterwarnings(
        action,
        message="Category 'chem_comp_bond' not found. No bonds will be parsed",
        category=UserWarning,
        module="biotite.structure.io.pdbx.convert",
    )
    warnings.filterwarnings(
        action,
        message=".*index_reduce.*is in beta.*API may change.*",
        category=UserWarning,
    )
    # Lightning sync_dist warnings - we handle distributed aggregation manually
    warnings.filterwarnings(
        action,
        message=".*recommended.*sync_dist=True.*",
        category=UserWarning,
    )


class VerbosityFilter(logging.Filter):
    """Keep user-facing status and actionable warnings when verbose output is disabled."""

    def filter(self, record: logging.LogRecord) -> bool:
        return VERBOSE.get() or record.levelno >= logging.WARNING or getattr(record, "always_visible", False)


def queue_startup_warnings(caught: list[warnings.WarningMessage]) -> None:
    """Defer executable import warnings until Hydra has resolved the verbose setting."""
    _STARTUP_WARNINGS.extend(caught)


def configure_verbosity(verbose: bool) -> None:
    """Apply verbosity in the current process, retaining its existing log destinations."""
    VERBOSE.set(verbose)
    for name in _FILTERED_LOGGERS:
        logger = logging.getLogger(name)
        logger.filters[:] = [
            item
            for item in logger.filters
            if not isinstance(item, CachedDataFilter | HydrogenRemovalFilter | LightningTipFilter)
        ]
    _filter_routine_warnings("default" if verbose else "ignore")
    if not verbose:
        # Suppress atomworks cached data warnings (logging filter)
        logging.getLogger("atomworks.ml").addFilter(CachedDataFilter())
        logging.getLogger("atomworks.io.utils.standardize").addFilter(HydrogenRemovalFilter())
        logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(LightningTipFilter())
        logging.getLogger("lightning.fabric.utilities.rank_zero").addFilter(LightningTipFilter())
        logging.getLogger("lightning_fabric.utilities.rank_zero").addFilter(LightningTipFilter())

    loggers = [
        *(logging.getLogger(name) for name in _FILTERED_LOGGERS),
        logging.getLogger(),
        *(v for v in logging.Logger.manager.loggerDict.values() if isinstance(v, logging.Logger)),
    ]
    for logger in loggers:
        for handler in logger.handlers:
            if not any(isinstance(item, VerbosityFilter) for item in handler.filters):
                handler.addFilter(VerbosityFilter())
    caught = list(_STARTUP_WARNINGS)
    _STARTUP_WARNINGS.clear()
    for warning in caught:
        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)


def initialize_worker_verbosity(
    worker_id: int = 0, *, verbose: bool, worker_init_fn: Callable[[int], None] | None = None
) -> None:
    """Initialize worker output policy, then run any caller-provided worker initializer."""
    configure_verbosity(verbose)
    if worker_init_fn is not None:
        worker_init_fn(worker_id)


@contextmanager
def verbosity(verbose: bool) -> Iterator[None]:
    """Apply verbosity for a call and restore the caller's warning and logging policy."""
    loggers = [
        *(logging.getLogger(name) for name in _FILTERED_LOGGERS),
        logging.getLogger(),
        *(v for v in logging.Logger.manager.loggerDict.values() if isinstance(v, logging.Logger)),
    ]
    objects = [*loggers, *{handler for logger in loggers for handler in logger.handlers}]
    previous_filters = [list(item.filters) for item in objects]
    token = VERBOSE.set(verbose)
    try:
        with warnings.catch_warnings():
            configure_verbosity(verbose)
            yield
    finally:
        VERBOSE.reset(token)
        for item, filters in zip(objects, previous_filters, strict=True):
            item.filters[:] = filters


def configured_verbosity(default: bool) -> Callable:
    """Apply a Hydra config's verbose setting to an entry point or programmatic call."""

    def decorate(function: Callable) -> Callable:
        @wraps(function)
        def wrapped(cfg: Any, *args: Any, **kwargs: Any) -> Any:
            with verbosity(cfg.get("verbose", default)):
                return function(cfg, *args, **kwargs)

        return wrapped

    return decorate
