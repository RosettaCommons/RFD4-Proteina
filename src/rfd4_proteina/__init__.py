"""Public RFD4-Proteina API: ``import rfd4_proteina as rfd4``."""

from importlib.metadata import version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rfproteina.proteina import Proteina

__version__ = version("rfd4-proteina")
__all__ = ["Proteina", "__version__"]


def __getattr__(name: str) -> Any:
    """Load the model on demand to avoid initializing Torch during package inspection."""
    if name == "Proteina":
        from rfproteina.proteina import Proteina

        globals()[name] = Proteina
        return Proteina
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
