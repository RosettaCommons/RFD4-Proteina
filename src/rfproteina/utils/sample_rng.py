"""Independent per-design random streams for batched inference."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

import torch

_GENERATORS: ContextVar[tuple[torch.Generator, ...] | None] = ContextVar("sample_rng_generators", default=None)


@contextmanager
def independent_sample_rng(states: Sequence[torch.Tensor], device: torch.device | str) -> Iterator[None]:
    """Continue each row's random stream from a preprocessing RNG snapshot.

    Args:
        states: CPU byte tensors from ``torch.get_rng_state`` (CPU) or
            ``torch.cuda.get_rng_state(device)`` (CUDA), in batch-row order.
        device: Device on which the snapshots were captured and noise is drawn.

    Rows must use the same padded shape as their singleton execution. The scope
    affects the sampling helpers only; global RNG states and training defaults
    remain unchanged. GPU arithmetic can differ across batch sizes.
    """
    if not states:
        raise ValueError("Independent sample RNG requires at least one row state")
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    generators = []
    for state in states:
        if state.device.type != "cpu" or state.dtype != torch.uint8 or state.ndim != 1:
            raise ValueError("Sample RNG states must be one-dimensional CPU byte tensors")
        generator = torch.Generator(device=resolved)
        generator.set_state(state.clone())
        generators.append(generator)
    token = _GENERATORS.set(tuple(generators))
    try:
        yield
    finally:
        _GENERATORS.reset(token)


def sample_rng_active() -> bool:
    """Whether a caller has supplied independent inference row streams."""
    return _GENERATORS.get() is not None


def sample_rng_states() -> tuple[torch.Tensor, ...]:
    """Snapshot active row streams so per-design reward callbacks can continue them."""
    generators = _GENERATORS.get()
    if generators is None:
        raise RuntimeError("RNG snapshots require independent_sample_rng")
    return tuple(generator.get_state().clone() for generator in generators)


@contextmanager
def sample_rng_row(index: int) -> Iterator[None]:
    """Temporarily use one row's stream for row-specific group operations."""
    generators = _GENERATORS.get()
    if generators is None:
        raise RuntimeError("A row RNG scope requires independent_sample_rng")
    token = _GENERATORS.set((generators[index],))
    try:
        yield
    finally:
        _GENERATORS.reset(token)


def _validate_rows(shape: Sequence[int], device: torch.device | str) -> tuple[torch.Generator, ...]:
    generators = _GENERATORS.get()
    if generators is None or not shape or shape[0] != len(generators):
        raise ValueError("Noise leading dimension must match the independent sample RNG rows")
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if any(generator.device != resolved for generator in generators):
        raise ValueError("Noise device must match the independent sample RNG device")
    return generators


def sample_randn(shape: Sequence[int], *, device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Draw Gaussian noise, preserving singleton draw shapes for each row."""
    if _GENERATORS.get() is None:
        return torch.randn(tuple(shape), device=device, dtype=dtype)
    generators = _validate_rows(shape, device)
    row_shape = (1, *shape[1:])
    return torch.cat(
        [torch.randn(row_shape, device=device, dtype=dtype, generator=generator) for generator in generators], dim=0
    )


def sample_randn_like(tensor: torch.Tensor) -> torch.Tensor:
    """Draw Gaussian noise with each singleton row's dtype and memory layout."""
    if _GENERATORS.get() is None:
        return torch.randn_like(tensor)
    generators = _validate_rows(tensor.shape, tensor.device)
    return torch.cat(
        [torch.empty_like(tensor[i : i + 1]).normal_(generator=generator) for i, generator in enumerate(generators)],
        dim=0,
    )
