"""Time-dependent guidance weight schedules.

A schedule is a callable ``t -> signed weight`` that folds magnitude, sign, and timing
into one place (there is no separate weight field). Use them in a guidance config as
``_partial_`` Hydra targets, e.g.::

    bb_ca:
      _target_: rfproteina.guidance.schedules.window
      _partial_: true
      w: -0.05
      t_start: 0.5
      t_end: 1.0

A negative ``w`` minimizes the metric (guidance always *maximizes* the weighted objective).
``t`` is the flow-matching time in ``[0, 1]`` for the relevant modality.
"""


def constant(t: float, *, w: float) -> float:
    """Return a constant weight ``w`` for all ``t``."""
    return w


def window(t: float, *, w: float, t_start: float, t_end: float) -> float:
    """Return ``w`` inside ``[t_start, t_end]`` and ``0`` outside."""
    return w if t_start <= t <= t_end else 0.0
