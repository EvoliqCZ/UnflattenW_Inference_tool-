"""S4.5 learning-rate schedule, integer arithmetic only (design D3; dormant).

``lr_scale_micro(step, policy)`` is the schedule multiplier in micro units
(``1_000_000`` == 1.0); ``lr_at(step, policy, group)`` multiplies it into the
policy's per-group base rate.

* Warmup: ``step < warmup_steps`` -> ``(step + 1) * 1e6 // warmup_steps``
  (floor; the first update already trains, the last warmup step is < 1.0).
* ``constant_after_warmup``: 1.0 afterwards.
* ``linear_warmup_cosine``: ``0.5 * (1 + cos(pi * p)) == cos(pi * p / 2) ** 2``
  with ``p = (step - warmup) / (max_update_steps - warmup)`` clamped to [0, 1],
  decaying to exactly 0 at ``max_update_steps``. The cosine is a fixed-point
  Taylor series (1e30 scale), so no float and no libm enters; the result is
  rounded half-up to micro.

Units: ``lr_micro`` is in 1e-6, so ``lr_micro * scale_micro`` is in 1e-12
(``LR_UNIT``). Rounding that product to whole 1e-6 would collapse a 1e-5 rate to
ten distinct values, so ``lr_at`` returns the unrounded 1e-12 integer and the
optimizer receives ``value * 1e-12`` -- the only place a float is made, and it
is never hashed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from unflatten.model.rl.scorers import MICRO

LR_UNIT: Final = 1e-12  # unit of `lr_at`'s result
_ONE: Final = 10**30
_PI: Final = 3141592653589793238462643383279  # pi * 1e30


class LrScheduleRefused(ValueError):
    """A schedule request the policy cannot answer (unfrozen field, bad step)."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _cos_squared_half_micro(progress_num: int, progress_den: int) -> int:
    """``round_half_up(1e6 * cos(pi/2 * num/den) ** 2)`` for ``0 <= num <= den``."""
    x = _PI * progress_num // (2 * progress_den)
    x2 = x * x // _ONE
    term = total = _ONE
    for k in range(1, 40):
        term = -term * x2 // _ONE // ((2 * k - 1) * (2 * k))
        total += term
        if term == 0:
            break
    return (total * total * MICRO + _ONE * _ONE // 2) // (_ONE * _ONE)


def lr_scale_micro(step: int, policy: Mapping[str, Any]) -> int:
    if not _is_int(step) or step < 0:
        raise LrScheduleRefused(f"step must be a non-negative int, got {step!r}")
    warmup, total = policy["warmup_steps"], policy["max_update_steps"]
    schedule = policy["schedule"]
    if warmup is None or schedule is None:
        raise LrScheduleRefused("warmup_steps / schedule are not frozen in the training policy")
    if step < warmup:
        return (step + 1) * MICRO // warmup
    if schedule == "constant_after_warmup":
        return MICRO
    if schedule == "linear_warmup_cosine":
        if total is None:
            raise LrScheduleRefused("max_update_steps is not frozen in the training policy")
        span = total - warmup
        if span <= 0 or step >= total:
            return 0
        return _cos_squared_half_micro(step - warmup, span)
    raise LrScheduleRefused(f"unknown schedule {schedule!r}")


def lr_at(step: int, policy: Mapping[str, Any], group: str) -> int:
    """Learning rate of ``group`` at ``step`` in units of 1e-12 (``LR_UNIT``)."""
    base = policy["lr_micro"].get(group)
    if base is None:
        raise LrScheduleRefused(f"no frozen learning rate for group {group!r}")
    return base * lr_scale_micro(step, policy)
