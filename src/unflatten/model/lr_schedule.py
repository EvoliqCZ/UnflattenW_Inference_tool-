"""S4 learning-rate tail -- dormant, opt-in, bound in the frozen run manifest.

Every frozen S4 run so far trained at a CONSTANT `--lr` / `--lr-reduced`. A schedule
is therefore a property of the run, not a loose flag: `s4_run_manifest` carries it as
an optional `lr_schedule` block (written only when selected) and `s4_train` refuses a
CLI that disagrees with it, on every invocation including `--resume`.

`linear_tail`: the multiplier is 1.0 up to and including `start_step`, then falls
linearly to `floor_milli / 1000` on the run's last step (`total_steps`). Both groups
(standard and reduced) are scaled by the same multiplier from their CLI base LR, so
the reduced/standard ratio never moves.

A step is the 1-based optimizer step about to be taken (`run_s4_training`'s `step`
after `step += 1`); step 0 is "before any step" and always sees the bases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

LR_SCHEDULE_KINDS = ("linear_tail",)
_MANIFEST_KEYS = frozenset({"kind", "start_step", "floor_milli"})


@dataclass(frozen=True)
class LrSchedule:
    kind: str
    start_step: int
    floor_milli: int

    def __post_init__(self) -> None:
        if self.kind not in LR_SCHEDULE_KINDS:
            raise ValueError(f"lr_schedule kind {self.kind!r} not in {LR_SCHEDULE_KINDS}")
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError(f"lr_schedule start_step must be an int >= 0, got {self.start_step!r}")
        if type(self.floor_milli) is not int or not 1 <= self.floor_milli <= 1000:
            raise ValueError(
                f"lr_schedule floor_milli must be an int in [1, 1000], got {self.floor_milli!r}"
            )


def require_within_budget(schedule: LrSchedule, *, total_steps: int) -> None:
    """The tail must start before the last step, or it never decays."""
    if schedule.start_step >= total_steps:
        raise ValueError(
            f"lr_schedule start_step {schedule.start_step} is not before the run budget "
            f"{total_steps}: the tail would never start"
        )


def multiplier(schedule: LrSchedule | None, step: int, *, total_steps: int) -> float:
    if schedule is None or step <= schedule.start_step:
        return 1.0
    if step > total_steps:
        raise ValueError(f"step {step} is past the run budget {total_steps}")
    floor = schedule.floor_milli / 1000
    progress = (step - schedule.start_step) / (total_steps - schedule.start_step)
    return 1.0 - (1.0 - floor) * progress


def expected_group_lrs(
    schedule: LrSchedule | None, step: int, *, total_steps: int, bases: Sequence[float]
) -> list[float]:
    """The LR each param group must carry right after `step` was taken."""
    scale = multiplier(schedule, step, total_steps=total_steps)
    return [base * scale for base in bases]


def apply(
    optimizer: torch.optim.Optimizer,
    schedule: LrSchedule | None,
    step: int,
    *,
    total_steps: int,
    bases: Sequence[float],
) -> None:
    """Set each group's LR to `base * multiplier` -- from the base, never compounding."""
    if schedule is None:
        return
    for group, lr in zip(
        optimizer.param_groups,
        expected_group_lrs(schedule, step, total_steps=total_steps, bases=bases),
        strict=True,
    ):
        group["lr"] = lr


def to_manifest_block(schedule: LrSchedule) -> dict[str, Any]:
    return {
        "kind": schedule.kind,
        "start_step": schedule.start_step,
        "floor_milli": schedule.floor_milli,
    }


def from_manifest_block(block: Mapping[str, Any] | None) -> LrSchedule | None:
    """`None` (key absent) is the constant LR -- never a silent default schedule."""
    if block is None:
        return None
    extra = set(block) - _MANIFEST_KEYS
    if extra:
        raise ValueError(f"lr_schedule block has unexpected keys {sorted(extra)}")
    return LrSchedule(
        kind=block["kind"], start_step=block["start_step"], floor_milli=block["floor_milli"]
    )
