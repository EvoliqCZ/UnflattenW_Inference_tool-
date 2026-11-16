"""Measured cost constants for `drafter.VerifyCostModel` (2026-09-28).

`drafter.py` already owns confidence-scheduled verification end to end:
`VerifyCostModel`, `choose_verify_length`, and `run_speculative_decode`'s
`verify_cost_model=` parameter (duck-typed on a drafter's optional
`draft_with_confidence`). This module adds exactly the one thing that was
still missing: measuring `VerifyCostModel`'s constants on the ALREADY-LOADED
model/state at eval time, instead of trusting `s7_dflash_policy.json`'s
placeholder numbers (`is_placeholder: true`) indefinitely.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from unflatten.model.speculative.drafter import VerifyCostModel


def calibrate_verify_cost(
    *,
    plain_step: Callable[[], None],
    verify_round: Callable[[], None],
    drafter_step: Callable[[], None] | None = None,
    repeats: int = 3,
    now: Callable[[], float] = time.perf_counter,
) -> VerifyCostModel:
    """Measure `plain_step()`, `verify_round()` and (if given)
    `drafter_step()` on whatever already-loaded model/state the caller's
    closures capture, `repeats` calls each, averaged, reported in
    milliseconds -- `VerifyCostModel`'s own units. `now` is injectable so
    this is exercisable on CPU with a fake clock (no real GPU/model
    required to test the averaging and result shape). `drafter_step=None`
    (default) leaves `drafter_cost_ms` at `VerifyCostModel`'s own default
    (`0.0`) -- it is informational only, `choose_verify_length` never reads
    it (`drafter.py`'s own docstring: the drafter's cost is already sunk by
    the time a verify-length choice is made)."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")

    def _average_ms(call: Callable[[], None]) -> float:
        total = 0.0
        for _ in range(repeats):
            start = now()
            call()
            total += (now() - start) * 1000.0
        return total / repeats

    plain_step_cost_ms = _average_ms(plain_step)
    verify_round_cost_ms = _average_ms(verify_round)
    drafter_cost_ms = _average_ms(drafter_step) if drafter_step is not None else 0.0
    return VerifyCostModel(
        plain_step_cost_ms=plain_step_cost_ms,
        verify_round_cost_ms=verify_round_cost_ms,
        drafter_cost_ms=drafter_cost_ms,
    )
