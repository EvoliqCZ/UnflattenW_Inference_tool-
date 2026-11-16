"""RFC004 §37: acceptance/latency accounting for `run_speculative_decode`.

A pure wrapper around `run_speculative_decode` (`drafter.py`) -- it changes
nothing about the decode itself, only observes it: one `RoundStats` per
loop iteration (`draft()` call), recording how many tokens the drafter
proposed, how many `run_speculative_decode` actually committed that round
(drafted-and-verified, or the single-token primary-only fallback), and how
many REAL primary forward calls that round cost
(`RoundStats.primary_forward_calls`). `speedup_estimate` is
`total_accepted / primary_calls`, where `primary_calls` now sums real
forward calls rather than rounds: a `PrimaryDecoder` with no
`verify_block` (RFC004 §37.1/§37.4) still costs one hidden-state
computation per verified token -- so a perfectly-drafted block over such a
primary reports NO speedup (`speedup_estimate == 1.0`), honestly, since it
genuinely is not faster. Only a `verify_block`-capable primary (one real
forward pass over the whole block, `decode_integration.
FreeRunningPrimaryDecoder`) reports `primary_forward_calls == 1` for that
round, which is where a real speedup shows up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from unflatten.model.speculative.drafter import (
    DecoderSnapshot,
    PrimaryDecoder,
    SpeculativeDrafter,
    _default_commit,
    run_speculative_decode,
)


@dataclass(frozen=True, slots=True)
class RoundStats:
    proposed: int
    accepted: int
    #: Real primary forward calls this round cost. Default `1` matches a
    #: round backed by a `verify_block`-capable (or trivial-fallback)
    #: primary -- also what every pre-existing hand-constructed `RoundStats`
    #: in this test suite already assumed before this field existed.
    primary_forward_calls: int = 1


@dataclass(frozen=True, slots=True)
class AccelerationReport:
    rounds: tuple[RoundStats, ...] = ()

    @property
    def primary_calls(self) -> int:
        return sum(round_.primary_forward_calls for round_ in self.rounds)

    @property
    def total_accepted(self) -> int:
        return sum(round_.accepted for round_ in self.rounds)

    @property
    def speedup_estimate(self) -> float:
        """Accepted tokens per REAL primary forward call; `0.0` for zero
        rounds (nothing decoded), never a division-by-zero crash on an
        empty decode."""
        if self.primary_calls == 0:
            return 0.0
        return self.total_accepted / self.primary_calls


@dataclass
class _InstrumentedDrafter:
    inner: SpeculativeDrafter
    proposed_counts: list[int] = field(default_factory=list)

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]:
        proposed = self.inner.draft(snapshot)
        self.proposed_counts.append(len(proposed))
        return proposed


@dataclass
class _InstrumentedPrimary:
    """Wraps `primary` to count REAL forward calls: `verify_step` and
    `next_token` cost one call each; `verify_block` -- one forward pass over
    the whole candidate block -- costs exactly one call regardless of block
    size. `verify_block` is exposed on this wrapper only when `inner`
    itself has one, so `drafter._verify_prefix`'s own
    `getattr(primary, "verify_block", None)` dispatch sees exactly the same
    capability the wrapped primary has -- wrapping a plain `verify_step`-only
    primary must never fake block support it does not have."""

    inner: PrimaryDecoder
    forward_calls: int = 0

    def __post_init__(self) -> None:
        if hasattr(self.inner, "verify_block"):
            self.verify_block = self._verify_block  # type: ignore[method-assign]

    def verify_step(self, prefix: tuple[int, ...], candidate_token: int) -> bool:
        self.forward_calls += 1
        return self.inner.verify_step(prefix, candidate_token)

    def next_token(self, prefix: tuple[int, ...]) -> int | None:
        self.forward_calls += 1
        return self.inner.next_token(prefix)

    def _verify_block(self, prefix: tuple[int, ...], candidates: tuple[int, ...]) -> int:
        self.forward_calls += 1
        return self.inner.verify_block(prefix, candidates)


def run_speculative_decode_with_accounting(
    primary: PrimaryDecoder,
    drafter: SpeculativeDrafter,
    *,
    max_tokens: int,
    start: DecoderSnapshot | None = None,
    commit: Callable[[DecoderSnapshot, tuple[int, ...]], DecoderSnapshot] = _default_commit,
) -> tuple[DecoderSnapshot, AccelerationReport]:
    """Exactly `run_speculative_decode`, plus the `AccelerationReport` above."""
    instrumented_drafter = _InstrumentedDrafter(inner=drafter)
    instrumented_primary = _InstrumentedPrimary(inner=primary)
    accepted_counts: list[int] = []
    forward_call_counts: list[int] = []
    calls_before = 0

    def _wrapped_commit(snapshot: DecoderSnapshot, accepted: tuple[int, ...]) -> DecoderSnapshot:
        nonlocal calls_before
        accepted_counts.append(len(accepted))
        forward_call_counts.append(instrumented_primary.forward_calls - calls_before)
        calls_before = instrumented_primary.forward_calls
        return commit(snapshot, accepted)

    result = run_speculative_decode(
        instrumented_primary,
        instrumented_drafter,
        max_tokens=max_tokens,
        start=start,
        commit=_wrapped_commit,
    )
    if len(instrumented_drafter.proposed_counts) == len(accepted_counts) + 1:
        # The round that found the end of the stream (`next_token` returned
        # `None`) drafted and paid forwards but committed nothing.
        accepted_counts.append(0)
        forward_call_counts.append(instrumented_primary.forward_calls - calls_before)
    rounds = tuple(
        RoundStats(proposed=proposed, accepted=accepted, primary_forward_calls=calls)
        for proposed, accepted, calls in zip(
            instrumented_drafter.proposed_counts,
            accepted_counts,
            forward_call_counts,
            strict=True,
        )
    )
    return result, AccelerationReport(rounds=rounds)
