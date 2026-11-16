"""Reasoning telemetry aggregation -- RFC004-ERRATA-005 E3 (§22.7/§43.1).

E3 requires the runtime to report, without any document content: overall
and per-slice/per-class gate activation rate, gate entropy, dead-path
detection (activation 0 on eligible slots), and latency/VRAM overhead.
`ReasoningTelemetryAggregator` is a pure aggregator over per-call
`ReasoningRecord`s -- it holds counters, not tensors of document state, and
never sees a `ReasoningResult.auxiliary_predictions` value.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningRecord:
    """One call's telemetry input, keyed by caller-supplied slice/class
    labels -- never document text or node content, only identifiers the
    caller already has a governance basis for logging (§43.2)."""

    activated: bool
    eligible: bool
    gate_probability: float
    slice_id: str
    failure_class: str | None = None
    latency_ms: float | None = None
    vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.gate_probability <= 1.0:
            raise ValueError(f"gate_probability must be in [0, 1], got {self.gate_probability!r}")
        if self.activated and not self.eligible:
            raise ValueError("a record cannot be activated while ineligible")


@dataclass(frozen=True)
class ReasoningTelemetryReport:
    """The aggregated snapshot `ReasoningTelemetryAggregator.report()` returns."""

    call_count: int
    eligible_count: int
    activation_rate_overall: float
    activation_rate_per_slice: dict[str, float]
    activation_rate_per_class: dict[str, float]
    gate_entropy_mean: float
    dead_paths: frozenset[str]
    mean_latency_ms: float | None
    mean_vram_bytes: float | None


class ReasoningTelemetryAggregator:
    """Accumulate `ReasoningRecord`s and produce a `ReasoningTelemetryReport`.

    Pure in the sense that it only ever reads its own accumulated state and
    the records handed to `record()`; it makes no model calls and touches no
    document content.
    """

    def __init__(self) -> None:
        self._records: list[ReasoningRecord] = []

    def record(self, record: ReasoningRecord) -> None:
        self._records.append(record)

    def report(self) -> ReasoningTelemetryReport:
        eligible = [r for r in self._records if r.eligible]
        activated = [r for r in eligible if r.activated]

        per_slice_eligible: dict[str, int] = defaultdict(int)
        per_slice_activated: dict[str, int] = defaultdict(int)
        for r in eligible:
            per_slice_eligible[r.slice_id] += 1
            if r.activated:
                per_slice_activated[r.slice_id] += 1
        activation_rate_per_slice = {
            slice_id: per_slice_activated.get(slice_id, 0) / count
            for slice_id, count in per_slice_eligible.items()
        }

        class_counts: Counter[str] = Counter(
            r.failure_class for r in eligible if r.failure_class is not None
        )
        class_activated: Counter[str] = Counter(
            r.failure_class
            for r in activated
            if r.failure_class is not None
        )
        activation_rate_per_class = {
            cls: class_activated.get(cls, 0) / count for cls, count in class_counts.items()
        }

        dead_paths = frozenset(
            slice_id
            for slice_id, count in per_slice_eligible.items()
            if count > 0 and per_slice_activated.get(slice_id, 0) == 0
        )

        entropies = [_binary_entropy(r.gate_probability) for r in self._records]
        gate_entropy_mean = sum(entropies) / len(entropies) if entropies else 0.0

        latencies = [r.latency_ms for r in self._records if r.latency_ms is not None]
        vram = [r.vram_bytes for r in self._records if r.vram_bytes is not None]

        return ReasoningTelemetryReport(
            call_count=len(self._records),
            eligible_count=len(eligible),
            activation_rate_overall=(len(activated) / len(eligible)) if eligible else 0.0,
            activation_rate_per_slice=dict(activation_rate_per_slice),
            activation_rate_per_class=dict(activation_rate_per_class),
            gate_entropy_mean=gate_entropy_mean,
            dead_paths=dead_paths,
            mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
            mean_vram_bytes=(sum(vram) / len(vram)) if vram else None,
        )


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
