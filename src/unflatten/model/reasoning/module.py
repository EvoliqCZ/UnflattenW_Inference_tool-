"""`ReasoningModule` contract -- RFC004-ERRATA-005 E1 §19 (~:150-163).

The contract is deliberately narrow: a reasoning specialization consumes the
shared document/visual state plus an eligibility mask and returns a residual
into that same shared state, never an independent document (rule 2), never
free-text chain-of-thought (rule 3, §61.22), and never a field forbidden to
the router (rule 5: no split, source, label grade, TaskView id, target
values, hidden PDF text, filename or provenance).

Nothing here instantiates a specialization -- see `registry.py`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReasoningTelemetry:
    """Per-call diagnostics, **without document content** (E1 rule 3, §43.2).

    `gate_activated` and `eligibility_masked` are per-call booleans;
    `telemetry.py` aggregates many of these into rates. `attach_point`
    records which point (OPEN-004-014) produced this call. Latency/VRAM are
    optional because a CPU-only or synthetic call may not measure them.
    """

    gate_activated: bool
    eligibility_masked: bool
    attach_point: str
    latency_ms: float | None = None
    vram_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.attach_point not in ("token", "node"):
            raise ValueError(
                f"attach_point must be 'token' or 'node' (OPEN-004-014), got "
                f"{self.attach_point!r}"
            )
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {self.latency_ms!r}")
        if self.vram_bytes is not None and self.vram_bytes < 0:
            raise ValueError(f"vram_bytes must be >= 0, got {self.vram_bytes!r}")


@dataclass(frozen=True)
class ReasoningResult:
    """The full output of one `ReasoningModule.forward` call (E1 §19).

    `residual` shares `d_model` with the document state it will be added to
    (§23). `gate` is the per-slot `{0,1}` decision that produced this call
    (0 everywhere when `reasoning=off`, rule 1). `auxiliary_predictions` may
    only be populated by heads registered in a `reasoning-aux-head-registry`
    (rule 10) -- this contract does not enforce that registration itself,
    since no such registry exists yet in Wave 1; it only carries the field.
    `confidence` is diagnostic, never authoritative (§36).
    """

    residual: Tensor
    gate: Tensor
    auxiliary_predictions: Mapping[str, Tensor]
    confidence: Tensor
    telemetry: ReasoningTelemetry

    def __post_init__(self) -> None:
        # `gate` is per-slot ({batch, slots}); it either matches `residual`
        # exactly (elementwise gating) or matches all but the trailing
        # feature dim (per-slot scalar gate broadcast over `d_model`).
        slot_shape = self.residual.shape[:-1]
        if self.gate.shape not in (self.residual.shape, slot_shape):
            raise ValueError(
                f"gate shape {tuple(self.gate.shape)!r} must equal residual shape "
                f"{tuple(self.residual.shape)!r} or its slot shape {tuple(slot_shape)!r}"
            )
        if not torch.all((self.gate == 0) | (self.gate == 1)):
            raise ValueError("gate must be hard {0,1} valued in a ReasoningResult")


class ReasoningModule(Protocol):
    """A registered reasoning specialization (E1 §19).

    `spec_id` must be `reasoning.<family>.<version>` -- see `registry.py` for
    the pattern this is validated against at registration time, not here:
    the protocol itself does not enforce naming, only shape.
    """

    spec_id: str

    def forward(
        self,
        document_state: Tensor,
        visual_state: Tensor,
        attention_mask: Tensor,
        eligibility_mask: Tensor,
        context: Mapping[str, Any],
    ) -> ReasoningResult: ...
