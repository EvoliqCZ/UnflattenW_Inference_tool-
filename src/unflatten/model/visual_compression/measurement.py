"""RFC005-ERRATA-007 E5 rule 2: the VC-0 bypass measurement harness.

VC-0 measures the *bypass* path on the frozen primary — no gradient, no
compressor — so the owner can decide whether compression has anything to
save before VC-1 through VC-4 spend any training budget (rule 2: "VC-0 môže
ukončiť program"). It needs the phases split (encoder vs. prefill vs.
autoregressive decode step) because neither number the errata cites splits
them, and post-encoder compression only ever touches the last two.

Runs on CPU with a stub candidate; peak VRAM is ``None`` off CUDA, never a
fabricated number.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch


class MeasuredCandidate(Protocol):
    """The three phases VC-0 times separately."""

    def encode(self, page: Any) -> torch.Tensor:
        """Vision encoder: page -> visual tokens, shape (..., tokens, d_model)."""
        ...

    def encode_prefix(self, encoded_pages: Sequence[torch.Tensor]) -> Any:
        """Prefill: encoded pages -> decoder prefix state."""
        ...

    def encode_step(self, prefix_state: Any) -> Any:
        """One autoregressive decode step from the prefix state."""
        ...


@dataclass(frozen=True)
class PhaseTiming:
    encoder_s: float
    encode_prefix_s: float
    encode_step_s: float


@dataclass(frozen=True)
class SliceQuality:
    """One per-slice quality reading (RFC005 §21.2 "small fonts, diacritics, ...")."""

    slice_name: str
    score: float


@dataclass(frozen=True)
class BaselineMeasurement:
    token_count: int
    peak_vram_bytes: int | None
    timing: PhaseTiming
    slice_quality: tuple[SliceQuality, ...]


def _default_vram_probe() -> int | None:
    if torch.cuda.is_available():  # pragma: no cover — exercised only on a CUDA host
        return torch.cuda.max_memory_allocated()
    return None


def measure_compression_baseline(
    candidate: MeasuredCandidate,
    pages: Sequence[Any],
    *,
    quality_hook: Callable[[Sequence[torch.Tensor]], Iterable[SliceQuality]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    vram_probe: Callable[[], int | None] = _default_vram_probe,
) -> BaselineMeasurement:
    """Measure the bypass path over ``pages`` with an injectable ``clock``.

    ``clock`` defaults to ``time.perf_counter`` but tests should inject a
    deterministic fake — real wall-clock timing has no place in an assertion.
    """
    if not pages:
        raise ValueError("measure_compression_baseline requires at least one page")

    t0 = clock()
    encoded = [candidate.encode(page) for page in pages]
    t1 = clock()
    prefix_state = candidate.encode_prefix(encoded)
    t2 = clock()
    candidate.encode_step(prefix_state)
    t3 = clock()

    token_count = sum(tensor.shape[-2] for tensor in encoded)
    quality = tuple(quality_hook(encoded)) if quality_hook is not None else ()

    return BaselineMeasurement(
        token_count=token_count,
        peak_vram_bytes=vram_probe(),
        timing=PhaseTiming(encoder_s=t1 - t0, encode_prefix_s=t2 - t1, encode_step_s=t3 - t2),
        slice_quality=quality,
    )
