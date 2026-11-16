"""RFC004 §37.2 offline export step (owner note, 2026-09-27): the full target
cannot run a full 300 DPI page prefill on a 12 GiB RTX 4070 (weights alone
~9 GiB, chunked prefill still OOMs) -- so the frozen target's own greedy
decode is run once, elsewhere (the H200), and only the *pinned hidden-state
taps* it produces plus its greedy token stream are written to disk. The
local 4070 trainer (:mod:`s7_dflash_train`) reads only these records; it
never instantiates the full target (see that module's docstring for the
proof).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


class HiddenTapPrimaryDecoder(Protocol):
    """Like `drafter.PrimaryDecoder`, plus the pinned hidden-state tap for
    the token it is about to emit. Only this export step needs both at once
    -- the drafter's own runtime adapter (`dflash.DflashSpeculativeDrafter`)
    takes taps from a caller-supplied provider instead, decoupled from any
    particular primary implementation."""

    def next_token_and_hidden(
        self, prefix: tuple[int, ...]
    ) -> tuple[int | None, torch.Tensor]: ...


class MultiLayerHiddenTapPrimaryDecoder(Protocol):
    """DSpark variant of :class:`HiddenTapPrimaryDecoder` (2026-09-27):
    a TUPLE of tapped-layer hidden states per token instead of one --
    `candidate_model.QwenDenseCandidateModel.encode_step`'s `hidden_taps`
    parameter is the intended source, one call per emitted token, matching
    `HiddenTapPrimaryDecoder`'s own single-token-at-a-time export shape."""

    def next_token_and_hidden_taps(
        self, prefix: tuple[int, ...]
    ) -> tuple[int | None, tuple[torch.Tensor, ...]]: ...


@dataclass(frozen=True, slots=True)
class DflashTapsRecord:
    """One page's offline training record: greedy token stream + the hidden
    tap recorded at each emitted position, bound to the exact target/config
    that produced them via `drafter_target_compatibility_hash`."""

    page_id: str
    hidden_taps: torch.Tensor  # (T, hidden_dim), row i is the tap for token i
    token_stream: tuple[int, ...]
    drafter_target_compatibility_hash: str
    dtype: str


def export_dflash_taps_record(
    *,
    primary: HiddenTapPrimaryDecoder,
    page_id: str,
    compatibility_hash: str,
    max_tokens: int,
    dtype: torch.dtype = torch.float32,
) -> DflashTapsRecord:
    """Greedy-walk `primary` up to `max_tokens`, recording the hidden tap at
    each step. No drafter, no gradient -- purely the frozen target's own
    greedy decode, matching `drafter.OracleDrafter`'s walk shape."""
    prefix: list[int] = []
    taps: list[torch.Tensor] = []
    for _ in range(max_tokens):
        token, hidden = primary.next_token_and_hidden(tuple(prefix))
        if token is None:
            break
        taps.append(hidden.detach().to(dtype).reshape(-1))
        prefix.append(token)
    stacked = torch.stack(taps) if taps else torch.empty(0, dtype=dtype)
    return DflashTapsRecord(
        page_id=page_id,
        hidden_taps=stacked,
        token_stream=tuple(prefix),
        drafter_target_compatibility_hash=compatibility_hash,
        dtype=str(dtype).removeprefix("torch."),
    )


def save_taps_record(record: DflashTapsRecord, path: Path) -> None:
    """Serialize one record to `path` (a single `torch.save` payload)."""
    payload = {
        "page_id": record.page_id,
        "hidden_taps": record.hidden_taps,
        "token_stream": list(record.token_stream),
        "drafter_target_compatibility_hash": record.drafter_target_compatibility_hash,
        "dtype": record.dtype,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_taps_record(path: Path, *, mmap: bool = True) -> DflashTapsRecord:
    """Load a record written by :func:`save_taps_record`. `mmap=True`
    (default, 2026-09-28 OOM fix) memory-maps `hidden_taps` instead of
    copying it into anonymous RAM: a trainer that never touches a page's
    tap tensor never pays for it, and the OS can reclaim the mapped pages
    under memory pressure instead of the kernel OOM-killing the process.
    `weights_only=True` is safe here -- the payload is only plain Python
    metadata (str/list) plus the tensor, all on the default safe-globals
    list; this file is our own trusted export, not an untrusted checkpoint."""
    payload = torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)
    return DflashTapsRecord(
        page_id=payload["page_id"],
        hidden_taps=payload["hidden_taps"],
        token_stream=tuple(payload["token_stream"]),
        drafter_target_compatibility_hash=payload["drafter_target_compatibility_hash"],
        dtype=payload["dtype"],
    )


@dataclass(frozen=True, slots=True)
class DSparkTapsRecord:
    """DSpark variant of :class:`DflashTapsRecord`: `hidden_taps` is
    `(T, n_taps, hidden_dim)` -- row `i` is the STACK of tapped-layer hidden
    states recorded for token `i`, in `dspark_hidden_taps` order."""

    page_id: str
    hidden_taps: torch.Tensor
    token_stream: tuple[int, ...]
    drafter_target_compatibility_hash: str
    dtype: str
    #: Optional `(T, d_model)` final (dense-adapted) hidden state at each
    #: fed token, row-aligned with `hidden_taps` -- what MTP heads read
    #: (`MtpDrafter.hidden_state_at`). Carried here (DSpark rehearsal
    #: 2026-09-27) so both drafters train from ONE export pass; `None`
    #: for records written without it.
    final_hidden: torch.Tensor | None = None


def export_dspark_taps_record(
    *,
    primary: MultiLayerHiddenTapPrimaryDecoder,
    page_id: str,
    compatibility_hash: str,
    max_tokens: int,
    dtype: torch.dtype = torch.float32,
) -> DSparkTapsRecord:
    """Greedy-walk `primary` up to `max_tokens`, recording the stacked
    multi-layer taps at each step -- otherwise identical to
    :func:`export_dflash_taps_record`."""
    prefix: list[int] = []
    taps: list[torch.Tensor] = []
    for _ in range(max_tokens):
        token, hidden_taps = primary.next_token_and_hidden_taps(tuple(prefix))
        if token is None:
            break
        taps.append(
            torch.stack([tap.detach().to(dtype).reshape(-1) for tap in hidden_taps])
        )
        prefix.append(token)
    stacked = torch.stack(taps) if taps else torch.empty(0, 0, dtype=dtype)
    return DSparkTapsRecord(
        page_id=page_id,
        hidden_taps=stacked,
        token_stream=tuple(prefix),
        drafter_target_compatibility_hash=compatibility_hash,
        dtype=str(dtype).removeprefix("torch."),
    )


def save_dspark_taps_record(record: DSparkTapsRecord, path: Path) -> None:
    """Serialize one DSpark record to `path`, mirroring :func:`save_taps_record`."""
    payload = {
        "page_id": record.page_id,
        "hidden_taps": record.hidden_taps,
        "token_stream": list(record.token_stream),
        "drafter_target_compatibility_hash": record.drafter_target_compatibility_hash,
        "dtype": record.dtype,
    }
    if record.final_hidden is not None:
        payload["final_hidden"] = record.final_hidden
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


class TeacherForcedMultiLayerPrimaryDecoder(Protocol):
    """Teacher-forced counterpart of :class:`MultiLayerHiddenTapPrimaryDecoder`
    (2026-09-28): ONE decoder forward over a whole gold token sequence
    (`encode_steps` with `k = len(gold_tokens)`, chunked if memory
    requires) instead of `len(gold_tokens)` free-running greedy steps.

    Off-policy (the gold sequence is the page's exact-text target, not
    necessarily what the frozen target would decode on its own) rather than
    on-policy (:class:`MultiLayerHiddenTapPrimaryDecoder`'s free-running
    greedy decode) -- see the DSpark rehearsal doc §17. The two agree
    wherever the target's own decode happens to equal the gold sequence
    (`test_matches_free_running_record_when_gold_equals_greedy_decode`)."""

    def hidden_taps_for_gold_sequence(
        self, gold_tokens: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def export_dspark_taps_record_teacher_forced(
    *,
    primary: TeacherForcedMultiLayerPrimaryDecoder,
    page_id: str,
    compatibility_hash: str,
    gold_tokens: Sequence[int],
    dtype: torch.dtype = torch.float32,
) -> DSparkTapsRecord:
    """Teacher-forced export: produces the SAME :class:`DSparkTapsRecord`
    shape and "tap from the last committed token" convention as
    :func:`export_dspark_taps_record` (`hidden_taps[i]`/`final_hidden[i]`
    is what feeding `gold_tokens[i]` taps, exactly what
    `TapStore.taps_after(i + 1)` reads online) -- so `TapStore`,
    `build_dspark_batch`, `measure_dspark_acceptance` and both trainers
    read either export unchanged. Costs ONE decoder forward per page
    instead of `len(gold_tokens)`."""
    gold = tuple(gold_tokens)
    if not gold:
        return DSparkTapsRecord(
            page_id=page_id,
            hidden_taps=torch.empty(0, 0, dtype=dtype),
            token_stream=(),
            drafter_target_compatibility_hash=compatibility_hash,
            dtype=str(dtype).removeprefix("torch."),
            final_hidden=torch.empty(0, 0, dtype=dtype),
        )
    hidden_taps, final_hidden = primary.hidden_taps_for_gold_sequence(gold)
    return DSparkTapsRecord(
        page_id=page_id,
        hidden_taps=hidden_taps.detach().to(dtype),
        token_stream=gold,
        drafter_target_compatibility_hash=compatibility_hash,
        dtype=str(dtype).removeprefix("torch."),
        final_hidden=final_hidden.detach().to(dtype),
    )


def load_dspark_taps_record(path: Path, *, mmap: bool = True) -> DSparkTapsRecord:
    """Load a record written by :func:`save_dspark_taps_record`. Same trust
    boundary and `mmap=True` default as :func:`load_taps_record` -- both
    `hidden_taps` (the multi-layer DSpark taps) and `final_hidden` (the MTP
    input) stay memory-mapped until a caller actually indexes into them."""
    payload = torch.load(path, map_location="cpu", mmap=mmap, weights_only=True)
    return DSparkTapsRecord(
        page_id=payload["page_id"],
        hidden_taps=payload["hidden_taps"],
        token_stream=tuple(payload["token_stream"]),
        drafter_target_compatibility_hash=payload["drafter_target_compatibility_hash"],
        dtype=payload["dtype"],
        final_hidden=payload.get("final_hidden"),
    )
