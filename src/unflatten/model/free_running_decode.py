"""Deterministic constrained greedy free-running decode -- design §5.7 D7,
§8 M3 items 1 and 3.

`decode_free_running` is the "no gold prefix" path design §8's M3 exit
criterion names: every accepted token comes from `model` + `controller`,
never from a target sequence.

**KV-cache path (design §20), the DEFAULT since that design.** A previous
revision of this module recomputed the WHOLE sequence through
`QwenDenseCandidateModel.encode_hidden` at every step and this docstring
cited `candidate_loader.CandidateDecoderAdapter.decode_step` as a
KV-cache contract for "a LATER stage" -- no such class or method exists,
or ever existed, anywhere in this tree; that was a forward reference to a
design section that had not been written yet, and it should not have kept
citing something unbuilt as though it were a real seam. Design §20 measured
the real cost of not having it: 437 s/page, ~10.8M token-forwards, because
the ~8,580-token visual prefix was recomputed at every one of ~1,200 steps.
`decode_free_running` now builds the prefix ONCE
(`QwenDenseCandidateModel.encode_prefix`, `use_cache=True`) and extends it
one token at a time (`encode_step`) -- ~9,800 token-forwards instead,
roughly a thousandfold less work. `use_kv_cache=False` forces the old
full-recompute path (`encode_hidden`, every step); it exists specifically
so the two paths can be decoded against each other and their accepted
token sequences compared byte-for-byte (`tests/model/test_free_running_
decode_kv_cache.py`) -- a cache that changed even one greedy decision would
have changed what the model says, not just how fast it says it.

Stop/error semantics (design §5.7, §16 criterion 6, task hard constraint
"never truncate-and-accept"):

- reaching `mode == "terminated"` (a legal `END_DOCUMENT` at empty stack in
  document mode, or a legal `END_FRAGMENT` at balanced stack in fragment
  mode -- `grammar.py`'s fragment protocol, §18.8) is the only success path,
- running out of `max_steps` first raises :class:`DecodeMaxStepsExceeded`,
  never returns a truncated hypothesis,
- a legal-token-set that comes back empty before termination raises
  :class:`DecodeVocabularyExhausted` -- this should be unreachable (the
  grammar always offers >= 1 legal continuation until it terminates) but is
  checked explicitly rather than let an argmax over an all-`-inf` row pick
  an arbitrary index and call it a decision.

The greedy tie-break is PINNED to `"lowest_packed_id"`, matching
`composite_vocab.build_composite_vocabulary`'s own `greedy_tie_break`
default -- `DecodeConfig` accepts the field to make the pin visible and
auditable, not to make it configurable; any other value raises rather than
silently decoding under an unpinned policy.

**Visual placeholder prefix, matching `candidate_model.py`'s native
injection.** `QwenDenseCandidateModel.encode_hidden` now scatters visual
content INTO the composite sequence at `image_token_id` placeholder
positions (`inject_visual`, `candidate_loader.py`) rather than a separate
prefix span. `composite_input_ids` must therefore already carry one packed
`QwenRef(image_token_id)` per visual token, in order, ahead of whatever the
grammar generates -- this module builds that fixed placeholder block once
from `visual_features.shape[1]` and prepends it to every step's growing
sequence. The placeholder block is never grammar-controlled (it never
passes through `controller.step`, never creates a `NodeState` -- it is
materialized visual context, not a decoded event) and never appears in the
returned `RawDocumentHypothesis.accepted_tokens`, which holds only the
grammar-generated tail.

This module authorizes no gradient update: `torch.inference_mode()` wraps
the whole decode loop, and nothing here builds an optimizer or calls
`.backward()`.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from unflatten.model import events_vocab, header_length_cap, page_start_policy, story_retry
from unflatten.model.candidate_model import (
    KVCacheSnapshot,
    QwenDenseCandidateModel,
    TeacherForcedBatch,
)
from unflatten.model.capability_profile import (
    STANDALONE_PAGE_PROFILE_IDS,
)
from unflatten.model.composite_grammar import CompositeGrammarController
from unflatten.model.composite_vocab import (
    ByteControlRef,
    ByteValueRef,
    CompositeVocabulary,
    PackedRanges,
    QwenRef,
    StructuralRef,
    pack,
    unpack,
)
from unflatten.model.loop_guard_v2_caps import (
    PERIOD_MAX,
    LoopGuardV2Caps,
    load_loop_guard_v2_caps,
)
from unflatten.model.node_lifecycle import NodeLifecycleTracker, NodeState
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.model.region_decode import decode_regions
from unflatten.model.typed_decode import decode_typed

#: The only tie-break policy this module implements (see module docstring).
PINNED_TIE_BREAK = "lowest_packed_id"

_TEXT_END_LOCAL_ID = events_vocab.TOKEN_IDS[events_vocab.TEXT_END]

#: DEC-031 loop guard v1 caps -- `LoopGuardV1`'s own constructor defaults.
#: `MAX_CONSECUTIVE_TAB` and `MAX_CONSECUTIVE_EMPTY_PARAGRAPH` were revised
#: by DEC-033 / RFC003-ERRATA-002 / RFC004-ERRATA-002 (R3/R9): under
#: `STANDALONE_V1_PROFILE_ID` the DEC-033 page projection guarantees the
#: post-projection data maximum for both patterns is 3, so the caps are set
#: to that measured maximum rather than the pre-projection 202/53. Every
#: OTHER capability profile -- including the pre-DEC-033
#: `s2-typed-widened-v1` -- must keep reproducing its v12 targets exactly,
#: so `decode_free_running` selects `_LEGACY_MAX_CONSECUTIVE_TAB`/
#: `_LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPH` for those instead (same
#: `capability_profile_id` gate as `CompositeGrammarController.
#: _dec033_bounds_active`). `MAX_CONSECUTIVE_EMPTY_ROW` is unchanged for
#: every profile (projection does not shorten empty rows).
#: `MAX_STRUCTURAL_WITHOUT_TEXT` is left at its pre-projection value here
#: and will be re-measured on the rebuilt cache with `ops/runlen.py` (R9);
#: DEC-031 will record the new number.
MAX_CONSECUTIVE_TAB = 3
MAX_CONSECUTIVE_EMPTY_PARAGRAPH = 3
MAX_CONSECUTIVE_EMPTY_ROW = 75
MAX_STRUCTURAL_WITHOUT_TEXT = 2881

#: Pre-DEC-033 caps (DEC-031's original measurement), still the correct
#: bound for every capability profile other than `STANDALONE_V1_PROFILE_ID`.
_LEGACY_MAX_CONSECUTIVE_TAB = 202
_LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPH = 53

_TAB_NAME = events_vocab.LEAF_TOKEN["tab"]
_PARAGRAPH_OPEN_NAME = events_vocab.OPEN_TOKEN["paragraph"]
_PARAGRAPH_CLOSE_NAME = events_vocab.CLOSE_TOKEN["paragraph"]
_ROW_OPEN_NAME = events_vocab.OPEN_TOKEN["row"]
_ROW_CLOSE_NAME = events_vocab.CLOSE_TOKEN["row"]
_CELL_OPEN_NAME = events_vocab.OPEN_TOKEN["cell"]
_CELL_CLOSE_NAME = events_vocab.CLOSE_TOKEN["cell"]
_TEXT_BEGIN_NAME = events_vocab.TEXT_BEGIN
_END_FRAGMENT_NAME = events_vocab.END_FRAGMENT
_END_DOCUMENT_NAME = events_vocab.END_DOCUMENT


class LoopGuardV1:
    """DEC-031's structural loop guard: level 1 bans a pattern from
    repeating past its measured maximum, level 2 forces a close-out once a
    page has gone too long without any text at all. Model-free and pure
    enough to test without a model, controller or vocabulary construction
    beyond `_legal_packed_ids`'s own -- `filter`/`observe` take packed ids
    and token names only.

    **Empty-paragraph variant (plan task 2, "choose ONE and explain
    why").** DEC-031's primary phrasing bans the token that would OPEN the
    next repetition of a saturated pattern, once that repetition's own run
    has completed -- exactly how the TAB and empty-row bans below already
    work (TAB itself is banned once its own run hits the cap; `ROW_OPEN`
    is banned once the empty-row run hits the cap and the immediately
    preceding token was that row's `ROW_CLOSE`). Banning `PARAGRAPH_OPEN`
    under the same rule (once the empty-paragraph run hits the cap and the
    preceding token was that pair's `PARAGRAPH_CLOSE`) keeps all three
    level-1 bans one shape -- "ban the token that starts the next
    repetition" -- rather than two shapes plus a special case for
    paragraphs; the plan's stated alternative (ban `PARAGRAPH_CLOSE` right
    after an `PARAGRAPH_OPEN`, forcing that one paragraph to carry content)
    is equally correct but would be the only ban of the three that fires
    on the OPEN side of the pattern instead of the CLOSE side.

    **Row/paragraph run tracking uses a simple reset**, not
    `ops/runlen.py`'s exact backtracking scan: any token that does not
    continue the pattern currently being matched resets that run to zero
    immediately, rather than the batch algorithm's ability to still credit
    a later, cleanly-matching row/pair. This can only make the guard fire
    LATER than the measurement's own maxima would predict, never earlier
    (plan's "Note on resets") -- the guard is at least as permissive as
    what `ops/runlen.py` measured, so the ratified caps still mean what
    DEC-031 says they mean.
    """

    def __init__(
        self,
        *,
        max_consecutive_tab: int = MAX_CONSECUTIVE_TAB,
        max_consecutive_empty_paragraph: int = MAX_CONSECUTIVE_EMPTY_PARAGRAPH,
        max_consecutive_empty_row: int = MAX_CONSECUTIVE_EMPTY_ROW,
        max_structural_without_text: int = MAX_STRUCTURAL_WITHOUT_TEXT,
    ) -> None:
        self._max_tab = max_consecutive_tab
        self._max_ep = max_consecutive_empty_paragraph
        self._max_er = max_consecutive_empty_row
        self._max_struct_no_text = max_structural_without_text

        self._tab_run = 0
        self._ep_run = 0
        self._er_run = 0
        self._awaiting_paragraph_close = False
        self._row_state = "outside"
        self._row_cells = 0
        self._structural_since_text = 0
        self._last_name: str | None = None

        #: DEC-031 point 5: every guard-touched page carries its own hit
        #: counts, never a silent trim.
        self.counters: dict[str, int] = {
            "ban_tab": 0,
            "ban_empty_paragraph": 0,
            "ban_empty_row": 0,
            "closeout": 0,
            "noop": 0,
        }

    def snapshot(self) -> LoopGuardV1:
        """Independent point-in-time copy for speculative decode's per-
        position guard state (`decode_integration.py`'s `_guard_snapshot`).
        Every field here is a scalar or a small fixed-shape dict, so a
        plain field copy is enough -- see `LoopGuardV2.snapshot` for why
        this replaces `copy.deepcopy`."""
        clone = LoopGuardV1.__new__(LoopGuardV1)
        clone._max_tab = self._max_tab
        clone._max_ep = self._max_ep
        clone._max_er = self._max_er
        clone._max_struct_no_text = self._max_struct_no_text
        clone._tab_run = self._tab_run
        clone._ep_run = self._ep_run
        clone._er_run = self._er_run
        clone._awaiting_paragraph_close = self._awaiting_paragraph_close
        clone._row_state = self._row_state
        clone._row_cells = self._row_cells
        clone._structural_since_text = self._structural_since_text
        clone._last_name = self._last_name
        clone.counters = dict(self.counters)
        return clone

    def filter(
        self, legal_ids: list[int], *, vocabulary: CompositeVocabulary, mode: str
    ) -> tuple[list[int], str | None]:
        """Narrow `legal_ids` per the current run-length state; call once
        per step, BEFORE `_select_greedy`, on the set `_legal_packed_ids`
        just built. Returns `(ids, event)` -- `event` names the last
        counter this call incremented (`None` if neither level fired).
        Never returns an empty list (DEC-031 point 3): a filter that would
        leave nothing keeps the set it started from and records `noop`.

        `mode` is the grammar's current `AllowedDomain.mode`
        (`"structure"`/`"text"`/`"byte"`, `_legal_packed_ids`'s own
        values). Two fast paths avoid `_structural_names` -- which
        unpacks every id in `legal_ids` -- unless there is real work to
        do (must_fix, coordinator follow-up: this was costing a text
        step the same ~148k-entry unpack `_packed_text_allowlist`'s own
        docstring measured at 22.6% of a decode, AND inflating `noop`
        once close-out was active, since every text step "failed" a
        filter that made no sense there):

        1. `mode != "structure"`: TAB/PARAGRAPH_OPEN/ROW_OPEN/`*_CLOSE`/
           `END_*` are never legal outside structure mode, so there is
           nothing for this guard to filter -- return `legal_ids`
           unchanged, no counter touched.
        2. No level-1 ban pending AND close-out not yet triggered: return
           `legal_ids` unchanged before computing `names` at all."""
        if mode != "structure":
            return legal_ids, None

        ban_name, ban_event = self._level1_ban()
        closeout_pending = self._structural_since_text >= self._max_struct_no_text
        if ban_name is None and not closeout_pending:
            return legal_ids, None

        names = _structural_names(legal_ids, vocabulary)
        ids = legal_ids
        event: str | None = None

        if ban_name is not None:
            narrowed = [i for i in ids if names.get(i) != ban_name]
            if narrowed:
                ids = narrowed
                self.counters[ban_event] += 1
                event = ban_event
            else:
                self.counters["noop"] += 1
                event = "noop"

        if names and closeout_pending:
            closing = {n for n in names.values() if n.endswith("_CLOSE")} | {
                _END_FRAGMENT_NAME,
                _END_DOCUMENT_NAME,
            }
            narrowed = [i for i in ids if names.get(i) in closing]
            if narrowed:
                if narrowed != ids:
                    self.counters["closeout"] += 1
                    event = "closeout"
                ids = narrowed
            else:
                self.counters["noop"] += 1
                event = "noop"

        return ids, event

    def _level1_ban(self) -> tuple[str | None, str | None]:
        if self._tab_run >= self._max_tab:
            return _TAB_NAME, "ban_tab"
        if self._ep_run >= self._max_ep and self._last_name == _PARAGRAPH_CLOSE_NAME:
            return _PARAGRAPH_OPEN_NAME, "ban_empty_paragraph"
        if self._er_run >= self._max_er and self._last_name == _ROW_CLOSE_NAME:
            return _ROW_OPEN_NAME, "ban_empty_row"
        return None, None

    def observe(self, name: str | None) -> None:
        """Update run-length state after `name` -- the structural token
        name actually accepted, or `None` for a text/byte-mode token --
        was chosen. Call AFTER `_select_greedy`: `filter` reads the state
        `observe` last left behind."""
        self._tab_run = self._tab_run + 1 if name == _TAB_NAME else 0

        if name == _PARAGRAPH_OPEN_NAME:
            self._awaiting_paragraph_close = True
        elif name == _PARAGRAPH_CLOSE_NAME and self._awaiting_paragraph_close:
            self._ep_run += 1
            self._awaiting_paragraph_close = False
        else:
            self._ep_run = 0
            self._awaiting_paragraph_close = False

        self._observe_row(name)

        if name == _TEXT_BEGIN_NAME:
            self._structural_since_text = 0
        elif name is not None:
            self._structural_since_text += 1

        self._last_name = name

    def _observe_row(self, name: str | None) -> None:
        if name == _ROW_OPEN_NAME:
            self._row_state = "expect_cell_open"
            self._row_cells = 0
            return
        if self._row_state == "outside":
            return
        if self._row_state == "expect_cell_open":
            if name == _CELL_OPEN_NAME:
                self._row_state = "expect_para_open"
            elif name == _ROW_CLOSE_NAME:
                self._er_run = self._er_run + 1 if self._row_cells >= 1 else 0
                self._row_state = "outside"
            else:
                self._er_run = 0
                self._row_state = "dirty"
            return
        if self._row_state == "expect_para_open":
            if name == _PARAGRAPH_OPEN_NAME:
                self._row_state = "expect_para_close"
            else:
                self._er_run = 0
                self._row_state = "dirty"
            return
        if self._row_state == "expect_para_close":
            if name == _PARAGRAPH_CLOSE_NAME:
                self._row_state = "expect_cell_close"
            else:
                self._er_run = 0
                self._row_state = "dirty"
            return
        if self._row_state == "expect_cell_close":
            if name == _CELL_CLOSE_NAME:
                self._row_cells += 1
                self._row_state = "expect_cell_open"
            else:
                self._er_run = 0
                self._row_state = "dirty"
            return
        if self._row_state == "dirty" and name == _ROW_CLOSE_NAME:
            self._row_state = "outside"


def _structural_names(
    legal_ids: Iterable[int], vocabulary: CompositeVocabulary
) -> dict[int, str]:
    """`{packed_id: token_name}` for exactly the `legal_ids` that unpack to
    a `StructuralRef` -- text/byte-mode ids (never a loop-guard concern)
    are simply absent from the result, rather than mapped to `None`."""
    names: dict[int, str] = {}
    for packed_id in legal_ids:
        token = vocabulary.unpack(packed_id)
        if isinstance(token, StructuralRef):
            names[packed_id] = _structural_name(token.local_id)
    return names


_CLOSE_NAME_TO_NODE_TYPE = {name: node_type for node_type, name in events_vocab.CLOSE_TOKEN.items()}
_TERMINATOR_NAMES = frozenset({_END_FRAGMENT_NAME, _END_DOCUMENT_NAME})
_CLOSER_NAMES = frozenset({events_vocab.TEXT_END}) | _TERMINATOR_NAMES


def _closer_name(token: Any) -> str | None:
    """The token's name if it is a closer (design §2 common definitions):
    `*_CLOSE`, `TEXT_END`, `END_FRAGMENT`, `END_DOCUMENT` or `byte_end`."""
    if isinstance(token, StructuralRef):
        name = _structural_name(token.local_id)
        return name if name.endswith("_CLOSE") or name in _CLOSER_NAMES else None
    if isinstance(token, ByteControlRef) and token.kind == "byte_end":
        return "byte_end"
    return None


def _depth_delta(token: Any) -> int:
    """+1 opener (`*_OPEN`, `TEXT_BEGIN`, `byte_begin`), -1 closer, 0
    everything else (leaves, Qwen tokens, byte values)."""
    if _closer_name(token) is not None:
        return -1
    if isinstance(token, StructuralRef):
        name = _structural_name(token.local_id)
        return 1 if name.endswith("_OPEN") or name == _TEXT_BEGIN_NAME else 0
    if isinstance(token, ByteControlRef) and token.kind == "byte_begin":
        return 1
    return 0


class LoopGuardV2:
    """DEC-036 loop guard v2 (workflow `spec-loop-guard-v2`, design §2):
    three rules on top of `LoopGuardV1`, which still runs first and
    unchanged.

    1. `text_run`: a Qwen token may not extend a run of itself past
       `caps.text_run_cap`. Any non-Qwen token resets the run.
    2. `periodic`: one detector over the whole packed stream for periods
       1..32. `unit_counts` holds, per canonical unit (smallest rotation),
       the completed repeats on this page; it is never reset, so a loop
       that is broken and re-entered resumes its count (owner: "strop sa
       nenuluje iným tokenom"). Only net-depth-0 units count; the veto
       lands only on the unit's anchor -- the first minimum-depth position
       of the canonical rotation, provably never a closer for a net-0
       unit -- once the count exceeds `caps.periodic_cap[p]`.
    3. `close_recovery`: when the unmasked argmax is an illegal closer
       (and, under variant B/C, names a frame that is actually open), the
       innermost legal close is emitted instead.

    Rules 1 and 2 are asked about the token greedy already chose; a
    vetoed id is excluded and greedy runs again (§2 "Veto semantics"), so
    an intervention is recorded only when it changed the decision.

    `record_only=True` is the measurement mode `scripts/loop_guard_v2_
    caps.py` runs over gold streams: nothing is vetoed, and
    `max_text_run`/`max_anchor_count` record the largest run and the
    largest count held at an anchor event per period -- the same
    quantities the veto compares against the caps, computed by this same
    class.
    """

    #: Rule 1 can veto one id per step and rule 2 one per period.
    MAX_VETOED_PER_STEP = 1 + PERIOD_MAX

    def __init__(
        self,
        *,
        ranges: PackedRanges,
        caps: LoopGuardV2Caps | None = None,
        record_only: bool = False,
    ) -> None:
        if caps is None and not record_only:
            raise ValueError("LoopGuardV2 needs caps unless record_only=True")
        self._ranges = ranges
        self._caps = caps
        self._record_only = record_only
        self._hist: list[int] = []
        self._runs = [0] * (PERIOD_MAX + 1)
        self._tokens: dict[int, Any] = {}
        self._windows: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
        self._anchors: dict[tuple[int, ...], int] = {}
        self._text_run_token: int | None = None
        self._text_run_len = 0

        #: `C` in the design: completed repeats per canonical unit.
        self.unit_counts: dict[tuple[int, ...], int] = {}
        self.counters: dict[str, int] = {
            "text_run": 0,
            "periodic": 0,
            "close_recovery": 0,
            "close_recovery_unavailable": 0,
            "noop": 0,
        }
        self.interventions: list[dict[str, Any]] = []
        self.max_text_run = 0
        self.max_anchor_count: dict[int, int] = {}

    def snapshot(self) -> LoopGuardV2:
        """Independent point-in-time copy of the guard's DECODE state --
        `_hist`, `_runs`, `unit_counts`, the text-run fields, `counters`,
        `interventions`, `max_text_run`, `max_anchor_count` -- cheap enough
        to take once per candidate position in speculative decode's block
        verification (`decode_integration.py`'s `_store`, one call per
        accepted position).

        `_tokens`/`_windows`/`_anchors` are memoisation caches, pure
        functions of their keys (`_token`/`_window`/`_anchor` above never
        consult anything but `self._ranges`, fixed for the guard's whole
        life) -- they are SHARED by reference with the clone, never copied.
        A page carries thousands of these entries by the time the guard's
        `_hist` is long, so `copy.deepcopy`, which does not know they are
        safe to alias, used to re-walk and re-copy all of them on every
        single accepted token -- the dominant cost this method removes
        (`docs/runs/dspark-rehearsal-2026-09-27.md` §9 point 4). Sharing is
        safe in both directions: if either the original or the clone later
        computes a new cache entry, the other sees it too, but every cached
        value is the unique correct answer for its key regardless of which
        guard computed it first, so an extra entry never changes a
        decision -- exactly what `test_decode_integration_guard_block.py`'s
        own `_guard_state` helper already assumes when it leaves these
        three caches out of its equivalence check."""
        clone = LoopGuardV2.__new__(LoopGuardV2)
        clone._ranges = self._ranges
        clone._caps = self._caps
        clone._record_only = self._record_only
        clone._hist = list(self._hist)
        clone._runs = list(self._runs)
        clone._tokens = self._tokens
        clone._windows = self._windows
        clone._anchors = self._anchors
        clone._text_run_token = self._text_run_token
        clone._text_run_len = self._text_run_len
        clone.unit_counts = dict(self.unit_counts)
        clone.counters = dict(self.counters)
        clone.interventions = list(self.interventions)
        clone.max_text_run = self.max_text_run
        clone.max_anchor_count = dict(self.max_anchor_count)
        return clone

    # -- token classification -------------------------------------------

    def _token(self, packed_id: int) -> Any:
        token = self._tokens.get(packed_id)
        if token is None:
            token = unpack(packed_id, self._ranges)
            self._tokens[packed_id] = token
        return token

    def _window(self, window: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
        """`(canonical rotation, net depth)` of `window`, memoised."""
        cached = self._windows.get(window)
        if cached is None:
            canon = min(window[k:] + window[:k] for k in range(len(window)))
            net = sum(_depth_delta(self._token(x)) for x in window)
            cached = (canon, net)
            self._windows[window] = cached
        return cached

    def _anchor(self, window: tuple[int, ...]) -> int:
        """Index into `window` of the unit's anchor: the first position of
        the canonical rotation whose before-depth is minimal, mapped back.
        Call only for net-depth-0 windows (design §2.2 proof)."""
        cached = self._anchors.get(window)
        if cached is not None:
            return cached
        canon, _net = self._window(window)
        k = next(k for k in range(len(window)) if window[k:] + window[:k] == canon)
        depth, best, best_at = 0, None, 0
        for j, packed_id in enumerate(canon):
            if best is None or depth < best:
                best, best_at = depth, j
            depth += _depth_delta(self._token(packed_id))
        index = (best_at + k) % len(window)
        if _closer_name(self._token(window[index])) is not None:
            raise RuntimeError(f"periodic anchor is a closer: window {window}, index {index}")
        self._anchors[window] = index
        return index

    # -- rules 1 and 2 ----------------------------------------------------

    def observe(self, packed_id: int) -> None:
        """Accept `packed_id` (call once per accepted token, after
        `controller.step`)."""
        if isinstance(self._token(packed_id), QwenRef):
            if packed_id == self._text_run_token:
                self._text_run_len += 1
            else:
                self._text_run_token, self._text_run_len = packed_id, 1
            self.max_text_run = max(self.max_text_run, self._text_run_len)
        else:
            self._text_run_token, self._text_run_len = None, 0

        hist, runs = self._hist, self._runs
        n = len(hist)
        for p in range(1, PERIOD_MAX + 1):
            runs[p] = runs[p] + 1 if n >= p and packed_id == hist[n - p] else 0
        hist.append(packed_id)
        for p in range(1, PERIOD_MAX + 1):
            if runs[p] >= p and runs[p] % p == 0:
                canon, net = self._window(tuple(hist[-p:]))
                if net == 0:
                    self.unit_counts[canon] = self.unit_counts.get(canon, 0) + 1

    def veto(self, packed_id: int) -> tuple[str, int | None, int] | None:
        """Would rule 1 or 2 refuse `packed_id` next? `(rule, period,
        count)` or `None`. In `record_only` mode this records the maxima
        and never vetoes."""
        if (
            not self._record_only
            and packed_id == self._text_run_token
            and self._text_run_len >= self._caps.text_run_cap
        ):
            return ("text_run", None, self._text_run_len)

        hist = self._hist
        n = len(hist)
        for p in range(1, min(PERIOD_MAX, n) + 1):
            if packed_id != hist[n - p]:
                continue
            window = tuple(hist[n - p :])
            canon, net = self._window(window)
            if net != 0 or self._anchor(window) != 0:
                continue
            count = self.unit_counts.get(canon, 0)
            if self._record_only:
                self.max_anchor_count[p] = max(self.max_anchor_count.get(p, 0), count)
            elif count > self._caps.periodic_cap[p]:
                return ("periodic", p, count)
        return None

    def select(
        self,
        logits_row: torch.Tensor,
        legal_ids: list[int],
        *,
        step: int,
        greedy: int | None = None,
    ) -> int:
        """Greedy over `legal_ids`, re-run while rule 1 or 2 vetoes the
        choice. If every legal id is vetoed, greedy's own choice is kept
        and a `noop` is recorded (DEC-031 point 3)."""
        if greedy is None:
            greedy = _select_greedy(logits_row, legal_ids)
        chosen = greedy
        vetoed: list[int] = []
        first_event = None
        while (event := self.veto(chosen)) is not None:
            first_event = first_event or event
            vetoed.append(chosen)
            if len(vetoed) > self.MAX_VETOED_PER_STEP:
                raise RuntimeError(f"loop guard v2 vetoed {len(vetoed)} ids in one step")
            excluded = set(vetoed)
            remaining = [i for i in legal_ids if i not in excluded]
            if not remaining:
                self._record(step, "noop", greedy, greedy, vetoed)
                return greedy
            chosen = _select_greedy(logits_row, remaining)
        if first_event is not None:
            rule, period, count = first_event
            self._record(step, rule, greedy, chosen, vetoed, period=period, count=count)
        return chosen

    # -- rule 3 -----------------------------------------------------------

    def decide(
        self,
        logits_row: torch.Tensor,
        legal_ids: list[int],
        *,
        domain: Any,
        state: Any,
        step: int,
    ) -> int:
        """One decode step (design §2.4 steps 3-4): rule 3, otherwise
        greedy plus the rule 1/2 veto loop. `legal_ids` is the post-v1
        set; `domain` is the grammar's own `AllowedDomain` for `state`."""
        greedy = _select_greedy(logits_row, legal_ids)
        recovered = self._close_recovery(
            logits_row, legal_ids, domain=domain, state=state, step=step, greedy=greedy
        )
        if recovered is not None:
            return recovered
        return self.select(logits_row, legal_ids, step=step, greedy=greedy)

    def _close_recovery(
        self,
        logits_row: torch.Tensor,
        legal_ids: list[int],
        *,
        domain: Any,
        state: Any,
        step: int,
        greedy: int,
    ) -> int | None:
        if state.mode not in ("structure", "text"):
            return None
        best = logits_row.max()
        unmasked = int((logits_row == best).nonzero()[0].item())
        token = self._token(unmasked)
        name = _closer_name(token)
        if name is None or _closer_is_legal(token, domain):
            return None
        variant = self._caps.close_recovery_variant
        if variant == "C" and not any(self.unit_counts.values()):
            return None
        stack = state.structural_state.stack
        if not stack:
            close = (
                self._legal_packed(_END_FRAGMENT_NAME, legal_ids)
                if name in _TERMINATOR_NAMES
                else None
            )
        elif variant in ("B", "C") and not _names_open_frame(name, stack):
            return None
        else:
            close = self._innermost_close(state, legal_ids)
        if close is None:
            self.counters["close_recovery_unavailable"] += 1
            return None
        if close != greedy:
            self._record(step, "close_recovery", greedy, close, [], unmasked_argmax=unmasked)
        return close

    def _legal_packed(self, name: str, legal_ids: list[int]) -> int | None:
        packed_id = pack(StructuralRef(events_vocab.TOKEN_IDS[name]), self._ranges)
        return packed_id if packed_id in legal_ids else None

    def _innermost_close(self, state: Any, legal_ids: list[int]) -> int | None:
        if state.mode == "text":
            return self._legal_packed(events_vocab.TEXT_END, legal_ids)
        top_close = events_vocab.CLOSE_TOKEN[state.structural_state.stack[-1].node_type]
        legal_closes = {
            name
            for name in (_closer_name(self._token(i)) for i in legal_ids)
            if name is not None and name.endswith("_CLOSE")
        }
        if len(legal_closes) > 1 or legal_closes - {top_close}:
            raise RuntimeError(
                f"close_recovery: >1 legal close or not the top's: {sorted(legal_closes)} "
                f"(top {top_close})"
            )
        return self._legal_packed(top_close, legal_ids)

    # -- evidence ---------------------------------------------------------

    def _record(
        self,
        step: int,
        rule: str,
        greedy: int,
        chosen: int,
        vetoed: list[int],
        *,
        period: int | None = None,
        count: int | None = None,
        unmasked_argmax: int | None = None,
    ) -> None:
        self.counters[rule] += 1
        self.interventions.append(
            {
                "step": step,
                "rule": rule,
                "greedy": greedy,
                "chosen": chosen,
                "vetoed": list(vetoed),
                "period": period,
                "count": count,
                "unmasked_argmax": unmasked_argmax,
            }
        )

    def record(self, v1_counters: Mapping[str, int]) -> dict[str, Any]:
        """The `loop_guard_v2` issue entry (design §5)."""
        return {
            "kind": "loop_guard_v2",
            "caps_sha256": self._caps.sha256,
            "counters": {"v1": dict(v1_counters), **self.counters},
            "interventions": [dict(i) for i in self.interventions],
        }


def _closer_is_legal(token: Any, domain: Any) -> bool:
    if domain.mode == "structure":
        return isinstance(token, StructuralRef) and token.local_id in domain.structural_local_ids
    # text mode: the only closer the grammar ever offers is TEXT_END.
    return (
        isinstance(token, StructuralRef)
        and _structural_name(token.local_id) == events_vocab.TEXT_END
        and domain.allow_text_end
    )


def _names_open_frame(name: str, stack: tuple[Any, ...]) -> bool:
    """Variant B: `END_*` always qualify; `X_CLOSE` iff `X` is on the
    stack; `TEXT_END`/`byte_end` name no structural frame."""
    if name in _TERMINATOR_NAMES:
        return True
    node_type = _CLOSE_NAME_TO_NODE_TYPE.get(name)
    return node_type is not None and any(frame.node_type == node_type for frame in stack)


class DecodeMaxStepsExceeded(RuntimeError):
    """Free-running decode did not reach `mode == "terminated"` within
    `DecodeConfig.max_steps`. Design §5.7: "maximum steps... nie sú úspech"
    -- this typed failure is what keeps a truncated stream from ever being
    returned as if it were complete.

    Carries `accepted_tokens`/`node_states` -- exactly what the loop had
    accumulated at the moment it hit the cap (bug `max-steps-discards-the-
    evidence`). This is diagnostic evidence about a FAILURE, never a
    hypothesis: nothing about raising this exception changes -- the raise
    still happens unconditionally, and no caller may construct a
    `RawDocumentHypothesis` from these fields and treat it as a truncated-
    but-accepted result (the "never truncate-and-accept" rule, §14.4 item
    13, stays exactly as strict). Default `()` for both so a caller that
    constructs this exception directly (existing tests) is unaffected.

    `loop_guard_counters` (DEC-031 task 2): `LoopGuardV1.counters` at the
    moment this raised, or `None` when `config.loop_guard` was not `"v1"`
    -- a page that still hit `max_steps`/vocabulary exhaustion under the
    guard is evidence the guard did not cover that page's loop shape, not
    evidence the guard ran and did nothing.

    `loop_guard_record` (DEC-036): the full `loop_guard_v2` issue entry
    (caps hash, counters, interventions) when `config.loop_guard == "v2"`,
    else `None`; `loop_guard_counters` then holds that entry's counters.

    `page_start_record` (DEC-061, PROPOSED): the `page_start_policy` issue
    entry (policy identity, pattern, counters) when `config.page_start_policy`
    is set, else `None`."""

    def __init__(
        self,
        message: str,
        *,
        accepted_tokens: tuple[int, ...] = (),
        node_states: tuple[NodeState, ...] = (),
        loop_guard_counters: Mapping[str, int] | None = None,
        loop_guard_record: Mapping[str, Any] | None = None,
        page_start_record: Mapping[str, Any] | None = None,
        story_length_cap_record: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.accepted_tokens = accepted_tokens
        self.node_states = node_states
        self.loop_guard_counters = loop_guard_counters
        self.loop_guard_record = loop_guard_record
        self.page_start_record = page_start_record
        self.story_length_cap_record = story_length_cap_record


class DecodeVocabularyExhausted(RuntimeError):
    """The grammar offered zero legal continuations before terminating.
    Should be unreachable while `mode != "terminated"`; see module
    docstring.

    Carries `accepted_tokens`/`node_states` for the same reason
    `DecodeMaxStepsExceeded` does -- should this ever actually fire, the
    tokens that led up to it are exactly the evidence a reader would need,
    and they would otherwise vanish at the moment they became interesting.

    `loop_guard_counters`, `page_start_record`: see `DecodeMaxStepsExceeded`."""

    def __init__(
        self,
        message: str,
        *,
        accepted_tokens: tuple[int, ...] = (),
        node_states: tuple[NodeState, ...] = (),
        loop_guard_counters: Mapping[str, int] | None = None,
        loop_guard_record: Mapping[str, Any] | None = None,
        page_start_record: Mapping[str, Any] | None = None,
        story_length_cap_record: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.accepted_tokens = accepted_tokens
        self.node_states = node_states
        self.loop_guard_counters = loop_guard_counters
        self.loop_guard_record = loop_guard_record
        self.page_start_record = page_start_record
        self.story_length_cap_record = story_length_cap_record


@dataclass(frozen=True)
class DecodeConfig:
    capability_profile_id: str
    max_steps: int = 4096
    tie_break: str = PINNED_TIE_BREAK
    #: DEC-031 task 2: `None` (default) keeps decode byte-identical to
    #: before this guard existed; `"v1"` turns on `LoopGuardV1` with its
    #: module-constant caps. The switch itself is part of decode identity
    #: (DEC-031 point 4), never a silent default -- `decode.py` carries it
    #: through to the sealed artefact. `"v2"` (DEC-036) adds `LoopGuardV2`
    #: on top of v1 and is refused outside `STANDALONE_V1_PROFILE_ID`.
    loop_guard: str | None = None
    #: AST Builder T1: apply the typed heads to the decoded node anchors
    #: (`typed_decode.decode_typed`) and return them in
    #: `RawDocumentHypothesis.typed_predictions`. Off by default so every
    #: existing decode stays byte-identical.
    typed_decode: bool = False
    #: Explicit T4 opt-in: emit closed node region hypotheses.  It retains
    #: visual hidden states from the existing prefix forward; it never runs a
    #: second decoder forward merely for the aligner.
    region_decode: bool = False
    #: W0-11 opt-in: `None` (default) keeps the prefix pass exactly today's
    #: single `use_cache=True` call, byte-identical to before this field
    #: existed. A positive int instead feeds the prefix through
    #: `QwenDenseCandidateModel.encode_prefix`/`encode_prefix_with_visual_
    #: hidden` in chunks of that many tokens (`candidate_model.py`'s
    #: `_prefill_chunked`) -- lets a 300 DPI page's ~8,580-token prefix fit
    #: a 12 GiB GPU that OOMs on the single-call peak. Part of decode
    #: identity, never a silent default, so a decode run with chunking on
    #: is distinguishable from one without it.
    prefill_chunk_tokens: int | None = None
    #: DEC-055 (PROPOSED, default off): which loop guard v3 arm a
    #: `loop_guard="v3"` decode runs (`loop_guard_v3.parse_arm`); `None`
    #: means the policy's `default_arm`. Refused for any other `loop_guard`.
    loop_guard_v3_arm: str | None = None
    #: DEC-061 (PROPOSED, default off): `None` keeps the decode byte-identical;
    #: `"page-start-story.v0"` lets `decode_free_running(page_start_pattern=...)`
    #: narrow the first section's header/footer decision to the given pattern.
    #: Refused outside standalone-page profiles. The policy file is read only
    #: when this is set.
    page_start_policy: str | None = None
    #: `header-length-cap.v0` (PROPOSED, default off): close a header/footer that is
    #: still open after the policy's cap of accepted tokens. `None` keeps the decode
    #: byte-identical; the policy file is read only when this is set.
    #: `story_length_cap_policy_path` overrides the packaged file (measurement arms).
    story_length_cap: str | None = None
    story_length_cap_policy_path: str | None = None
    #: `story-retry.v0` (PROPOSED, default off): a finished decode whose header/footer holds
    #: more than the policy's threshold tokens is decoded again with that story kind banned
    #: (`StoryPattern` from the model's own first decode, never from gold). `None` keeps the
    #: decode byte-identical; the policy file is read only when this is set.
    story_retry: str | None = None
    story_retry_policy_path: str | None = None

    def __post_init__(self) -> None:
        if self.tie_break != PINNED_TIE_BREAK:
            raise ValueError(
                f"tie_break must be {PINNED_TIE_BREAK!r} (design §5.7's pinned "
                f"greedy tie-break policy); got {self.tie_break!r}"
            )
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.loop_guard not in (None, "v1", "v2", "v3"):
            raise ValueError(
                f"loop_guard must be None, 'v1', 'v2' or 'v3', got {self.loop_guard!r}"
            )
        if self.loop_guard in ("v2", "v3"):
            _refuse_unless_loop_guard_v2_applies(self.capability_profile_id)
        if self.loop_guard == "v3":
            from unflatten.model.loop_guard_v3 import load_loop_guard_v3_policy, resolve_arm

            resolve_arm(load_loop_guard_v3_policy(), self.loop_guard_v3_arm)
        elif self.loop_guard_v3_arm is not None:
            raise ValueError(
                f"loop_guard_v3_arm {self.loop_guard_v3_arm!r} needs loop_guard='v3', "
                f"got {self.loop_guard!r}"
            )
        if self.prefill_chunk_tokens is not None and self.prefill_chunk_tokens <= 0:
            raise ValueError(
                f"prefill_chunk_tokens must be positive, got {self.prefill_chunk_tokens}"
            )
        if self.page_start_policy is not None:
            _refuse_unless_page_start_policy_applies(
                self.page_start_policy, self.capability_profile_id
            )
        if self.story_length_cap is not None:
            _refuse_unless_story_length_cap_applies(
                self.story_length_cap,
                self.capability_profile_id,
                self.story_length_cap_policy_path,
            )
        elif self.story_length_cap_policy_path is not None:
            raise ValueError("story_length_cap_policy_path needs story_length_cap")
        if self.story_retry is not None:
            _refuse_unless_story_retry_applies(
                self.story_retry,
                self.capability_profile_id,
                self.story_retry_policy_path,
                page_start_policy=self.page_start_policy,
            )
        elif self.story_retry_policy_path is not None:
            raise ValueError("story_retry_policy_path needs story_retry")


def default_loop_guard_v1(capability_profile_id: str) -> LoopGuardV1:
    """The v1 caps a real decode uses for `capability_profile_id`: the DEC-033
    widened caps for every standalone-page profile (by SCOPE, not one id --
    2026-09-24: the v15 profile silently got the legacy caps), the pre-DEC-033
    legacy caps otherwise so older profiles reproduce their targets exactly."""
    if capability_profile_id in STANDALONE_PAGE_PROFILE_IDS:
        return LoopGuardV1()
    return LoopGuardV1(
        max_consecutive_tab=_LEGACY_MAX_CONSECUTIVE_TAB,
        max_consecutive_empty_paragraph=_LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPH,
    )


def _refuse_unless_loop_guard_v2_applies(capability_profile_id: str) -> None:
    """DEC-036 design §3: v2 caps exist only for the profile they were
    measured under, so `"v2"` is refused for every other profile (old
    profiles cannot reach v2 at all). This is the first point that loads
    the caps file -- lazily, and only for `"v2"`."""
    if capability_profile_id not in STANDALONE_PAGE_PROFILE_IDS:
        raise ValueError(
            "loop_guard 'v2' requires a standalone-page capability profile "
            f"({sorted(STANDALONE_PAGE_PROFILE_IDS)}); got {capability_profile_id!r}"
        )
    caps = load_loop_guard_v2_caps()
    if caps.capability_profile_id != capability_profile_id:
        raise ValueError(
            f"loop_guard 'v2' caps {caps.path} were measured under "
            f"{caps.capability_profile_id!r}, not {capability_profile_id!r}"
        )


def _refuse_unless_page_start_policy_applies(policy_id: str, capability_profile_id: str) -> None:
    """DEC-061: the story patterns are measured on standalone pages only. This is
    the first point that reads the policy file -- lazily, only when it is set."""
    if policy_id != page_start_policy.POLICY_ID:
        raise ValueError(
            f"page_start_policy must be None or {page_start_policy.POLICY_ID!r}, got {policy_id!r}"
        )
    if capability_profile_id not in STANDALONE_PAGE_PROFILE_IDS:
        raise ValueError(
            "page_start_policy requires a standalone-page capability profile "
            f"({sorted(STANDALONE_PAGE_PROFILE_IDS)}); got {capability_profile_id!r}"
        )
    page_start_policy.load_page_start_policy()


def _refuse_unless_story_length_cap_applies(
    policy_id: str, capability_profile_id: str, policy_path: str | None
) -> None:
    """header-length-cap.v0: standalone-page profiles only, and only a policy with
    a frozen cap. The first point that reads the policy file."""
    if policy_id != header_length_cap.POLICY_ID:
        raise ValueError(
            f"story_length_cap must be None or {header_length_cap.POLICY_ID!r}, got {policy_id!r}"
        )
    if capability_profile_id not in STANDALONE_PAGE_PROFILE_IDS:
        raise ValueError(
            "story_length_cap requires a standalone-page capability profile "
            f"({sorted(STANDALONE_PAGE_PROFILE_IDS)}); got {capability_profile_id!r}"
        )
    header_length_cap.refuse_unless_policy_usable(
        header_length_cap.load_header_length_cap_policy(policy_path)
    )


def _refuse_unless_story_retry_applies(
    policy_id: str,
    capability_profile_id: str,
    policy_path: str | None,
    *,
    page_start_policy: str | None,
) -> None:
    """story-retry.v0: standalone-page profiles only, a frozen threshold, and not combined with
    a page-start policy (the retry IS a page-start pattern). The first point that reads the file."""
    if policy_id != story_retry.POLICY_ID:
        raise ValueError(
            f"story_retry must be None or {story_retry.POLICY_ID!r}, got {policy_id!r}"
        )
    if capability_profile_id not in STANDALONE_PAGE_PROFILE_IDS:
        raise ValueError(
            "story_retry requires a standalone-page capability profile "
            f"({sorted(STANDALONE_PAGE_PROFILE_IDS)}); got {capability_profile_id!r}"
        )
    if page_start_policy is not None:
        raise ValueError(
            "story_retry cannot be combined with page_start_policy "
            f"({page_start_policy!r}): the retry applies its own page-start pattern"
        )
    story_retry.refuse_unless_policy_usable(story_retry.load_story_retry_policy(policy_path))


def _decode_with_story_retry(kwargs: dict[str, Any]) -> RawDocumentHypothesis:
    """`story-retry.v0`: decode once; when a story exceeds the threshold decode again under the
    pattern that keeps the short kinds. A second decode that raises returns the FIRST hypothesis
    with the failure named in its `story_retry` record (never silently)."""
    config: DecodeConfig = kwargs["config"]
    policy = story_retry.load_story_retry_policy(config.story_retry_policy_path)
    base = dataclasses.replace(
        config, story_retry=None, story_retry_policy_path=None
    )
    first = decode_free_running(**{**kwargs, "config": base})
    stories = story_retry.story_lengths(first.accepted_tokens, kwargs["vocabulary"])
    pattern = story_retry.retry_pattern(stories, policy.threshold_tokens)
    if pattern is None:
        record = story_retry.retry_record(policy, stories, None, "not_needed")
        return dataclasses.replace(first, issues=(*first.issues, record))
    retry_config = dataclasses.replace(
        base, page_start_policy=page_start_policy.POLICY_ID
    )
    try:
        second = decode_free_running(
            **{**kwargs, "config": retry_config, "page_start_pattern": pattern}
        )
    except (DecodeMaxStepsExceeded, DecodeVocabularyExhausted) as exc:
        record = story_retry.retry_record(
            policy, stories, pattern, f"retry_failed:{type(exc).__name__}"
        )
        return dataclasses.replace(first, issues=(*first.issues, record))
    record = story_retry.retry_record(policy, stories, pattern, "retried")
    return dataclasses.replace(second, issues=(*second.issues, record))


_PAGE_START_COUNTERS = (
    "decision_steps",
    "masked_steps",
    "fallback_empty",
    "unexpected_state",
    "forced_story_reopen",
)


def _page_start_applicability(config: DecodeConfig, pattern: Any) -> tuple[Any, str | None]:
    """`(policy, reason)` for one decode: `policy is None` when the policy is off;
    `reason is None` when the pattern is applied, else `no_pattern` /
    `below_threshold`. A pattern without a policy is a caller error."""
    if config.page_start_policy is None:
        if pattern is not None:
            raise ValueError(
                "page_start_pattern needs config.page_start_policy "
                f"(got pattern from {pattern.source!r}, policy None)"
            )
        return None, None
    policy = page_start_policy.load_page_start_policy()
    if pattern is None:
        return policy, "no_pattern"
    if pattern.confidence is not None and pattern.confidence < policy.confidence_threshold:
        return policy, "below_threshold"
    return policy, None


def _page_start_record(
    policy: Any, pattern: Any, reason: str | None, counters: Mapping[str, int]
) -> dict[str, Any]:
    """The `page_start_policy` issue entry (design §3.4)."""
    return {
        "kind": "page_start_policy",
        "policy_id": policy.policy_id,
        "policy_status": policy.status,
        "policy_sha256": policy.sha256,
        "pattern": (
            None
            if pattern is None
            else {"has_header": pattern.has_header, "has_footer": pattern.has_footer}
        ),
        "pattern_source": None if pattern is None else pattern.source,
        # integer ppm: issue records are hashed through CBOR, which has no floats
        "confidence_ppm": (
            None
            if pattern is None or pattern.confidence is None
            else round(pattern.confidence * 1_000_000)
        ),
        "applied": reason is None,
        "reason": reason,
        "counters": dict(counters),
    }


def _story_pattern_controller(controller: CompositeGrammarController, pattern: Any) -> Any:
    from unflatten.model.page_start_controller import StoryPatternController

    return StoryPatternController.wrap(controller, pattern=pattern)


@functools.lru_cache(maxsize=8)
def _packed_text_allowlist(vocabulary: CompositeVocabulary) -> tuple[int, ...]:
    """The packed, sorted form of `vocabulary.qwen_text_allowlist` -- the
    part of a text-mode legal set that is identical on every step for a
    given vocabulary (only `allow_text_end`/`allow_byte_begin` vary per
    step, and those add at most two ids on top of this).

    `CompositeVocabulary` is frozen and hashable (all fields -- including
    the `frozenset` allowlist -- are hashable), so `functools.lru_cache`
    keys correctly on *value*: two vocabularies with different allowlists
    or ranges get distinct cache entries, and nothing is ever attached to
    a `CompositeVocabulary` instance itself. The result is a `tuple`, so
    it is immutable by construction -- `_legal_packed_ids` always builds a
    fresh `list` from it, never hands out this cached object itself.
    """
    return tuple(
        vocabulary.pack(QwenRef(native_id)) for native_id in sorted(vocabulary.qwen_text_allowlist)
    )


def _legal_packed_ids(
    controller: CompositeGrammarController, state, vocabulary: CompositeVocabulary
) -> list[int]:
    """The exact set of packed ids legal from `state` -- built from
    `AllowedDomain`'s own compact ranges.

    For `mode == "text"`, the allowlist portion is no longer enumerated
    and packed fresh on every call: `_packed_text_allowlist` computes it
    once per distinct vocabulary and this function copies that cached
    tuple into a new list before appending the (at most two) per-step
    conditional ids. Measured cost before this change: with a real
    ~148k-entry allowlist, sorting the frozenset and calling
    `vocabulary.pack` per entry on every text-mode step was ~22.6% of a
    full free-running decode -- comparable to the model forward itself."""
    domain = controller.allowed(state)
    ids: list[int] = []
    if domain.mode == "structure":
        ids.extend(
            vocabulary.pack(StructuralRef(local_id)) for local_id in domain.structural_local_ids
        )
    elif domain.mode == "text":
        ids.extend(_packed_text_allowlist(vocabulary))
        if domain.allow_text_end:
            ids.append(vocabulary.pack(StructuralRef(_TEXT_END_LOCAL_ID)))
        if domain.allow_byte_begin:
            ids.append(vocabulary.pack(ByteControlRef("byte_begin")))
    elif domain.mode == "byte":
        for lo, hi in domain.allowed_byte_ranges:
            ids.extend(vocabulary.pack(ByteValueRef(value)) for value in range(lo, hi))
        if domain.allow_byte_end:
            ids.append(vocabulary.pack(ByteControlRef("byte_end")))
    return ids


def _select_greedy(logits_row: torch.Tensor, legal_ids: list[int]) -> int:
    """`PINNED_TIE_BREAK`: among the legal ids with the (jointly) highest
    logit, pick the lowest packed id -- written explicitly rather than
    relying on `torch.argmax`'s undocumented tie behaviour.

    The result does not depend on the order of `legal_ids` (max, then the
    minimum id among the tied), so the list is not sorted; `np.fromiter`
    builds the index tensor ~3x faster than `torch.tensor(list)`. Over the
    ~148k-entry text allowlist the old `torch.tensor(sorted(...))` cost ~6 ms
    of CPU per decoded token (measured 2026-09-28)."""
    legal = torch.from_numpy(
        np.fromiter(legal_ids, dtype=np.int64, count=len(legal_ids))
    ).to(logits_row.device)
    scores = logits_row.index_select(0, legal)
    best_score = scores.max()
    tied = legal[scores == best_score]
    return int(tied.min().item())


def decode_free_running(
    *,
    model: QwenDenseCandidateModel,
    controller: CompositeGrammarController,
    vocabulary: CompositeVocabulary,
    visual_features: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    config: DecodeConfig,
    use_kv_cache: bool = True,
    loop_guard_factory: Any = None,
    initial_cache_factory: Any = None,
    deepstack_visual_embeds: tuple[torch.Tensor, ...] | None = None,
    page_start_pattern: Any = None,
) -> RawDocumentHypothesis:
    """Free-running constrained greedy decode: raw visual features in, a
    grammar-complete `RawDocumentHypothesis` out -- no gold prefix anywhere
    in the loop below.

    `image_grid_thw` and `image_token_id` are `inject_visual`'s own
    contract (`candidate_loader.py`): the former is passed straight through
    to its native positional index, the latter names which packed id the
    fixed visual placeholder block uses (module docstring).

    `use_kv_cache` (design §20): `True` (default) is the fast path --
    `model.encode_prefix` once, `model.encode_step` per accepted token.
    `False` forces the old full-recompute path (`model.encode_hidden` every
    step) that this module used before design §20; it exists so a caller
    (namely the equivalence test) can run both arms over the identical
    inputs and diff `accepted_tokens`, not as a normal production knob.

    `config.loop_guard` (DEC-031 task 2): `None` (default) means this
    function's control flow is untouched from before the guard existed --
    no `LoopGuardV1` is even constructed. `"v1"` builds one `LoopGuardV1`
    (via `loop_guard_factory`, see below) for this call and threads it
    through `_legal_packed_ids`'s output (`filter`, before
    `_select_greedy`) and every accepted token (`observe`, after). Its
    counters ride along on the returned `RawDocumentHypothesis.issues` (a
    `{"kind": "loop_guard_v1", "counters": ...}` entry) for a terminated
    page, or on `DecodeMaxStepsExceeded.loop_guard_counters`/
    `DecodeVocabularyExhausted.loop_guard_counters` for the two typed
    failures -- neither touches `prediction_ir.py`'s own schema to carry
    this.

    `"v2"` (DEC-036) additionally builds one `LoopGuardV2` from the frozen
    caps (`load_loop_guard_v2_caps`, loaded only here and in `DecodeConfig`
    for `"v2"`). v1 still filters first, unchanged; v2 then decides the
    step (`LoopGuardV2.decide`: close recovery, else greedy plus the text-
    run/periodic veto loop) and observes every accepted id. Its
    `{"kind": "loop_guard_v2", ...}` entry (v1's counters nested inside)
    replaces the v1 entry on `issues`, and rides on the typed failures as
    `loop_guard_record`.

    `loop_guard_factory` (test-only knob, plan task 2: "make the
    constants overridable only through the guard constructor in tests,
    NOT through the CLI"): `None` (default) means a real `"v1"` decode
    picks `LoopGuardV1`'s caps itself, gated by `config.
    capability_profile_id` exactly like `CompositeGrammarController.
    _dec033_bounds_active` -- `STANDALONE_V1_PROFILE_ID` gets the DEC-033
    widened caps (`MAX_CONSECUTIVE_TAB`/`MAX_CONSECUTIVE_EMPTY_PARAGRAPH`,
    3/3), every other profile gets the pre-DEC-033 legacy caps
    (`_LEGACY_MAX_CONSECUTIVE_TAB`/`_LEGACY_MAX_CONSECUTIVE_EMPTY_PARAGRAPH`,
    202/53) so it keeps reproducing its v12 targets exactly. Passing a
    factory explicitly bypasses this gate entirely -- nothing in
    `DecodeConfig`/the CLI can reach this parameter, only a caller
    constructing this function's kwargs directly (a test) can pass e.g.
    `lambda: LoopGuardV1(max_consecutive_tab=3)` to prove the guard's
    effect within a handful of steps instead of the real caps' hundreds.
    Under `"v2"` the factory builds the v1 part only.

    `initial_cache_factory` (owner requirement 2026-09-27, quantized KV
    cache; region-decode support added 2026-09-28): `None` (default) is
    exactly today's behaviour -- `model.encode_prefix`/`model.
    encode_prefix_with_visual_hidden` builds its own cache. When given, it
    is called with no arguments once (e.g. `Float8KVCache`, `StaticKVCache`)
    and the result is passed as `initial_cache` to whichever of the two
    prefix methods `config.region_decode` selects, so the whole decode
    accumulates K/V in that cache object either way. Only applies on the
    `use_kv_cache=True`, non-chunked-prefill path -- both prefix methods
    refuse `initial_cache` together with `prefill_chunk_tokens`.

    `deepstack_visual_embeds` (DEC-056): the page's DeepStack levels for a
    model built with `deepstack=True` (it refuses a batch without them, and
    a plain model refuses them); `None` (default) is today's call exactly.
    Only the prefix carries visual positions, so only the batches built
    here pass them on.

    `page_start_pattern` (DEC-061, PROPOSED): a `page_start_policy.StoryPattern`
    for the first section's header/footer stories. `None` (default) is today's
    decode exactly. Needs `config.page_start_policy` (else `ValueError`). Applied
    only when `confidence is None or confidence >= policy.confidence_threshold`,
    by wrapping the controller (after the v3 force-content wrap, before `fresh`)
    so it narrows the legal set the loop guards and the greedy step both see.
    A `page_start_policy` record (counters per design §3.4) rides on
    `RawDocumentHypothesis.issues` and on the typed failures; when the policy is
    on but the pattern is not applied the record says why and the tokens are
    those of a decode without the policy.
    """
    if config.story_retry is not None:
        return _decode_with_story_retry(
            {
                "model": model,
                "controller": controller,
                "vocabulary": vocabulary,
                "visual_features": visual_features,
                "image_grid_thw": image_grid_thw,
                "image_token_id": image_token_id,
                "config": config,
                "use_kv_cache": use_kv_cache,
                "loop_guard_factory": loop_guard_factory,
                "initial_cache_factory": initial_cache_factory,
                "deepstack_visual_embeds": deepstack_visual_embeds,
                "page_start_pattern": page_start_pattern,
            }
        )
    if visual_features.dim() != 3 or visual_features.shape[0] != 1:
        raise ValueError(
            "decode_free_running is single-page/batch=1 in M3 scope, got "
            f"visual_features shape {tuple(visual_features.shape)}"
        )
    device = visual_features.device
    guard_v3 = _build_loop_guard_v3(config, vocabulary) if config.loop_guard == "v3" else None
    if guard_v3 is not None and guard_v3._arm.force_content:
        controller = _force_content_controller(controller)
    page_start, page_start_reason = _page_start_applicability(config, page_start_pattern)
    page_start_counters = dict.fromkeys(_PAGE_START_COUNTERS, 0)
    story_controller = None
    if page_start is not None and page_start_reason is None:
        controller = story_controller = _story_pattern_controller(controller, page_start_pattern)
    state = controller.fresh(config.capability_profile_id)
    tracker = NodeLifecycleTracker()
    accepted: list[int] = []
    story_cap = (
        header_length_cap.StoryLengthCap(
            header_length_cap.load_header_length_cap_policy(config.story_length_cap_policy_path),
            vocabulary,
        )
        if config.story_length_cap is not None
        else None
    )
    if config.loop_guard in (None, "v2", "v3"):
        # DEC-036 addendum 2 (owner 19.9., re-confirmed 24.9.): v2 REPLACES v1 --
        # v1's empty-paragraph ban vetoed gold on 11 % of val pages.
        guard = None
    elif loop_guard_factory is not None:
        guard = loop_guard_factory()
    else:
        guard = default_loop_guard_v1(config.capability_profile_id)
    guard_v2 = (
        LoopGuardV2(ranges=vocabulary.ranges, caps=load_loop_guard_v2_caps())
        if config.loop_guard == "v2"
        else None
    )

    def _page_start_entry() -> dict[str, Any]:
        return _page_start_record(
            page_start, page_start_pattern, page_start_reason, page_start_counters
        )

    def _guard_evidence() -> dict[str, Any]:
        extra = {"page_start_record": _page_start_entry()} if page_start is not None else {}
        if story_cap is not None:
            extra["story_length_cap_record"] = story_cap.record()
        if guard_v3 is not None:
            record = guard_v3.record()
            return {
                "loop_guard_counters": record["counters"],
                "loop_guard_record": record,
                **extra,
            }
        if guard_v2 is not None:
            record = guard_v2.record(guard.counters if guard is not None else {})
            return {
                "loop_guard_counters": record["counters"],
                "loop_guard_record": record,
                **extra,
            }
        return {
            "loop_guard_counters": dict(guard.counters) if guard is not None else None,
            **extra,
        }

    visual_tokens = visual_features.shape[1]
    placeholder_id = vocabulary.pack(QwenRef(image_token_id))
    # Training feeds `[placeholders][BOS, t0, .., t_{n-2}]`
    # (`training_materializer`: `input_ids = [bos_id, *packed_ids[:-1]]`,
    # BOS = the END_DOCUMENT structural id), so target `t` is predicted at
    # composite position `V + t`. The decode prefix must end with the same
    # BOS, or every step conditions one position off (measured 2026-09-22:
    # step 0 hidden off by 3.9, later steps by ~0.03-0.06).
    bos_local_id = events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT]
    bos_id = vocabulary.pack(StructuralRef(local_id=bos_local_id))
    placeholder_block = [placeholder_id] * visual_tokens + [bos_id]

    with torch.inference_mode():
        prefix_batch = TeacherForcedBatch(
            visual_features=visual_features,
            composite_input_ids=torch.tensor([placeholder_block], dtype=torch.long, device=device),
            composite_attention_mask=torch.ones(
                1, len(placeholder_block), dtype=torch.long, device=device
            ),
            image_grid_thw=image_grid_thw,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        visual_hidden = None
        # `prefill_chunk_tokens=None` (the default) omits the keyword
        # entirely rather than passing `None` through -- a `model`/`retain`
        # fake built against the pre-W0-11 `encode_prefix`/`encode_prefix_
        # with_visual_hidden` signature (no such parameter at all) must keep
        # working unchanged; only an explicit `prefill_chunk_tokens` opt-in
        # requires the real signature.
        prefill_kwargs = (
            {} if config.prefill_chunk_tokens is None
            else {"prefill_chunk_tokens": config.prefill_chunk_tokens}
        )
        if use_kv_cache:
            if initial_cache_factory is not None:
                prefill_kwargs = {
                    **prefill_kwargs,
                    "initial_cache": initial_cache_factory(),
                }
            if config.region_decode:
                retain = getattr(model, "encode_prefix_with_visual_hidden", None)
                if retain is None:
                    raise ValueError(
                        "region_decode requires encode_prefix_with_visual_hidden support"
                    )
                cache_state, next_position_hidden, visual_hidden = retain(
                    prefix_batch, **prefill_kwargs
                )
            else:
                cache_state, next_position_hidden = model.encode_prefix(
                    prefix_batch, **prefill_kwargs
                )
        else:
            hidden, _prefix_len = model.encode_hidden(prefix_batch)
            next_position_hidden = hidden[:, -1, :]
            if config.region_decode:
                # The composite range is [visual block][BOS]; a merging
                # post-encoder compressor shrinks the visual block inside the
                # model, so the span is read from the hidden state, never from
                # `visual_features.shape[1]` (which would run into BOS when a
                # merge saves exactly one token and still pass the grid check).
                composite_visual_tokens = hidden.shape[1] - _prefix_len - 1
                visual_hidden = hidden[
                    :, _prefix_len : _prefix_len + composite_visual_tokens, :
                ]

        if guard_v3 is not None:
            # R2 rewind bookkeeping (DEC-055): grammar state after every
            # prefix length, the hidden state before every structural token
            # (the tracker's anchor, reused as the resume hidden), and the
            # KV length the prefix pass left, so a rewind crops the cache
            # with the speculative path's own `rollback_to`.
            v3_states = [state]
            v3_anchors: dict[int, torch.Tensor] = {}
            v3_base_len = cache_state.snapshot().composite_len if use_kv_cache else 0

        for _step in range(config.max_steps):
            if state.mode == "terminated":
                break
            if story_controller is not None:
                _count_story_decision(story_controller.decision(state), page_start_counters)
            legal_ids = _legal_packed_ids(controller, state, vocabulary)
            if story_cap is not None:
                legal_ids = story_cap.narrow(legal_ids, state, len(accepted))
            if guard is not None:
                legal_ids, _guard_event = guard.filter(
                    legal_ids, vocabulary=vocabulary, mode=state.mode
                )
            if not legal_ids:
                raise DecodeVocabularyExhausted(
                    f"grammar mode {state.mode!r} offered zero legal continuations "
                    "before terminating -- this should be unreachable",
                    accepted_tokens=tuple(accepted),
                    node_states=tracker.states,
                    **_guard_evidence(),
                )

            logits = model.composite_head(next_position_hidden)[0]

            if guard_v3 is not None:
                decision = guard_v3.decide(
                    logits,
                    legal_ids,
                    domain=controller.allowed(state),
                    state=state,
                    step=len(accepted),
                )
                if isinstance(decision, int):
                    chosen_id = decision
                else:
                    b = decision.position
                    if b not in v3_anchors:
                        raise RuntimeError(
                            f"loop guard v3 rewind to {b}: not a structural position"
                        )
                    next_position_hidden = v3_anchors[b].unsqueeze(0)
                    del accepted[b:]
                    del v3_states[b + 1 :]
                    state = v3_states[b]
                    v3_anchors = {i: a for i, a in v3_anchors.items() if i < b}
                    tracker = _replay_tracker(accepted, v3_anchors, vocabulary)
                    guard_v3.rewind_to(decision)
                    if story_cap is not None:
                        story_cap.rewind(b)
                    if use_kv_cache:
                        cache_state = cache_state.rollback_to(
                            KVCacheSnapshot(composite_len=v3_base_len + b)
                        )
                    continue
            elif guard_v2 is None:
                chosen_id = _select_greedy(logits, legal_ids)
            else:
                chosen_id = guard_v2.decide(
                    logits,
                    legal_ids,
                    domain=controller.allowed(state),
                    state=state,
                    step=len(accepted),
                )
            token = vocabulary.unpack(chosen_id)
            state = controller.step(state, token)
            accepted.append(chosen_id)
            if story_cap is not None:
                story_cap.observe(state, len(accepted))

            structural_name = (
                _structural_name(token.local_id) if isinstance(token, StructuralRef) else None
            )
            if guard is not None:
                guard.observe(structural_name)
            if guard_v2 is not None:
                guard_v2.observe(chosen_id)
            if guard_v3 is not None:
                guard_v3.observe(chosen_id)
                v3_states.append(state)

            if isinstance(token, StructuralRef):
                name = structural_name
                # `next_position_hidden[0]` is a VIEW into a base tensor
                # that, in the non-cache arm, spans the whole growing
                # sequence (visual placeholder block + accepted tokens) --
                # keeping that view alive would pin every step's full
                # hidden state for as long as the resulting NodeState
                # lives. `.detach().clone()` copies just the one row into
                # its own storage before it reaches the tracker, in both
                # arms.
                anchor = next_position_hidden[0].detach().clone()
                tracker.apply_structural_token(
                    name, event_index=len(accepted) - 1, anchor=anchor
                )
                if guard_v3 is not None:
                    v3_anchors[len(accepted) - 1] = anchor

            if state.mode == "terminated":
                break

            if use_kv_cache:
                cache_state, next_position_hidden = model.encode_step(cache_state, chosen_id)
            else:
                input_ids = torch.tensor(
                    [placeholder_block + accepted], dtype=torch.long, device=device
                )
                attention_mask = torch.ones_like(input_ids)
                batch = TeacherForcedBatch(
                    visual_features=visual_features,
                    composite_input_ids=input_ids,
                    composite_attention_mask=attention_mask,
                    image_grid_thw=image_grid_thw,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                )
                hidden, _prefix_len = model.encode_hidden(batch)
                next_position_hidden = hidden[:, -1, :]

    if state.mode != "terminated":
        raise DecodeMaxStepsExceeded(
            f"free-running decode did not terminate within {config.max_steps} steps",
            accepted_tokens=tuple(accepted),
            node_states=tracker.states,
            **_guard_evidence(),
        )

    if guard_v3 is not None:
        issues: tuple[Any, ...] = (guard_v3.record(),)
    elif guard_v2 is not None:
        issues = (guard_v2.record(guard.counters if guard is not None else {}),)
    elif guard is not None:
        issues = ({"kind": "loop_guard_v1", "counters": dict(guard.counters)},)
    else:
        issues = ()
    if page_start is not None:
        issues = (*issues, _page_start_entry())
    if story_cap is not None:
        issues = (*issues, story_cap.record())
    typed = decode_typed(model, tracker.states) if config.typed_decode else ()
    regions = (
        decode_regions(
            model.node_visual_aligner,
            tracker.states,
            visual_hidden=visual_hidden[0],
            image_grid_thw=image_grid_thw,
        )
        if config.region_decode
        else ()
    )
    return RawDocumentHypothesis(
        accepted_tokens=tuple(accepted),
        node_states=tracker.states,
        typed_predictions=typed,
        region_predictions=regions,
        terminated=True,
        issues=issues,
    )


def _build_loop_guard_v3(config: DecodeConfig, vocabulary: CompositeVocabulary) -> Any:
    """DEC-055 loop guard v3 for one decode; imported only for `"v3"`."""
    from unflatten.model.loop_guard_v3 import LoopGuardV3, load_loop_guard_v3_policy, resolve_arm

    policy = load_loop_guard_v3_policy()
    return LoopGuardV3(
        ranges=vocabulary.ranges,
        policy=policy,
        arm=resolve_arm(policy, config.loop_guard_v3_arm),
        v2_caps=load_loop_guard_v2_caps(),
    )


def _count_story_decision(decision: Any, counters: dict[str, int]) -> None:
    """Fold one `StoryPatternController.decision` into the record's counters."""
    if decision is None:
        return
    counters["decision_steps"] += 1
    counters["masked_steps"] += bool(decision.banned_local_ids)
    counters["fallback_empty"] += decision.fallback_empty
    counters["unexpected_state"] += decision.unexpected_state
    counters["forced_story_reopen"] += decision.forced_story_reopen


def _force_content_controller(controller: CompositeGrammarController) -> Any:
    from unflatten.model.loop_guard_v3 import ForceContentController

    return ForceContentController.wrap(controller)


def _replay_tracker(
    accepted: list[int], anchors: Mapping[int, torch.Tensor], vocabulary: CompositeVocabulary
) -> NodeLifecycleTracker:
    """A fresh tracker over `accepted` (after an R2 rewind), with each
    structural token's original anchor."""
    tracker = NodeLifecycleTracker()
    for index, packed_id in enumerate(accepted):
        token = vocabulary.unpack(packed_id)
        if isinstance(token, StructuralRef):
            tracker.apply_structural_token(
                _structural_name(token.local_id), event_index=index, anchor=anchors[index]
            )
    return tracker


_TOKEN_NAME_BY_ID = {local_id: name for name, local_id in events_vocab.TOKEN_IDS.items()}


def _structural_name(local_id: int) -> str:
    return _TOKEN_NAME_BY_ID[local_id]
