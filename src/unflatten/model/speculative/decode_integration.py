"""RFC004 §37.4-5: the opt-in wire-up between `run_speculative_decode`
(`drafter.py`) and the real grammar-constrained free-running decode
(`free_running_decode.decode_free_running`).

`speculative_decode_free_running`'s `speculator` parameter defaults to
`None`, in which case this function does nothing but call
`decode_free_running` itself, unmodified -- byte-identical to the
acceleration-off path by construction, not merely by proof (pinned by
`test_decode_integration.py::test_default_speculator_is_byte_identical_to_
decode_free_running`).

Passing `speculator` a FACTORY (`Callable[[FreeRunningPrimaryDecoder],
SpeculativeDrafter]`) switches to `FreeRunningPrimaryDecoder` below, which
reproduces `decode_free_running`'s own per-step decision
(`_legal_packed_ids` + `_select_greedy`, imported rather than
re-implemented) behind the `PrimaryDecoder` protocol, so `run_speculative_
decode`'s draft/verify/commit loop can drive it. It is a factory rather
than a ready-made drafter because `MtpDrafter.hidden_state_at` needs to
read hidden states off THIS SPECIFIC `FreeRunningPrimaryDecoder` instance
(built fresh per call, from this call's own KV-cache prefix pass) --
`speculator(primary_decoder)` is invoked once, after that instance exists,
so e.g. `lambda primary: MtpDrafter(mtp_heads=heads,
hidden_state_at=primary.hidden_before)` can close over it. A drafter with
no such need (`OffDrafter`, `OracleDrafter`) just ignores the argument:
`lambda _primary: OffDrafter()`.

Node-lifecycle tracking and the typed-head tail are applied only to tokens
`run_speculative_decode` actually *commits* -- never to a
drafted-but-rejected candidate -- so a rejected draft can never leave a
trace in the returned hypothesis (§37.4's lossless-equivalence
requirement).

**`config.loop_guard` and `config.region_decode` (W6-2, DEC-047 dormant)**:
both now work, over COMMITTED tokens only.

- `loop_guard`: `FreeRunningPrimaryDecoder` builds the same `LoopGuardV1`/
  `LoopGuardV2` object `decode_free_running` would (`default_loop_guard_v1`
  / `LoopGuardV2`), and applies `filter`/`decide`/`observe` at exactly the
  places `decode_free_running` does. Guard state is snapshotted per prefix
  length alongside the grammar state (`_by_length`) so a `rollback_to`
  undoes a guard observation exactly like it undoes a KV-cache append, and
  the snapshot stored for length `n` is the state `decode_free_running`'s
  guard holds after its `n`-th committed token
  (`test_decode_integration_guard_block.py` pins the final state against
  `decode_free_running` and every per-length snapshot against the
  per-token path).
  **Decision vs evidence (2026-09-27)**: a guard call does two things -- it
  DECIDES (a pure function of the run-length state, the logits and the
  grammar state) and it RECORDS (counters, v2 interventions). The adapter
  decides each position once (`_decision`, cached per prefix length on the
  current path) and immediately takes the recorded evidence back out of the
  live guard; `_advance_guard` puts it back, followed by `observe`, only
  when the path actually moves past that position with the chosen token.
  So a position decided twice (a drafter probing, then a verify), or
  decided and then rolled over, records its evidence exactly once, as in
  `decode_free_running`. (Before this, the per-token path under a guard
  kept the tokens right but dropped interventions whenever a real drafter
  was active -- measured on the new tests: `_commit` rolled the cache back
  and re-walked it without re-deciding.)
- **Guard replay (block verification under loop guard v2)**: `verify_block`
  is bound with no guard AND with loop guard v2 -- the frozen v15b policy --
  so a drafted block costs one `encode_steps` forward there too. After that
  forward, `_verify_block` replays the guard over the block, position by
  position, with state only (`_replay_block_position`): the primary's
  decision at each position is `LoopGuardV2.decide` over the guard state
  reached after the candidates accepted so far; if it equals the candidate,
  the guard advances by it (evidence, then `observe`) and the next position
  is judged; the first position where the guard vetoes, recovers a close, or
  greedy simply disagrees ends the accepted prefix, and the KV cache is
  cropped there. The decision made at that position stays cached, so
  `run_speculative_decode`'s single-token fallback commits the guard's
  forced token without another forward. Loop guard v1 keeps the
  per-token path (`verify_step` per candidate) -- it shares the decision
  cache, but block verification under v1 is not the served policy and is
  not claimed here.
- `region_decode`: the KV prefix is built with `model.encode_prefix_with_
  visual_hidden` instead of `model.encode_prefix` (matching `decode_free_
  running`'s own branch), and `region_decode.decode_regions` runs once at
  the end over `tracker.states` -- the committed structural tokens only,
  same as the non-speculative path.

Every other `DecodeConfig` field `decode_free_running` understands
(`typed_decode`, `max_steps`; `prefill_chunk_tokens` is NOT supported here,
since the prefix pass below always calls the encode-prefix method with no
extra kwargs) behaves the same as the non-speculative path.

**Grammar masking for any drafter, including DSpark (2026-09-27)**: no
drafter here (`dflash`, `mtp`, or the new `dspark` variant) filters its own
proposals against the grammar before returning them -- there is nothing to
reuse from a masking call, because none exists. Grammar compliance comes
for free from THIS module's own verify step instead: `_verify_block`/
`verify_step` compare each candidate against the primary's OWN greedy pick
among `_legal_ids_at`'s legal set, so a grammar-illegal drafted token simply
fails to match and is rejected at that position (falling back to primary
decode for it), exactly like a correctly-typed but wrong token would. A
`SpeculativeDrafter` never needs its own copy of the grammar controller.

**`config.loop_guard="v2"` and block verification**: see "Guard replay"
above -- under the frozen v15b policy every drafter here, including DSpark,
now verifies a whole block in one forward pass, with output identical to
`decode_free_running` (tokens, node states and the `loop_guard_v2` record).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from unflatten.model import events_vocab
from unflatten.model.candidate_model import (
    KVCacheSnapshot,
    QwenDenseCandidateModel,
    TeacherForcedBatch,
)
from unflatten.model.composite_grammar import CompositeGrammarController, CompositeGrammarState
from unflatten.model.composite_vocab import CompositeVocabulary, QwenRef, StructuralRef
from unflatten.model.free_running_decode import (
    DecodeConfig,
    DecodeMaxStepsExceeded,
    LoopGuardV1,
    LoopGuardV2,
    _legal_packed_ids,
    _select_greedy,
    _structural_name,
    decode_free_running,
    default_loop_guard_v1,
)
from unflatten.model.loop_guard_v2_caps import load_loop_guard_v2_caps
from unflatten.model.node_lifecycle import NodeLifecycleTracker
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.model.region_decode import decode_regions
from unflatten.model.speculative.drafter import (
    DecoderSnapshot,
    SpeculativeDrafter,
    VerifyCostModel,
    run_speculative_decode,
)
from unflatten.model.typed_decode import decode_typed


def _guard_snapshot(
    guard: LoopGuardV1 | None, guard_v2: LoopGuardV2 | None
) -> tuple[LoopGuardV1 | None, LoopGuardV2 | None]:
    """Independent copy of the two (mutually-exclusive-in-practice) guard
    objects, stored per prefix length in `FreeRunningPrimaryDecoder.
    _by_length` so a `rollback_to` can restore guard state exactly like it
    restores the KV cache and grammar state. `None`/`None` copies to itself
    trivially.

    Each guard's own `snapshot()` (2026-09-28 fix, `docs/runs/dspark-
    rehearsal-2026-09-27.md` §9 point 4) replaces the `copy.deepcopy` this
    used to call: `LoopGuardV2` carries memoisation caches that grow with
    page length, and generic deepcopy re-walked and re-copied all of them
    on every single accepted token, measured at ~2.9 ms/call after 1 000
    positions and growing. `snapshot()` copies only the guard's own decode
    state and shares those caches by reference (safe -- see its
    docstring), so this call's cost no longer grows with page length.
    Used both to STORE a length's state and to REINSTALL a stored one as
    the live guard (`_sync_to`'s rollback branch) -- the same "give me an
    independent copy" operation either way."""
    return (
        guard.snapshot() if guard is not None else None,
        guard_v2.snapshot() if guard_v2 is not None else None,
    )


class FreeRunningPrimaryDecoder:
    """`PrimaryDecoder` (drafter.py) over a real `QwenDenseCandidateModel` +
    `CompositeGrammarController`, reproducing `decode_free_running`'s own
    greedy+grammar step exactly. Keyed-by-length caches (grammar state, KV
    cache snapshot, hidden state) mirror `test_drafter.py`'s
    `_ModelPrimaryDecoder` -- valid under the same assumption stated there:
    a round only ever walks the ONE growing prefix forward or rolls it back
    to an earlier point on that same path, never sideways.
    """

    def __init__(
        self,
        *,
        model: QwenDenseCandidateModel,
        controller: CompositeGrammarController,
        vocabulary: CompositeVocabulary,
        capability_profile_id: str,
        prefix_state: Any,
        prefix_hidden: torch.Tensor,
        guard: LoopGuardV1 | None = None,
        guard_v2: LoopGuardV2 | None = None,
    ) -> None:
        self._model = model
        self._controller = controller
        self._vocabulary = vocabulary
        self._cache_state = prefix_state
        self._cached_prefix: tuple[int, ...] = ()
        self.guard = guard
        self.guard_v2 = guard_v2
        fresh = controller.fresh(capability_profile_id)
        self._by_length: dict[
            int, tuple[CompositeGrammarState, Any, torch.Tensor, tuple[Any, Any]]
        ] = {0: (fresh, prefix_state.snapshot(), prefix_hidden, _guard_snapshot(guard, guard_v2))}
        #: The primary's decision at each prefix length on the CURRENT path,
        #: with the guard evidence that decision produced (module docstring,
        #: "guard replay"). An entry is dropped whenever `_by_length` for
        #: that length is rewritten, so it never outlives the state it was
        #: computed from.
        self._decisions: dict[int, tuple[int | None, _GuardEvidence | None]] = {}
        # Block verification (`verify_block`, one `encode_steps` call for a
        # whole draft) is bound as an attribute -- not defined on the class
        # -- so `drafter._verify_prefix`'s `getattr(primary, "verify_block",
        # None)` dispatch sees it only where it is proven equivalent: with
        # no guard, or with loop guard v2 (replayed per accepted position,
        # module docstring). Loop guard v1 keeps the per-token path.
        if guard is None:
            self.verify_block = self._verify_block

    def _sync_to(self, prefix: tuple[int, ...]) -> None:
        if prefix == self._cached_prefix:
            return
        if len(prefix) < len(self._cached_prefix):
            _, kv_snapshot, _, guard_snapshot = self._by_length[len(prefix)]
            self._cache_state = self._cache_state.rollback_to(kv_snapshot)
            self.guard, self.guard_v2 = _guard_snapshot(*guard_snapshot)
            self._cached_prefix = prefix
            return
        if (
            len(prefix) == len(self._cached_prefix) + 1
            and prefix[: len(self._cached_prefix)] == self._cached_prefix
        ):
            new_token_id = prefix[-1]
            length = len(self._cached_prefix)
            grammar_state, _, hidden_before, _ = self._by_length[length]
            if self.guard is not None or self.guard_v2 is not None:
                # The guard's evidence for this position comes from the
                # decision made there; make sure one exists (a no-op on
                # every path the decode loop takes, which always decided
                # this position before walking past it).
                self._decision(length, grammar_state, hidden_before)
            new_grammar_state = self._controller.step(
                grammar_state, self._vocabulary.unpack(new_token_id)
            )
            self._cache_state, hidden = self._model.encode_step(self._cache_state, new_token_id)
            self._advance_guard(length, new_token_id)
            self._store(len(prefix), new_grammar_state, self._cache_state.snapshot(), hidden)
            self._cached_prefix = prefix
            return
        raise AssertionError(f"unexpected prefix jump from {self._cached_prefix} to {prefix}")

    def _store(
        self, length: int, grammar_state: CompositeGrammarState, kv_snapshot: Any,
        hidden: torch.Tensor,
    ) -> None:
        """Record the state after `length` tokens on the current path (the
        guard is snapshotted as it is now) and drop the decision cached for
        a previous occupant of that length."""
        self._by_length[length] = (
            grammar_state, kv_snapshot, hidden, _guard_snapshot(self.guard, self.guard_v2)
        )
        self._decisions.pop(length, None)

    def _advance_guard(self, length: int, token_id: int) -> None:
        """Move the live guard(s) from prefix length `length` to `length +
        1` by `token_id`, exactly as `decode_free_running` does for one
        committed step: first the evidence the decision at `length`
        recorded (only if that decision chose `token_id` -- a token the
        primary did not choose is never committed, so its path is always
        rolled back), then `observe`."""
        if self.guard is None and self.guard_v2 is None:
            return
        decision = self._decisions.get(length)
        if decision is not None and decision[0] == token_id and decision[1] is not None:
            decision[1].apply(self.guard, self.guard_v2)
        self._observe(token_id)

    def _observe(self, token_id: int) -> None:
        """Feed one token to the guard(s)' run-length state -- only ever a
        token on the current path, which a later `rollback_to` undoes via
        `_by_length` (module docstring)."""
        token = self._vocabulary.unpack(token_id)
        name = _structural_name(token.local_id) if isinstance(token, StructuralRef) else None
        if self.guard is not None:
            self.guard.observe(name)
        if self.guard_v2 is not None:
            self.guard_v2.observe(token_id)

    @property
    def vocabulary(self) -> CompositeVocabulary:
        return self._vocabulary

    def hidden_before(self, prefix: tuple[int, ...]) -> torch.Tensor:
        # A prefix of the current path is already in `_by_length`: read it
        # without rolling the cache back. `_commit` asks for every position
        # of a just-verified block; a rollback + re-encode there would
        # re-pay, one `encode_step` at a time, the forwards `verify_block`
        # just saved.
        if (
            len(prefix) <= len(self._cached_prefix)
            and self._cached_prefix[: len(prefix)] == prefix
        ):
            return self._by_length[len(prefix)][2]
        self._sync_to(prefix)
        return self._by_length[len(prefix)][2]

    def mode_at(self, prefix: tuple[int, ...]) -> str:
        self._sync_to(prefix)
        return self._by_length[len(prefix)][0].mode

    @property
    def controller(self) -> CompositeGrammarController:
        return self._controller

    def grammar_state_at(self, prefix: tuple[int, ...]) -> CompositeGrammarState:
        """The grammar state after `prefix` -- for drafters that mask or walk
        their proposal under the grammar (`speculative.grammar_draft`). A
        prefix of the current path is read without moving the KV cache,
        like `hidden_before`."""
        if (
            len(prefix) <= len(self._cached_prefix)
            and self._cached_prefix[: len(prefix)] == prefix
        ):
            return self._by_length[len(prefix)][0]
        self._sync_to(prefix)
        return self._by_length[len(prefix)][0]

    def _legal_ids_at(
        self, grammar_state: CompositeGrammarState
    ) -> list[int]:
        legal_ids = _legal_packed_ids(self._controller, grammar_state, self._vocabulary)
        if self.guard is not None:
            legal_ids, _event = self.guard.filter(
                legal_ids, vocabulary=self._vocabulary, mode=grammar_state.mode
            )
        return legal_ids

    def _decide(
        self, logits_row: torch.Tensor, legal_ids: list[int], grammar_state: CompositeGrammarState,
        prefix_len: int,
    ) -> int:
        if self.guard_v2 is not None:
            return self.guard_v2.decide(
                logits_row,
                legal_ids,
                domain=self._controller.allowed(grammar_state),
                state=grammar_state,
                step=prefix_len,
            )
        return _select_greedy(logits_row, legal_ids)

    def _decision(
        self,
        length: int,
        grammar_state: CompositeGrammarState,
        hidden: torch.Tensor,
        logits_row: torch.Tensor | None = None,
    ) -> int | None:
        """The primary's choice after `length` tokens of the current path
        (`grammar_state`/`hidden` are that position's), computed once and
        cached in `_decisions`. The guard(s) are left exactly as they were:
        whatever evidence (counters, interventions) deciding recorded is
        taken back out and kept with the decision, to be applied by
        `_advance_guard` if and when the path moves past this position with
        the chosen token -- so a position decided twice, or decided and
        then rolled back, records its evidence once, like
        `decode_free_running` does.

        `logits_row` (2026-09-28 fix): the already-computed composite-head
        row for this position, when the caller (`_verify_block`) already
        batched the whole drafted block through `composite_head` in ONE
        call. `composite_head`'s Qwen sub-projection reads the full
        vocab-sized frozen weight regardless of row count (measured
        memory-bandwidth-bound: ~2.1 ms whether called with 1 row or 8), so
        calling it once per candidate in a block -- as this method used to
        do unconditionally -- paid that cost `len(candidates)` times over
        for no reason (`docs/runs/dspark-rehearsal-2026-09-27.md`'s "kde
        ide čas" section; measured ~14 ms/round for an 8-candidate block,
        ~4 ms for 3). `None` (the single-token path, `next_token`) keeps
        computing it here, unchanged."""
        cached = self._decisions.get(length)
        if cached is not None:
            return cached[0]

        def decide() -> int | None:
            if grammar_state.mode == "terminated":
                return None
            legal_ids = self._legal_ids_at(grammar_state)
            if not legal_ids:
                return None
            # Same calling convention as `decode_free_running` itself:
            # `hidden` is `(batch, d_model)`, and `composite_head` returns
            # `(batch, total_count)` -- a single `[0]` index reads the one
            # row this single-page (batch=1) decode ever has.
            row = logits_row if logits_row is not None else self._model.composite_head(hidden)[0]
            return self._decide(row, legal_ids, grammar_state, length)

        chosen, evidence = _decide_taking_back_evidence(self.guard, self.guard_v2, decide)
        self._decisions[length] = (chosen, evidence)
        return chosen

    def next_token(self, prefix: tuple[int, ...]) -> int | None:
        self._sync_to(prefix)
        grammar_state, _, hidden, _ = self._by_length[len(prefix)]
        return self._decision(len(prefix), grammar_state, hidden)

    def verify_step(self, prefix: tuple[int, ...], candidate_token: int) -> bool:
        return self.next_token(prefix) == candidate_token

    def _verify_block(self, prefix: tuple[int, ...], candidates: tuple[int, ...]) -> int:
        """RFC004 §37.1/§37.4: verify a whole drafted block against the
        primary in ONE `QwenDenseCandidateModel.encode_steps` forward pass
        instead of one `encode_step`/`next_token` call per candidate.
        Causal attention makes this safe: `encode_steps`'s hidden state at
        position `i` depends only on `candidates[:i+1]`, exactly the value
        a sequential `encode_step` after `candidates[:i]` would produce --
        regardless of whether a later candidate in the block ever gets
        accepted (`test_candidate_model_encode_steps.py`).

        Then the guard replay (module docstring): position by position, the
        primary's decision -- greedy, or `LoopGuardV2.decide` over the guard
        state reached so far -- is compared with the candidate
        (`_replay_block_position`); an accepted candidate advances the guard
        before the next position is judged. The first position where the
        decision differs (a veto, a close recovery, or plain disagreement)
        ends the accepted prefix; the decision made there stays cached, so
        the caller's single-token fallback commits it without another
        forward.

        Returns the number of LEADING candidates accepted (0 if the first
        is already wrong). The KV cache is cropped back to exactly that
        accepted length -- `encode_steps` speculatively extended it by
        `len(candidates)` positions regardless of acceptance."""
        self._sync_to(prefix)
        if not candidates:
            return 0
        grammar_state, _, hidden, _ = self._by_length[len(prefix)]
        base_composite_len = self._cache_state.composite_len
        hidden_batch, new_state = self._model.encode_steps(self._cache_state, candidates)
        # 2026-09-28 fix: one `composite_head` call for the whole block
        # instead of one per candidate (see `_decision`'s docstring --
        # measured ~2.1 ms per call regardless of row count, so `len
        # (candidates)` separate calls paid that memory-bandwidth-bound
        # cost that many times for nothing). Position `i`'s decision needs
        # the hidden state BEFORE candidate `i` is applied: `hidden` (the
        # prefix's own state) for `i == 0`, `hidden_batch[:, i - 1, :]`
        # otherwise -- `hidden_batch[:, i, :]` is the state AFTER candidate
        # `i`, which only matters for `_store`ing what the NEXT position
        # reads.
        decision_hidden = torch.cat([hidden.unsqueeze(1), hidden_batch[:, :-1, :]], dim=1)
        block_logits = self._model.composite_head(decision_hidden)[0]
        accepted = 0
        current_grammar = grammar_state
        current_hidden = hidden
        for i, candidate_token in enumerate(candidates):
            length = len(prefix) + i
            if not self._replay_block_position(
                length, candidate_token, current_grammar, current_hidden, block_logits[i]
            ):
                break
            accepted += 1
            current_grammar = self._controller.step(
                current_grammar, self._vocabulary.unpack(candidate_token)
            )
            current_hidden = hidden_batch[:, i, :]
            self._store(
                length + 1,
                current_grammar,
                KVCacheSnapshot(composite_len=base_composite_len + accepted),
                current_hidden,
            )
        self._cache_state = new_state.crop(len(candidates) - accepted)
        self._cached_prefix = prefix + tuple(candidates[:accepted])
        return accepted

    def _replay_block_position(
        self,
        length: int,
        candidate_token: int,
        grammar_state: CompositeGrammarState,
        hidden: torch.Tensor,
        logits_row: torch.Tensor,
    ) -> bool:
        """One step of the guard replay: does the primary (with its guard,
        in the state reached after `length` tokens) choose
        `candidate_token` here? If so, advance the guard past it exactly as
        a committed single-token step would, and say so. `logits_row` is
        this position's pre-batched `composite_head` row (`_verify_block`)."""
        if self._decision(length, grammar_state, hidden, logits_row) != candidate_token:
            return False
        self._advance_guard(length, candidate_token)
        return True


@dataclass(frozen=True, slots=True)
class _GuardEvidence:
    """What one guard decision recorded, as opposed to what it decided:
    counter increments (v1 `filter`, v2 `decide`) and appended v2
    interventions. `FreeRunningPrimaryDecoder._decision` takes it out of
    the live guard(s) right after deciding (`_decide_taking_back_evidence`)
    and `_advance_guard` puts it back when the path moves past that
    position, so the guard's record counts each committed position exactly
    once however often that position is decided or rolled over. Every other
    effect of `filter`/`decide` is a memo cache keyed by its own inputs
    (`LoopGuardV2._tokens`/`_windows`/`_anchors`); the decode-state fields
    only ever move in `observe`."""

    v1_counters: dict[str, int]
    v2_counters: dict[str, int]
    v2_interventions: tuple[dict[str, Any], ...]

    def apply(self, guard: LoopGuardV1 | None, guard_v2: LoopGuardV2 | None) -> None:
        """Add this (delta) evidence to the guard(s)' record."""
        if guard is not None:
            for key, value in self.v1_counters.items():
                guard.counters[key] = guard.counters.get(key, 0) + value
        if guard_v2 is not None:
            for key, value in self.v2_counters.items():
                guard_v2.counters[key] = guard_v2.counters.get(key, 0) + value
            guard_v2.interventions.extend(dict(i) for i in self.v2_interventions)


def _decide_taking_back_evidence(
    guard: LoopGuardV1 | None,
    guard_v2: LoopGuardV2 | None,
    decide: Callable[[], int | None],
) -> tuple[int | None, _GuardEvidence | None]:
    """Run `decide()` and return its choice plus the evidence it recorded,
    with the guard(s)' record restored to what it was before the call."""
    if guard is None and guard_v2 is None:
        return decide(), None
    v1_before = dict(guard.counters) if guard is not None else {}
    v2_before = dict(guard_v2.counters) if guard_v2 is not None else {}
    n_before = len(guard_v2.interventions) if guard_v2 is not None else 0
    chosen = decide()
    v1_delta: dict[str, int] = {}
    if guard is not None:
        v1_delta = _counter_delta(v1_before, guard.counters)
        guard.counters.clear()
        guard.counters.update(v1_before)
    v2_delta: dict[str, int] = {}
    added: tuple[dict[str, Any], ...] = ()
    if guard_v2 is not None:
        v2_delta = _counter_delta(v2_before, guard_v2.counters)
        guard_v2.counters.clear()
        guard_v2.counters.update(v2_before)
        added = tuple(guard_v2.interventions[n_before:])
        del guard_v2.interventions[n_before:]
    return chosen, _GuardEvidence(
        v1_counters=v1_delta, v2_counters=v2_delta, v2_interventions=added
    )


def _counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: value - before.get(key, 0)
        for key, value in after.items()
        if value != before.get(key, 0)
    }


def _commit(
    adapter: FreeRunningPrimaryDecoder,
    tracker: NodeLifecycleTracker,
    snapshot: DecoderSnapshot,
    accepted: tuple[int, ...],
) -> DecoderSnapshot:
    committed = snapshot.committed_tokens
    for i, token_id in enumerate(accepted):
        prefix_before = committed + accepted[:i]
        hidden = adapter.hidden_before(prefix_before)
        token = adapter.vocabulary.unpack(token_id)
        if isinstance(token, StructuralRef):
            name = _structural_name(token.local_id)
            anchor = hidden[0].detach().clone()
            tracker.apply_structural_token(name, event_index=len(committed) + i, anchor=anchor)
    return DecoderSnapshot(committed_tokens=committed + accepted)


def speculative_decode_free_running(
    *,
    model: QwenDenseCandidateModel,
    controller: CompositeGrammarController,
    vocabulary: CompositeVocabulary,
    visual_features: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_token_id: int,
    config: DecodeConfig,
    speculator: Callable[[FreeRunningPrimaryDecoder], SpeculativeDrafter] | None = None,
    verify_cost_model: VerifyCostModel | None = None,
    initial_cache_factory: Any = None,
) -> RawDocumentHypothesis:
    """`speculator=None` (default): exactly `decode_free_running` -- see
    module docstring. Otherwise `speculator` is called once with the
    fresh `FreeRunningPrimaryDecoder` built for this decode, and its
    result drives `run_speculative_decode`. `verify_cost_model` (2026-09-28,
    optional): forwarded to `run_speculative_decode` unchanged -- see its
    own docstring for the confidence-scheduled truncation it enables when
    `speculator` returns a `draft_with_confidence`-capable drafter.

    `initial_cache_factory` (2026-09-28, ported from `decode_free_running`'s
    own parameter of the same name -- module docstring's "the prefix pass
    below always calls the encode-prefix method with no extra kwargs" used
    to make this parameter simply not exist here): `None` (default) is
    exactly today's behaviour. When given, it is called with no arguments
    once (e.g. `Float8KVCache`, `StaticKVCache`) and the result is passed as
    `initial_cache` to whichever prefix method `config.region_decode`
    selects, exactly like `decode_free_running`."""
    if speculator is None:
        return decode_free_running(
            model=model,
            controller=controller,
            vocabulary=vocabulary,
            visual_features=visual_features,
            image_grid_thw=image_grid_thw,
            image_token_id=image_token_id,
            config=config,
            initial_cache_factory=initial_cache_factory,
        )
    if config.loop_guard == "v3":
        # DEC-055 loop guard v3 (PROPOSED) rewinds and bans by position,
        # which the draft/verify/commit loop does not replay -- refused
        # rather than silently decoded under v2 semantics.
        raise ValueError("loop_guard 'v3' is not supported with a speculator")
    if visual_features.dim() != 3 or visual_features.shape[0] != 1:
        raise ValueError(
            "speculative_decode_free_running is single-page/batch=1, got "
            f"visual_features shape {tuple(visual_features.shape)}"
        )
    device = visual_features.device
    visual_tokens = visual_features.shape[1]
    placeholder_id = vocabulary.pack(QwenRef(image_token_id))
    bos_local_id = events_vocab.TOKEN_IDS[events_vocab.END_DOCUMENT]
    bos_id = vocabulary.pack(StructuralRef(local_id=bos_local_id))
    placeholder_block = [placeholder_id] * visual_tokens + [bos_id]

    # W6-2 (DEC-047): same guard construction `decode_free_running` uses --
    # `"v1"`/`"v2"` are mutually exclusive, `None` builds neither.
    guard = (
        default_loop_guard_v1(config.capability_profile_id) if config.loop_guard == "v1" else None
    )
    guard_v2 = (
        LoopGuardV2(ranges=vocabulary.ranges, caps=load_loop_guard_v2_caps())
        if config.loop_guard == "v2"
        else None
    )

    tracker = NodeLifecycleTracker()
    with torch.inference_mode():
        prefix_batch = TeacherForcedBatch(
            visual_features=visual_features,
            composite_input_ids=torch.tensor([placeholder_block], dtype=torch.long, device=device),
            composite_attention_mask=torch.ones(
                1, len(placeholder_block), dtype=torch.long, device=device
            ),
            image_grid_thw=image_grid_thw,
        )
        visual_hidden = None
        prefix_kwargs = (
            {} if initial_cache_factory is None
            else {"initial_cache": initial_cache_factory()}
        )
        if config.region_decode:
            retain = getattr(model, "encode_prefix_with_visual_hidden", None)
            if retain is None:
                raise ValueError(
                    "region_decode requires encode_prefix_with_visual_hidden support"
                )
            prefix_state, prefix_hidden, visual_hidden = retain(prefix_batch, **prefix_kwargs)
        else:
            prefix_state, prefix_hidden = model.encode_prefix(prefix_batch, **prefix_kwargs)
        adapter = FreeRunningPrimaryDecoder(
            model=model,
            controller=controller,
            vocabulary=vocabulary,
            capability_profile_id=config.capability_profile_id,
            prefix_state=prefix_state,
            prefix_hidden=prefix_hidden,
            guard=guard,
            guard_v2=guard_v2,
        )
        drafter = speculator(adapter)
        result = run_speculative_decode(
            adapter,
            drafter,
            max_tokens=config.max_steps,
            commit=lambda snap, acc: _commit(adapter, tracker, snap, acc),
            verify_cost_model=verify_cost_model,
        )
        # Sync to the committed stream BEFORE reading the guard: a
        # per-token verify round can leave the cache one token short of
        # what it committed, and the guard state lives with the cache.
        terminated = adapter.mode_at(result.committed_tokens) == "terminated"
        guard_evidence = _guard_evidence(adapter)
        if not terminated:
            raise DecodeMaxStepsExceeded(
                f"speculative free-running decode did not terminate within "
                f"{config.max_steps} steps",
                accepted_tokens=result.committed_tokens,
                node_states=tracker.states,
                **guard_evidence,
            )
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

    if adapter.guard_v2 is not None:
        issues: tuple[Any, ...] = (
            adapter.guard_v2.record(adapter.guard.counters if adapter.guard is not None else {}),
        )
    elif adapter.guard is not None:
        issues = ({"kind": "loop_guard_v1", "counters": dict(adapter.guard.counters)},)
    else:
        issues = ()

    return RawDocumentHypothesis(
        accepted_tokens=result.committed_tokens,
        node_states=tracker.states,
        typed_predictions=typed,
        region_predictions=regions,
        terminated=True,
        issues=issues,
    )


def _guard_evidence(adapter: FreeRunningPrimaryDecoder) -> dict[str, Any]:
    """`DecodeMaxStepsExceeded`'s `loop_guard_counters`/`loop_guard_record`
    kwargs, mirroring `decode_free_running`'s own `_guard_evidence`
    closure, over the COMMITTED guard state `adapter` carries at the point
    of the raise."""
    v1_counters = adapter.guard.counters if adapter.guard is not None else {}
    if adapter.guard_v2 is not None:
        record = adapter.guard_v2.record(v1_counters)
        return {"loop_guard_counters": record["counters"], "loop_guard_record": record}
    return {"loop_guard_counters": dict(v1_counters) if adapter.guard is not None else None}
