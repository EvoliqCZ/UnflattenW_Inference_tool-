"""RFC004 §37.1, §37.4-5: SpeculativeDrafter protocol and a token-level harness.

**Scope and gap.** §37.1's normative loop is
``snapshot -> draft -> primary+grammar verify -> commit -> rollback`` over the
*authoritative* decoder state (grammar stack, KV cache, NodeStateStore,
emitted counters). This module implements that loop faithfully at the
token/grammar level with an opaque, immutable :class:`DecoderSnapshot` — it
does **not** wire into the real production state:

- ``composite_grammar.CompositeGrammarState`` is immutable (`step` returns a
  new instance, never mutates), so grammar rollback is free there.
- ``candidate_model.IncrementalDecodeState`` now HAS an additive, opt-in
  crop/restore API (``snapshot`` / ``crop`` / ``rollback_to``, DEC-047) --
  closing the gap this docstring used to name. ``test_drafter.py``'s
  ``TestModelBackedPrimaryDecoder`` wires a real ``QwenDenseCandidateModel``
  (stub backend) behind ``PrimaryDecoder`` using exactly that primitive, so
  the protocol proven here is also proven against a real KV cache, not only
  an opaque token list. The production decode path
  (``free_running_decode.py``) still never calls any of it -- wiring an
  active speculator into production inference is a separate, unratified
  change.

Given that, this harness proves the *protocol* — snapshot immutability
already makes rollback structural (an exception before a successful commit
call can never have mutated the pre-commit snapshot, because nothing here
mutates anything) — and the lossless-equivalence property, on a token stream
standing in for the real one (plus, now, on a real model's KV cache).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class DecoderSnapshot:
    """Opaque authoritative state: RFC004 §37.1's "authoritative DecoderState".

    Immutable by construction, exactly like ``CompositeGrammarState`` — a
    round that commits or rolls back can only ever produce a *new* instance.
    """

    committed_tokens: tuple[int, ...] = ()


class PrimaryDecoder(Protocol):
    """Stand-in for "primary decoder + grammar verify against the snapshot" (§37.1)."""

    def verify_step(self, prefix: tuple[int, ...], candidate_token: int) -> bool:
        """True iff the primary decoder + grammar would emit ``candidate_token`` next."""
        ...

    def next_token(self, prefix: tuple[int, ...]) -> int | None:
        """Primary-only greedy decode: the next token, or ``None`` at end of stream."""
        ...

    def verify_block(self, prefix: tuple[int, ...], candidates: tuple[int, ...]) -> int:
        """OPTIONAL (RFC004 §37.1/§37.4): verify a whole drafted block in
        ONE primary forward pass instead of one ``verify_step`` call per
        candidate, returning the number of LEADING candidates accepted (0
        if the first is already wrong). ``_verify_prefix`` below only calls
        this when the concrete ``primary`` object actually defines it
        (``getattr(primary, "verify_block", None)``) -- a ``PrimaryDecoder``
        without one (e.g. this module's own ``_StubPrimary``-shaped test
        harnesses) keeps using ``verify_step`` per candidate, unchanged."""
        ...


class SpeculativeDrafter(Protocol):
    """RFC004 §37.1: "speculator proposes tokens/events without side effects"."""

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]: ...

    def draft_with_confidence(
        self, snapshot: DecoderSnapshot
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """OPTIONAL (2026-09-28, confidence-scheduled verification): same
        proposal as `draft`, plus one per-candidate survival probability
        (`p_i` in `[0, 1]`, DSpark v2's confidence head). `run_speculative_
        decode` only calls this when the concrete drafter actually defines
        it (`getattr(drafter, "draft_with_confidence", None)`) -- a drafter
        without one (every existing one) keeps using `draft`, unchanged."""
        ...


@dataclass(frozen=True, slots=True)
class OffDrafter:
    """§37.5 ``runtime_active_speculator: off`` — proposes nothing, ever."""

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]:
        return ()


@dataclass(frozen=True, slots=True)
class OracleDrafter:
    """Lossless-equivalence harness drafter: proposes exactly what the primary
    decoder would produce, up to ``max_block``. Used to prove that going
    through the draft/verify/commit loop changes nothing observable versus
    ``off`` (§37.4)."""

    primary: PrimaryDecoder
    max_block: int = 4

    def draft(self, snapshot: DecoderSnapshot) -> tuple[int, ...]:
        prefix = list(snapshot.committed_tokens)
        proposed: list[int] = []
        for _ in range(self.max_block):
            token = self.primary.next_token(tuple(prefix))
            if token is None:
                break
            proposed.append(token)
            prefix.append(token)
        return tuple(proposed)


def _verify_prefix(
    primary: PrimaryDecoder, committed: tuple[int, ...], proposed: tuple[int, ...]
) -> tuple[int, ...]:
    if not proposed:
        return ()
    verify_block = getattr(primary, "verify_block", None)
    if verify_block is not None:
        # RFC004 §37.1/§37.4: ONE primary forward pass over the whole
        # block, instead of `len(proposed)` `verify_step` calls -- see
        # `PrimaryDecoder.verify_block`'s docstring.
        accepted_count = verify_block(committed, proposed)
        return proposed[:accepted_count]
    prefix = list(committed)
    accepted: list[int] = []
    for token in proposed:
        if not primary.verify_step(tuple(prefix), token):
            break
        accepted.append(token)
        prefix.append(token)
    return tuple(accepted)


def _default_commit(snapshot: DecoderSnapshot, accepted: tuple[int, ...]) -> DecoderSnapshot:
    return DecoderSnapshot(committed_tokens=snapshot.committed_tokens + accepted)


@dataclass(frozen=True, slots=True)
class VerifyCostModel:
    """Confidence-scheduled verification's cost constants (2026-09-28,
    DSpark v2). Owner instruction: "policy fields with placeholders" --
    these are read from `s7_dflash_policy.json`'s `verify_cost_model` block
    (`is_placeholder: true` until an owner freezes them from real
    measurement), never a code literal here. All three are milliseconds,
    the same unit the 2026-09-27/28 rehearsal doc measured
    (`docs/runs/dspark-rehearsal-2026-09-27.md` §14): plain step ~34 ms,
    one verify-round forward ~61 ms at real context, drafter forward
    ~16 ms (DSpark) / ~3 ms (MTP). `drafter_cost_ms` is informational only
    here -- it is already sunk by the time a verify-length choice is made
    (the drafter has already produced its confidences), so it does not
    enter :func:`choose_verify_length`'s comparison, which is about
    whether to pay `verify_round_cost_ms` at all."""

    plain_step_cost_ms: float
    verify_round_cost_ms: float
    drafter_cost_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.plain_step_cost_ms <= 0:
            raise ValueError("plain_step_cost_ms must be > 0")
        if self.verify_round_cost_ms <= 0:
            raise ValueError("verify_round_cost_ms must be > 0")


def choose_verify_length(
    survival_probs: tuple[float, ...], cost_model: VerifyCostModel
) -> int:
    """`L` in `0..len(survival_probs)` maximizing expected NET benefit over
    doing a single plain step instead. Expected accepted tokens at length
    `L` is the cumulative-survival sum `sum_{i=1}^{L} prod_{j<i} p_j`
    (`p_j` = `survival_probs[j-1]`, independence approximation over slots --
    the same approximation the paper's confidence scheduling makes).
    Benefit at `L >= 1` is `expected_accepted(L) * plain_step_cost_ms -
    verify_round_cost_ms` (replacing up to `L` plain steps with one verify
    round); benefit at `L = 0` (skip speculation, do a plain step) is `0` by
    definition -- the baseline every other length is compared against. `L =
    0` wins whenever every positive length's benefit is `<= 0`, so
    speculation is never slower than plain decode by more than the
    drafter's own (already-sunk) cost."""
    best_length = 0
    best_benefit = 0.0
    cumulative_survival = 1.0
    expected_accepted = 0.0
    for i, p in enumerate(survival_probs, start=1):
        cumulative_survival *= p
        expected_accepted += cumulative_survival
        benefit = (
            expected_accepted * cost_model.plain_step_cost_ms - cost_model.verify_round_cost_ms
        )
        if benefit > best_benefit:
            best_benefit = benefit
            best_length = i
    return best_length


def _draft_with_optional_confidence(
    drafter: SpeculativeDrafter, snapshot: DecoderSnapshot
) -> tuple[tuple[int, ...], tuple[float, ...] | None]:
    draft_with_confidence = getattr(drafter, "draft_with_confidence", None)
    if draft_with_confidence is not None:
        return draft_with_confidence(snapshot)
    return drafter.draft(snapshot), None


def run_speculative_decode(
    primary: PrimaryDecoder,
    drafter: SpeculativeDrafter,
    *,
    max_tokens: int,
    start: DecoderSnapshot | None = None,
    commit: Callable[[DecoderSnapshot, tuple[int, ...]], DecoderSnapshot] = _default_commit,
    verify_cost_model: VerifyCostModel | None = None,
) -> DecoderSnapshot:
    """RFC004 §37.1 loop: draft, verify the accepted prefix, commit atomically.

    If ``commit`` raises, the exception propagates before ``snapshot`` in
    this function is ever reassigned — the caller's ``start`` (or the last
    successfully committed snapshot) is left exactly as it was. "Partially
    committed draft" (§37.1, forbidden) cannot happen here because
    :class:`DecoderSnapshot` is immutable and ``commit`` either returns a
    whole new snapshot or does not return at all.

    ``verify_cost_model`` (2026-09-28, optional, additive): when given AND
    ``drafter`` implements ``draft_with_confidence``, the proposed block is
    first truncated to :func:`choose_verify_length`'s chosen length before
    verification -- possibly to zero, in which case ``_verify_prefix``
    returns ``()`` immediately and the existing §37.1 fallback below runs
    exactly one plain ``primary.next_token`` step, unchanged. ``None``
    (default) or a drafter with no ``draft_with_confidence`` leaves this
    function byte-identical to before this parameter existed.
    """
    snapshot = start if start is not None else DecoderSnapshot()
    while len(snapshot.committed_tokens) < max_tokens:
        # Never commit past `max_tokens`: a non-speculative decode stops at
        # exactly that many tokens, so a drafted block may not overshoot it.
        proposed, confidences = _draft_with_optional_confidence(drafter, snapshot)
        proposed = proposed[: max_tokens - len(snapshot.committed_tokens)]
        if verify_cost_model is not None and confidences is not None:
            confidences = confidences[: len(proposed)]
            length = choose_verify_length(confidences, verify_cost_model)
            proposed = proposed[:length]
        accepted = _verify_prefix(primary, snapshot.committed_tokens, proposed)
        if not accepted:
            # §37.1 fallback: nothing verified, primary decode continues alone.
            token = primary.next_token(snapshot.committed_tokens)
            if token is None:
                break
            accepted = (token,)
        snapshot = commit(snapshot, accepted)
    return snapshot


def token_stream_ast_hash(tokens: tuple[int, ...]) -> str:
    """Deterministic stand-in for a Canonical AST hash over a token stream.

    This is NOT :func:`unflatten.ast.hashing.ast_hash` — this harness has no
    real AST, only a token stream standing in for one (module docstring). It
    exists so the lossless-equivalence test has something hash-shaped to
    compare, matching §37.4's "final Canonical AST bytes/hash" axis.
    """
    payload = ",".join(str(token) for token in tokens).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


#: RFC004 §37.5 "exact registry groups `speculator.dflash_speculator` and
#: `speculator.mtp_speculator`".
_VALID_SPECULATORS: tuple[str, ...] = ("dflash_speculator", "mtp_speculator")


@dataclass(frozen=True, slots=True)
class SpeculativeDecodingConfig:
    """RFC004 §37.5 Component Lock fields, kept as a data contract only."""

    mtp_aux_training_used: bool
    installed_speculators: tuple[str, ...] = ()
    runtime_active_speculator: str = "off"
    drafter_target_compatibility_hash: str | None = None
    transaction_protocol_version: str = "0.1.0"

    def __post_init__(self) -> None:
        for name in self.installed_speculators:
            if name not in _VALID_SPECULATORS:
                raise ValueError(
                    f"unknown speculator {name!r}, expected one of {_VALID_SPECULATORS}"
                )
        if self.runtime_active_speculator != "off":
            if self.runtime_active_speculator not in self.installed_speculators:
                raise ValueError(
                    f"runtime_active_speculator {self.runtime_active_speculator!r} "
                    f"is not in installed_speculators {self.installed_speculators}"
                )
            if not self.drafter_target_compatibility_hash:
                raise ValueError("an active speculator requires drafter_target_compatibility_hash")
