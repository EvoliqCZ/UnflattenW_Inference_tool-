"""Decoder training targets from a Canonical Target Bundle.

The last step of the dataset→model bridge. `encode.encode_structural`
already turns a bundle's structural stream into the RFC004 §18.2 token
alphabet; this module turns that token list into what a training step
needs — target ids, teacher-forcing inputs, and the grammar mask.

Two properties are load-bearing and both are tested rather than assumed:

**Padding is `IGNORE_INDEX`, not a token.** A padded position has no
target. Encoding it as a real token id would train the decoder to emit
that token after the document ends, and RFC005 is explicit that a missing
target is never encoded as a value.

**The grammar must permit the data.** The mask comes from the §18.4
pushdown controller, and the controller is derived from RFC001's
parent-child rules while the tokens come from the dataset. If the two
ever drift, the correct target becomes forbidden at some step and the
loss is unlearnable — silently, because a masked-out target simply
receives no probability mass. `allowed_mask_sequence` is therefore
checked against the real targets in the tests, not trusted.
"""

from __future__ import annotations

from typing import Any

import torch

from unflatten.dataset.fragments.envelope import supervised_event_ranges
from unflatten.model import grammar
from unflatten.model.encode import encode_structural
from unflatten.model.encode_fragment import (
    encode_fragment,
    encode_fragment_with_origins,
)
from unflatten.model.events_vocab import END_DOCUMENT, TOKEN_IDS

#: Loss-ignored label, matching `torch.nn.CrossEntropyLoss` default.
IGNORE_INDEX = -100

#: Capability profile the grammar controller runs under.
#:
#: Flipped to `word-visual-v1` with DEC-003 §1: the `m2a-v3` store
#: declares that profile on every document, and a v0 controller REFUSES
#: a v1 token sequence outright (`grammar.GrammarViolation` on the first
#: `HEADER_OPEN`) — so the mask could not be built for the very corpus
#: it exists to constrain. The v1 profile is a superset, so a v0
#: document still masks correctly under it; `build_cache` records the
#: profile in the cache identity, so a cache built under either one is
#: never confused for the other.
#:
#: **Kept as a literal for the historical S1 feature-build reproduction
#: only (DEC-024 wiring, `spec-viacstranove-dokumenty-plan.md`).** Every
#: RATIFIED `capability_profile.CapabilityProfile` -- including
#: `PRIMARY_S1_CAPABILITY_PROFILE` -- declares `ast_profile =
#: "word-visual-v1.1"`, ratified after this literal was already load-
#: bearing for the caches on disk today. So this constant is not "the S1
#: grammar" in general, only the specific historical one; a NEW build
#: choosing a capability profile explicitly (`build_cache.py`'s
#: `--capability-profile`) derives its grammar from that profile's
#: `ast_profile` instead, never from this literal.
DEFAULT_PROFILE = "word-visual-v1"

#: Token the decoder is primed with before the first real step. The
#: alphabet has no dedicated BOS: the first target is always a document
#: open, so priming with the termination token is unambiguous.
BOS_TOKEN_ID = TOKEN_IDS[END_DOCUMENT]


def token_sequence(
    structural: dict[str, Any], fragment: dict[str, Any] | None = None
) -> list[str]:
    """Token names for a document, or for one page of it.

    With `fragment`, the sequence covers only what that page contains
    (RFC003 §26). A training sample is one page, so using the whole
    document would ask page 1 to predict page 7 — unlearnable for
    anything multi-page, which is most of the corpus.
    """
    if fragment is not None:
        return encode_fragment(fragment, structural)
    return encode_structural(structural)


def declares_supervision_scope(fragment: dict[str, Any]) -> bool:
    """Does this fragment state which of its slices may be supervised?

    Fragments written before RFC003 §26 `page_evidence` carry ownership
    and nothing else. They cannot be read either way in good faith:
    treating them as fully supervised is the fabricated positive this
    change removes, and treating them as fully withdrawn silently
    empties a cache. A caller must see the difference and count it.
    """
    return any(
        "supervision_mask" in entry for entry in fragment.get("slices", ())
    )


def supervised_event_indices(fragment: dict[str, Any]) -> set[int]:
    """Document event indices this fragment may be supervised on.

    `envelope.supervised_event_ranges` is the sanctioned reading —
    `structural_event_ranges` is ownership, which is partition
    bookkeeping and says nothing about whether a label is true.
    """
    return {
        index
        for span in supervised_event_ranges(fragment)
        for index in range(int(span["start"]), int(span["end"]))
    }


def present_supervised_event_indices(
    fragment: dict[str, Any],
    structural: dict[str, Any],
    *,
    dropped_node_ids: frozenset[str] = frozenset(),
) -> set[int]:
    """Document event indices supervised under `s2-standalone-page-v1`
    (RFC003-ERRATA-002 E1).

    A page is a closed document of its own (DEC-033 §1): a node's
    structure is supervised here iff it has a slice on THIS page with
    `supervision_mask` true — never the page-of-owner-only reading
    `supervised_event_indices` gives, which is the sanctioned reading for
    every other profile and stays unchanged. The union of `events_of`
    (open + text + close, from the document-level event stream, exactly
    the same events `encode_standalone_fragment_with_origins` emits for a
    present node) over exactly those slices, so the invariant an
    unsupervised piece asserts holds by construction: effective
    (supervised) here iff the node has a supervised slice on this page.

    `dropped_node_ids` is the same DEC-033 §5/§6 page-normalization
    projection (`page_projection.project`) the encoder removes from
    `present` with: a dropped node is absent from this page, not merely
    unsupervised on it, so its events are excluded here too.
    """
    node_ids = {
        entry["node_id"]
        for entry in fragment.get("slices", ())
        if entry.get("supervision_mask")
    } - dropped_node_ids
    indices: set[int] = set()
    for event in structural.get("events", ()):
        if event.get("node_id") in node_ids:
            indices.add(event["index"])
    return indices


def unsupervised_target_steps(
    structural: dict[str, Any], fragment: dict[str, Any]
) -> torch.Tensor:
    """`(steps,)` bool: True where the target must not be scored.

    RFC003 §29.3 is one-directional — `supervision_mask` implies
    `observable_mask` — and an `unresolved` node has no page anyone can
    point at, so nothing about it may reach a loss. Its *structural*
    events are included: the decoder trains on that stream, so leaving
    them scored would train the fabricated placement itself.

    Ownership is untouched. The node keeps its single owner and its
    `structural_event_ranges`, and its tokens stay in the sequence — the
    event partition is about completeness, not about whether a label is
    true, and removing the tokens would make the stream ungrammatical.

    The envelope's own tokens (`FRAGMENT_BEGIN`, the resume chain, the
    suspends, `END_FRAGMENT`) have no origin event: they describe the
    window, not a node, and stay scored. A page where nothing else does
    is refused by the caller rather than trained as an empty page.
    """
    supervised = supervised_event_indices(fragment)
    _tokens, origins = encode_fragment_with_origins(fragment, structural)
    return torch.tensor(
        [origin is not None and origin not in supervised for origin in origins],
        dtype=torch.bool,
    )


def decoder_targets(
    structural: dict[str, Any],
    *,
    fragment: dict[str, Any] | None = None,
    pad_to: int | None = None,
    profile: str = DEFAULT_PROFILE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(target_ids, input_ids)` for teacher-forced training.

    `input_ids` is `target_ids` shifted right and primed, so step *i*
    predicts `target_ids[i]` having consumed `input_ids[:i+1]`. Padding
    beyond the sequence is `IGNORE_INDEX` in the targets and repeats the
    priming token in the inputs, which the loss never scores.

    Positions belonging to an `unresolved` placement are withdrawn the
    same way — present in `input_ids`, `IGNORE_INDEX` in `target_ids`.
    It is not optional, because a caller who forgets it trains the
    fabricated positives this exists to remove; on a fragment without
    `page_evidence` it is a no-op.

    `profile` is accepted for call-site symmetry with `allowed_mask_sequence`
    (a caller building both from the same `structural`/`fragment` names one
    profile once) but is NOT consulted here: `token_sequence` maps AST node
    types to tokens directly and never runs the grammar controller --
    profile eligibility is a mask `allowed_mask_sequence` computes, not a
    constraint this function enforces.
    """
    del profile  # unused -- see docstring
    tokens = token_sequence(structural, fragment)
    ids = torch.tensor([TOKEN_IDS[t] for t in tokens], dtype=torch.long)
    inputs = torch.cat([torch.tensor([BOS_TOKEN_ID], dtype=torch.long), ids[:-1]])
    if fragment is not None:
        # Withdrawal applies to the targets only: the tokens stay in the
        # inputs so teacher forcing sees a grammatical page.
        ids = ids.masked_fill(
            unsupervised_target_steps(structural, fragment), IGNORE_INDEX
        )
    if pad_to is None:
        return ids, inputs
    if pad_to < ids.shape[0]:
        raise ValueError(
            f"pad_to {pad_to} is shorter than the {ids.shape[0]}-token sequence"
        )
    gap = pad_to - ids.shape[0]
    ids = torch.cat([ids, torch.full((gap,), IGNORE_INDEX, dtype=torch.long)])
    inputs = torch.cat([inputs, torch.full((gap,), BOS_TOKEN_ID, dtype=torch.long)])
    return ids, inputs


def allowed_mask_sequence(
    structural: dict[str, Any],
    *,
    fragment: dict[str, Any] | None = None,
    pad_to: int | None = None,
    profile: str = DEFAULT_PROFILE,
) -> torch.Tensor:
    """`(steps, vocab)` boolean mask of grammar-legal tokens per step.

    Row *i* is what the §18.4 controller permits after consuming the
    first *i* tokens — i.e. the choices available when predicting
    `target_ids[i]`.
    """
    tokens = token_sequence(structural, fragment)
    steps = pad_to if pad_to is not None else len(tokens)
    if steps < len(tokens):
        raise ValueError(
            f"pad_to {steps} is shorter than the {len(tokens)}-token sequence"
        )
    mask = torch.zeros(steps, len(TOKEN_IDS), dtype=torch.bool)
    state = grammar.fresh(profile)
    for index, token in enumerate(tokens):
        for allowed in grammar.allowed_events(state):
            mask[index, TOKEN_IDS[allowed]] = True
        state = grammar.step(state, token)
    # padded rows stay all-False: nothing is legal past the end, and the
    # loss ignores those positions anyway
    return mask
