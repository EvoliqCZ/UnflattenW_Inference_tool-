"""The model-arm adapter (design §2.6 U19, plan P7): a `"terminated"`
sealed `PageDecodeResult` -> `hy.PageHypothesis`, over the exact same
anchor rule `hypothesis.from_model_hypothesis` uses, without ever
importing it.

`page_hypothesis` itself reads only `sealed.PageDecodeResult`/
`sealed.NodeStateProjection` -- plain dataclasses over `int`/`str`/dict --
and never imports `unflatten.model.prediction_ir`, whose
`RawDocumentHypothesis`/`TypedPrediction` this module deliberately does
not construct. (`hy.from_token_ids` itself still pulls in
`unflatten.model.node_lifecycle`, whose `NodeState.anchor` is a torch
tensor, transitively through `hypothesis.py` -- that dependency is
`hypothesis.py`'s own, not something this module avoids; `sealed.py`'s
torch-free promise is about the SEALED ARTEFACT shape, which this module
also never widens.) `test_sealed_adapter.py` pins that this hand-mirrored
rule agrees with `from_model_hypothesis` on a real page, so a future edit
to either side that silently diverges is caught there, not here.

**R12 (design §2.6, DEC-038: S2 is retired).** The model arm's identity
carries no stage field -- an S2 and an S3 artefact are indistinguishable
by `role`/`arm` alone (design §2.6 MEASURED, both `s2_exit_iid_provisional`
/ `trained`). `refuse_unless_checkpoint_ok` is the CLI's only guard against
ever scoring a retired checkpoint again: it requires the caller's
`--checkpoint-id` to equal the artefact's own AND refuses the one retired
id outright, even when they agree.
"""

from __future__ import annotations

from typing import Any

from unflatten.assembly import hypothesis as hy
from unflatten.eval.s2_gate import sealed
from unflatten.model import composite_vocab as cv

#: The `s2-discovery-v13` checkpoint (DEC-038, S2 retired 2026-09-22): the
#: sole entry this project ever names here. A P7 mutant deletes this set or
#: the comparison in `refuse_unless_checkpoint_ok` -- both must fail a test.
RETIRED_CHECKPOINT_IDS: frozenset[str] = frozenset(
    {"ckpt_3561da0944cb07f62d522e3f83afbe9f90e6d7577b7b3f5f39aaf40cae0afb4c"}
)

#: U20: pinned page geometry. `assemble([page], ...)` with exactly one
#: page never reads `page_ordinal` against a neighbour (`assemble.py`'s own
#: seam logic only fires between two consecutive ordinals), and no builder
#: pointer or derivation reads `width_px`/`height_px` for a standalone
#: page either -- the values are carried only because `PageInput` requires
#: them, never because anything downstream inspects them here.
PAGE_INPUT = hy.PageInput(page_ordinal=0, width_px=2480, height_px=3508)


class SealedArtefactRefused(ValueError):
    """A sealed artefact or its `--checkpoint-id` failed a P7 guard."""


def refuse_unless_checkpoint_ok(artefact: sealed.SealedCohortArtefact, checkpoint_id: str) -> None:
    """R12: refuse unless `checkpoint_id` equals the artefact's own AND is
    not in `RETIRED_CHECKPOINT_IDS` -- checked in that order so a caller
    passing the retired id explicitly (matching it on purpose) is refused
    for the retirement, not merely told it "matched"."""
    actual = artefact.eval_cache_identity.get("checkpoint_id")
    if actual != checkpoint_id:
        raise SealedArtefactRefused(
            f"--checkpoint-id {checkpoint_id!r} does not match the artefact's own "
            f"eval_cache_identity.checkpoint_id {actual!r}"
        )
    if checkpoint_id in RETIRED_CHECKPOINT_IDS:
        raise SealedArtefactRefused(
            f"checkpoint_id {checkpoint_id!r} is retired (DEC-038: S2 is retired, "
            "no S2 runs or S3-vs-S2 comparisons ever again)"
        )


def refuse_unless_vocabulary_matches(
    vocabulary: cv.CompositeVocabulary, artefact: sealed.SealedCohortArtefact
) -> None:
    """U19: the run's OWN rebuilt vocabulary must hash-equal the artefact's
    `eval_cache_identity.composite_vocabulary_hash` -- never trust the
    artefact's claimed hash without an independent rebuild."""
    expected = artefact.eval_cache_identity.get("composite_vocabulary_hash")
    if vocabulary.artifact_hash != expected:
        raise SealedArtefactRefused(
            f"rebuilt composite_vocabulary_hash={vocabulary.artifact_hash!r} does not "
            f"match the artefact's eval_cache_identity.composite_vocabulary_hash={expected!r}"
        )


def build_vocabulary_and_tokenizer(
    artefact: sealed.SealedCohortArtefact,
) -> tuple[cv.CompositeVocabulary, Any]:
    """The real construction path (design §2.6 U19): the DEC-007 D3-pinned
    tokenizer and the run's own allowlist (`qwen_text_allowlist.
    build_qwen_text_binding`, the exact builder `build_exact_text_cache.
    _real_binding` uses), refused unless it hash-matches the artefact."""
    from unflatten.model import build_exact_text_cache as bec
    from unflatten.model import qwen_text_allowlist as qtal
    from unflatten.model import qwen_tokenizer_artifacts as qta

    tokenizer, snapshot_dir = qta.load_pinned_tokenizer()
    artifact_set_hash = qta.tokenizer_artifact_set_hash(
        snapshot_dir, model_id=qta.QWEN_CANDIDATE_MODEL_ID, revision=qta.QWEN_CANDIDATE_REVISION
    )
    width = bec.candidate_native_vocab_size()
    bec.check_native_vocab_size(width, tokenizer_len=len(tokenizer))
    binding = qtal.build_qwen_text_binding(
        tokenizer=tokenizer, tokenizer_artifact_set_hash=artifact_set_hash, native_vocab_size=width
    )
    vocabulary = cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=artifact_set_hash, allowlist_payload=binding.allowlist
    )
    refuse_unless_vocabulary_matches(vocabulary, artefact)
    return vocabulary, tokenizer


def page_hypothesis(
    page: sealed.PageDecodeResult,
    *,
    vocabulary: cv.CompositeVocabulary,
    tokenizer: Any,
    page_input: hy.PageInput = PAGE_INPUT,
) -> hy.PageHypothesis:
    """The same anchor rule as `hypothesis.from_model_hypothesis`
    (`hypothesis.py:175-177`): a typed prediction attaches to its node's
    `open_event_index` when the node is `opened_node`, else its
    `leaf_event_index`. Reimplemented rather than called so this module
    never imports `prediction_ir` (module docstring)."""
    if page.outcome != "terminated" or page.accepted_tokens is None or page.node_states is None:
        raise ValueError(
            f"{page.document_bundle_id}: only a 'terminated' page carries a page "
            f"hypothesis (outcome={page.outcome!r})"
        )
    node_by_id = {state.local_node_id: state for state in page.node_states}
    typed_by_anchor: dict[int, dict[str, Any]] = {}
    for prediction in page.typed_predictions:
        local_node_id = prediction["local_node_id"]
        node_type = prediction["node_type"]
        pointer = prediction["pointer"]
        state = node_by_id.get(local_node_id)
        if state is None:
            raise hy.PageStreamInvalid(
                f"typed prediction names unknown decode node {local_node_id!r}"
            )
        if state.node_type != node_type:
            raise hy.PageStreamInvalid(
                f"typed prediction for {node_type!r} names "
                f"{state.local_node_id} of type {state.node_type!r}"
            )
        position = (
            state.open_event_index if state.slot_kind == "opened_node" else state.leaf_event_index
        )
        slot = typed_by_anchor.setdefault(position, {})
        if pointer in slot:
            raise hy.PageStreamInvalid(
                f"{state.local_node_id}: two typed predictions for {pointer}"
            )
        slot[pointer] = prediction["value"]
    return hy.from_token_ids(
        page.accepted_tokens,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        typed_by_anchor=typed_by_anchor,
        page_input=page_input,
    )


def parser_local_ids_by_model_id(
    page: sealed.PageDecodeResult, *, vocabulary: cv.CompositeVocabulary
) -> dict[str, str]:
    """Map sealed decoder ids to parser ids by structural-token anchors.

    The two lifecycle trackers own distinct id namespaces.  Their shared,
    sealed fact is the open/leaf token position, never tree-list position or
    a coincidentally equal ``n<k>`` spelling.
    """
    if page.accepted_tokens is None or page.node_states is None:
        raise ValueError("only a carried terminated page has structural anchors")
    from unflatten.model import events_vocab
    from unflatten.model.node_lifecycle import NodeLifecycleTracker

    token_name_by_id = {token_id: name for name, token_id in events_vocab.TOKEN_IDS.items()}
    node_tokens = frozenset(events_vocab.OPEN_TOKEN.values()) | frozenset(
        events_vocab.LEAF_TOKEN.values()
    )
    structural_tokens = node_tokens | frozenset(events_vocab.CLOSE_TOKEN.values())
    parser = NodeLifecycleTracker()
    parser_by_anchor: dict[int, tuple[str, str]] = {}
    for position, token_id in enumerate(page.accepted_tokens):
        ref = vocabulary.unpack(int(token_id))
        if not isinstance(ref, cv.StructuralRef):
            continue
        name = token_name_by_id.get(ref.local_id)
        if name not in structural_tokens:
            continue
        state = parser.apply_structural_token(name, event_index=position)
        if name in node_tokens and state is not None:
            parser_by_anchor[position] = (state.local_node_id, state.node_type)
    result: dict[str, str] = {}
    for state in page.node_states:
        anchor = (
            state.open_event_index
            if state.slot_kind == "opened_node"
            else state.leaf_event_index
        )
        parser_state = parser_by_anchor.get(anchor)
        if parser_state is None or parser_state[1] != state.node_type:
            raise ValueError(f"{state.local_node_id}: sealed node has no matching parser anchor")
        result[state.local_node_id] = parser_state[0]
    return result
