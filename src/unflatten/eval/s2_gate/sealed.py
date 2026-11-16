"""The phase A -> phase B contract (RFC005 §36.1 M4).

**Deliberately torch-free.** Phase B (`score.py`) is CPU-only and must stay
loadable on a machine with no GPU stack at all; phase A (`decode.py`) is
GPU-only. If the shared types lived in either phase's own module, the
other phase would import torch (or the frozen-candidate machinery)
transitively just to read an enum. Nothing in this file imports `torch`,
`unflatten.model.prediction_ir` or `unflatten.model.node_lifecycle` --
`node_lifecycle.NodeState.anchor: torch.Tensor` means importing so much as
one constant from either of those two pulls torch in.

**The prediction-IR hash domain and projection shape are duplicated here,
not imported**, for exactly that reason. `test_sealed.py` pins that the
duplicate agrees with `prediction_ir`'s own domain string, schema version
and projection shape by building a REAL `RawDocumentHypothesis`, sealing
it through `decode.seal_page` (which uses the authoritative
`prediction_ir` functions), and checking that
`recompute_prediction_ir_hash` over the resulting `PageDecodeResult`
reproduces the exact same hash -- so a divergence between the two shapes
is caught by that test, not discovered silently downstream.

**Why `PageDecodeResult` must carry the hypothesis, not just its hash.**
RFC005 §36.1 M4's whole point is that phase B recomputes `prediction_ir_hash`
independently and refuses on mismatch -- a seal whose subject is not
carried alongside it cannot be verified by anyone, which defeats the
seal. So a `"terminated"` page's record carries `accepted_tokens` and the
node-state projection `prediction_ir_hash` was computed over. Carrying its
own hypothesis is not a reference-text leak: it is phase A's own output,
never anything read from the reference (RFC005 §36.1's separation phase A
enforces is about INPUT, not about what phase A is allowed to emit).

`decode.py` imports from here and PRODUCES a `SealedCohortArtefact`.
`score.py` imports from here and CONSUMES one.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from unflatten.model.hashing import domain_hash

#: MUST equal `unflatten.model.prediction_ir._PREDICTION_IR_DOMAIN`
#: (duplicated, not imported -- module docstring). Pinned equal by
#: `test_sealed.py::test_recompute_prediction_ir_hash_matches_the_real_seal`.
_PREDICTION_IR_DOMAIN = "rfc004.model-prediction-ir/v0"

#: MUST equal `unflatten.model.prediction_ir.MODEL_PREDICTION_IR_SCHEMA_VERSION`.
#: A page sealed under an older IR schema version recomputes to a DIFFERENT
#: hash than the one stored on it at seal time (the version string is part of
#: the hashed payload) -- expected, not a regression: `prediction_ir_hash` is
#: this version's own seal, and an artefact sealed under a prior version is
#: re-verified only by rebuilding it under the version that sealed it, never
#: by silently accepting a stale schema version here.
_MODEL_PREDICTION_IR_SCHEMA_VERSION = "0.4.0"

#: MUST equal `unflatten.model.prediction_ir.InferenceScope.kind`'s only V0
#: value and `unflatten.model.prediction_ir.ModelPredictionIR.phase`'s only
#: value, respectively.
_SCOPE_KIND: Literal["single_page"] = "single_page"
_ASSEMBLED_PHASE: Literal["assembled"] = "assembled"

#: RFC004 inference-mode literal §36.1 makes mandatory on every prediction
#: manifest. Free-running is the only mode this pipeline produces.
INFERENCE_MODE: Literal["free_running"] = "free_running"


class DecodeArm:
    """The two arms §15.4/DEC-009 dodatok C name. Not a bare string
    constant: a caller passing an arbitrary label could silently make
    phase B's "an arm never compares against itself" refusal pass on a
    typo instead of a real mismatch."""

    TRAINED: Literal["trained"] = "trained"
    FROZEN_INIT_BASELINE: Literal["frozen_init_baseline"] = "frozen_init_baseline"


VALID_ARMS: frozenset[str] = frozenset({DecodeArm.TRAINED, DecodeArm.FROZEN_INIT_BASELINE})

DecodeOutcome = Literal[
    "terminated",
    "max_steps_exceeded",
    "vocabulary_exhausted",
    "unsupported_resume",
    "not_terminated",
    "empty_hypothesis",
]

#: The typed pre-decode errors a page can carry straight from the cohort
#: manifest (design §12 table) -- none of them ever reaches
#: `decode.decode_page`, because none of them has a cached feature entry
#: to decode from. `unplaced`/`no_observation`/`fully_unsupervised`
#: (DEC-033 multi_page_pages slice, 18.9.): the feature build could not
#: supervise the page (`build_cache.prepare_page`'s reason of the same
#: name); `projection_invariant`: the exact-text build refused the page on
#: `page_projection`'s run-bound postcondition; `exact_text_grammar_refused`:
#: the exact-text grammar refused a page that has a feature row (the
#: owner-accepted cell-tail loss, RFC004-ERRATA-002 E1 point 2 -- not
#: `grammar_refused`, which the FEATURE build decides). All five are kept
#: in the denominator as errors, never dropped from it.
PreDecodeDisposition = Literal[
    "capability_excluded",
    "grammar_refused",
    "text_encoding_failed",
    "over_length",
    "unplaced",
    "no_observation",
    "fully_unsupervised",
    "projection_invariant",
    "exact_text_grammar_refused",
]

#: Dispositions added after sealed artefacts and cohort manifests already
#: existed: `CohortAccounting` defaults them to 0 and `as_dict` omits them
#: at 0, so a cohort without such a page serialises byte-identically to
#: before and old payloads still load.
ADDITIVE_DISPOSITIONS: tuple[str, ...] = (
    "unplaced",
    "no_observation",
    "fully_unsupervised",
    "projection_invariant",
    "exact_text_grammar_refused",
)

ATTEMPT_DISPOSITION: Literal["attempt"] = "attempt"

#: A page this ratified exclusion withdraws before it is even attempted
#: (DEC-009 dodatok D, the sole ratified exclusion). Distinct from the
#: `PreDecodeDisposition` values: those stay in the `attempted`
#: denominator as typed errors; this one does not.
WITHDRAWN_DISPOSITION: Literal["withdrawn_text_coverage_floor"] = "withdrawn_text_coverage_floor"

ALL_DISPOSITIONS: frozenset[str] = frozenset(
    {
        ATTEMPT_DISPOSITION,
        WITHDRAWN_DISPOSITION,
        "capability_excluded",
        "grammar_refused",
        "text_encoding_failed",
        "over_length",
        *ADDITIVE_DISPOSITIONS,
    }
)

PRE_DECODE_ERROR_DISPOSITIONS: frozenset[str] = frozenset(
    {
        "capability_excluded",
        "grammar_refused",
        "text_encoding_failed",
        "over_length",
        *ADDITIVE_DISPOSITIONS,
    }
)

#: Dispositions whose page must have NO row in the candidate feature cache:
#: `capability_excluded`/`grammar_refused`/`over_length` are decided by the
#: FEATURE-cache build itself (task 0's build summary: 218 + 26 + 1 = 245
#: of 449 never got a written row).
NO_FEATURES_DISPOSITIONS: frozenset[str] = frozenset(
    {
        "capability_excluded",
        "grammar_refused",
        "over_length",
        "unplaced",
        "no_observation",
        "fully_unsupervised",
    }
)

#: The complementary set: `attempt`/`withdrawn_text_coverage_floor`/
#: `text_encoding_failed` pages DO have a feature-cache row (204 of 449 --
#: `text_encoding_failed` and the ratified floor are decided by the
#: EXACT-TEXT cache build, which reads reference text the feature cache
#: never touches, so those 9 + 12 pages still have visual features even
#: though phase A does not attempt to decode them).
HAS_FEATURES_DISPOSITIONS: frozenset[str] = frozenset(
    {
        ATTEMPT_DISPOSITION,
        WITHDRAWN_DISPOSITION,
        "text_encoding_failed",
        "projection_invariant",
        "exact_text_grammar_refused",
    }
)


@dataclass(frozen=True, slots=True)
class NodeStateProjection:
    """The JSON-serializable projection of one `node_lifecycle.NodeState`
    (mirrors `prediction_ir._node_state_projection` exactly -- the runtime
    tensor fields `anchor`/`content_summary` are excluded there too, so
    there is nothing torch-shaped to carry here in the first place)."""

    local_node_id: str
    slot_kind: str
    node_type: str
    parent_local_node_id: str | None
    open_event_index: int | None = None
    leaf_event_index: int | None = None
    resume_event_index: int | None = None
    close_event_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_node_id": self.local_node_id,
            "slot_kind": self.slot_kind,
            "node_type": self.node_type,
            "parent_local_node_id": self.parent_local_node_id,
            "open_event_index": self.open_event_index,
            "leaf_event_index": self.leaf_event_index,
            "resume_event_index": self.resume_event_index,
            "close_event_index": self.close_event_index,
        }


@dataclass(frozen=True, slots=True)
class SealedRegion:
    """One V0 region after the torch-free sealed-reader validation."""

    page_ordinal: int
    bbox_ppm: tuple[int, int, int, int]
    continuation: Literal["complete"]
    confidence_ppm: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_ordinal": self.page_ordinal,
            "bbox_ppm": list(self.bbox_ppm),
            "continuation": self.continuation,
            "confidence_ppm": self.confidence_ppm,
        }


@dataclass(frozen=True, slots=True)
class SealedRegionPrediction:
    """Closed V0 alignment record consumed by the builder chain."""

    local_node_id: str
    node_type: str
    alignment_status: Literal[
        "region", "abstain_not_visible", "abstain_container", "abstain_degenerate"
    ]
    regions: tuple[SealedRegion, ...]
    confidence_ppm: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "local_node_id": self.local_node_id,
            "node_type": self.node_type,
            "alignment_status": self.alignment_status,
            "regions": [region.as_dict() for region in self.regions],
            "confidence_ppm": self.confidence_ppm,
        }


@dataclass(frozen=True, slots=True)
class UnterminatedPrefix:
    """Diagnostic evidence for a `max_steps_exceeded` (or, should it ever
    fire, `vocabulary_exhausted`) page: the tokens/node-state projections
    `free_running_decode.DecodeMaxStepsExceeded`/`DecodeVocabularyExhausted`
    had accumulated before the cap (bug `max-steps-discards-the-evidence`).

    **Never a hypothesis.** No `prediction_ir_hash` is ever computed over
    this, `score.py` never reads this field (a non-`"terminated"` page is
    scored with an empty-string hypothesis regardless of whether this is
    populated), and the field name is deliberately not `accepted_tokens`/
    `node_states` -- those two names are reserved, on this same dataclass's
    sibling fields, for a `"terminated"` page's SEALED hypothesis. This is
    evidence about why a page failed, carried so a human can inspect it
    (a loop? an unclosed node? endless text?), never a partial result a
    scorer could mistake for one."""

    accepted_tokens: tuple[int, ...]
    node_states: tuple[NodeStateProjection, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_tokens": list(self.accepted_tokens),
            "node_states": [ns.as_dict() for ns in self.node_states],
        }


@dataclass(frozen=True, slots=True)
class CohortAccounting:
    """M6's denominator block: every count a reported rate must carry
    beside it (design §6 "every reported value carries its denominator")."""

    eligible: int
    withdrawn_text_coverage_floor: int
    attempted: int
    decoded: int
    capability_excluded: int
    grammar_refused: int
    text_encoding_failed: int
    over_length: int
    #: `ADDITIVE_DISPOSITIONS`: default 0, omitted from `as_dict` at 0.
    unplaced: int = 0
    no_observation: int = 0
    fully_unsupervised: int = 0
    projection_invariant: int = 0
    exact_text_grammar_refused: int = 0

    def as_dict(self) -> dict[str, int]:
        out = dataclasses.asdict(self)
        for key in ADDITIVE_DISPOSITIONS:
            if not out[key]:
                del out[key]
        return out


@dataclass(frozen=True, slots=True)
class PageDecodeResult:
    """One page's phase-A record. Every eligible page in the cohort
    manifest gets exactly one of these -- an `"attempt"` page after a real
    (or faked, in tests) decode, every other disposition straight from the
    manifest with no decode attempted at all (design §12 table, governance
    M6).

    For a `"terminated"` outcome, `accepted_tokens`/`node_states`/
    `typed_predictions`/`resource_predictions`/`issues` carry EXACTLY what
    `prediction_ir_hash` was computed over (M4 follow-up) -- phase B
    recomputes the hash from these fields via `recompute_prediction_ir_hash`
    and refuses on mismatch. `None` for every other outcome: there is no
    sealed hypothesis to carry.

    `decode_fn_identity`/`max_steps` (code review follow-up, "carry gate
    13's decode_fn identity through to the artefact"): gate 13's own
    wiring evidence (`s1_runner.gate_13_free_running_no_silent_truncation`'s
    `evidence["decode_fn"]`/`["max_steps"]`) -- which callee actually ran
    the decode for THIS page, for every attempted outcome, not only
    `"terminated"`.

    `decode_fn_identity` alone names the WRAPPER (`s1_runner_inputs.
    HypothesisCapture`), which is identical for a real `decode_free_running`
    run and for a test's fake decode function -- true of both, so it
    cannot tell them apart (coordinator follow-up, second round: the same
    label-not-evidence defect as the retired `no_grammar_bypass = True`
    literal). `decode_fn_inner_identity` is the field that actually
    distinguishes them: what `HypothesisCapture` wraps
    (`self._decode_fn`), `None` only if the callee is not a wrapper of
    that shape at all. `decode_free_running` has no bypass mode anywhere
    in its module -- no flag, no parameter -- so naming IT (not its
    wrapper) is the whole proof that no unconstrained path was
    substituted for the constrained one; a reader derives
    `no_grammar_bypass` from `decode_fn_inner_identity` rather than
    reading an asserted boolean (memory `literal-true-when-written`)."""

    document_bundle_id: str
    outcome: str
    prediction_ir_hash: str | None = None
    inference_mode: str | None = None
    input_parity_proof_hash: str | None = None
    scope: str | None = None
    detail: str | None = None
    accepted_tokens: tuple[int, ...] | None = None
    node_states: tuple[NodeStateProjection, ...] | None = None
    typed_predictions: tuple[Any, ...] = ()
    region_predictions: tuple[Any, ...] = ()
    resource_predictions: tuple[Any, ...] = ()
    issues: tuple[Any, ...] = ()
    decode_fn_identity: str | None = None
    max_steps: int | None = None
    decode_fn_inner_identity: str | None = None
    #: `UnterminatedPrefix` for a `"max_steps_exceeded"`/`"vocabulary_
    #: exhausted"` page whose decode captured one (module docstring's
    #: `UnterminatedPrefix`) -- diagnostic evidence only, `None` for every
    #: other outcome including `"terminated"` (whose hypothesis lives in
    #: `accepted_tokens`/`node_states` above, not here).
    unterminated_prefix: UnterminatedPrefix | None = None


@dataclass(frozen=True, slots=True)
class SealedCohortArtefact:
    """Phase A's output: the eval-cache identity and the arm that produced
    it (so phase B can never compare an arm's hypotheses against
    themselves), the cohort's own accounting, and one `PageDecodeResult`
    per eligible page. `arm` lives once here, not per page."""

    role: str
    arm: str
    eval_cache_identity: Mapping[str, Any]
    cohort_manifest_sha256: str
    accounting: CohortAccounting
    pages: tuple[PageDecodeResult, ...]


def _fragment_projection(page: PageDecodeResult) -> dict[str, Any]:
    """Mirrors `prediction_ir._fragment_projection` exactly, reconstructed
    from `page`'s carried fields rather than a live `FragmentHypothesis`
    (module docstring: no torch/`prediction_ir` import here)."""
    if page.accepted_tokens is None or page.node_states is None:
        raise ValueError(
            f"{page.document_bundle_id}: cannot reconstruct a fragment projection -- "
            "accepted_tokens/node_states are not carried (only a 'terminated' page "
            "carries them)"
        )
    return {
        "scope": {"kind": _SCOPE_KIND},
        "accepted_tokens": list(page.accepted_tokens),
        "node_states": [ns.as_dict() for ns in page.node_states],
        "typed_predictions": [dict(t) for t in page.typed_predictions],
        "region_predictions": [_sealed_region_prediction_dict(r) for r in page.region_predictions],
        "resource_predictions": list(page.resource_predictions),
        "issues": list(page.issues),
    }


def recompute_prediction_ir_hash(page: PageDecodeResult) -> str:
    """Phase B's own recomputation of `page.prediction_ir_hash` from the
    fields `page` carries -- the seal's actual verification step (RFC005
    §36.1 M4). Raises `ValueError` if `page` carries nothing to recompute
    from (any outcome other than `"terminated"`)."""
    payload = {
        "model_prediction_ir_schema_version": _MODEL_PREDICTION_IR_SCHEMA_VERSION,
        "phase": _ASSEMBLED_PHASE,
        "document_hypotheses": [{"fragments": [_fragment_projection(page)]}],
    }
    return domain_hash(_PREDICTION_IR_DOMAIN, payload)


# ---------------------------------------------------------------------------
# JSON serialisation both phases use
# ---------------------------------------------------------------------------


def page_to_dict(page: PageDecodeResult) -> dict[str, Any]:
    payload = dataclasses.asdict(page)
    payload["region_predictions"] = [
        _sealed_region_prediction_dict(region) for region in page.region_predictions
    ]
    return payload


def page_from_dict(payload: Mapping[str, Any]) -> PageDecodeResult:
    # A page with no `region_predictions` key was sealed before IR 0.3.0,
    # by a model with no aligner: `page_to_dict` has written the key on
    # every page since. It loads as `()`, exactly like its siblings
    # `typed_predictions` and `resource_predictions` already do for pages
    # that predate THEM. Refusing it made every existing artefact
    # unreadable (the 100-page S3 artefact was the one that showed it).
    node_states_raw = payload.get("node_states")
    node_states = (
        tuple(NodeStateProjection(**ns) for ns in node_states_raw)
        if node_states_raw is not None
        else None
    )
    accepted_tokens_raw = payload.get("accepted_tokens")
    accepted_tokens = tuple(accepted_tokens_raw) if accepted_tokens_raw is not None else None
    unterminated_prefix_raw = payload.get("unterminated_prefix")
    unterminated_prefix = (
        UnterminatedPrefix(
            accepted_tokens=tuple(unterminated_prefix_raw["accepted_tokens"]),
            node_states=tuple(
                NodeStateProjection(**ns) for ns in unterminated_prefix_raw["node_states"]
            ),
        )
        if unterminated_prefix_raw is not None
        else None
    )
    return PageDecodeResult(
        document_bundle_id=payload["document_bundle_id"],
        outcome=payload["outcome"],
        prediction_ir_hash=payload.get("prediction_ir_hash"),
        inference_mode=payload.get("inference_mode"),
        input_parity_proof_hash=payload.get("input_parity_proof_hash"),
        scope=payload.get("scope"),
        detail=payload.get("detail"),
        accepted_tokens=accepted_tokens,
        node_states=node_states,
        typed_predictions=tuple(payload.get("typed_predictions") or ()),
        region_predictions=_sealed_region_predictions(payload.get("region_predictions") or []),
        resource_predictions=tuple(payload.get("resource_predictions") or ()),
        issues=tuple(payload.get("issues") or ()),
        decode_fn_identity=payload.get("decode_fn_identity"),
        max_steps=payload.get("max_steps"),
        decode_fn_inner_identity=payload.get("decode_fn_inner_identity"),
        unterminated_prefix=unterminated_prefix,
    )


_REGION_PREDICTION_KEYS = frozenset(
    {"local_node_id", "node_type", "alignment_status", "regions", "confidence_ppm"}
)
_REGION_KEYS = frozenset({"page_ordinal", "bbox_ppm", "continuation", "confidence_ppm"})
_REGION_STATUSES = frozenset(
    {"region", "abstain_not_visible", "abstain_container", "abstain_degenerate"}
)


def _sealed_region_predictions(raw: Any) -> tuple[SealedRegionPrediction, ...]:
    if not isinstance(raw, list):
        raise ValueError("sealed region_predictions must be an array")
    return tuple(_sealed_region_prediction(item) for item in raw)


def _sealed_region_prediction(raw: Any) -> SealedRegionPrediction:
    if not isinstance(raw, Mapping) or set(raw) != _REGION_PREDICTION_KEYS:
        raise ValueError("sealed region prediction must have the exact V0 key set")
    local_node_id, node_type, status = (
        raw["local_node_id"], raw["node_type"], raw["alignment_status"]
    )
    if not isinstance(local_node_id, str) or not local_node_id:
        raise ValueError("sealed region prediction local_node_id must be a non-empty string")
    if not isinstance(node_type, str) or not node_type:
        raise ValueError("sealed region prediction node_type must be a non-empty string")
    if status not in _REGION_STATUSES:
        raise ValueError("sealed region prediction has an invalid alignment_status")
    confidence = _ppm(raw["confidence_ppm"], "confidence_ppm")
    regions_raw = raw["regions"]
    if not isinstance(regions_raw, list):
        raise ValueError("sealed region prediction regions must be an array")
    if len(regions_raw) != (1 if status == "region" else 0):
        raise ValueError("sealed region prediction status and region cardinality disagree")
    return SealedRegionPrediction(
        local_node_id,
        node_type,
        status,
        tuple(_sealed_region(item) for item in regions_raw),
        confidence,
    )


def _sealed_region(raw: Any) -> SealedRegion:
    if not isinstance(raw, Mapping) or set(raw) != _REGION_KEYS:
        raise ValueError("sealed region must have the exact V0 key set")
    page_ordinal = raw["page_ordinal"]
    if isinstance(page_ordinal, bool) or page_ordinal != 0:
        raise ValueError("sealed V0 region page_ordinal must be exactly 0")
    bbox_raw = raw["bbox_ppm"]
    if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
        raise ValueError("sealed region bbox_ppm must be a four-item array")
    bbox = tuple(_ppm(value, "bbox_ppm") for value in bbox_raw)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("sealed region bbox_ppm must have positive half-open area")
    if raw["continuation"] != "complete":
        raise ValueError("sealed V0 region continuation must be complete")
    confidence = _ppm(raw["confidence_ppm"], "confidence_ppm")
    return SealedRegion(page_ordinal, bbox, "complete", confidence)


def _ppm(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000:
        raise ValueError(f"sealed region {name} must be an integer within 0..1000000")
    return value


def _sealed_region_prediction_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, SealedRegionPrediction):
        raise ValueError("sealed page has unvalidated region prediction")
    return raw.as_dict()


def artefact_to_dict(artefact: SealedCohortArtefact) -> dict[str, Any]:
    return {
        "role": artefact.role,
        "arm": artefact.arm,
        "eval_cache_identity": dict(artefact.eval_cache_identity),
        "cohort_manifest_sha256": artefact.cohort_manifest_sha256,
        "accounting": artefact.accounting.as_dict(),
        "pages": [page_to_dict(p) for p in artefact.pages],
    }


def artefact_from_dict(payload: Mapping[str, Any]) -> SealedCohortArtefact:
    return SealedCohortArtefact(
        role=payload["role"],
        arm=payload["arm"],
        eval_cache_identity=dict(payload["eval_cache_identity"]),
        cohort_manifest_sha256=payload["cohort_manifest_sha256"],
        accounting=CohortAccounting(**payload["accounting"]),
        pages=tuple(page_from_dict(p) for p in payload["pages"]),
    )


def artefact_to_json(artefact: SealedCohortArtefact) -> str:
    return json.dumps(artefact_to_dict(artefact), sort_keys=True)


def artefact_from_json(raw: str) -> SealedCohortArtefact:
    return artefact_from_dict(json.loads(raw))
