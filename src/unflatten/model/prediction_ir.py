"""Prediction IR -- design §5.8 D8, §8 M3 exit criterion.

The model never returns an AST directly (design §5.8): `decode_free_running`
(`free_running_decode.py`) produces a `RawDocumentHypothesis`; THIS module
seals it into a `FragmentHypothesis` and assembles fragments into a
`ModelPredictionIR` at `phase = "assembled"`. Design §5.8 is explicit that
single-page is not a bypass: "Aj jediný úspešne ukončený scope sa sealed do
jedného FragmentHypothesis" -- so even the V0 single-fragment case goes
through the same seal/assemble steps a future multi-fragment caller would
use, rather than a single-page special case that skips them.

`AstPromoter.promote` (design §5.8's next stage, consuming a
`ModelPredictionIR`) is OUT OF SCOPE here -- M3's exit criterion is a
schema-valid Prediction IR, explicitly not `strict_valid` (required typed
properties are still unresolved; typed heads are M4's job).

Runtime tensor fields on `NodeState` (`anchor`, `content_summary`) are
deliberately excluded from the JSON projection/schema: they are decode-time
hidden states, not part of the logical document-structure contract the AST
Builder consumes downstream (see the schema file's own `$comment`).

This module authorizes no gradient update: it declares dataclasses and pure
functions only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash
from unflatten.model.node_lifecycle import NodeState
from unflatten.model.region_codec import Region

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_PREDICTION_IR_DOMAIN = "rfc004.model-prediction-ir/v0"

MODEL_PREDICTION_IR_SCHEMA_VERSION = "0.4.0"  # 0.3.0: closed region_predictions records (T4);
# 0.4.0: typed_predictions[].value widened to allow JSON null (RFC004-ERRATA-003 E1: the
# palette "no colour" class and a null borders/page_borders/sides container both decode to
# None) -- the same kind of shape widening 0.1.0 -> 0.2.0 closed typed_predictions with.

#: V0 only knows one scope kind (design §5.6/§9: multi-page/bounded windows
#: are deferred). Not an enum of one by oversight -- widening this is a
#: later milestone's job, not a silent default.
ScopeKind = Literal["single_page"]


class PredictionIrInvalid(ValueError):
    """A Prediction IR payload fails its closed schema, or a structural
    invariant this module enforces beyond what the schema can express
    (design §5.8; M1-internal, not an RFC004 registry code)."""


@dataclass(frozen=True, slots=True)
class InferenceScope:
    kind: ScopeKind = "single_page"


@dataclass(frozen=True, slots=True)
class RawDocumentHypothesis:
    """design §5.8's dataclass, PLUS one field the design's illustrative
    snippet does not carry: `terminated`. `typed_predictions` holds
    `typed_decode.TypedPrediction`s when `DecodeConfig.typed_decode` is on
    (AST Builder T1) and is empty otherwise; `resource_predictions` is always
    empty -- the resource branch is not built yet; `issues` carries typed
    decode-time findings that are not themselves fatal (M3 raises for fatal
    ones instead of returning a partial hypothesis, design §5.7).

    **`terminated`, added 2026-09-05, review item 3 (not a silent
    extension -- named here because it changes the design's "verbatim
    field set").** Before this field, `seal_fragment` (below) could only
    check `node_states` non-empty, and `s1_runner.gate_13_free_running_no_silent_truncation`
    could only check `accepted_tokens` non-empty -- neither is "reached a
    valid mode terminator". `decode_free_running`'s control flow already
    guarantees it (raises `DecodeMaxStepsExceeded`/`DecodeVocabularyExhausted`
    rather than ever constructing a hypothesis for a truncated stream), but
    that guarantee lived ENTIRELY in the decode loop, not as data on the
    object itself -- a caller substituting a different `decode_fn` (gate
    13's own `decode_fn` parameter exists exactly to allow this, for its
    mutation proof) that swallows the typed failure and returns a non-empty
    but genuinely incomplete hypothesis was indistinguishable from a real
    terminator to anything reading only `accepted_tokens`/`node_states`.

    Defaults to `False` -- fail-closed (D8: insufficient evidence is
    `blocked`/`fail`, never a silent pass). `decode_free_running` sets it
    `True` explicitly, and only on its one success path (`state.mode ==
    "terminated"`); every other caller that constructs a
    `RawDocumentHypothesis` by hand (a test double, a truncating wrapper) gets
    `False` unless it deliberately claims otherwise, which is the honest
    default for something that did not go through real free-running decode.
    """

    accepted_tokens: tuple[int, ...]
    node_states: tuple[NodeState, ...]
    typed_predictions: tuple[Any, ...] = ()
    region_predictions: tuple[Any, ...] = ()
    resource_predictions: tuple[Any, ...] = ()
    issues: tuple[Any, ...] = ()
    terminated: bool = False


@dataclass(frozen=True, slots=True)
class FragmentHypothesis:
    scope: InferenceScope
    raw: RawDocumentHypothesis


@dataclass(frozen=True, slots=True)
class DocumentHypothesis:
    fragments: tuple[FragmentHypothesis, ...]


@dataclass(frozen=True, slots=True)
class ModelPredictionIR:
    phase: Literal["assembled"]
    document_hypotheses: tuple[DocumentHypothesis, ...]


def seal_fragment(raw: RawDocumentHypothesis, scope: InferenceScope) -> FragmentHypothesis:
    """`RawDocumentHypothesis -> FragmentHypothesis` (design §5.8's
    `PredictionIrAssembler.seal_fragment`). `raw` must already be
    grammar-complete -- `decode_free_running` guarantees this by raising a
    typed failure rather than ever returning an incomplete hypothesis
    (design §5.7), so this function's own checks are a defence against a
    caller that built a `RawDocumentHypothesis` some other way, not a
    duplicate of decode's own completeness proof.

    `raw.terminated` is checked explicitly (2026-09-05, review item 3) --
    before this field existed, only `node_states` non-empty was checked,
    which a truncated-but-nonempty hypothesis would also satisfy."""
    if not raw.terminated:
        raise PredictionIrInvalid(
            "a RawDocumentHypothesis with terminated=False is not sealable -- it did "
            "not reach a valid mode terminator (design §5.7: max_steps/an unresolved "
            "required property are never success)"
        )
    if not raw.node_states:
        raise PredictionIrInvalid("a RawDocumentHypothesis with no node_states is not sealable")
    return FragmentHypothesis(scope=scope, raw=raw)


def assemble(fragments: Sequence[FragmentHypothesis]) -> ModelPredictionIR:
    """`fragments -> ModelPredictionIR` at `phase = "assembled"` (design
    §5.8's `PredictionIrAssembler.assemble`). V0 always calls this with
    exactly one fragment (single-page canary); the shape supports more
    because design §5.8 requires the SAME path stay usable for a later
    multi-fragment mode, not a single-fragment special case."""
    if not fragments:
        raise PredictionIrInvalid("assemble() requires at least one FragmentHypothesis")
    return ModelPredictionIR(
        phase="assembled", document_hypotheses=(DocumentHypothesis(fragments=tuple(fragments)),)
    )


# ---------------------------------------------------------------------------
# JSON projection, schema validation, hashing
# ---------------------------------------------------------------------------


def _node_state_projection(node: NodeState) -> dict[str, Any]:
    return {
        "local_node_id": node.local_node_id,
        "slot_kind": node.slot_kind,
        "node_type": node.node_type,
        "parent_local_node_id": node.parent_local_node_id,
        "open_event_index": node.open_event_index,
        "leaf_event_index": node.leaf_event_index,
        "resume_event_index": node.resume_event_index,
        "close_event_index": node.close_event_index,
    }


def typed_prediction_projection(prediction: Any) -> dict[str, Any]:
    """The hashed record of one typed prediction; the s2 gate seals exactly this."""
    return {
        "local_node_id": prediction.local_node_id,
        "node_type": prediction.node_type,
        "pointer": prediction.pointer,
        "head_id": prediction.head_id,
        "value": prediction.value,
    }


def region_prediction_projection(prediction: Any) -> dict[str, Any]:
    """The closed, tensor-free record of one node visual alignment result."""
    return {
        "local_node_id": prediction.local_node_id,
        "node_type": prediction.node_type,
        "alignment_status": prediction.alignment_status,
        "regions": [
            {
                "page_ordinal": region.page_ordinal,
                "bbox_ppm": list(region.bbox_ppm),
                "continuation": region.continuation,
                "confidence_ppm": region.confidence_ppm,
            }
            for region in prediction.regions
        ],
        "confidence_ppm": prediction.confidence_ppm,
    }


def _fragment_projection(fragment: FragmentHypothesis) -> dict[str, Any]:
    return {
        "scope": {"kind": fragment.scope.kind},
        "accepted_tokens": list(fragment.raw.accepted_tokens),
        "node_states": [_node_state_projection(n) for n in fragment.raw.node_states],
        "typed_predictions": [
            typed_prediction_projection(t) for t in fragment.raw.typed_predictions
        ],
        "region_predictions": [
            region_prediction_projection(r) for r in fragment.raw.region_predictions
        ],
        "resource_predictions": list(fragment.raw.resource_predictions),
        "issues": list(fragment.raw.issues),
    }


def prediction_ir_projection(ir: ModelPredictionIR) -> dict[str, Any]:
    """The JSON-serializable, schema-validatable projection of `ir`."""
    return {
        "model_prediction_ir_schema_version": MODEL_PREDICTION_IR_SCHEMA_VERSION,
        "phase": ir.phase,
        "document_hypotheses": [
            {"fragments": [_fragment_projection(f) for f in doc.fragments]}
            for doc in ir.document_hypotheses
        ],
    }


@cache
def _validator(name: str) -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_model_prediction_ir(payload: Mapping[str, Any]) -> None:
    """Validate `payload` against `model_prediction_ir.schema.json`."""
    errors = list(_validator("model_prediction_ir").iter_errors(payload))
    if errors:
        raise PredictionIrInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )
    for document in payload["document_hypotheses"]:
        for fragment in document["fragments"]:
            for prediction in fragment["region_predictions"]:
                try:
                    for region in prediction["regions"]:
                        Region(**region)
                except ValueError as exc:
                    raise PredictionIrInvalid(f"invalid closed region prediction: {exc}") from exc


def prediction_ir_hash(payload: Mapping[str, Any]) -> str:
    return domain_hash(_PREDICTION_IR_DOMAIN, payload)
