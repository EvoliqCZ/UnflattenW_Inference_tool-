from __future__ import annotations

from dataclasses import dataclass

import pytest

from unflatten.model.node_lifecycle import NodeState
from unflatten.model.prediction_ir import PredictionIrInvalid, RawDocumentHypothesis
from unflatten.service.model_runtime import (
    DiagnosticModelRuntime,
    ForbiddenModelChannel,
    build_single_page_input,
    seal_diagnostic_prediction,
)


@dataclass(frozen=True, slots=True)
class _StubTypedPrediction:
    local_node_id: str = "n0"
    node_type: str = "run"
    pointer: str = "/style/bold"
    head_id: str = "run.style.bold"
    value: bool = True


def _terminated(
    value: bool = True,
    *,
    typed_predictions: tuple = (),
    resource_predictions: tuple = (),
    issues: tuple = (),
) -> RawDocumentHypothesis:
    return RawDocumentHypothesis(
        accepted_tokens=(1, 2),
        node_states=(
            NodeState(
                local_node_id="n0",
                slot_kind="opened_node",
                node_type="document",
                parent_local_node_id=None,
            ),
        ),
        typed_predictions=typed_predictions,
        resource_predictions=resource_predictions,
        issues=issues,
        terminated=value,
    )


def test_builds_explicit_single_page_word_core_300_dpi_input():
    model_input = build_single_page_input(width_px=2, height_px=1, rgb_pixels=b"\0" * 6)

    assert len(model_input.planes) == 1
    assert model_input.input_mode == "single_page"
    assert model_input.profile == "word-core-v0"
    assert model_input.scale.status == "known"
    assert model_input.scale.dpi == 300
    assert model_input.masks[0].bitmap is None
    assert model_input.transforms[0].denom == 1


def test_forbidden_channel_is_refused_before_model_input_construction():
    with pytest.raises(ForbiddenModelChannel, match="hidden_pdf_text"):
        build_single_page_input(
            width_px=1,
            height_px=1,
            rgb_pixels=b"\0" * 3,
            host_channels={"hidden_pdf_text": "must not enter"},
        )


def test_only_terminated_hypothesis_is_sealed_and_validated():
    payload = seal_diagnostic_prediction(_terminated())
    assert payload["phase"] == "assembled"

    with pytest.raises(PredictionIrInvalid, match="terminated"):
        seal_diagnostic_prediction(_terminated(False))


def test_diagnostic_runtime_returns_verified_artifact_only_after_decode():
    class Candidate:
        def decode(self, model_input):
            assert model_input.profile == "word-core-v0"
            return _terminated()

    artifact = DiagnosticModelRuntime(Candidate()).predict(
        build_single_page_input(width_px=1, height_px=1, rgb_pixels=b"\0" * 3)
    )
    assert artifact.raw_bytes.startswith(b"{")


def test_diagnostic_runtime_never_artifacts_nontermination():
    class Candidate:
        def decode(self, model_input):
            return _terminated(False)

    with pytest.raises(PredictionIrInvalid, match="terminated"):
        DiagnosticModelRuntime(Candidate()).predict(
            build_single_page_input(width_px=1, height_px=1, rgb_pixels=b"\0" * 3)
        )


def test_typed_predictions_and_loop_guard_v2_issue_are_sealed():
    """D14: typed predictions are authorized now (DEC-021/§24 supersedes the
    S2-era refusal); a `loop_guard_v2` decode issue rides through unchanged."""
    hypothesis = _terminated(
        typed_predictions=(_StubTypedPrediction(),),
        issues=({"kind": "loop_guard_v2", "loop_guard_counters": {"tab": 0}},),
    )

    payload = seal_diagnostic_prediction(hypothesis)

    fragment = payload["document_hypotheses"][0]["fragments"][0]
    assert fragment["typed_predictions"] == [
        {
            "local_node_id": "n0",
            "node_type": "run",
            "pointer": "/style/bold",
            "head_id": "run.style.bold",
            "value": True,
        }
    ]
    assert fragment["issues"] == [{"kind": "loop_guard_v2", "loop_guard_counters": {"tab": 0}}]


def test_a_null_typed_value_is_sealed_and_schema_valid():
    """Follow-up to T5 (defect found before acceptance): a JSON `null` typed
    value is legitimate model output, not a schema violation --
    (a) the palette "no colour" class decodes to `None` (`typed_decode.
    _palette_value`, RFC004-ERRATA-003 E1), (b) a border object decodes to
    `None` when its `borders_is_null` head fires (`typed_decode.
    BORDER_OBJECT_POINTERS`). Before the IR schema widened `typed_predictions
    [].value` to allow `null`, this raised `PredictionIrInvalid` and any real
    page with a null colour or null borders failed the whole diagnostic job
    with `PREDICTION_IR_INVALID`."""
    from dataclasses import replace

    null_highlight = replace(
        _StubTypedPrediction(),
        pointer="/style/highlight",
        head_id="run.style.highlight",
        value=None,
    )
    null_borders = _StubTypedPrediction(
        local_node_id="p0", node_type="paragraph", pointer="/style/borders",
        head_id="paragraph.style.borders", value=None,
    )
    hypothesis = _terminated(typed_predictions=(null_highlight, null_borders))

    payload = seal_diagnostic_prediction(hypothesis)  # must not raise

    fragment = payload["document_hypotheses"][0]["fragments"][0]
    assert [p["value"] for p in fragment["typed_predictions"]] == [None, None]


def test_resource_predictions_are_refused():
    hypothesis = _terminated(resource_predictions=({"kind": "image_crop"},))

    with pytest.raises(PredictionIrInvalid, match="resource predictions"):
        seal_diagnostic_prediction(hypothesis)


def test_diagnostic_model_runtime_delegates_decode_attrs():
    class Candidate:
        tokenizer = object()
        decode_policy = object()
        eval_cache_identity = {"component_lock_hash": "sha256:" + "1" * 64}
        component_lock_hash = "sha256:" + "1" * 64
        selection_policy_hash = "sha256:" + "2" * 64

    runtime = DiagnosticModelRuntime(Candidate())

    assert runtime.tokenizer is Candidate.tokenizer
    assert runtime.decode_policy is Candidate.decode_policy
    assert runtime.eval_cache_identity == Candidate.eval_cache_identity
    assert runtime.component_lock_hash == Candidate.component_lock_hash
    assert runtime.selection_policy_hash == Candidate.selection_policy_hash
