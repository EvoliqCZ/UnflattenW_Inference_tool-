"""CPU-only checks for page tensor ownership at the native engine boundary."""

from __future__ import annotations

import weakref

import pytest
import torch

from unflatten.eval.s2_gate.decode import seal_page
from unflatten.model.candidate_runtime import PreparedCandidateInput
from unflatten.model.node_lifecycle import NodeState
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.service.engines.native import NativeEngine
from unflatten.service.inference_engine import PageInput
from unflatten.service.model_runtime import DiagnosticModelRuntime


def _seal_page(raw, prepared):
    return seal_page(
        document_bundle_id="job#page",
        hypothesis=raw,
        eval_cache_identity={"model": "fixed-test-model"},
        visual_features=prepared.visual_features,
        component_lock_hash="sha256:" + "a" * 64,
        selection_policy_hash="sha256:" + "b" * 64,
        decode_fn_identity="decode_free_running",
        max_steps=100,
    )


class _TensorCandidate:
    def __init__(self, fail_stage: str) -> None:
        self.fail_stage = fail_stage
        self.refs: list[weakref.ReferenceType] = []
        self.sealed: list[object] = []
        self.page_seals: list[object] = []
        self.pages = 0

    def _tensor(self, values, dtype):
        tensor = torch.tensor(values, dtype=dtype, requires_grad=dtype.is_floating_point)
        self.refs.append(weakref.ref(tensor))
        return tensor

    def extract(self, _input):
        assert all(ref() is None for ref in self.refs), "previous page tensors retained"
        self.pages += 1
        prepared = PreparedCandidateInput(
            self._tensor([[1.25, -2.5]], torch.bfloat16),
            self._tensor([[1, 2, 3]], torch.int64),
            (self._tensor([[3.5]], torch.float32),),
        )
        if self.fail_stage == "extract" and self.pages == 1:
            raise RuntimeError("synthetic page extraction failure")
        return prepared

    def decode_features(self, prepared):
        if self.fail_stage == "decode" and self.pages == 1:
            raise RuntimeError("synthetic page decode failure")
        raw = RawDocumentHypothesis(
            accepted_tokens=(1, 2, 3),
            node_states=(NodeState(
                "n0", "opened_node", "Document", None,
                open_event_index=0, close_event_index=2,
                anchor=self._tensor([4.5], torch.float32),
                content_summary=self._tensor([-5.5], torch.bfloat16),
            ),),
            terminated=True,
        )
        self.sealed.append(DiagnosticModelRuntime.seal(raw))
        self.page_seals.append(_seal_page(raw, prepared))
        return raw


class _Progress:
    def __init__(self, candidate):
        self.candidate = candidate

    def enter_stage(self, _stage):
        pass

    def page_done(self, _result):
        assert all(ref() is None for ref in self.candidate.refs), "page tensors retained"


@pytest.mark.parametrize("fail_stage", ["none", "extract", "decode"])
def test_decode_pages_completed_or_failed_page_releases_original_tensors(fail_stage, monkeypatch):
    candidate = _TensorCandidate(fail_stage)
    transfers = []
    original_to = torch.Tensor.to

    def record_to(tensor, *args, **kwargs):
        transfers.append(args)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_to)
    engine = NativeEngine.from_loaded(DiagnosticModelRuntime(candidate))
    results = engine.decode_pages(
        [PageInput(("job", index), None) for index in range(2)],
        progress=_Progress(candidate),
    )
    assert [result.error_code for result in results] == (
        ["INTERNAL_ERROR", None] if fail_stage != "none" else [None, None]
    )
    # Five retained tensors plus the independent seal's feature hash per success.
    assert transfers.count(("cpu",)) == (12 if fail_stage == "none" else 6)
    for result, sealed, page_seal in zip(
        (r for r in results if r.raw is not None), candidate.sealed, candidate.page_seals,
    ):
        assert engine.seal(result.raw) == sealed
        assert _seal_page(result.raw, result.prepared) == page_seal
        tensors = (
            result.prepared.visual_features,
            result.prepared.image_grid_thw,
            *result.prepared.deepstack_visual_embeds,
            result.raw.node_states[0].anchor,
            result.raw.node_states[0].content_summary,
        )
        assert [tensor.tolist() for tensor in tensors] == [
            [[1.25, -2.5]], [[1, 2, 3]], [[3.5]], [4.5], [-5.5],
        ]
        assert [tensor.dtype for tensor in tensors] == [
            torch.bfloat16, torch.int64, torch.float32, torch.float32, torch.bfloat16,
        ]
        assert all(tensor.device.type == "cpu" and not tensor.requires_grad for tensor in tensors)
