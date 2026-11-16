"""Contract suite for `InferenceEngine` implementations (design §17.2, T05).

Parametrized over engine factories via `_ENGINE_FACTORIES` so `NativeEngine`
(T11, `engines/native.py`) runs the same suite as `FakeEngine`. Adding
`NativeEngine` pulls in `torch` transitively (`candidate_runtime.py`
imports it directly) -- the T05 "no torch" guarantee held only until a real
engine joined the parametrization; `FakeEngine` itself still needs none.
`_NativeEngineForContract` below wires a fake candidate into `NativeEngine`
so this suite exercises the real `decode_pages`/except-ladder/`seal` without
a model bundle or a GPU -- bundle loading (`verify_bundle` +
`load_candidate_runtime`) is T17's integration surface, not this contract.
"""

from __future__ import annotations

import dataclasses

import pytest
from tests.service.fake_engine import FakeEngine

from unflatten.service.contracts import JOB_STAGES
from unflatten.service.engines.native import NativeEngine
from unflatten.service.inference_engine import (
    EngineCapabilities,
    EngineRegistry,
    ModelIdentity,
    PageInput,
)
from unflatten.service.model_runtime import DiagnosticModelRuntime


@dataclasses.dataclass(frozen=True, slots=True)
class _FakeCandidateRaw:
    accepted_tokens: tuple[int, ...] = (1, 2, 3)


class _FakeCandidate:
    """Today's fake `Candidate` idiom (`test_model_runtime.py`'s `class
    Candidate`), extended with `extract`/`decode_features` and a
    `fail_pages` set keyed by the rendered `model_input` string -- the
    contract suite's `PageInput.model_input` carries no structured key, so
    matching on its exact string is simpler than parsing it back apart."""

    tokenizer = "fake-tokenizer"
    decode_policy = "fake-decode-policy"
    eval_cache_identity = "fake-eval-cache-identity"
    component_lock_hash = "fake-component-lock-hash"
    selection_policy_hash = "fake-selection-policy-hash"
    vocabulary = "fake-vocabulary"

    def __init__(self, fail_model_inputs: frozenset[str] = frozenset()) -> None:
        self._fail_model_inputs = fail_model_inputs

    def extract(self, model_input: str) -> str:
        return model_input

    def decode_features(self, prepared: str) -> _FakeCandidateRaw:
        if prepared in self._fail_model_inputs:
            raise RuntimeError("synthetic decode failure")
        return _FakeCandidateRaw()


class _NativeEngineForContract(NativeEngine):
    """`NativeEngine` with `load`/`unload` swapped for a fake candidate;
    `decode_pages`/`seal`/`device_facts`/`capabilities` are the real ones."""

    def __init__(self, fail_pages: frozenset[tuple[str, int]] = frozenset()) -> None:
        super().__init__()
        self._fail_model_inputs = frozenset(
            f"model-input-{job_id}-{index}" for job_id, index in fail_pages
        )

    def load(self, *_args: object, **_kwargs: object) -> ModelIdentity:
        model = DiagnosticModelRuntime(_FakeCandidate(fail_model_inputs=self._fail_model_inputs))
        self._model = model
        return ModelIdentity(
            tokenizer=model.tokenizer,
            decode_policy=model.decode_policy,
            eval_cache_identity=model.eval_cache_identity,
            component_lock_hash=model.component_lock_hash,
            selection_policy_hash=model.selection_policy_hash,
            vocabulary=model.vocabulary,
        )

    def unload(self) -> None:
        self._model = None


_ENGINE_FACTORIES = {
    "fake": FakeEngine,
    "native": _NativeEngineForContract,
}


class _RecordingProgress:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    def enter_stage(self, stage: str) -> None:
        self.events.append(("enter_stage", stage))

    def page_done(self, result: object) -> None:
        self.events.append(("page_done", result))


@pytest.fixture(params=sorted(_ENGINE_FACTORIES))
def engine_factory(request: pytest.FixtureRequest):
    return _ENGINE_FACTORIES[request.param]


def _page(job_id: str, index: int) -> PageInput:
    return PageInput(key=(job_id, index), model_input=f"model-input-{job_id}-{index}")


def test_load_unload_idempotent(engine_factory):
    engine = engine_factory()
    engine.load()
    engine.load()
    engine.unload()
    engine.unload()


def test_decode_pages_preserves_order_and_echoes_key(engine_factory):
    engine = engine_factory()
    engine.load()
    pages = [_page("job-1", 0), _page("job-1", 1), _page("job-1", 2)]
    results = engine.decode_pages(pages, progress=_RecordingProgress())
    assert [result.key for result in results] == [page.key for page in pages]


def test_page_error_isolated_in_page_result(engine_factory):
    engine = engine_factory(fail_pages=frozenset({("job-1", 1)}))
    engine.load()
    pages = [_page("job-1", 0), _page("job-1", 1), _page("job-1", 2)]
    results = engine.decode_pages(pages, progress=_RecordingProgress())
    assert results[0].error_code is None
    assert results[0].raw is not None
    assert results[1].error_code is not None
    assert results[1].raw is None
    assert results[2].error_code is None
    assert results[2].raw is not None


def test_progress_enter_stage_then_page_done(engine_factory):
    engine = engine_factory()
    engine.load()
    progress = _RecordingProgress()
    engine.decode_pages([_page("job-1", 0)], progress=progress)
    kinds = [kind for kind, _ in progress.events]
    assert kinds[-1] == "page_done"
    assert kinds.count("page_done") == 1
    assert all(kind == "enter_stage" for kind in kinds[:-1])


def test_enter_stage_values_in_job_stages(engine_factory):
    engine = engine_factory()
    engine.load()
    progress = _RecordingProgress()
    engine.decode_pages([_page("job-1", 0)], progress=progress)
    stages = [value for kind, value in progress.events if kind == "enter_stage"]
    assert stages
    assert all(stage in JOB_STAGES for stage in stages)


def test_identity_has_six_docx_stage_attributes(engine_factory):
    engine = engine_factory()
    identity = engine.load()
    for attribute in (
        "tokenizer",
        "decode_policy",
        "eval_cache_identity",
        "component_lock_hash",
        "selection_policy_hash",
        "vocabulary",
    ):
        assert hasattr(identity, attribute)


def test_device_facts_none_without_cuda(engine_factory, monkeypatch):
    # `NativeEngine.device_facts` calls real `torch.cuda.is_available` --
    # forced False here so this test is hermetic on a host that happens to
    # have a GPU (this task runs with none claimed, per the plan's GPU:
    # nie). `FakeEngine` ignores the patch; it never calls torch.
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    engine = engine_factory()
    assert engine.device_facts() is None


def test_capabilities_frozen(engine_factory):
    engine = engine_factory()
    with pytest.raises(dataclasses.FrozenInstanceError):
        engine.capabilities.engine_id = "candidate"


def test_registry_available_excludes_unproven():
    unproven = EngineCapabilities(
        engine_id="candidate", batches_pages=True, max_batch_pages=8, parity_status="unproven"
    )
    registry = EngineRegistry(factories={"candidate": lambda: FakeEngine(capabilities=unproven)})
    assert registry.available() == ()


def test_registry_available_includes_proven_and_reference():
    reference = EngineCapabilities(
        engine_id="native", batches_pages=False, max_batch_pages=1, parity_status="reference"
    )
    proven = EngineCapabilities(
        engine_id="candidate", batches_pages=True, max_batch_pages=8, parity_status="proven"
    )
    registry = EngineRegistry(
        factories={
            "native": lambda: FakeEngine(capabilities=reference),
            "candidate": lambda: FakeEngine(capabilities=proven),
        }
    )
    assert set(registry.available()) == {"native", "candidate"}
