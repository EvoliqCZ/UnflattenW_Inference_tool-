"""`FakeEngine` -- an `InferenceEngine` (design §17.2) test double.

Shared by `test_inference_engine_contract.py` (T05) and, later, by
`test_worker.py`/`fake_worker.py` (T11, T37) so a real `DiagnosticWorker` can
run end to end without a GPU. Deliberately does not import `torch`: `raw` on
a successful page is `_FakeRaw`, a stand-in for `RawDocumentHypothesis` --
building a real one needs `unflatten.model.node_lifecycle.NodeState`, which
imports `torch` at module load (see `inference_engine.py`'s module
docstring for why `seal` stays in the protocol for the same reason).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from unflatten.service.inference_engine import (
    DeviceFacts,
    EngineCapabilities,
    EngineProgress,
    ModelIdentity,
    PageInput,
    PageResult,
)

_DEFAULT_STAGES: tuple[str, ...] = ("feature_extraction", "model_decode")


def _default_capabilities() -> EngineCapabilities:
    return EngineCapabilities(
        engine_id="native", batches_pages=False, max_batch_pages=1, parity_status="reference"
    )


def _default_identity() -> ModelIdentity:
    return ModelIdentity(
        tokenizer="fake-tokenizer",
        decode_policy="fake-decode-policy",
        eval_cache_identity="fake-eval-cache-identity",
        component_lock_hash="fake-component-lock-hash",
        selection_policy_hash="fake-selection-policy-hash",
        vocabulary="fake-vocabulary",
    )


@dataclass(frozen=True, slots=True)
class _FakeRaw:
    """Stand-in for `RawDocumentHypothesis` -- see module docstring."""

    key: tuple[str, int]


@dataclass
class FakeEngine:
    capabilities: EngineCapabilities = field(default_factory=_default_capabilities)
    identity: ModelIdentity = field(default_factory=_default_identity)
    fail_pages: frozenset[tuple[str, int]] = field(default_factory=frozenset)
    stages: tuple[str, ...] = _DEFAULT_STAGES
    device_facts_value: DeviceFacts | None = None
    loaded: bool = field(default=False, init=False)

    def load(self, *_args: Any, **_kwargs: Any) -> ModelIdentity:
        self.loaded = True
        return self.identity

    def unload(self) -> None:
        self.loaded = False

    def decode_pages(
        self, pages: Sequence[PageInput], *, progress: EngineProgress
    ) -> list[PageResult]:
        results: list[PageResult] = []
        for page in pages:
            for stage in self.stages:
                progress.enter_stage(stage)
            if page.key in self.fail_pages:
                result = PageResult(
                    key=page.key,
                    raw=None,
                    prepared=None,
                    decode_steps=None,
                    seconds=0.0,
                    error_code="INTERNAL_ERROR",
                )
            else:
                result = PageResult(
                    key=page.key,
                    raw=_FakeRaw(key=page.key),
                    prepared=page.model_input,
                    decode_steps=1,
                    seconds=0.001,
                    error_code=None,
                )
            progress.page_done(result)
            results.append(result)
        return results

    def seal(self, raw: Any) -> Any:
        """Not exercised by the T05 contract suite (§13 does not list a
        `seal` test); kept minimal so this module stays torch-free."""
        return raw

    def device_facts(self) -> DeviceFacts | None:
        return self.device_facts_value
