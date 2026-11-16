"""`InferenceEngine` plugin seam (design §17.2, plan T05).

`DiagnosticWorker` (T11) will drive one resident engine through this
protocol instead of calling `DiagnosticModelRuntime` directly: `load` ->
repeated `decode_pages` -> `unload`. `NativeEngine` (T11, `engines/native.py`)
wraps today's `_load_variant`/`extract`/`decode_features` path byte-for-byte.
`EngineRegistry` excludes any engine whose parity is not `reference`/`proven`
(§17.5).

Deliberately import-light: importing this module pulls in no `torch`
(§13/T05 acceptance -- the contract suite runs without it installed). The
real domain types this protocol references
(`ModelSafeInput`, `RawDocumentHypothesis`, `PredictionIrArtifact`,
`RuntimeIdentityPin`, `QuantizationSpec`) are only ever used as annotations
(`from __future__ import annotations` defers their evaluation) and imported
under `TYPE_CHECKING` -- most of them import `torch` transitively
(`unflatten.model.node_lifecycle` for `RawDocumentHypothesis`,
`unflatten.model.candidate_runtime` for `RuntimeIdentityPin`).

**Decision (§17.7 point 2, made in this task):** `seal` stays in the
protocol. The design's speculative removal was conditional on `FakeEngine`
being able to produce a *valid* `raw` -- but a valid `RawDocumentHypothesis`
needs `NodeState`, which imports `torch` at module load, so satisfying that
condition would break the "no torch" contract-suite guarantee. `seal` is
kept.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from unflatten.model.candidate_runtime import RuntimeIdentityPin
    from unflatten.model.input_contract import ModelSafeInput
    from unflatten.model.prediction_ir import RawDocumentHypothesis
    from unflatten.model.quantization.spec import QuantizationSpec
    from unflatten.service.prediction_ir_artifact import PredictionIrArtifact

EngineId = Literal["native"]
ParityStatus = Literal["reference", "proven", "unproven"]

@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    engine_id: EngineId
    batches_pages: bool
    max_batch_pages: int
    parity_status: ParityStatus


@dataclass(frozen=True, slots=True)
class PageInput:
    key: tuple[str, int]
    model_input: ModelSafeInput


@dataclass(frozen=True, slots=True)
class PageResult:
    key: tuple[str, int]
    raw: RawDocumentHypothesis | None
    prepared: Any | None
    decode_steps: int | None
    seconds: float
    error_code: str | None


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Exactly the six attributes `docx_stage` reads from a runtime today
    (`docx_stage.py:246-281, 427-464`; `model_runtime.py:31-77`)."""

    tokenizer: Any
    decode_policy: Any
    eval_cache_identity: Any
    component_lock_hash: str
    selection_policy_hash: str
    vocabulary: Any


@dataclass(frozen=True, slots=True)
class DeviceFacts:
    free_vram_mib: float
    total_vram_mib: float
    compute_capability: tuple[int, int]


class EngineProgress(Protocol):
    """Engine never sees `JobStore` or a lease -- only this narrow callback."""

    def enter_stage(self, stage: str) -> None: ...
    def page_done(self, result: PageResult) -> None: ...


class InferenceEngine(Protocol):
    capabilities: EngineCapabilities

    def load(
        self,
        bundle_dir: Path,
        *,
        quantization: QuantizationSpec,
        device: str,
        identity_pin: RuntimeIdentityPin,
        decode_steps_hard_ceiling: int,
    ) -> ModelIdentity: ...

    def unload(self) -> None: ...

    def decode_pages(
        self, pages: Sequence[PageInput], *, progress: EngineProgress
    ) -> list[PageResult]: ...

    def seal(self, raw: RawDocumentHypothesis) -> PredictionIrArtifact: ...

    def device_facts(self) -> DeviceFacts | None: ...


@dataclass(frozen=True, slots=True)
class EngineRegistry:
    """`{engine_id: factory}` (§17.2). `available()` is a pure function over
    each factory's constructed `capabilities` -- it never calls `load`, so
    checking availability never touches a GPU or a model bundle. An engine
    is available iff its factory can be called (proxy for "importable": a
    factory that imports its runtime lazily lets `ImportError` propagate
    here) and its `parity_status` is `reference` or
    `proven` -- `unproven` is excluded (§17.5's fail-closed rule)."""

    factories: Mapping[EngineId, Callable[[], InferenceEngine]]

    def available(self) -> tuple[EngineId, ...]:
        available: list[EngineId] = []
        for engine_id, factory in self.factories.items():
            try:
                engine = factory()
            except ImportError:
                continue
            if engine.capabilities.parity_status in ("reference", "proven"):
                available.append(engine_id)
        return tuple(available)
