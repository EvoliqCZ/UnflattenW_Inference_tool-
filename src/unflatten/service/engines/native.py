"""`NativeEngine` -- the reference `InferenceEngine` (design §17.3).

`load` is today's `_load_variant` (`worker_cli.py`) starting right after
`verify_bundle`: `load_candidate_runtime` + `DiagnosticModelRuntime`.
`decode_pages` iterates pages sequentially -- `extract` then
`decode_features` -- with the per-page except-ladder copied byte-for-byte
from `worker.py:361-394` (the design's own words: "the except-ladder ... is
temporarily copied here; `worker.py`'s copy is deleted by T17 in one merge
window"). `unload` is today's `_free_resident_model`
(`worker_cli.py:395-403`).

This module is not wired into `worker.py`/`worker_cli.py` yet (T17).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unflatten.model.candidate_runtime import (
    DecodePolicy,
    PreparedCandidateInput,
    load_candidate_runtime,
)
from unflatten.model.free_running_decode import DecodeMaxStepsExceeded
from unflatten.model.prediction_ir import PredictionIrInvalid, RawDocumentHypothesis
from unflatten.service.device_memory import cuda_memory_info
from unflatten.service.inference_engine import (
    DeviceFacts,
    EngineCapabilities,
    EngineProgress,
    ModelIdentity,
    PageInput,
    PageResult,
)
from unflatten.service.model_bundle import (
    ALLOWLIST_FILENAME,
    CHECKPOINT_FILENAME,
    COMPONENT_LOCK_FILENAME,
    DECODE_POLICY_FILENAME,
    verify_bundle,
)
from unflatten.service.model_runtime import DiagnosticModelRuntime, ForbiddenModelChannel
from unflatten.service.prediction_ir_artifact import PredictionIrArtifactInvalid

if TYPE_CHECKING:
    from unflatten.model.candidate_runtime import RuntimeIdentityPin
    from unflatten.model.quantization.spec import QuantizationSpec
    from unflatten.service.prediction_ir_artifact import PredictionIrArtifact

LOGGER = logging.getLogger(__name__)


def _load_decode_policy_block(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / DECODE_POLICY_FILENAME).read_text(encoding="utf-8"))


def _completed_page_to_cpu(prepared: Any, raw: Any) -> tuple[Any, Any]:
    """Keep completed pages for sealing without retaining their device tensors."""
    if isinstance(prepared, PreparedCandidateInput):
        prepared = replace(
            prepared,
            visual_features=prepared.visual_features.detach().to("cpu"),
            image_grid_thw=prepared.image_grid_thw.detach().to("cpu"),
            deepstack_visual_embeds=(
                tuple(tensor.detach().to("cpu") for tensor in prepared.deepstack_visual_embeds)
                if prepared.deepstack_visual_embeds is not None else None
            ),
        )
    if isinstance(raw, RawDocumentHypothesis):
        raw = replace(raw, node_states=tuple(
            replace(
                node,
                anchor=node.anchor.detach().to("cpu") if node.anchor is not None else None,
                content_summary=(
                    node.content_summary.detach().to("cpu")
                    if node.content_summary is not None else None
                ),
            )
            for node in raw.node_states
        ))
    return prepared, raw


class NativeEngine:
    """Sequential, single-page-at-a-time engine over today's candidate
    runtime path. Always available: `parity_status="reference"`, no import
    beyond what the service already needs."""

    capabilities = EngineCapabilities(
        engine_id="native", batches_pages=False, max_batch_pages=1, parity_status="reference"
    )

    def __init__(self) -> None:
        self._model: DiagnosticModelRuntime | None = None

    @classmethod
    def from_loaded(cls, model: DiagnosticModelRuntime) -> NativeEngine:
        """(T17): wrap an already-loaded `DiagnosticModelRuntime` -- used by
        `DiagnosticWorker`, which still receives its resident model from a
        `ModelProvider` (unchanged, `worker_cli.py`), not from `load()`."""
        engine = cls()
        engine._model = model
        return engine

    def load(
        self,
        bundle_dir: Path,
        *,
        quantization: QuantizationSpec,
        device: str,
        identity_pin: RuntimeIdentityPin,
        decode_steps_hard_ceiling: int,
    ) -> ModelIdentity:
        manifest = verify_bundle(bundle_dir, expected_precision_spec_hash=quantization.spec_hash)
        identities = manifest["identities"]
        decode_block = _load_decode_policy_block(bundle_dir)
        decode_policy = DecodePolicy(
            loop_guard=decode_block["decode"].get("loop_guard"),
            max_steps=decode_block["decode"]["max_steps"],
            typed_decode=decode_block["decode"]["typed_decode"],
            region_decode=decode_block["decode"]["region_decode"],
            selection_policy_hash=decode_block["selection_policy_hash"],
            prefill_chunk_tokens=decode_block["decode"].get("prefill_chunk_tokens"),
        )
        candidate = load_candidate_runtime(
            checkpoint_path=bundle_dir / CHECKPOINT_FILENAME,
            component_lock_path=bundle_dir / COMPONENT_LOCK_FILENAME,
            allowlist_path=bundle_dir / ALLOWLIST_FILENAME,
            tokenizer_artifact_set_hash=identities["tokenizer_artifact_set_hash"],
            pin=identity_pin,
            device=device,
            decode_policy=decode_policy,
            caps_sha256_pin=identities["loop_guard_v2_caps_sha256"],
            decode_steps_hard_ceiling=decode_steps_hard_ceiling,
            quantization=quantization,
        )
        model = DiagnosticModelRuntime(candidate)
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
        import gc

        import torch

        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def decode_pages(
        self, pages: Sequence[PageInput], *, progress: EngineProgress
    ) -> list[PageResult]:
        model = self._model
        if model is None:
            raise RuntimeError("NativeEngine.decode_pages called before load()")
        results: list[PageResult] = []
        for page in pages:
            started = time.monotonic()
            progress.enter_stage("feature_extraction")
            prepared = raw = None
            try:
                prepared = model.extract(page.model_input)
                progress.enter_stage("model_decode")
                raw = model.decode_features(prepared)
                prepared, raw = _completed_page_to_cpu(prepared, raw)
            except (ForbiddenModelChannel, PredictionIrInvalid, PredictionIrArtifactInvalid):
                result = self._failed(page, started, "PREDICTION_IR_INVALID")
            except DecodeMaxStepsExceeded:
                result = self._failed(page, started, "MODEL_DECODE_MAX_STEPS")
            except (OSError, ValueError, RuntimeError) as exc:
                # The type only -- an exception message can carry document
                # text, which never enters the service log (worker.py:384).
                LOGGER.warning(
                    "native_engine_page_internal_error job_id=%s page_index=%d exc_type=%s",
                    page.key[0],
                    page.key[1],
                    type(exc).__name__,
                )
                result = self._failed(page, started, "INTERNAL_ERROR")
            else:
                result = PageResult(
                    key=page.key,
                    raw=raw,
                    prepared=prepared,
                    decode_steps=len(raw.accepted_tokens),
                    seconds=time.monotonic() - started,
                    error_code=None,
                )
            finally:
                # A failed decode also leaves its extracted features here.
                # Clear both before callbacks or the next page's extraction.
                prepared = raw = None
            progress.page_done(result)
            results.append(result)
        return results

    @staticmethod
    def _failed(page: PageInput, started: float, error_code: str) -> PageResult:
        return PageResult(
            key=page.key,
            raw=None,
            prepared=None,
            decode_steps=None,
            seconds=time.monotonic() - started,
            error_code=error_code,
        )

    def seal(self, raw: RawDocumentHypothesis) -> PredictionIrArtifact:
        model = self._model
        if model is None:
            raise RuntimeError("NativeEngine.seal called before load()")
        return model.seal(raw)

    def device_facts(self) -> DeviceFacts | None:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = cuda_memory_info()
        capability = torch.cuda.get_device_capability()
        return DeviceFacts(
            free_vram_mib=free_bytes / (1024 * 1024),
            total_vram_mib=total_bytes / (1024 * 1024),
            compute_capability=capability,
        )
