"""Dedicated diagnostic worker process."""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from PIL import Image, UnidentifiedImageError

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.model.prediction_ir import PredictionIrInvalid
from unflatten.service.artifact_store import ArtifactStore, ArtifactStoreError
from unflatten.service.contracts import validate_contract
from unflatten.service.docx_stage import (
    DocxStageResult,
    MultiPageDocxStageResult,
    PageAttempt,
    build_diagnostic_docx,
    build_diagnostic_docx_multi,
)
from unflatten.service.engines.native import NativeEngine
from unflatten.service.errors import problem_response
from unflatten.service.inference_engine import PageInput, PageResult
from unflatten.service.job_store import (
    ArtifactMetadata,
    ArtifactRecord,
    FenceRejected,
    JobRecord,
    JobStore,
    Lease,
)
from unflatten.service.model_runtime import (
    DiagnosticModelRuntime,
    ForbiddenModelChannel,
    build_single_page_input,
)
from unflatten.service.pdf_sandbox import (
    RASTER_ROUNDING_PX,
    PdfAdmissionReport,
    PdfSandboxFailure,
)
from unflatten.service.policy import runtime_policy_hash, validate_runtime_policy
from unflatten.service.prediction_ir_artifact import (
    PredictionIrArtifact,
    PredictionIrArtifactInvalid,
    verify_prediction_ir_artifact,
)
from unflatten.service.preview import (
    PreviewRenderFailed,
    render_docx_previews,
    render_source_preview,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_TRANSIENT_FAILURE_CODES = frozenset(
    {"POPPLER_TIMEOUT", "POPPLER_RESOURCE_LIMIT", "ARTIFACT_WRITE_FAILED"}
)
_DIAGNOSTIC_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    """Private admitted source and its terminal cleanup operation."""

    path: Path
    delete: Callable[[], None]


class Sandbox(Protocol):
    def inspect(self, source: Path) -> PdfAdmissionReport: ...

    def raster(
        self, source: Path, report: PdfAdmissionReport, output: Path, *, page_index: int = 0
    ) -> None: ...

    def cancel(self) -> None: ...


class ModelProvider(Protocol):
    """DEC-044 Dodatok D §1 fix (2026-09-27 field report): resolves the
    resident model for one job, switching if `variant_id` names a different
    one than is currently loaded. `variant_id=None` means 'no models config
    is configured' or 'use the default variant' -- the provider decides
    which."""

    def ensure_loaded(self, variant_id: str | None) -> DiagnosticModelRuntime: ...


class _StaticModelProvider:
    """Wraps one fixed `DiagnosticModelRuntime` -- today's behaviour,
    byte-identical, no switching capability. The default when a caller
    passes `model=` instead of `model_provider=`."""

    def __init__(self, model: DiagnosticModelRuntime) -> None:
        self._model = model

    def ensure_loaded(self, variant_id: str | None) -> DiagnosticModelRuntime:
        return self._model


class WorkerFailure(RuntimeError):
    """A classified, content-free worker-stage failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _as_engine(model: Any) -> Any:
    """(T17, design §17.3/§17.4): `run_once` always drives
    `engine.decode_pages`, never `extract`/`decode_features` directly. A
    `ModelProvider` still hands back a `DiagnosticModelRuntime`
    (`worker_cli.py` is unchanged) -- wrap it in `NativeEngine` (T11). An
    object that is already engine-shaped (e.g. `FakeEngine` in tests) is
    used as-is, duck-typed on the one method that matters here."""
    if hasattr(model, "decode_pages"):
        return model
    return NativeEngine.from_loaded(model)


class _WorkerEngineProgress:
    """(T17, design §17.4): the `EngineProgress` `DiagnosticWorker` hands to
    `engine.decode_pages`. `enter_stage` is today's `_enter_stage`
    (heartbeat + `update_stage`, unchanged). `page_done` folds in both a
    raster/validation failure recorded before `decode_pages` even runs
    (`_run_pages`) and the engine's own per-page result, republishing
    `pages[]` sorted by page index each time -- only when `pages_total > 1`,
    preserving the single-page contract of never touching those columns
    (`worker.py`'s old `_execute_multi:395-405`)."""

    def __init__(self, worker: DiagnosticWorker, lease: Lease, pages_total: int) -> None:
        self._worker = worker
        self._lease = lease
        self._pages_total = pages_total
        self._outcomes: dict[int, PageResult] = {}

    def enter_stage(self, stage: str) -> None:
        self._worker._enter_stage(self._lease, stage)

    def page_done(self, result: PageResult) -> None:
        self._outcomes[result.key[1]] = result
        if self._pages_total > 1:
            pages = self.page_reports()
            self._worker._jobs.update_page_progress(
                self._lease,
                pages_total=self._pages_total,
                pages_done=len(pages),
                current_page=result.key[1],
                pages=pages,
                now=self._worker._clock(),
            )

    def outcome(self, index: int) -> PageResult:
        return self._outcomes[index]

    def page_reports(self) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "state": "succeeded" if self._outcomes[index].error_code is None else "failed",
                "error_code": self._outcomes[index].error_code,
                "decode_steps": self._outcomes[index].decode_steps,
                "seconds": self._outcomes[index].seconds,
            }
            for index in sorted(self._outcomes)
        ]


class DiagnosticWorker:
    """One-resident-model, one-active-job diagnostic worker."""

    def __init__(
        self,
        *,
        worker_id: str,
        jobs: JobStore,
        artifacts: ArtifactStore,
        sandbox: Sandbox,
        model: DiagnosticModelRuntime | None = None,
        model_provider: ModelProvider | None = None,
        source_resolver: Callable[[JobRecord], SourceArtifact] | None = None,
        policy: Mapping[str, Any],
        component_lock_hash: str,
        work_root: Path,
        clock: Callable[[], float] = time.time,
        readiness_heartbeat: Callable[[float], None] | None = None,
        transient_failure_codes: frozenset[str] = DEFAULT_TRANSIENT_FAILURE_CODES,
        diagnostic_docx_capable: bool | None = None,
    ) -> None:
        validate_runtime_policy(policy)
        if not worker_id:
            raise ValueError("worker_id must be non-empty")
        if policy["worker_inference_concurrency"] != 1:
            raise ValueError("diagnostic worker requires inference concurrency exactly one")
        if not isinstance(component_lock_hash, str) or not component_lock_hash:
            raise ValueError("component_lock_hash must be non-empty")
        if diagnostic_docx_capable is False:
            raise ValueError(
                "worker refuses to start: diagnostic_docx_capability is False (design D10)"
            )
        if (model is None) == (model_provider is None):
            raise ValueError("exactly one of model or model_provider is required")
        self.worker_id = worker_id
        self._jobs = jobs
        self._artifacts = artifacts
        self._sandbox = sandbox
        self._model_provider: ModelProvider = model_provider or _StaticModelProvider(model)
        # `ensure_loaded(None)`: "give me whatever is already resident" --
        # never a reload, even for a real switching provider (design below).
        if model is None:
            self._model_provider.ensure_loaded(None)
        # Only the provider owns an idle GPU model. Commands may switch it
        # before the first document, or between completed documents.
        self._model: DiagnosticModelRuntime | None = None
        self._engine = None
        self._source_resolver = source_resolver or self._artifact_source
        self._policy = dict(policy)
        self._policy_hash = runtime_policy_hash(policy)
        self._component_lock_hash = component_lock_hash
        self._work_root = work_root
        self._clock = clock
        self._readiness_heartbeat = readiness_heartbeat
        self._transient_codes = transient_failure_codes

    def _artifact_source(self, job: JobRecord) -> SourceArtifact:
        artifact, path = self._artifacts.find_staged_source(
            principal_id=job.principal_id, job_id=job.job_id
        )
        return SourceArtifact(path=path, delete=lambda: self._artifacts.discard_staged(artifact))

    def run_once(self) -> bool:
        """Claim and execute at most one job, returning whether one was claimed."""
        now = self._clock()
        self._publish_readiness(now)
        self._jobs.requeue_expired(now=now)
        lease = self._jobs.claim_next(worker_id=self.worker_id, now=now)
        if lease is None:
            return False
        LOGGER.info("worker_job_claimed job_id=%s attempt=%d", lease.job_id, lease.attempt)
        source: SourceArtifact | None = None
        self._stage: str | None = None
        delete_source = False
        attempt_root = self._work_root / lease.execution_id
        raster_path = attempt_root / "page.png"
        try:
            job = self._jobs.get_claimed_job(lease, now=self._clock())
            self._verify_runtime_identity(job)
            # Drop this worker's reference first: a switch can only free the
            # resident model if nothing but the provider still holds it.
            self._engine = None
            self._model = None
            self._model = self._run_stage(
                lease, "loading_model", lambda: self._model_provider.ensure_loaded(
                    job.model_variant
                ),
            )
            self._engine = _as_engine(self._model)
            source = self._source_resolver(job)
            attempt_root.mkdir(mode=0o700, parents=True, exist_ok=False)
            report = self._run_stage(
                lease, "pdf_inspection", lambda: self._sandbox.inspect(source.path)
            )
            if report.page_count > 1:
                ir_artifacts, docx_result, _page_reports, source_pngs = self._execute_multi(
                    lease, job, source.path, attempt_root, report
                )
                self._finalize_multi(lease, job, ir_artifacts, docx_result, source_pngs)
            else:
                artifact, docx_result, png_bytes = self._execute(
                    lease, job, source.path, raster_path, report
                )
                self._finalize(lease, job, artifact, docx_result, png_bytes)
            delete_source = True
            LOGGER.info("worker_job_succeeded job_id=%s attempt=%d", lease.job_id, lease.attempt)
        except FenceRejected:
            self._sandbox.cancel()
            delete_source = self._finish_cancellation(lease)
            LOGGER.info("worker_fence_lost job_id=%s attempt=%d", lease.job_id, lease.attempt)
        except (PdfSandboxFailure, ArtifactStoreError, WorkerFailure) as exc:
            code = exc.code
            delete_source = self._record_failure(lease, code)
        except (ForbiddenModelChannel, PredictionIrInvalid, PredictionIrArtifactInvalid):
            # `DecodeMaxStepsExceeded`/generic decode errors never reach here
            # any more (T17): `self._engine.decode_pages` (NativeEngine, or
            # `FakeEngine` in tests) catches them per page and returns
            # `PageResult.error_code`; `_execute` re-raises a single page's
            # error as `WorkerFailure(error_code)` for the whole-job-fails
            # single-page contract, caught above.
            delete_source = self._record_failure(lease, "PREDICTION_IR_INVALID")
        except (OSError, ValueError, RuntimeError) as exc:
            # The type and stage only: an exception message can carry document
            # text, which never enters the service log.
            LOGGER.warning(
                "worker_internal_error job_id=%s attempt=%d stage=%s exc_type=%s",
                lease.job_id, lease.attempt, self._stage, type(exc).__name__,
            )
            delete_source = self._record_failure(lease, "INTERNAL_ERROR")
        finally:
            self._engine = None
            self._model = None
            shutil.rmtree(attempt_root, ignore_errors=True)
            if delete_source and source is not None:
                source.delete()
        return True

    def _verify_runtime_identity(self, job: JobRecord) -> None:
        if job.policy_hash != self._policy_hash:
            raise WorkerFailure("RUNTIME_POLICY_MISMATCH")
        if job.component_lock_hash != self._component_lock_hash:
            raise WorkerFailure("MODEL_COMPONENT_LOCK_MISMATCH")

    def _run_pages(
        self,
        lease: Lease,
        source: Path,
        attempt_root: Path,
        report: PdfAdmissionReport,
    ) -> tuple[list[PageResult], list[tuple[int, bytes]], list[dict[str, Any]]]:
        """(T17, design §17.4): rasterize/validate/project every page first,
        engine-agnostically, then issue exactly ONE `engine.decode_pages`
        call for the whole job -- `N=1` is simply the single-page case of
        this same path. Shared by `_execute` and `_execute_multi`, which
        differ only in how they seal/build a docx from the results. A
        raster/validation failure is recorded per page (never aborting the
        others) UNLESS there is only one page, in which case it propagates
        exactly as before (the whole job fails, no page bookkeeping)."""
        pages_total = report.page_count
        if pages_total > 1:
            self._jobs.update_page_progress(
                lease, pages_total=pages_total, pages_done=0, current_page=0, pages=[],
                now=self._clock(),
            )
        progress = _WorkerEngineProgress(self, lease, pages_total)
        page_inputs: list[PageInput] = []
        source_pngs: list[tuple[int, bytes]] = []
        for page_index in range(pages_total):
            raster_path = attempt_root / f"page_{page_index}.png"
            started = self._clock()
            try:
                self._run_stage(
                    lease, "pdf_rasterization",
                    lambda p=raster_path, i=page_index: self._sandbox.raster(
                        source, report, p, page_index=i
                    ),
                )
                width, height, pixels, page_png_bytes = self._run_stage(
                    lease, "png_validation", lambda p=raster_path, i=page_index:
                    self._load_rgb(p, report, page_index=i)
                )
            except (PdfSandboxFailure, WorkerFailure) as exc:
                if pages_total == 1:
                    raise
                progress.page_done(
                    PageResult(
                        key=(lease.job_id, page_index), raw=None, prepared=None,
                        decode_steps=None, seconds=self._clock() - started,
                        error_code=exc.code,
                    )
                )
                continue
            source_pngs.append((page_index, page_png_bytes))
            model_input = self._run_stage(
                lease, "model_input_projection",
                lambda w=width, h=height, px=pixels: build_single_page_input(
                    width_px=w, height_px=h, rgb_pixels=px, host_channels={}
                ),
            )
            page_inputs.append(
                PageInput(key=(lease.job_id, page_index), model_input=model_input)
            )
        if page_inputs:
            self._run_stage(
                lease, "model_decode",
                lambda: self._engine.decode_pages(page_inputs, progress=progress),
            )
        results = [progress.outcome(i) for i in range(pages_total)]
        return results, source_pngs, progress.page_reports()

    def _execute(
        self,
        lease: Lease,
        job: JobRecord,
        source: Path,
        raster_path: Path,
        report: PdfAdmissionReport,
    ) -> tuple[PredictionIrArtifact, DocxStageResult | None, bytes]:
        results, source_pngs, _page_reports = self._run_pages(
            lease, source, raster_path.parent, report
        )
        result = results[0]
        if result.error_code is not None:
            # (T17): the engine never raises for one page's own failure
            # (design §17.2); a single-page job still fails the whole job,
            # via the generic `WorkerFailure` branch run_once already has.
            raise WorkerFailure(result.error_code)
        png_bytes = source_pngs[0][1]
        artifact = self._run_stage(
            lease, "prediction_ir_sealing", lambda: self._seal_verified(result.raw)
        )
        docx_result: DocxStageResult | None = None
        if job.requested_output == "diagnostic_docx":
            docx_result = self._run_stage(
                lease,
                "ast_promotion",
                lambda: build_diagnostic_docx(
                    result.raw,
                    result.prepared,
                    png_bytes,
                    runtime=self._model,
                    job_id=lease.job_id,
                    work_dir=raster_path.parent,
                    max_docx_bytes=self._policy["max_diagnostic_docx_bytes"],
                    expected_prediction_ir_hash=artifact.semantic_hash,
                ),
            )
            # G1: the chain's own stage already renewed the lease when it
            # finished (`_run_stage`'s trailing heartbeat); entering the next
            # named stage is bookkeeping only -- the gate decision itself was
            # already made inside `build_diagnostic_docx`.
            self._enter_stage(lease, "renderer_gate")
        return artifact, docx_result, png_bytes

    def _execute_multi(
        self,
        lease: Lease,
        job: JobRecord,
        source: Path,
        attempt_root: Path,
        report: PdfAdmissionReport,
    ) -> tuple[
        list[tuple[int, PredictionIrArtifact]],
        MultiPageDocxStageResult | None,
        list[dict[str, Any]],
        list[tuple[int, bytes]],
    ]:
        """DEC-044 Dodatok D §1 (B1): decode every page of a multi-page PDF
        independently; a single page's failure never aborts the others (only
        `FenceRejected` -- the whole job's lease is gone -- propagates)."""
        results, source_pngs, page_reports = self._run_pages(
            lease, source, attempt_root, report
        )
        attempts = [
            PageAttempt(
                index=r.key[1], seconds=r.seconds, raw=r.raw, prepared=r.prepared,
                error_code=r.error_code,
            )
            for r in results
        ]
        ir_artifacts: list[tuple[int, PredictionIrArtifact]] = []
        for attempt in attempts:
            if attempt.raw is None:
                continue
            artifact = self._run_stage(
                lease, "prediction_ir_sealing", lambda a=attempt: self._seal_verified(a.raw)
            )
            ir_artifacts.append((attempt.index, artifact))
        docx_result: MultiPageDocxStageResult | None = None
        if job.requested_output == "diagnostic_docx":
            docx_result = self._run_stage(
                lease, "ast_promotion",
                lambda: build_diagnostic_docx_multi(
                    attempts, runtime=self._model, job_id=lease.job_id,
                    work_dir=attempt_root,
                    max_docx_bytes=self._policy["max_diagnostic_docx_bytes"],
                    page_rasters=dict(source_pngs),
                ),
            )
            self._enter_stage(lease, "renderer_gate")
        return ir_artifacts, docx_result, page_reports, source_pngs

    def _seal_verified(self, raw: Any) -> PredictionIrArtifact:
        artifact = self._engine.seal(raw)
        verify_prediction_ir_artifact(artifact)
        return artifact

    def _run_stage(self, lease: Lease, stage: str, operation: Callable[[], T]) -> T:
        """Fence a stage and renew its lease while a long operation remains active."""
        self._enter_stage(lease, stage)
        started = self._clock()
        stopped = threading.Event()
        heartbeat_failure: list[FenceRejected] = []

        def renew() -> None:
            while not stopped.wait(self._policy["heartbeat_interval_seconds"]):
                try:
                    self._heartbeat(lease)
                except FenceRejected as exc:
                    heartbeat_failure.append(exc)
                    self._sandbox.cancel()
                    return

        heartbeat = threading.Thread(target=renew, name="worker-heartbeat", daemon=True)
        heartbeat.start()
        try:
            result = operation()
        finally:
            stopped.set()
            heartbeat.join()
        if heartbeat_failure:
            raise heartbeat_failure[0]
        self._heartbeat(lease)
        # G1/P6: heartbeat renews the lease the moment the stage's own work
        # finishes -- before the content-free timing log below, and before
        # any later stage (e.g. `renderer_gate`) that stage's outcome feeds.
        LOGGER.info(
            "worker_stage_done job_id=%s stage=%s seconds=%.3f",
            lease.job_id,
            stage,
            self._clock() - started,
        )
        return result

    def _build_preview_artifacts(
        self,
        job: JobRecord,
        lease: Lease,
        *,
        source_pages: list[tuple[int, bytes]],
        docx_bytes: bytes,
    ) -> tuple[list[ArtifactRef], list[ArtifactMetadata], list[str]]:
        """DEC-044 Dodatok D (C): stage `source_preview` (one per admitted
        page) and `docx_preview` (one per rendered docx page -- may differ in
        count from the source pages, e.g. a refused seam adds a page break).
        Never raises: a LibreOffice/pdftoppm failure is recorded as
        `PREVIEW_RENDER_FAILED` and that preview role is simply omitted,
        never failing the job (its diagnostic_docx already exists)."""
        staged: list[ArtifactRef] = []
        metadata: list[ArtifactMetadata] = []
        reasons: list[str] = []
        for page_index, png_bytes in source_pages:
            try:
                preview_bytes = render_source_preview(png_bytes)
            except (OSError, ValueError, PreviewRenderFailed) as exc:
                LOGGER.warning(
                    "worker_source_preview_failed job_id=%s page_index=%d exc_type=%s",
                    lease.job_id, page_index, type(exc).__name__,
                )
                reasons.append("PREVIEW_RENDER_FAILED")
                continue
            ref = self._artifacts.stage(
                principal_id=job.principal_id, job_id=job.job_id,
                execution_id=lease.execution_id, role="source_preview", mime_type="image/png",
                chunks=(preview_bytes,), max_bytes=self._policy["max_png_bytes"],
            )
            staged.append(ref)
            metadata.append(self._artifact_metadata("source_preview", ref, page_index=page_index))
        try:
            docx_pages = render_docx_previews(docx_bytes)
        except PreviewRenderFailed as exc:
            LOGGER.info(
                "worker_docx_preview_failed job_id=%s error_class=%s",
                lease.job_id, type(exc).__name__,
            )
            reasons.append("PREVIEW_RENDER_FAILED")
        else:
            for page_index, preview_bytes in enumerate(docx_pages):
                ref = self._artifacts.stage(
                    principal_id=job.principal_id, job_id=job.job_id,
                    execution_id=lease.execution_id, role="docx_preview", mime_type="image/png",
                    chunks=(preview_bytes,), max_bytes=self._policy["max_png_bytes"],
                )
                staged.append(ref)
                metadata.append(
                    self._artifact_metadata("docx_preview", ref, page_index=page_index)
                )
        return staged, metadata, sorted(set(reasons))

    def _finalize(
        self,
        lease: Lease,
        job: JobRecord,
        artifact: PredictionIrArtifact,
        docx_result: DocxStageResult | None,
        png_bytes: bytes,
    ) -> None:
        self._enter_stage(lease, "artifact_finalization")
        staged_ir = self._stage_prediction_ir(job, lease, artifact)
        if docx_result is None:
            outcome = validate_contract(
                {
                    "prediction_ir_status": "sealed",
                    "claim_scope": "diagnostic",
                    "canonical_ast_produced": False,
                    "docx_produced": False,
                    "reason_codes": ["DIAGNOSTIC_DOCX_NOT_REQUESTED"],
                    "docx_chain": None,
                },
                "diagnostic_outcome",
            )
            self._publish_ir_only(
                lease, staged_ir, outcome, self._policy["prediction_ir_retention_seconds"]
            )
            return

        if not docx_result.produced:
            outcome = validate_contract(
                {
                    "prediction_ir_status": "sealed",
                    "claim_scope": "diagnostic",
                    "canonical_ast_produced": (
                        docx_result.docx_chain.get("ast_status") == "strict_valid"
                    ),
                    "docx_produced": False,
                    "reason_codes": list(docx_result.reason_codes),
                    "docx_chain": docx_result.docx_chain,
                },
                "diagnostic_outcome",
            )
            self._publish_ir_only(
                lease, staged_ir, outcome, self._policy["prediction_ir_retention_seconds"]
            )
            return

        canonical_ast_produced = docx_result.docx_chain.get("ast_status") == "strict_valid"
        preview_staged, preview_metadata, preview_reasons = self._build_preview_artifacts(
            job, lease, source_pages=[(0, png_bytes)], docx_bytes=docx_result.docx_bytes
        )
        outcome = validate_contract(
            {
                "prediction_ir_status": "sealed",
                "claim_scope": "diagnostic",
                "canonical_ast_produced": canonical_ast_produced,
                "docx_produced": docx_result.produced,
                "reason_codes": sorted(set(docx_result.reason_codes) | set(preview_reasons)),
                "docx_chain": docx_result.docx_chain,
            },
            "diagnostic_outcome",
        )

        def fence(published: list[ArtifactRef], publish: Callable[[], None]) -> bool:
            artifacts = [
                self._artifact_metadata("prediction_ir", published[0]),
                self._artifact_metadata("diagnostic_docx", published[1]),
            ]
            for meta, ref in zip(preview_metadata, published[2:], strict=True):
                artifacts.append(
                    self._artifact_metadata(meta.role, ref, page_index=meta.page_index)
                )
            self._jobs.finalize_success(
                lease,
                artifacts=artifacts,
                diagnostic_outcome=outcome,
                retention_seconds=self._policy["diagnostic_docx_retention_seconds"],
                now=self._clock(),
                publish=publish,
            )
            return True

        try:
            staged_docx = self._artifacts.stage(
                principal_id=job.principal_id,
                job_id=job.job_id,
                execution_id=lease.execution_id,
                role="diagnostic_docx",
                mime_type=_DIAGNOSTIC_DOCX_MIME,
                chunks=(docx_result.docx_bytes,),
                max_bytes=self._policy["max_diagnostic_docx_bytes"],
            )
            self._artifacts.publish_many(
                [staged_ir, staged_docx, *preview_staged], fence, allow_repeated_roles=True
            )
        except ArtifactStoreError as exc:
            # G1/P2: a failed second `os.replace` (design §D3) rolls BOTH
            # staged files back -- the already-sealed IR bytes still live in
            # `artifact.raw_bytes`, so re-stage and finalize IR-only rather
            # than losing the result the client already has a right to.
            if exc.role not in (
                None, "diagnostic_docx", "source_preview", "docx_preview",
            ) or exc.code not in {
                "ARTIFACT_PUBLICATION_FAILED",
                "ARTIFACT_WRITE_FAILED",
            }:
                raise
            LOGGER.info(
                "worker_docx_publish_failed job_id=%s error_class=%s",
                lease.job_id,
                type(exc.__cause__).__name__ if exc.__cause__ is not None else "Unknown",
            )
            fallback_chain = dict(docx_result.docx_chain)
            fallback_chain["error_class"] = (
                f"publish_failed:"
                f"{type(exc.__cause__).__name__ if exc.__cause__ is not None else 'Unknown'}"
            )
            fallback_outcome = validate_contract(
                {
                    "prediction_ir_status": "sealed",
                    "claim_scope": "diagnostic",
                    "canonical_ast_produced": canonical_ast_produced,
                    "docx_produced": False,
                    "reason_codes": ["DOCX_CHAIN_ERROR"],
                    "docx_chain": fallback_chain,
                },
                "diagnostic_outcome",
            )
            restaged_ir = self._stage_prediction_ir(job, lease, artifact)
            self._publish_ir_only(
                lease,
                restaged_ir,
                fallback_outcome,
                self._policy["prediction_ir_retention_seconds"],
            )

    def _finalize_multi(
        self,
        lease: Lease,
        job: JobRecord,
        ir_artifacts: list[tuple[int, PredictionIrArtifact]],
        docx_result: MultiPageDocxStageResult | None,
        source_pngs: list[tuple[int, bytes]],
    ) -> None:
        """DEC-044 Dodatok D §1 (B1): the multi-page counterpart of
        `_finalize` -- one `prediction_ir` artifact PER successfully-decoded
        page (`page_index` set), published atomically alongside the ONE
        merged `diagnostic_docx` (`page_index` the -1 'not page-scoped'
        sentinel), via `publish_many(..., allow_repeated_roles=True)`. Per-page
        progress (`pages_total`/`pages_done`/`pages`) was already published
        incrementally by `_execute_multi`; `finalize_success`'s UPDATE never
        touches those columns, so the final report survives untouched."""
        self._enter_stage(lease, "artifact_finalization")
        staged_irs = [
            (page_index, self._stage_prediction_ir(job, lease, artifact))
            for page_index, artifact in ir_artifacts
        ]
        if docx_result is None:
            outcome = validate_contract(
                {
                    "prediction_ir_status": "sealed",
                    "claim_scope": "diagnostic",
                    "canonical_ast_produced": False,
                    "docx_produced": False,
                    "reason_codes": ["DIAGNOSTIC_DOCX_NOT_REQUESTED"],
                    "docx_chain": None,
                },
                "diagnostic_outcome",
            )
            self._publish_ir_only_multi(
                lease, staged_irs, outcome, self._policy["prediction_ir_retention_seconds"]
            )
            return

        canonical_ast_produced = docx_result.docx_chain.get("ast_status") == "strict_valid"
        if not docx_result.produced:
            outcome = validate_contract(
                {
                    "prediction_ir_status": "sealed",
                    "claim_scope": "diagnostic",
                    "canonical_ast_produced": canonical_ast_produced,
                    "docx_produced": False,
                    "reason_codes": list(docx_result.reason_codes),
                    "docx_chain": docx_result.docx_chain,
                },
                "diagnostic_outcome",
            )
            self._publish_ir_only_multi(
                lease, staged_irs, outcome, self._policy["prediction_ir_retention_seconds"]
            )
            return

        preview_staged, preview_metadata, preview_reasons = self._build_preview_artifacts(
            job, lease, source_pages=source_pngs, docx_bytes=docx_result.docx_bytes
        )
        outcome = validate_contract(
            {
                "prediction_ir_status": "sealed",
                "claim_scope": "diagnostic",
                "canonical_ast_produced": canonical_ast_produced,
                "docx_produced": docx_result.produced,
                "reason_codes": sorted(set(docx_result.reason_codes) | set(preview_reasons)),
                "docx_chain": docx_result.docx_chain,
            },
            "diagnostic_outcome",
        )

        def fence(published: list[ArtifactRef], publish: Callable[[], None]) -> bool:
            published_irs = published[: len(staged_irs)]
            published_docx = published[len(staged_irs)]
            published_previews = published[len(staged_irs) + 1 :]
            artifacts = [
                self._artifact_metadata("prediction_ir", ref, page_index=page_index)
                for (page_index, _staged), ref in zip(staged_irs, published_irs, strict=True)
            ]
            artifacts.append(self._artifact_metadata("diagnostic_docx", published_docx))
            for meta, ref in zip(preview_metadata, published_previews, strict=True):
                artifacts.append(
                    self._artifact_metadata(meta.role, ref, page_index=meta.page_index)
                )
            self._jobs.finalize_success(
                lease,
                artifacts=artifacts,
                diagnostic_outcome=outcome,
                retention_seconds=self._policy["diagnostic_docx_retention_seconds"],
                now=self._clock(),
                publish=publish,
            )
            return True

        try:
            staged_docx = self._artifacts.stage(
                principal_id=job.principal_id,
                job_id=job.job_id,
                execution_id=lease.execution_id,
                role="diagnostic_docx",
                mime_type=_DIAGNOSTIC_DOCX_MIME,
                chunks=(docx_result.docx_bytes,),
                max_bytes=self._policy["max_diagnostic_docx_bytes"],
            )
            self._artifacts.publish_many(
                [ref for _, ref in staged_irs] + [staged_docx] + preview_staged,
                fence, allow_repeated_roles=True,
            )
        except ArtifactStoreError as exc:
            if exc.role not in (
                None, "diagnostic_docx", "source_preview", "docx_preview",
            ) or exc.code not in {
                "ARTIFACT_PUBLICATION_FAILED",
                "ARTIFACT_WRITE_FAILED",
            }:
                raise
            LOGGER.info(
                "worker_docx_publish_failed job_id=%s error_class=%s",
                lease.job_id,
                type(exc.__cause__).__name__ if exc.__cause__ is not None else "Unknown",
            )
            fallback_chain = dict(docx_result.docx_chain)
            fallback_chain["error_class"] = (
                f"publish_failed:"
                f"{type(exc.__cause__).__name__ if exc.__cause__ is not None else 'Unknown'}"
            )
            fallback_outcome = validate_contract(
                {
                    "prediction_ir_status": "sealed",
                    "claim_scope": "diagnostic",
                    "canonical_ast_produced": canonical_ast_produced,
                    "docx_produced": False,
                    "reason_codes": ["DOCX_CHAIN_ERROR"],
                    "docx_chain": fallback_chain,
                },
                "diagnostic_outcome",
            )
            restaged_irs = [
                (page_index, self._stage_prediction_ir(job, lease, artifact))
                for page_index, artifact in ir_artifacts
            ]
            self._publish_ir_only_multi(
                lease,
                restaged_irs,
                fallback_outcome,
                self._policy["prediction_ir_retention_seconds"],
            )

    def _publish_ir_only_multi(
        self,
        lease: Lease,
        staged_irs: list[tuple[int, ArtifactRef]],
        outcome: Mapping[str, Any],
        retention_seconds: int,
    ) -> None:
        def fence(published: list[ArtifactRef], publish: Callable[[], None]) -> bool:
            artifacts = [
                self._artifact_metadata("prediction_ir", ref, page_index=page_index)
                for (page_index, _staged), ref in zip(staged_irs, published, strict=True)
            ]
            self._jobs.finalize_success(
                lease,
                artifacts=artifacts,
                diagnostic_outcome=outcome,
                retention_seconds=retention_seconds,
                now=self._clock(),
                publish=publish,
            )
            return True

        self._artifacts.publish_many(
            [ref for _, ref in staged_irs], fence, allow_repeated_roles=True
        )

    def _stage_prediction_ir(
        self, job: JobRecord, lease: Lease, artifact: PredictionIrArtifact
    ) -> ArtifactRef:
        return self._artifacts.stage(
            principal_id=job.principal_id,
            job_id=job.job_id,
            execution_id=lease.execution_id,
            role="prediction_ir",
            mime_type="application/json",
            chunks=(artifact.raw_bytes,),
            max_bytes=self._policy["max_prediction_ir_bytes"],
        )

    def _publish_ir_only(
        self,
        lease: Lease,
        staged: ArtifactRef,
        outcome: Mapping[str, Any],
        retention_seconds: int,
    ) -> None:
        def fence(published: ArtifactRef, publish: Callable[[], None]) -> bool:
            self._jobs.finalize_success(
                lease,
                artifacts=[self._artifact_metadata("prediction_ir", published)],
                diagnostic_outcome=outcome,
                retention_seconds=retention_seconds,
                now=self._clock(),
                publish=publish,
            )
            return True

        self._artifacts.publish(staged, fence)

    @staticmethod
    def _artifact_metadata(
        role: str, ref: ArtifactRef, *, page_index: int = -1
    ) -> ArtifactMetadata:
        return ArtifactMetadata(
            role=role,
            ref=ref.ref,
            sha256=ref.sha256.removeprefix("sha256:"),
            size_bytes=ref.size_bytes,
            mime_type=ref.mime_type,
            page_index=page_index,
        )

    def _enter_stage(self, lease: Lease, stage: str) -> None:
        self._stage = stage
        self._heartbeat(lease)
        self._jobs.update_stage(lease, stage=stage, now=self._clock())

    def _heartbeat(self, lease: Lease) -> None:
        now = self._clock()
        self._jobs.heartbeat(lease, now=now)
        self._publish_readiness(now)

    def _publish_readiness(self, now: float) -> None:
        if self._readiness_heartbeat is not None:
            self._readiness_heartbeat(now)

    def _record_failure(self, lease: Lease, code: str) -> bool:
        retryable = code in self._transient_codes
        problem_code = code if code in _PUBLIC_FAILURE_CODES else "INTERNAL_ERROR"
        try:
            record = self._jobs.fail_attempt(
                lease,
                failure_class="transient_infrastructure" if retryable else "deterministic",
                retryable=retryable,
                error=problem_response(
                    code=problem_code,
                    title="Diagnostic processing failed",
                    status=503 if retryable else 422,
                    retryable=retryable,
                ),
                now=self._clock(),
            )
        except FenceRejected:
            self._sandbox.cancel()
            return self._finish_cancellation(lease)
        LOGGER.info(
            "worker_job_failed job_id=%s attempt=%d code=%s retryable=%s",
            lease.job_id,
            lease.attempt,
            problem_code,
            retryable,
        )
        return record.state == "failed"

    def _finish_cancellation(self, lease: Lease) -> bool:
        try:
            return self._jobs.complete_cancel(lease, now=self._clock()).state == "cancelled"
        except FenceRejected:
            return False

    @staticmethod
    def artifact_ref(record: ArtifactRecord) -> ArtifactRef:
        """Project published job metadata back to the shared ArtifactRef contract."""
        return ArtifactRef(
            ref=record.ref,
            sha256=f"sha256:{record.sha256}",
            size_bytes=record.size_bytes,
            mime_type=record.mime_type,
        )

    def _load_rgb(
        self, raster_path: Path, report: PdfAdmissionReport, *, page_index: int = 0
    ) -> tuple[int, int, bytes, bytes]:
        """(design D4): also returns the raw PNG bytes, read here -- once,
        already size-bounded -- because `finally` deletes `raster_path` at
        the end of the attempt (`worker.py:153`), before a later diagnostic
        Word stage could otherwise still read the file from disk."""
        try:
            geometry = report.page_geometries[page_index]
            if raster_path.stat().st_size > self._policy["max_png_bytes"]:
                raise WorkerFailure("PNG_INVALID")
            png_bytes = raster_path.read_bytes()
            with Image.open(raster_path) as image:
                image.load()
                # the adapter's one-pixel poppler rounding tolerance (pdf_sandbox)
                if (
                    abs(image.width - geometry.raster_width_pixels) > RASTER_ROUNDING_PX
                    or abs(image.height - geometry.raster_height_pixels) > RASTER_ROUNDING_PX
                ):
                    raise WorkerFailure("PNG_INVALID")
                rgb = image.convert("RGB")
                return rgb.width, rgb.height, rgb.tobytes(), png_bytes
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise WorkerFailure("PNG_INVALID") from exc


_PUBLIC_FAILURE_CODES = frozenset(
    {
        "RUNTIME_POLICY_MISMATCH",
        "MODEL_COMPONENT_LOCK_MISMATCH",
        "UPLOAD_TOO_LARGE",
        "PDF_SIGNATURE_INVALID",
        "PDF_MALFORMED",
        "PDF_ENCRYPTED",
        "PDF_ACTIVE_CONTENT",
        "PDF_EMBEDDED_FILE",
        "PDF_ANNOTATION_UNSUPPORTED",
        "PDF_HIDDEN_TEXT",
        "PDF_TEXT_VISIBILITY_AMBIGUOUS",
        "UNSUPPORTED_PAGE_COUNT",
        "PDF_PAGE_DIMENSIONS_EXCEEDED",
        "POPPLER_TIMEOUT",
        "POPPLER_RESOURCE_LIMIT",
        "RASTER_OUTPUT_INVALID",
        "RASTER_PIXEL_LIMIT_EXCEEDED",
        "PNG_INVALID",
        "PREDICTION_IR_INVALID",
        "MODEL_DECODE_MAX_STEPS",
        "INTERNAL_ERROR",
    }
)
