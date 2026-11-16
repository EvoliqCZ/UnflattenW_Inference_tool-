from __future__ import annotations

import ast
import dataclasses
import logging
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.service.fake_engine import FakeEngine

from unflatten.model.free_running_decode import DecodeMaxStepsExceeded
from unflatten.model.node_lifecycle import NodeState
from unflatten.model.prediction_ir import RawDocumentHypothesis
from unflatten.service import docx_stage as ds
from unflatten.service import worker as worker_module
from unflatten.service.artifact_store import ArtifactStore, ArtifactStoreError
from unflatten.service.job_store import JobStore
from unflatten.service.model_runtime import DiagnosticModelRuntime
from unflatten.service.pdf_sandbox import PdfAdmissionReport, PdfSandboxFailure
from unflatten.service.policy import load_runtime_policy, runtime_policy_hash
from unflatten.service.preview import PreviewRenderFailed
from unflatten.service.worker import DiagnosticWorker, SourceArtifact, WorkerFailure


@pytest.fixture(autouse=True)
def _no_real_previews(monkeypatch: pytest.MonkeyPatch) -> None:
    """DEC-044 Dodatok D (C): every worker test except the dedicated preview
    tests below disables preview rendering -- fast and hermetic, no real
    LibreOffice dependency; the real render path is covered end to end by
    tests/service/test_preview.py."""

    def _fail(*_args, **_kwargs):
        raise PreviewRenderFailed("previews disabled in this test")

    monkeypatch.setattr(worker_module, "render_source_preview", _fail)
    monkeypatch.setattr(worker_module, "render_docx_previews", _fail)


def _png() -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        value = kind + payload
        return struct.pack(">I", len(payload)) + value + struct.pack(">I", zlib.crc32(value))

    raw = b"\0\xff\0\0"
    return b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _report(page_count: int = 1) -> PdfAdmissionReport:
    return PdfAdmissionReport.from_mapping(
        {
            "schema_version": "1.2",
            "scanner_identity": (
                "unflatten-pdf-admission/qpdf-11.9.0-1.1ubuntu0.1-"
                "poppler-24.02.0-1ubuntu9.9/v13"
            ),
            "status": "accepted",
            "reason": None,
            "page_count": page_count,
            "page_width_points": 0.24,
            "page_height_points": 0.24,
            "raster_width_pixels": 1,
            "raster_height_pixels": 1,
            "raster_pixels": 1,
            "page_geometries": [
                {
                    "page_width_points": 0.24,
                    "page_height_points": 0.24,
                    "raster_width_pixels": 1,
                    "raster_height_pixels": 1,
                    "raster_pixels": 1,
                }
                for _ in range(page_count)
            ],
            "encrypted": False,
            "javascript_action_objects": 0,
            "attachment_objects": 0,
            "c2pa_metadata_objects": 0,
            "annotation_objects": 0,
            "text_visibility": "visible_only",
            "hidden_text_objects": 0,
            "ambiguous_text_objects": 0,
            "extreme_image_objects": 0,
            "object_count": 1,
        }
    )


class Sandbox:
    def __init__(
        self,
        calls: list[str],
        failure: str | None = None,
        *,
        report: PdfAdmissionReport | None = None,
    ) -> None:
        self.calls = calls
        self.failure = failure
        self.cancelled = False
        self._report = report

    def inspect(self, source: Path) -> PdfAdmissionReport:
        self.calls.append("inspect")
        if self.failure:
            raise PdfSandboxFailure(self.failure)
        return self._report if self._report is not None else _report()

    def raster(
        self, source: Path, report: PdfAdmissionReport, output: Path, *, page_index: int = 0
    ) -> None:
        self.calls.append("raster")
        output.write_bytes(_png())

    def cancel(self) -> None:
        self.cancelled = True


class Candidate:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def extract(self, model_input):
        self.calls.append("extract")
        return model_input

    def decode_features(self, prepared):
        self.calls.append("decode")
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
            terminated=True,
        )


def _store(tmp_path: Path) -> JobStore:
    policy = load_runtime_policy()
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:components",
        now=100.0,
    )
    return store


def _worker(tmp_path: Path, store: JobStore, calls: list[str]) -> DiagnosticWorker:
    return DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        model=DiagnosticModelRuntime(Candidate(calls)),
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )


@pytest.mark.parametrize(
    ("persisted_policy_hash", "persisted_component_hash", "expected_code"),
    [
        ("sha256:wrong-policy", "sha256:components", "RUNTIME_POLICY_MISMATCH"),
        (None, "sha256:wrong-components", "MODEL_COMPONENT_LOCK_MISMATCH"),
    ],
)
def test_worker_runtime_identity_mismatch_fails_before_processing_content_free(
    tmp_path: Path,
    persisted_policy_hash: str | None,
    persisted_component_hash: str,
    expected_code: str,
) -> None:
    policy = load_runtime_policy()
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="sensitive-request-digest",
        requested_output="prediction_ir",
        policy_hash=persisted_policy_hash or runtime_policy_hash(policy),
        component_lock_hash=persisted_component_hash,
        now=100.0,
    )
    calls: list[str] = []

    assert _worker(tmp_path, store, calls).run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "failed"
    assert job.error is not None
    assert job.error["code"] == expected_code
    assert "sensitive-request-digest" not in str(job.error)
    assert calls == []


def _source(tmp_path: Path) -> SourceArtifact:
    path = tmp_path / "source.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    return SourceArtifact(path=path, delete=lambda: path.unlink(missing_ok=True))


def test_worker_run_once_orders_stages_finalizes_diagnostic_and_cleans_inputs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    source = _source(tmp_path)
    calls: list[str] = []
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=artifacts,
        sandbox=Sandbox(calls),
        model=DiagnosticModelRuntime(Candidate(calls)),
        source_resolver=lambda _job: source,
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    with caplog.at_level(logging.INFO):
        assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None
    assert job.state == "succeeded"
    assert job.diagnostic_outcome == {
        "prediction_ir_status": "sealed",
        "claim_scope": "diagnostic",
        "canonical_ast_produced": False,
        "docx_produced": False,
        "reason_codes": ["DIAGNOSTIC_DOCX_NOT_REQUESTED"],
        "docx_chain": None,
    }
    assert calls == ["inspect", "raster", "extract", "decode"]
    assert not source.path.exists()
    assert not list((tmp_path / "work").glob("**/*.png"))
    published = store.list_artifacts(job.job_id)
    assert len(published) == 1
    assert artifacts.read_verified(
        principal_id=job.principal_id,
        job_id=job.job_id,
        role="prediction_ir",
        artifact=worker.artifact_ref(published[0]),
    ).startswith(b"{")
    assert "%PDF" not in caplog.text
    assert "n0" not in caplog.text


def test_worker_run_once_resolves_api_staged_source_from_artifact_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    staged = artifacts.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id="ingress_1234567890abcdef",
        role="source_pdf",
        mime_type="application/pdf",
        chunks=(b"%PDF-1.7\n",),
    )
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=artifacts,
        sandbox=Sandbox([]),
        model=DiagnosticModelRuntime(Candidate([])),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert not (artifacts.root / staged.ref).exists()


def test_worker_run_once_requeues_expired_leases_before_claiming(tmp_path: Path) -> None:
    class RecoveryStore(JobStore):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.recovery_calls: list[float] = []

        def requeue_expired(self, *, now: float) -> int:
            self.recovery_calls.append(now)
            return super().requeue_expired(now=now)

    policy = load_runtime_policy()
    store = RecoveryStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=policy["lease_duration_seconds"],
        max_attempts=policy["max_attempts"],
        global_active_jobs=policy["global_active_jobs"],
        active_jobs_per_principal=policy["active_jobs_per_principal"],
    )
    store.initialize()
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox([]),
        model=DiagnosticModelRuntime(Candidate([])),
        policy=policy,
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is False
    assert store.recovery_calls == [101.0]


@pytest.mark.parametrize(
    ("code", "expected_state"),
    [("POPPLER_TIMEOUT", "queued"), ("PDF_MALFORMED", "failed")],
)
def test_worker_run_once_retries_only_allowlisted_transient_failures(
    tmp_path: Path, code: str, expected_state: str
) -> None:
    store = _store(tmp_path)
    source = _source(tmp_path)
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox([], failure=code),
        model=DiagnosticModelRuntime(Candidate([])),
        source_resolver=lambda _job: source,
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == expected_state
    assert source.path.exists() is (expected_state == "queued")


def test_worker_run_once_cancellation_between_stages_fences_publication_and_cleans(
    tmp_path: Path,
) -> None:
    class CancellingStore(JobStore):
        def update_stage(self, lease, *, stage: str, now: float) -> None:
            super().update_stage(lease, stage=stage, now=now)
            if stage == "model_decode":
                self.request_cancel(lease.job_id, "principal_alpha", now=now)

    store = CancellingStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="prediction_ir",
        policy_hash=runtime_policy_hash(load_runtime_policy()),
        component_lock_hash="sha256:components",
        now=100.0,
    )
    source = _source(tmp_path)
    sandbox = Sandbox([])
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=sandbox,
        model=DiagnosticModelRuntime(Candidate([])),
        source_resolver=lambda _job: source,
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "cancelled"
    assert store.list_artifacts(job.job_id) == []
    assert sandbox.cancelled is True
    assert not source.path.exists()


def test_worker_source_has_no_renderer_or_promotion_import() -> None:
    worker_path = Path(__file__).parents[2] / "src/unflatten/service/worker.py"
    tree = ast.parse(worker_path.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported = " ".join(ast.unparse(node) for node in imports)
    assert "renderer" not in imported
    assert "ast_promotion" not in imported
    assert "compile_docx" not in worker_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# T5: diagnostic_docx chain (design D4/D18, G1/G4/G9)
# ---------------------------------------------------------------------------


class DocxCandidate:
    """A `candidate_runtime`-shaped fake carrying everything
    `diagnostic_docx_capability`/`build_diagnostic_docx` read off it."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.tokenizer = object()
        self.decode_policy = SimpleNamespace(
            max_steps=4608, loop_guard="v2", typed_decode=True, region_decode=True,
            selection_policy_hash="sha256:selection", prefill_chunk_tokens=None,
        )
        self.eval_cache_identity = {"dataset_release_root_hash": "sha256:release"}
        self.component_lock_hash = "sha256:lock"
        self.selection_policy_hash = "sha256:selection"
        self.vocabulary = object()
        self.raw = RawDocumentHypothesis(
            accepted_tokens=(1, 2),
            node_states=(
                NodeState(
                    local_node_id="n0", slot_kind="opened_node", node_type="document",
                    parent_local_node_id=None,
                ),
            ),
            terminated=True,
        )

    def extract(self, model_input):
        self.calls.append("extract")
        return SimpleNamespace(visual_features=None)

    def decode_features(self, prepared):
        self.calls.append("decode")
        return self.raw


def _expected_ir_hash(raw: RawDocumentHypothesis) -> str:
    from unflatten.model.prediction_ir import prediction_ir_hash
    from unflatten.service.model_runtime import seal_diagnostic_prediction

    return prediction_ir_hash(seal_diagnostic_prediction(raw))


def _store_diagnostic_docx(tmp_path: Path) -> JobStore:
    policy = load_runtime_policy()
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output="diagnostic_docx",
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:components",
        now=100.0,
    )
    return store


def _docx_row(*, docx_path: Path, policy_counts: dict | None = {}) -> SimpleNamespace:  # noqa: B006
    # The shape of `run.ChainRow` after DEC-044 Dodatok B: the chain carries the
    # counts of every applied policy (`{}` when a policy ran and edited nothing,
    # never None once it is on) and the derived AST profile (E4).
    return SimpleNamespace(
        page_outcome="terminated",
        policy_counts=policy_counts,
        geometry_counts={},
        story_dedup_counts={},
        leader_fill_counts={},
        leader_fill_v1_counts={},
        recovered_image_counts={},
        run_typography_snap_counts={},
        image_region_snap_counts={},
        tab_stops_counts={},
        profile="word-core-v0",
        envelope=SimpleNamespace(
            axes=SimpleNamespace(ast_status="strict_valid"),
            renderer=SimpleNamespace(status="render_ready", docx_path=docx_path),
        ),
    )


def _docx_worker(
    tmp_path: Path, store: JobStore, candidate: DocxCandidate, **overrides
) -> DiagnosticWorker:
    kwargs = dict(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(candidate.calls),
        model=DiagnosticModelRuntime(candidate),
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )
    kwargs.update(overrides)
    return DiagnosticWorker(**kwargs)


def test_worker_diagnostic_docx_publishes_both_artifacts_and_ratified_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)
    written_docx_paths: list[Path] = []

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        written_docx_paths.append(docx_path)
        return _docx_row(docx_path=docx_path, policy_counts={"default:run:/language": 1})

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.diagnostic_outcome["docx_produced"] is True
    assert job.diagnostic_outcome["canonical_ast_produced"] is True
    # PREVIEW_RENDER_FAILED: previews are disabled by the autouse fixture above.
    assert job.diagnostic_outcome["reason_codes"] == [
        "HEADLESS_DEFAULTS_V0_APPLIED", "PREVIEW_RENDER_FAILED",
    ]
    published = {a.role for a in store.list_artifacts(job.job_id)}
    assert published == {"prediction_ir", "diagnostic_docx"}
    # the work dir (and its .docx) is gone once the attempt finished
    assert not written_docx_paths[0].exists()


def test_worker_publishes_source_and_docx_previews_when_rendering_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)
    monkeypatch.setattr(worker_module, "render_source_preview", lambda *a, **k: b"src-preview")
    monkeypatch.setattr(
        worker_module, "render_docx_previews", lambda *a, **k: [b"docx-p0", b"docx-p1"]
    )

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert "PREVIEW_RENDER_FAILED" not in job.diagnostic_outcome["reason_codes"]
    by_key = {(a.role, a.page_index): a for a in store.list_artifacts(job.job_id)}
    assert set(by_key) == {
        ("prediction_ir", -1), ("diagnostic_docx", -1),
        ("source_preview", 0), ("docx_preview", 0), ("docx_preview", 1),
    }


def test_worker_docx_preview_failure_alone_does_not_fail_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A docx_preview render failure never fails the job, and never drops an
    already-succeeded source_preview -- each preview role is independent."""
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    def _raise(*_a, **_k):
        raise PreviewRenderFailed("boom")

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)
    monkeypatch.setattr(worker_module, "render_source_preview", lambda *a, **k: b"src-preview")
    monkeypatch.setattr(worker_module, "render_docx_previews", _raise)

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.diagnostic_outcome["docx_produced"] is True
    assert "PREVIEW_RENDER_FAILED" in job.diagnostic_outcome["reason_codes"]
    by_key = {(a.role, a.page_index) for a in store.list_artifacts(job.job_id)}
    assert by_key == {("prediction_ir", -1), ("diagnostic_docx", -1), ("source_preview", 0)}


def test_worker_diagnostic_docx_publish_many_failure_falls_back_to_ir_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)

    from unflatten.service import artifact_store as artifact_store_module

    real_publish_many = artifact_store_module.ArtifactStore.publish_many

    def failing_publish_many(self, staged, fence, **kwargs):
        if len(staged) == 2:
            raise ArtifactStoreError(
                "ARTIFACT_PUBLICATION_FAILED", "no space left", role="diagnostic_docx"
            )
        return real_publish_many(self, staged, fence, **kwargs)

    monkeypatch.setattr(artifact_store_module.ArtifactStore, "publish_many", failing_publish_many)

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.diagnostic_outcome["docx_produced"] is False
    assert job.diagnostic_outcome["reason_codes"] == ["DOCX_CHAIN_ERROR"]
    assert job.diagnostic_outcome["docx_chain"]["error_class"].startswith("publish_failed:")
    published = {a.role for a in store.list_artifacts(job.job_id)}
    assert published == {"prediction_ir"}
    downloaded = ArtifactStore(tmp_path / "artifacts").read_verified(
        principal_id=job.principal_id,
        job_id=job.job_id,
        role="prediction_ir",
        artifact=worker.artifact_ref(store.list_artifacts(job.job_id)[0]),
    )
    assert downloaded.startswith(b"{")


def test_worker_diagnostic_docx_staging_failure_falls_back_to_ir_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    monkeypatch.setattr(
        ds,
        "seal_page",
        lambda **_kw: SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw)),
    )

    def fake_process_model_page(_page, *, render_dir, **_kw):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(
        ds,
        "process_model_page",
        fake_process_model_page,
    )
    real_stage = ArtifactStore.stage

    def fail_docx_stage(self, **kwargs):
        if kwargs["role"] == "diagnostic_docx":
            raise ArtifactStoreError("ARTIFACT_WRITE_FAILED", "disk full", role="diagnostic_docx")
        return real_stage(self, **kwargs)

    monkeypatch.setattr(ArtifactStore, "stage", fail_docx_stage)

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.diagnostic_outcome["reason_codes"] == ["DOCX_CHAIN_ERROR"]
    assert {artifact.role for artifact in store.list_artifacts(job.job_id)} == {"prediction_ir"}


def test_worker_diagnostic_docx_heartbeats_after_the_chain_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = DocxCandidate([])
    heartbeats: list[float] = []
    clock_values = iter([101.0] * 1000)

    class RecordingStore(JobStore):
        def heartbeat(self, lease, *, now: float):
            heartbeats.append(now)
            return super().heartbeat(lease, now=now)

    recording_store = RecordingStore(
        tmp_path / "jobs2.sqlite3", lease_duration_seconds=900, max_attempts=2,
        global_active_jobs=4, active_jobs_per_principal=2,
    )
    recording_store.initialize()
    recording_store.create_job(
        job_id="job_1234567890abcdef", principal_id="principal_alpha",
        idempotency_key_hash="key", request_digest="digest",
        requested_output="diagnostic_docx", policy_hash=runtime_policy_hash(load_runtime_policy()),
        component_lock_hash="sha256:components", now=100.0,
    )
    worker = _docx_worker(tmp_path, recording_store, candidate, clock=lambda: next(clock_values))

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)

    heartbeats_before_finalize = None

    real_finalize = DiagnosticWorker._finalize

    def spying_finalize(self, lease, job, artifact, docx_result, png_bytes):
        nonlocal heartbeats_before_finalize
        heartbeats_before_finalize = len(heartbeats)
        return real_finalize(self, lease, job, artifact, docx_result, png_bytes)

    monkeypatch.setattr(DiagnosticWorker, "_finalize", spying_finalize)

    assert worker.run_once() is True
    # `ast_promotion`'s own `_run_stage` heartbeat, then `renderer_gate`'s
    # `_enter_stage` heartbeat, both happened before `_finalize` ran.
    assert heartbeats_before_finalize is not None and heartbeats_before_finalize >= 2


def test_worker_diagnostic_docx_fence_lost_during_chain_is_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CancellingStore(JobStore):
        def update_stage(self, lease, *, stage: str, now: float) -> None:
            super().update_stage(lease, stage=stage, now=now)
            if stage == "ast_promotion":
                self.request_cancel(lease.job_id, "principal_alpha", now=now)

    store = CancellingStore(
        tmp_path / "jobs.sqlite3", lease_duration_seconds=900, max_attempts=2,
        global_active_jobs=4, active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef", principal_id="principal_alpha",
        idempotency_key_hash="key", request_digest="digest",
        requested_output="diagnostic_docx", policy_hash=runtime_policy_hash(load_runtime_policy()),
        component_lock_hash="sha256:components", now=100.0,
    )
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "cancelled"
    assert store.list_artifacts(job.job_id) == []


@pytest.mark.parametrize(
    ("row_outcome", "reason"),
    [
        ("PageStreamInvalid", "DOCX_PAGE_STREAM_INVALID"),
        ("max_steps_exceeded", "DOCX_PAGE_NOT_TERMINATED"),
    ],
)
def test_worker_diagnostic_docx_chain_outcome_still_publishes_ir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row_outcome: str, reason: str
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(
        ds,
        "process_model_page",
        lambda page, *, render_dir, **kwargs: SimpleNamespace(
            # a non-terminated page: the policies ran before the outcome check
            # and edited nothing, so every count field is `{}` (never None)
            page_outcome=row_outcome, policy_counts={}, envelope=None,
            geometry_counts={}, story_dedup_counts={}, leader_fill_counts={},
            leader_fill_v1_counts={}, run_typography_snap_counts={},
            image_region_snap_counts={}, tab_stops_counts={},
            recovered_image_counts={},
            profile="word-core-v0",
        ),
    )

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.diagnostic_outcome["docx_produced"] is False
    assert job.diagnostic_outcome["reason_codes"] == [reason]
    published = {a.role for a in store.list_artifacts(job.job_id)}
    assert published == {"prediction_ir"}


def test_worker_diagnostic_docx_chain_exception_still_publishes_ir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    def boom(**kwargs):
        raise RuntimeError("chain exploded")

    monkeypatch.setattr(ds, "seal_page", boom)

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.diagnostic_outcome["docx_produced"] is False
    assert job.diagnostic_outcome["reason_codes"] == ["DOCX_CHAIN_ERROR"]
    assert "chain exploded" not in str(job.diagnostic_outcome)


def test_worker_constructor_refuses_when_diagnostic_docx_capability_is_false(
    tmp_path: Path,
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    with pytest.raises(ValueError, match="diagnostic_docx_capability"):
        _docx_worker(tmp_path, store, DocxCandidate([]), diagnostic_docx_capable=False)


def test_worker_logs_are_content_free_and_include_stage_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store_diagnostic_docx(tmp_path)
    candidate = DocxCandidate([])
    worker = _docx_worker(tmp_path, store, candidate)

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)

    with caplog.at_level(logging.INFO):
        assert worker.run_once() is True

    assert "worker_stage_done job_id=job_1234567890abcdef stage=ast_promotion" in caplog.text
    assert "docx-bytes" not in caplog.text
    assert "canonical_ast" not in caplog.text


def _admitted(width: int, height: int) -> PdfAdmissionReport:
    base = _report()
    geometry = dataclasses.replace(
        base.page_geometries[0],
        page_width_points=width * 72 / 300,
        page_height_points=height * 72 / 300,
        raster_width_pixels=width,
        raster_height_pixels=height,
        raster_pixels=width * height,
    )
    return dataclasses.replace(
        base,
        page_geometries=(geometry,),
        page_width_points=geometry.page_width_points,
        page_height_points=geometry.page_height_points,
        raster_width_pixels=width,
        raster_height_pixels=height,
        raster_pixels=width * height,
    )


def _png_file(tmp_path: Path, width: int, height: int) -> Path:
    from PIL import Image

    path = tmp_path / f"r-{width}x{height}.png"
    Image.new("RGB", (width, height), "white").save(path)
    return path


@pytest.mark.parametrize(("width", "height", "ok"),
                         [(4, 5, True), (5, 5, True), (6, 4, True), (7, 5, False), (5, 3, False)])
def test_load_rgb_accepts_poppler_rounding_within_one_pixel(tmp_path, width, height, ok):
    """Same one-pixel tolerance as `PdfSandbox.raster` (26.9. prod: a 595.44 pt
    page is admitted as 2482 px and rendered as 2481 px)."""
    worker = _worker(tmp_path, _store(tmp_path), [])
    report = _admitted(5, 5)
    path = _png_file(tmp_path, width, height)
    if ok:
        got_w, got_h, _, _ = worker._load_rgb(path, report)
        assert (got_w, got_h) == (width, height)
    else:
        with pytest.raises(WorkerFailure):
            worker._load_rgb(path, report)


class _ExplodingCandidate(Candidate):
    def decode_features(self, prepared):
        self.calls.append("decode")
        raise ValueError("secret page text: Jana Novakova, rodne cislo 123456/7890")


def test_worker_internal_error_logs_exception_type_and_stage_without_its_message(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """S5 control run 2026-09-26: two photo jobs failed as INTERNAL_ERROR after
    six minutes of decoding and the worker log said nothing about why. The
    handler must name the exception type, and must still never log the
    exception's message, which can carry document text (the service keeps
    job content out of its logs).

    T17 update: a page's own decode error is now caught inside
    `engine.decode_pages` (`NativeEngine`, design §17.3) rather than
    `run_once`'s own except-ladder -- the content-free warning log
    (`native_engine_page_internal_error`, still no message, still names the
    exception type) moved with it. `run_once` still fails the whole
    single-page job with `INTERNAL_ERROR` via the generic `WorkerFailure`
    path."""
    store = _store(tmp_path)
    calls: list[str] = []
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        model=DiagnosticModelRuntime(_ExplodingCandidate(calls)),
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    with caplog.at_level(logging.INFO):
        assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "failed"
    assert job.error is not None and job.error["code"] == "INTERNAL_ERROR"
    assert "native_engine_page_internal_error job_id=job_1234567890abcdef" in caplog.text
    assert "exc_type=ValueError" in caplog.text
    assert "Novakova" not in caplog.text
    assert "123456/7890" not in caplog.text


class _StepLimitCandidate(Candidate):
    def decode_features(self, prepared):
        self.calls.append("decode")
        raise DecodeMaxStepsExceeded("free-running decode did not terminate within 4096 steps")


def test_worker_decode_max_steps_exceeded_is_public_not_internal_error(
    tmp_path: Path,
) -> None:
    """26.9. addendum: a truncated free-running decode is a deterministic,
    content-free failure of its own and must not fall into the generic
    (OSError, ValueError, RuntimeError) INTERNAL_ERROR branch (S5 control
    run 2026-09-26 lost two photo jobs to INTERNAL_ERROR this way)."""
    store = _store(tmp_path)
    calls: list[str] = []
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        model=DiagnosticModelRuntime(_StepLimitCandidate(calls)),
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "failed"
    assert job.error is not None
    assert job.error["code"] == "MODEL_DECODE_MAX_STEPS"
    assert job.error["retryable"] is False


# ---------------------------------------------------------------------------
# B1: multi-page PDF (DEC-044 Dodatok D §1)
# ---------------------------------------------------------------------------


class MultiPageSandbox(Sandbox):
    """Rasters every page unless `fail_page_index` matches, then that ONE
    page's raster call raises -- the rest of the job must still proceed."""

    def __init__(
        self, calls: list[str], *, page_count: int, fail_page_index: int | None = None
    ) -> None:
        super().__init__(calls, report=_report(page_count))
        self._fail_page_index = fail_page_index

    def raster(
        self, source: Path, report: PdfAdmissionReport, output: Path, *, page_index: int = 0
    ) -> None:
        if page_index == self._fail_page_index:
            self.calls.append("raster_failed")
            raise PdfSandboxFailure("RASTER_OUTPUT_INVALID")
        self.calls.append(f"raster:{page_index}")
        output.write_bytes(_png())


def _store_multi_page(tmp_path: Path, *, requested_output: str = "diagnostic_docx") -> JobStore:
    policy = load_runtime_policy()
    store = JobStore(
        tmp_path / "jobs.sqlite3",
        lease_duration_seconds=900,
        max_attempts=2,
        global_active_jobs=4,
        active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef",
        principal_id="principal_alpha",
        idempotency_key_hash="key",
        request_digest="digest",
        requested_output=requested_output,
        policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:components",
        now=100.0,
    )
    return store


def _multi_page_worker(
    tmp_path: Path, store: JobStore, candidate: DocxCandidate, sandbox: Sandbox, **overrides
) -> DiagnosticWorker:
    kwargs = dict(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=sandbox,
        model=DiagnosticModelRuntime(candidate),
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )
    kwargs.update(overrides)
    return DiagnosticWorker(**kwargs)


def test_worker_multi_page_decodes_every_page_and_merges_into_one_docx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_multi_page(tmp_path)
    candidate = DocxCandidate([])
    sandbox = MultiPageSandbox(candidate.calls, page_count=2)
    worker = _multi_page_worker(tmp_path, store, candidate, sandbox)
    seen_pages: list = []

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_pages(pages, *, render_dir, **kwargs):
        seen_pages.extend(pages)
        assert set(kwargs["page_rasters"]) == {0, 1}
        assert all(data.startswith(b"\x89PNG") for data in kwargs["page_rasters"].values())
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"multi-docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_pages", fake_process_model_pages)

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.pages_total == 2
    assert job.pages_done == 2
    assert [p["state"] for p in job.pages] == ["succeeded", "succeeded"]
    assert len(seen_pages) == 2
    published = store.list_artifacts(job.job_id)
    by_key = {(a.role, a.page_index) for a in published}
    assert by_key == {("prediction_ir", 0), ("prediction_ir", 1), ("diagnostic_docx", -1)}
    assert sandbox.calls.count("raster:0") == 1
    assert sandbox.calls.count("raster:1") == 1


def test_worker_multi_page_one_failed_page_never_drops_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store_multi_page(tmp_path)
    candidate = DocxCandidate([])
    sandbox = MultiPageSandbox(candidate.calls, page_count=2, fail_page_index=1)
    worker = _multi_page_worker(tmp_path, store, candidate, sandbox)

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_pages(pages, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"multi-docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_pages", fake_process_model_pages)

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert [p["state"] for p in job.pages] == ["succeeded", "failed"]
    assert job.pages[1]["error_code"] == "RASTER_OUTPUT_INVALID"
    # only the succeeded page publishes its own prediction_ir
    published = store.list_artifacts(job.job_id)
    by_key = {(a.role, a.page_index) for a in published}
    assert by_key == {("prediction_ir", 0), ("diagnostic_docx", -1)}


def test_worker_multi_page_prediction_ir_only_publishes_one_per_page(
    tmp_path: Path,
) -> None:
    store = _store_multi_page(tmp_path, requested_output="prediction_ir")
    calls: list[str] = []
    sandbox = MultiPageSandbox(calls, page_count=3)
    worker = _multi_page_worker(tmp_path, store, Candidate(calls), sandbox)

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.pages_total == 3
    published = store.list_artifacts(job.job_id)
    assert {(a.role, a.page_index) for a in published} == {
        ("prediction_ir", 0), ("prediction_ir", 1), ("prediction_ir", 2),
    }


# ---------------------------------------------------------------------------
# DEC-044 Dodatok D §1 fix (2026-09-27 field report): live model switching
# ---------------------------------------------------------------------------


class _FakeSwitchingProvider:
    """A `ModelProvider` fake that records every `ensure_loaded` call and
    returns a distinct `Candidate`-backed runtime per requested variant."""

    def __init__(self, calls: list[str]) -> None:
        self.requests: list[str | None] = []
        self._resident: str | None = None
        self._runtimes: dict[str | None, DiagnosticModelRuntime] = {}
        self._calls = calls

    def _runtime_for(self, variant_id: str | None) -> DiagnosticModelRuntime:
        if variant_id not in self._runtimes:
            self._runtimes[variant_id] = DiagnosticModelRuntime(Candidate(self._calls))
        return self._runtimes[variant_id]

    def ensure_loaded(self, variant_id):
        self.requests.append(variant_id)
        if variant_id is not None:
            self._resident = variant_id
        return self._runtime_for(self._resident)


def test_worker_constructor_requires_exactly_one_of_model_or_model_provider(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    kwargs = dict(
        worker_id="worker_gpu_0",
        jobs=_store(tmp_path),
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )
    with pytest.raises(ValueError, match="exactly one"):
        DiagnosticWorker(**kwargs)
    with pytest.raises(ValueError, match="exactly one"):
        DiagnosticWorker(
            model=DiagnosticModelRuntime(Candidate(calls)),
            model_provider=_FakeSwitchingProvider(calls),
            **kwargs,
        )


def test_worker_calls_ensure_loaded_with_the_job_model_variant_each_run(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    store = _store(tmp_path)
    provider = _FakeSwitchingProvider(calls)
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        model_provider=provider,
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True

    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job.state == "succeeded"
    # index 0 is the constructor's own initial `ensure_loaded(None)` call;
    # index 1 is run_once's own call for a job with no configured variant.
    assert provider.requests == [None, None]
    assert "loading_model" not in calls  # stage names never leak into model calls


def test_worker_switches_model_when_a_job_names_a_different_variant(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    store = JobStore(
        tmp_path / "jobs.sqlite3", lease_duration_seconds=900, max_attempts=2,
        global_active_jobs=4, active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_variant_a", principal_id="principal_alpha", idempotency_key_hash="key-a",
        request_digest="digest-a", requested_output="prediction_ir",
        policy_hash=runtime_policy_hash(load_runtime_policy()),
        component_lock_hash="sha256:components", now=100.0, model_variant="fp8",
    )
    store.create_job(
        job_id="job_variant_b", principal_id="principal_alpha", idempotency_key_hash="key-b",
        request_digest="digest-b", requested_output="prediction_ir",
        policy_hash=runtime_policy_hash(load_runtime_policy()),
        component_lock_hash="sha256:components", now=100.0, model_variant="int4",
    )
    provider = _FakeSwitchingProvider(calls)
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        model_provider=provider,
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True
    assert worker.run_once() is True

    # index 0 is the constructor's own initial `ensure_loaded(None)` call.
    assert provider.requests == [None, "fp8", "int4"]
    for job_id in ("job_variant_a", "job_variant_b"):
        job = store.get_job(job_id, "principal_alpha")
        assert job.state == "succeeded"


# ---------------------------------------------------------------------------
# T17: `DiagnosticWorker` over `engine.decode_pages` (design §17.3/§17.4)
# ---------------------------------------------------------------------------


class _StageRecordingStore(JobStore):
    """Records every `stage` argument passed to `update_stage`, in order."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stages: list[str] = []

    def update_stage(self, lease, *, stage: str, now: float) -> None:
        self.stages.append(stage)
        super().update_stage(lease, stage=stage, now=now)


def _stage_store(tmp_path: Path) -> _StageRecordingStore:
    policy = load_runtime_policy()
    store = _StageRecordingStore(
        tmp_path / "jobs.sqlite3", lease_duration_seconds=900, max_attempts=2,
        global_active_jobs=4, active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef", principal_id="principal_alpha",
        idempotency_key_hash="key", request_digest="digest",
        requested_output="prediction_ir", policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:components", now=100.0,
    )
    return store


def test_stage_sequence_single_page_matches_baseline(tmp_path: Path) -> None:
    """(T17, design §17.4): a one-page job is `engine.decode_pages` called
    with exactly one `PageInput` -- the outer `model_decode` `_run_stage`
    (heartbeat bookkeeping) wraps `NativeEngine`'s own per-page
    `feature_extraction`/`model_decode` callbacks, which still wrap today's
    `extract`/`decode_features` byte-for-byte (design §17.3)."""
    store = _stage_store(tmp_path)
    calls: list[str] = []
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox(calls),
        model=DiagnosticModelRuntime(Candidate(calls)),
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert "extract" in calls and "decode" in calls
    assert store.stages == [
        "loading_model",
        "pdf_inspection",
        "pdf_rasterization",
        "png_validation",
        "model_input_projection",
        "model_decode",
        "feature_extraction",
        "model_decode",
        "prediction_ir_sealing",
        "artifact_finalization",
    ]


def test_stage_sequence_three_pages_matches_baseline(tmp_path: Path) -> None:
    """(T17): three pages rasterize/validate/project first, engine-agnostic
    and page-by-page (`PdfSandbox.raster(page_index=...)`, design §17.4),
    THEN one `engine.decode_pages` call decodes all three in order -- a
    documented, DECIDED change from today's per-page interleaving (raster
    then decode ONE page at a time): merging `_execute`/`_execute_multi`
    into a single `engine.decode_pages` call per job (§17.4) means the raster
    phase for every page precedes the decode phase for any page. The set and
    per-page order of stage KINDS is unchanged; only their grouping is
    batched."""
    policy = load_runtime_policy()
    store = _StageRecordingStore(
        tmp_path / "jobs.sqlite3", lease_duration_seconds=900, max_attempts=2,
        global_active_jobs=4, active_jobs_per_principal=2,
    )
    store.initialize()
    store.create_job(
        job_id="job_1234567890abcdef", principal_id="principal_alpha",
        idempotency_key_hash="key", request_digest="digest",
        requested_output="prediction_ir", policy_hash=runtime_policy_hash(policy),
        component_lock_hash="sha256:components", now=100.0,
    )
    calls: list[str] = []
    sandbox = MultiPageSandbox(calls, page_count=3)
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=sandbox,
        model=DiagnosticModelRuntime(Candidate(calls)),
        source_resolver=lambda _job: _source(tmp_path),
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert store.stages == [
        "loading_model",
        "pdf_inspection",
        "pdf_rasterization", "png_validation", "model_input_projection",
        "pdf_rasterization", "png_validation", "model_input_projection",
        "pdf_rasterization", "png_validation", "model_input_projection",
        "model_decode",
        "feature_extraction", "model_decode",
        "feature_extraction", "model_decode",
        "feature_extraction", "model_decode",
        "prediction_ir_sealing", "prediction_ir_sealing", "prediction_ir_sealing",
        "artifact_finalization",
    ]


def test_single_page_status_page_fields_null(tmp_path: Path) -> None:
    """(T17): the merged `_run_pages` only calls `update_page_progress` when
    `pages_total > 1` -- a single-page job's `pages_total`/`pages_done`/
    `current_page`/`pages` stay `None`, unchanged from today's contract."""
    store = _store(tmp_path)
    worker = _worker(tmp_path, store, [])

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.pages_total is None
    assert job.pages_done is None
    assert job.current_page is None
    assert job.pages is None


def test_worker_runs_over_fake_engine(tmp_path: Path) -> None:
    """(T17, design §13/§17.2): `DiagnosticWorker` drives `FakeEngine`
    exactly like `NativeEngine` -- via `model=`, duck-typed on
    `decode_pages` (`_as_engine`) -- with no GPU, no torch import beyond
    what the test harness itself already needs."""
    store = _store(tmp_path)
    source = _source(tmp_path)
    engine = FakeEngine(fail_pages=frozenset({("job_1234567890abcdef", 0)}))
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=store,
        artifacts=ArtifactStore(tmp_path / "artifacts"),
        sandbox=Sandbox([]),
        model=engine,
        source_resolver=lambda _job: source,
        policy=load_runtime_policy(),
        component_lock_hash="sha256:components",
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "failed"
    assert job.error is not None and job.error["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize("failure", ["processing", "policy_report"])
def test_multi_page_word_exception_preserves_every_sealed_ir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    store = _store_multi_page(tmp_path)
    candidate = DocxCandidate([])
    worker = _multi_page_worker(
        tmp_path, store, candidate, MultiPageSandbox(candidate.calls, page_count=2)
    )
    monkeypatch.setattr(
        ds, "seal_page", lambda **_: SimpleNamespace(
            prediction_ir_hash=_expected_ir_hash(candidate.raw)
        )
    )

    def failing_chain(*args, **kwargs):
        if failure == "processing":
            raise OSError("private-work-path must never appear in public status")
        row = _docx_row(docx_path=tmp_path / "missing.docx")
        row.run_typography_snap_counts = None
        return row

    monkeypatch.setattr(ds, "process_model_pages", failing_chain)
    assert worker.run_once() is True
    job = store.get_job("job_1234567890abcdef", "principal_alpha")
    assert job is not None and job.state == "succeeded"
    assert job.pages_done == job.pages_total == 2
    assert [page["state"] for page in job.pages] == ["succeeded", "succeeded"]
    outcome = job.diagnostic_outcome
    assert outcome["docx_produced"] is False
    assert outcome["reason_codes"] == ["DOCX_CHAIN_ERROR"]
    assert outcome["docx_chain"]["error_class"] == (
        "OSError" if failure == "processing" else "ValueError"
    )
    assert "private-work-path" not in str(outcome)
    artifacts = store.list_artifacts(job.job_id)
    assert {(a.role, a.page_index) for a in artifacts} == {
        ("prediction_ir", 0), ("prediction_ir", 1)
    }
    for artifact in artifacts:
        data = ArtifactStore(tmp_path / "artifacts").read_verified(
            principal_id=job.principal_id, job_id=job.job_id, role="prediction_ir",
            artifact=worker.artifact_ref(artifact),
        )
        assert data.startswith(b"{")
