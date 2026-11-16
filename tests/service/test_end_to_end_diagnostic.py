from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.service.test_api_endpoints import AUTH, _client, _submit
from tests.service.test_worker import (
    Candidate,
    DocxCandidate,
    Sandbox,
    _docx_row,
    _expected_ir_hash,
)

from unflatten.service import docx_stage as ds
from unflatten.service import worker as worker_module
from unflatten.service.model_runtime import DiagnosticModelRuntime
from unflatten.service.preview import PreviewRenderFailed
from unflatten.service.worker import DiagnosticWorker


def _fail_preview(*_args: object, **_kwargs: object) -> None:
    raise PreviewRenderFailed("previews disabled in this test")


def test_api_and_worker_stubbed_path_publishes_authorized_diagnostic_ir(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    submitted = _submit(client)
    dependencies = client.app.state.dependencies
    calls: list[str] = []
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=dependencies.job_store,
        artifacts=dependencies.artifact_store,
        sandbox=Sandbox(calls),
        model=DiagnosticModelRuntime(Candidate(calls)),
        policy=dependencies.runtime_policy,
        component_lock_hash=dependencies.component_lock_hash,
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    assert submitted.status_code == 202
    assert worker.run_once() is True

    status = client.get(submitted.json()["status_url"], headers=AUTH)
    downloaded = client.get(
        f"{submitted.json()['status_url']}/artifacts/prediction_ir", headers=AUTH
    )

    assert status.status_code == 200
    assert status.json()["state"] == "succeeded"
    assert status.json()["diagnostic_outcome"] == {
        "prediction_ir_status": "sealed",
        "claim_scope": "diagnostic",
        "canonical_ast_produced": False,
        "docx_produced": False,
        "reason_codes": ["DIAGNOSTIC_DOCX_NOT_REQUESTED"],
        "docx_chain": None,
    }
    assert status.json()["artifacts"] == ["prediction_ir"]
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "application/json"
    fragments = downloaded.json()["document_hypotheses"][0]["fragments"]
    assert fragments[0]["scope"] == {"kind": "single_page"}
    assert calls == ["inspect", "raster", "extract", "decode"]


def test_api_and_worker_diagnostic_docx_path_publishes_both_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path)
    submitted = _submit(client, **{"X-Unflatten-Output": "diagnostic_docx"})
    dependencies = client.app.state.dependencies
    candidate = DocxCandidate([])
    worker = DiagnosticWorker(
        worker_id="worker_gpu_0",
        jobs=dependencies.job_store,
        artifacts=dependencies.artifact_store,
        sandbox=Sandbox(candidate.calls),
        model=DiagnosticModelRuntime(candidate),
        policy=dependencies.runtime_policy,
        component_lock_hash=dependencies.component_lock_hash,
        work_root=tmp_path / "work",
        clock=lambda: 101.0,
    )

    def fake_seal_page(**kwargs):
        return SimpleNamespace(prediction_ir_hash=_expected_ir_hash(candidate.raw))

    def fake_process_model_page(page, *, render_dir, **kwargs):
        docx_path = Path(render_dir) / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        return _docx_row(docx_path=docx_path)

    monkeypatch.setattr(ds, "seal_page", fake_seal_page)
    monkeypatch.setattr(ds, "process_model_page", fake_process_model_page)
    # DEC-044 Dodatok D (C): previews are exercised end to end in
    # tests/service/test_preview.py and tests/service/test_worker.py;
    # disabled here so this test's assertions stay about the docx path only.
    monkeypatch.setattr(worker_module, "render_source_preview", _fail_preview)
    monkeypatch.setattr(worker_module, "render_docx_previews", _fail_preview)

    assert submitted.status_code == 202
    assert worker.run_once() is True

    status = client.get(submitted.json()["status_url"], headers=AUTH)
    downloaded_docx = client.get(
        f"{submitted.json()['status_url']}/artifacts/diagnostic_docx", headers=AUTH
    )

    assert status.status_code == 200
    assert status.json()["state"] == "succeeded"
    assert status.json()["artifacts"] == ["prediction_ir", "diagnostic_docx"]
    assert status.json()["diagnostic_outcome"]["docx_produced"] is True
    assert status.json()["diagnostic_outcome"]["reason_codes"] == [
        "HEADLESS_DEFAULTS_V0_APPLIED", "PREVIEW_RENDER_FAILED",
    ]
    assert downloaded_docx.status_code == 200
    assert downloaded_docx.headers["content-type"] == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert downloaded_docx.content == b"docx-bytes"


def test_restarted_worker_reclaims_expired_lease_and_fences_stale_attempt(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    submitted = _submit(client)
    jobs = client.app.state.dependencies.job_store
    stale = jobs.claim_next(worker_id="worker_before_restart", now=100.0)
    assert stale is not None

    assert jobs.requeue_expired(now=1001.0) == 1
    current = jobs.claim_next(worker_id="worker_after_restart", now=1001.0)

    assert current is not None
    assert current.job_id == submitted.json()["job_id"]
    assert current.lease_epoch == stale.lease_epoch + 1
    assert current.execution_id != stale.execution_id
