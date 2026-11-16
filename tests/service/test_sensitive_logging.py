from __future__ import annotations

import logging
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
from tests.service.test_api_endpoints import _client, _submit
from tests.service.test_model_commands import _executor, _hf_variant, make_store
from tests.service.test_worker import Candidate, Sandbox

from unflatten.service import model_commands as mc
from unflatten.service.api import WebSettings
from unflatten.service.model_bundle import ModelBundleInvalid
from unflatten.service.model_runtime import DiagnosticModelRuntime
from unflatten.service.worker import DiagnosticWorker

_WEB_HEADERS = {
    "Content-Type": "application/json",
    "X-Unflatten-Client": "web",
    "Sec-Fetch-Site": "same-origin",
}


def test_api_and_worker_logs_exclude_sensitive_request_and_model_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    token = "a" * 32
    filename = "board-acquisition-secret.pdf"
    metadata = "Author=Highly Sensitive Person"
    document_text = "CONFIDENTIAL merger amount 123456789"
    subprocess_payload = "qpdf stderr contained customer payload"
    # DEC-052 design §3.2 (T25): the user-supplied label is never logged.
    label = "quarterly-earnings-embargoed.pdf"
    pdf = f"%PDF-1.7\n% {filename} {metadata} {document_text} {subprocess_payload}".encode()
    client = _client(tmp_path)

    with caplog.at_level(logging.INFO):
        submitted = _submit(
            client, body=pdf,
            **{"X-Unflatten-Label": urllib.parse.quote(label, safe="")},
        )
        dependencies = client.app.state.dependencies
        worker = DiagnosticWorker(
            worker_id="worker_gpu_0",
            jobs=dependencies.job_store,
            artifacts=dependencies.artifact_store,
            sandbox=Sandbox([]),
            model=DiagnosticModelRuntime(Candidate([])),
            policy=dependencies.runtime_policy,
            component_lock_hash=dependencies.component_lock_hash,
            work_root=tmp_path / "work",
            clock=lambda: 101.0,
        )
        assert worker.run_once() is True

    assert submitted.status_code == 202
    for sensitive in (
        token,
        filename,
        metadata,
        document_text,
        subprocess_payload,
        label,
        "%PDF",
        "PNG",
        "accepted_tokens",
        "canonical_ast",
    ):
        assert sensitive not in caplog.text


def test_web_session_logs_exclude_password_and_cookie_token(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """T21 (design §11 bullet 6/2): a setup/login password, and the raw
    session cookie token, must never appear in a log line."""
    password = "correct horse battery staple secret"
    client = _client(tmp_path, web=WebSettings(web_auth_path=tmp_path / "web-auth.json"))

    with caplog.at_level(logging.INFO):
        setup = client.post(
            "/v1/setup",
            json={"password": password},
            headers={"Authorization": "Bearer " + "a" * 32, **_WEB_HEADERS},
        )
        assert setup.status_code == 201
        token = client.cookies["unflatten_session"]
        whoami = client.get("/v1/session", headers={"Sec-Fetch-Site": "same-origin"})
        assert whoami.status_code == 200

    assert password not in caplog.text
    assert token not in caplog.text


def test_model_command_executor_never_logs_hf_token_url(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T42 (design §4.2/§4.3): a Hugging Face URL carrying a token can appear
    in a `snapshot_download`/`verify_bundle` failure message -- `_run_download`
    must only ever log the exception TYPE (`model_commands.py::_run_download`),
    never the message, so the token can never reach the log."""
    token_url = (
        "https://huggingface.co/api/models/secret/resolve/main/x?token=hf_SUPERSECRET1234567890"
    )
    jobs = make_store(tmp_path)
    variant = _hf_variant()
    variant_id = variant["variant_id"]

    def fake_snapshot_download(**kwargs: Any) -> str:
        return str(tmp_path / "snapshot")

    def fake_verify_bundle(bundle_dir: Path, **kwargs: Any) -> dict:
        raise ModelBundleInvalid(f"manifest mismatch, source={token_url}")

    monkeypatch.setattr(mc, "_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(mc, "verify_bundle", fake_verify_bundle)
    monkeypatch.setattr(mc, "delete_variant_snapshot", lambda *args, **kwargs: None)

    jobs.enqueue_model_command(kind="download", variant_id=variant_id, requested_by="p", now=0.0)
    executor = _executor(jobs, variants={variant_id: variant}, cache_dir=tmp_path)

    with caplog.at_level(logging.INFO):
        assert executor.run_once() is True

    assert jobs.last_model_command(variant_id, "download")["state"] == "failed"
    assert token_url not in caplog.text
    assert "hf_SUPERSECRET1234567890" not in caplog.text
