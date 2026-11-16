"""Exercise the standalone example against a synthetic HTTP service, without a GPU."""

import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import UUID

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "examples" / "headless_client.py"
TOKEN = "synthetic_headless_client_credential_12345678"
WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
WORD_BYTES = b"PK\x03\x04synthetic-word\x00\xff"


def status_payload(state="succeeded", *, docx=True):
    return {
        "schema_version": "1.1",
        "job_id": "job_example123",
        "state": state,
        "stage": "artifact_finalization",
        "attempt": 1,
        "requested_output": "diagnostic_docx",
        "claim_scope": "diagnostic",
        "product_quality_claim": False,
        "diagnostic_outcome": {"docx_produced": docx},
        "artifacts": [
            {
                "role": "diagnostic_docx",
                "page_index": None,
                "url": "/v1/conversions/job_example123/artifacts/diagnostic_docx",
            }
        ]
        if docx
        else [],
        "error": {"code": "INTERNAL_ERROR"} if state == "failed" else None,
        "model_variant": "dense-1.0-fp8-test",
        "pages_total": 1,
        "pages_done": 1,
        "current_page": None,
        "pages": [],
        "label": None,
        "input_kind": "pdf",
        "engine": "native",
        "created_at": "2026-11-16T10:00:00.000Z",
        "updated_at": "2026-11-16T10:00:01.000Z",
        "terminal_at": "2026-11-16T10:00:01.000Z",
        "expires_at": None,
    }


@contextmanager
def synthetic_api(
    *,
    statuses=None,
    submit_error=None,
    redirect=False,
    truncated=False,
    broken_chunked=False,
    dribble=None,
):
    requests = []
    statuses = statuses or [status_payload()]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, code, body, content_type="application/json", **headers):
            if dribble == "headers" and self.command == "GET":
                self.wfile.write(b"HTTP/1.0 200 OK\r\nX-Slow: ")
                if not self.write_slowly(b"a" * 22):
                    return
                self.wfile.write(
                    (
                        f"\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\n\r\n"
                    ).encode()
                    + body
                )
                return
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            extra = 12 if truncated and content_type == WORD_MIME else 0
            self.send_header("Content-Length", str(len(body) + extra))
            for name, value in headers.items():
                self.send_header(name, value)
            self.end_headers()
            if (dribble == "status" and self.command == "GET" and content_type != WORD_MIME) or (
                dribble == "word" and content_type == WORD_MIME
            ):
                self.write_slowly(body)
                return
            self.wfile.write(body)

        def write_slowly(self, body):
            chunk_size = max(1, len(body) // 22)
            try:
                for offset in range(0, len(body), chunk_size):
                    self.wfile.write(body[offset : offset + chunk_size])
                    self.wfile.flush()
                    time.sleep(0.04)
            except (BrokenPipeError, ConnectionResetError):
                return False
            return True

        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(("POST", self.path, dict(self.headers), body))
            if submit_error:
                self.reply(
                    submit_error,
                    json.dumps(
                        {
                            "code": "AUTHENTICATION_FAILED",
                            "detail": TOKEN,
                        }
                    ).encode(),
                )
                return
            if redirect:
                self.reply(307, b"{}", Location="/redirected")
                return
            self.reply(
                202,
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "job_id": "job_example123",
                        "state": "queued",
                        "requested_output": "diagnostic_docx",
                        "claim_scope": "diagnostic",
                        "product_quality_claim": False,
                        "status_url": "http://untrusted.invalid/do-not-forward-credentials",
                    }
                ).encode(),
            )

        def do_GET(self):
            requests.append(("GET", self.path, dict(self.headers), b""))
            if self.path.endswith("/artifacts/diagnostic_docx"):
                if broken_chunked:
                    self.send_response(200)
                    self.send_header("Content-Type", WORD_MIME)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    self.wfile.write(b"10\r\nshort")
                    return
                self.reply(200, WORD_BYTES, WORD_MIME)
                return
            current = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            self.reply(200, json.dumps(current).encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/proxy", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_client(tmp_path, base_url, *, suffix=".pdf", token=TOKEN, **settings):
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"synthetic-source\x00\xff")
    config = tmp_path / "request.json"
    config.write_text(
        json.dumps(
            {
                "base_url": base_url,
                "input": source.name,
                "output": "result.docx",
                "poll_interval": 0.01,
                "timeout": 2,
                **settings,
            }
        )
    )
    env = {**os.environ, "UNFLATTEN_API_TOKEN": token}
    return subprocess.run(
        [sys.executable, str(CLIENT), str(config)],
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )


@pytest.mark.parametrize(
    "suffix,mime",
    [
        (".pdf", "application/pdf"),
        (".png", "image/png"),
        (".jpg", "image/jpeg"),
        (".jpeg", "image/jpeg"),
    ],
)
def test_client_uploads_polls_and_downloads_binary_without_web_session(tmp_path, suffix, mime):
    with synthetic_api(statuses=[status_payload("running"), status_payload()]) as (url, calls):
        result = run_client(
            tmp_path, url, suffix=suffix, model="dense-1.0-fp8-test", label="Žiadosť"
        )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "result.docx").read_bytes() == WORD_BYTES
    assert TOKEN not in result.stdout + result.stderr
    assert [call[:2] for call in calls] == [
        ("POST", "/proxy/v1/conversions"),
        ("GET", "/proxy/v1/conversions/job_example123"),
        ("GET", "/proxy/v1/conversions/job_example123"),
        ("GET", "/proxy/v1/conversions/job_example123/artifacts/diagnostic_docx"),
    ]
    headers = {key.lower(): value for key, value in calls[0][2].items()}
    assert headers["content-type"] == mime
    assert headers["x-unflatten-profile"] == "word-core-v0"
    assert headers["x-unflatten-output"] == "diagnostic_docx"
    assert headers["x-unflatten-model"] == "dense-1.0-fp8-test"
    assert headers["x-unflatten-label"] == "%C5%BDiados%C5%A5"
    assert UUID(headers["idempotency-key"]).version == 4
    assert calls[0][3] == b"synthetic-source\x00\xff"
    assert all(call[2].get("Authorization") == f"Bearer {TOKEN}" for call in calls)
    assert all("Cookie" not in call[2] for call in calls)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "request.json",
        "result.docx",
        f"source{suffix}",
    ]


@pytest.mark.parametrize(
    "state,docx,message",
    [
        ("failed", False, "failed"),
        ("cancelled", False, "cancelled"),
        ("expired", False, "expired"),
        ("succeeded", False, "Word"),
    ],
)
def test_client_terminal_failure_preserves_existing_output(tmp_path, state, docx, message):
    output = tmp_path / "result.docx"
    output.write_bytes(b"existing-document")
    with synthetic_api(statuses=[status_payload(state, docx=docx)]) as (url, calls):
        result = run_client(tmp_path, url)
    assert result.returncode != 0
    assert message in result.stderr
    assert output.read_bytes() == b"existing-document"
    assert len(calls) == 2


def test_client_stops_polling_at_deadline_and_leaves_no_output(tmp_path):
    with synthetic_api(statuses=[status_payload("running")]) as (url, calls):
        result = run_client(tmp_path, url, timeout=0.12, poll_interval=0.02)
    assert result.returncode != 0
    assert "timeout" in result.stderr.lower() or "deadline" in result.stderr.lower()
    assert not (tmp_path / "result.docx").exists()
    assert 2 <= len(calls) < 15


def test_client_http_failure_does_not_echo_response_secrets(tmp_path):
    with synthetic_api(submit_error=401) as (url, calls):
        result = run_client(tmp_path, url)
    assert result.returncode != 0
    assert "401" in result.stderr
    assert TOKEN not in result.stdout + result.stderr
    assert len(calls) == 1


def test_client_does_not_follow_authenticated_redirect(tmp_path):
    with synthetic_api(redirect=True) as (url, calls):
        result = run_client(tmp_path, url)
    assert result.returncode != 0
    assert "307" in result.stderr
    assert len(calls) == 1


def test_client_incomplete_download_preserves_output_and_removes_temporary_file(tmp_path):
    (tmp_path / "result.docx").write_bytes(b"previous-document")
    with synthetic_api(truncated=True) as (url, calls):
        result = run_client(tmp_path, url)
    assert result.returncode != 0
    assert "incomplete" in result.stderr.lower()
    assert (tmp_path / "result.docx").read_bytes() == b"previous-document"
    assert len(calls) == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "request.json",
        "result.docx",
        "source.pdf",
    ]


def test_client_handles_broken_http_download_without_traceback_or_partial_output(tmp_path):
    (tmp_path / "result.docx").write_bytes(b"previous-document")
    with synthetic_api(broken_chunked=True) as (url, _calls):
        result = run_client(tmp_path, url)
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert (tmp_path / "result.docx").read_bytes() == b"previous-document"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("dribble", ["status", "word", "headers"])
def test_client_deadline_interrupts_continuously_dribbling_responses(tmp_path, dribble):
    output = tmp_path / "result.docx"
    output.write_bytes(b"previous-document")
    with synthetic_api(dribble=dribble) as (url, calls):
        started = time.monotonic()
        result = run_client(tmp_path, url, timeout=0.15)
        elapsed = time.monotonic() - started
    assert result.returncode != 0
    assert elapsed < 0.5, f"{dribble} exceeded total deadline: {elapsed:.3f}s"
    assert "deadline" in result.stderr.lower() or "timeout" in result.stderr.lower()
    assert "Traceback" not in result.stderr
    assert TOKEN not in result.stdout + result.stderr
    assert output.read_bytes() == b"previous-document"
    assert len(calls) == (3 if dribble == "word" else 2)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "settings,token",
    [
        ({"temperature": 0.5}, TOKEN),
        ({"poll_interval": 0}, TOKEN),
        ({"timeout": -1}, TOKEN),
        ({"timeout": True}, TOKEN),
        ({}, ""),
    ],
)
def test_client_rejects_invalid_configuration_before_sending_document(tmp_path, settings, token):
    with synthetic_api() as (url, calls):
        result = run_client(tmp_path, url, token=token, **settings)
    assert result.returncode != 0
    assert "Error:" in result.stderr
    assert not calls
    assert not (tmp_path / "result.docx").exists()
