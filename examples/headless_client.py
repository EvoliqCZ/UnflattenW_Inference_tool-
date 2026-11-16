#!/usr/bin/env python3
"""Convert a local PDF/image to Word through the API, using only Python's stdlib."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4

WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
INPUT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
CONFIG_KEYS = {"base_url", "input", "output", "model", "label", "poll_interval", "timeout"}


class ClientError(Exception):
    """An actionable client failure whose message contains no server response body."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward a Bearer credential to a redirect destination.
        return None


@contextmanager
def wallclock_timeout(seconds: float):
    """Bound all blocking I/O in this standalone POSIX main-thread client."""
    if not all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer")):
        raise ClientError("This client requires Linux/POSIX real-time signal timers.")

    def expired(_signum, _frame):
        raise ClientError("Conversion deadline reached (timeout); the server job may still run.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    # Socket timeouts reset whenever data arrives. One real-time alarm also
    # interrupts slow DNS/connect/TLS, uploads, headers and dribbling bodies.
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        delay, interval = previous_timer
        if delay:
            delay = max(delay - (time.monotonic() - started), 0.000001)
        signal.setitimer(signal.ITIMER_REAL, delay, interval)


def load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClientError("Cannot read configuration as a JSON object.") from exc
    if not isinstance(config, dict) or set(config) - CONFIG_KEYS:
        raise ClientError("Unsupported configuration; see examples/headless-request.example.json.")
    for name in ("base_url", "input", "output"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ClientError(f"Configuration requires a nonempty {name} string.")
    parsed = urlsplit(config["base_url"])
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ClientError("base_url must be an HTTP(S) service URL without credentials or query.")
    config["base_url"] = config["base_url"].rstrip("/")
    for name in ("input", "output"):
        if "://" in config[name]:
            raise ClientError(f"{name} must be a local file path.")
        config[name] = (path.resolve().parent / config[name]).expanduser().resolve()
    if config["input"] == config["output"]:
        raise ClientError("input and output must be different files.")
    if config["input"].suffix.lower() not in INPUT_TYPES or not config["input"].is_file():
        raise ClientError("input must be an existing local PDF, PNG or JPEG file.")
    for name in ("model", "label"):
        if name in config and (
            not isinstance(config[name], str)
            or not config[name].strip()
            or any(ord(char) < 32 or ord(char) == 127 for char in config[name])
        ):
            raise ClientError(f"{name} must be a nonempty string without control characters.")
    if "model" in config and not config["model"].isascii():
        raise ClientError("model must be a variant identifier from GET /v1/models.")
    for name, default in (("poll_interval", 2), ("timeout", 1800)):
        value = config.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ClientError(f"{name} must be a finite positive number of seconds.")
        config[name] = value
    return config


def remaining(deadline: float) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise ClientError("Conversion deadline reached (timeout); the server job may still run.")
    return seconds


def open_response(opener, request: Request, deadline: float):
    try:
        return opener.open(request, timeout=remaining(deadline))
    except HTTPError as exc:
        exc.close()
        raise ClientError(f"API returned HTTP {exc.code}; no response body was logged.") from exc
    except (URLError, TimeoutError) as exc:
        raise ClientError("Network request failed or reached its timeout.") from exc


def read_json(opener, request: Request, deadline: float, expected_status: int) -> dict:
    with open_response(opener, request, deadline) as response:
        if response.status != expected_status:
            raise ClientError("Unexpected HTTP status from API.")
        raw = response.read(4 * 1024 * 1024 + 1)
    remaining(deadline)
    try:
        payload = json.loads(raw) if len(raw) <= 4 * 1024 * 1024 else None
    except (ValueError, UnicodeError) as exc:
        raise ClientError("API returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ClientError("API returned an invalid JSON object or an oversized response.")
    return payload


def download_word(opener, request: Request, output: Path, deadline: float) -> None:
    temporary = None
    try:
        with open_response(opener, request, deadline) as response:
            if response.headers.get_content_type() != WORD_MIME:
                raise ClientError("Download did not return a Word document.")
            length = response.headers.get("Content-Length")
            try:
                expected = None if length is None else int(length)
            except ValueError as exc:
                raise ClientError("Download has an invalid content length.") from exc
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as target:
                temporary = Path(target.name)
                received = 0
                while True:
                    remaining(deadline)
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    received += len(chunk)
                if received == 0 or (expected is not None and received != expected):
                    raise ClientError("Word download was empty or incomplete.")
                target.flush()
                os.fsync(target.fileno())
        remaining(deadline)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _convert(config: dict, token: str) -> Path:
    if (
        not 32 <= len(token) <= 4096
        or not token.isascii()
        or any(char.isspace() or char == "," for char in token)
    ):
        raise ClientError("Set UNFLATTEN_API_TOKEN to a valid API token (without Bearer prefix).")
    opener = build_opener(NoRedirect())
    deadline = time.monotonic() + config["timeout"]
    base = config["base_url"]
    auth = {"Authorization": f"Bearer {token}"}
    headers = {
        **auth,
        "Idempotency-Key": str(uuid4()),
        "X-Unflatten-Profile": "word-core-v0",
        "X-Unflatten-Output": "diagnostic_docx",
        "Content-Type": INPUT_TYPES[config["input"].suffix.lower()],
        "Content-Length": str(config["input"].stat().st_size),
    }
    if "model" in config:
        headers["X-Unflatten-Model"] = config["model"]
    if "label" in config:
        headers["X-Unflatten-Label"] = quote(config["label"], safe="")
    with config["input"].open("rb") as source:
        submitted = read_json(
            opener,
            Request(f"{base}/v1/conversions", data=source, headers=headers, method="POST"),
            deadline,
            202,
        )
    job_id = submitted.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"job_[A-Za-z0-9_-]{1,128}", job_id) is None:
        raise ClientError("Submission did not return a valid job_id.")
    print(f"Submitted {job_id}", flush=True)
    # Construct same-origin paths rather than trusting response-provided URLs.
    job_url = f"{base}/v1/conversions/{job_id}"
    while True:
        status = read_json(opener, Request(job_url, headers=auth), deadline, 200)
        state = status.get("state")
        if state == "succeeded":
            outcome = status.get("diagnostic_outcome")
            if not isinstance(outcome, dict) or outcome.get("docx_produced") is not True:
                raise ClientError("Job succeeded without a Word document; inspect its API status.")
            break
        if state in {"failed", "cancelled", "expired"}:
            raise ClientError(f"Conversion {state}; inspect the job status for details.")
        if state not in {"queued", "running"}:
            raise ClientError("API returned an unknown job state.")
        time.sleep(min(config["poll_interval"], remaining(deadline)))
    download_word(
        opener,
        Request(f"{job_url}/artifacts/diagnostic_docx", headers=auth),
        config["output"],
        deadline,
    )
    return config["output"]


def convert(config: dict, token: str) -> Path:
    with wallclock_timeout(config["timeout"]):
        return _convert(config, token)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON configuration file")
    args = parser.parse_args()
    try:
        output = convert(load_config(args.config), os.environ.get("UNFLATTEN_API_TOKEN", ""))
    except ClientError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, HTTPException) as exc:
        # Avoid echoing server bodies, credentials or URLs in raw exception text.
        print(
            f"Error: local file or network operation failed ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 1
    print(f"Saved Word document: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
