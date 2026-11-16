"""Isolated, GPU-free real API for the Swagger browser regression."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import uvicorn
from tests.service.test_api_endpoints import TOKEN, _client

from unflatten.service.api import create_app


def main() -> None:
    port = int(os.environ.get("UNFLATTEN_DOCS_TEST_PORT", "8478"))
    token_path = Path(gettempdir()) / f"unflatten-e2e-bootstrap-token-{port}"
    with TemporaryDirectory(prefix="unflatten-docs-e2e-") as directory:
        data = Path(directory)
        dependencies = _client(data).app.state.dependencies
        app = create_app(replace(dependencies, web_ui="none"))
        token_path.write_text(TOKEN)
        token_path.chmod(0o600)
        worker = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.service.fake_worker",
                "--job-db",
                str(data / "jobs.sqlite3"),
                "--artifact-root",
                str(data / "artifacts"),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
        finally:
            worker.terminate()
            worker.wait(timeout=10)
            token_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
