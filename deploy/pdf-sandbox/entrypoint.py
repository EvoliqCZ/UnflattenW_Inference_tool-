from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scanner import inspect_pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("inspect", "raster"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--page-index", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.mode == "inspect":
        report = inspect_pdf(arguments.source)
        sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    if arguments.page_index < 0:
        return 24
    if not arguments.expected_sha256 or _sha256(arguments.source) != arguments.expected_sha256:
        return 20
    report = inspect_pdf(arguments.source)
    if report["status"] != "accepted":
        return 21
    if arguments.page_index >= int(report["page_count"]):
        return 24
    pdftoppm_page = arguments.page_index + 1
    output_prefix = Path("/tmp/page")
    result = subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(pdftoppm_page),
            "-l",
            str(pdftoppm_page),
            "-singlefile",
            "-cropbox",
            "-png",
            "-r",
            "300",
            str(arguments.source),
            str(output_prefix),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"},
    )
    output = output_prefix.with_suffix(".png")
    produced = list(output.parent.glob("page*"))
    if result.returncode != 0 or produced != [output] or not output.is_file():
        return 22
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sys.stdout.buffer.write(chunk)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # The host intentionally receives no parser diagnostics or document content.
        raise SystemExit(23) from None
