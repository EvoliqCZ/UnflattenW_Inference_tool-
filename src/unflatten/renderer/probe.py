"""LibreOffice headless open/render probe (RFC002 validation layer V5
"application open probe" / V6 "render probe").

Environment caveat discovered empirically in this sandbox: the snap-
packaged LibreOffice on this machine can only read/write files under a
*non-hidden* directory directly beneath `$HOME` — it fails with
"source file could not be loaded" for paths under `/tmp` or under any
dot-directory such as `~/.cache/...` (the snap `home` interface excludes
dotfiles/dot-directories by default). The probe therefore stages its
input under `~/.unflatten-probe-workdir`'s non-hidden sibling
`~/unflatten-probe/` before invoking `libreoffice`, and always passes an
isolated `-env:UserInstallation=` profile directory so concurrent probe
runs (and the developer's own LibreOffice session) never collide on the
same lock file.

That profile directory is **per process** (`profile-<pid>`). A single
shared one was the cause of the probe's long-standing flakiness: two
pytest processes — this suite and the dataset suite, or a parallel
worker — take the same LibreOffice profile lock, and the loser reports
`OPEN_PROBE_FAILED` with returncode 0 and empty stderr, which looks
exactly like a genuinely unopenable document. Building the profile is
the expensive part, so it is still reused across every call *within* a
process, and removed at exit.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from unflatten.ast.issues import Issue
from unflatten.libreoffice import profile_dir

_PROBE_ROOT = Path.home() / "unflatten-probe"
_TIMEOUT_SECONDS = 60

_available: bool | None = None
_profile_dir: Path | None = None


def _get_profile_dir() -> Path:
    """This process's own LibreOffice profile, built once and reused."""
    global _profile_dir
    if _profile_dir is None:
        _profile_dir = profile_dir("renderer", process_id=os.getpid())
        _profile_dir.mkdir(parents=True, exist_ok=True)
        atexit.register(shutil.rmtree, _profile_dir, ignore_errors=True)
    return _profile_dir


def _run_convert(input_path: Path, outdir: Path, profile_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "libreoffice",
            "--headless",
            f"-env:UserInstallation=file://{profile_dir}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(outdir),
            str(input_path),
        ],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_SECONDS,
    )


def probe_available() -> bool:
    """Self-test once (cached) whether the LibreOffice open probe can
    run at all in this environment: `libreoffice` on `PATH`, and a
    trivial `.docx` actually converts to a non-empty PDF.
    """
    global _available
    if _available is not None:
        return _available
    if shutil.which("libreoffice") is None:
        _available = False
        return _available

    from unflatten.renderer.api import RendererOptions
    from unflatten.renderer.compiler import compile_docx

    work = _PROBE_ROOT / "selftest"
    work.mkdir(parents=True, exist_ok=True)
    minimal_ast = {
        "ast_version": "0.1.0",
        "profile": "word-core-v0",
        "document_id": "probe-selftest",
        "metadata": {"language": None, "title": None, "subject": None},
        "resources": [],
        "sections": [
            {
                "id": "s1",
                "type": "section",
                "break_type": "next_page",
                "page": {"width_mpt": 595276, "height_mpt": 841890, "orientation": "portrait"},
                "margins": {
                    "top_mpt": 72000,
                    "right_mpt": 72000,
                    "bottom_mpt": 72000,
                    "left_mpt": 72000,
                    "gutter_mpt": 0,
                },
                "columns": {"count": 1, "equal_width": True, "gap_mpt": 18000, "widths_mpt": []},
                "children": [],
            }
        ],
        "extensions": {},
    }
    docx_path = work / "selftest.docx"
    result = compile_docx(minimal_ast, docx_path, RendererOptions())
    if result.status == "failed":
        _available = False
        return _available

    issue = libreoffice_open_probe(docx_path)
    _available = issue is None
    return _available


def libreoffice_pdf_text(docx_path: Path) -> tuple[str | None, Issue | None]:
    """Convert `docx_path` to PDF and return the PDF's extracted text.

    The open probe alone cannot tell a rendered header from a header
    that was parsed and then dropped: both produce a PDF and a zero
    exit code. Reading the text back is the only evidence that content
    outside `document.xml` actually reached the page.

    Returns `(text, None)` on success, `(None, Issue)` otherwise —
    including when `pdftotext` is not installed, which is reported
    rather than silently treated as "no text".
    """
    return _convert(docx_path, extract_text=True)


def libreoffice_open_probe(docx_path: Path) -> Issue | None:
    """Open `docx_path` in headless LibreOffice and convert it to PDF.

    Returns `None` on success (the PDF was produced), else a
    structured `Issue` (category `OPEN_PROBE`). Never raises for an
    ordinary open/convert failure or timeout — only for programming
    errors (e.g. `docx_path` not existing is reported as an Issue too).

    Each call stages its input/output under a fresh, unique
    subdirectory of `~/unflatten-probe/calls/<uuid4>/` so concurrent
    probe calls (e.g. a parametrized test run, or two test workers)
    never collide on the same bare filename or output PDF. The `-env:
    UserInstallation=` profile directory stays shared across calls
    (it's the expensive part to (re)build) at `~/unflatten-probe/
    profile/`. The call's staging directory is removed in `finally`,
    even on error/timeout.
    """
    _text, issue = _convert(docx_path, extract_text=False)
    return issue


def libreoffice_pdf_bytes(docx_path: Path) -> tuple[bytes | None, Issue | None]:
    """Convert `docx_path` to PDF and return the raw PDF bytes.

    Test support only — for proving page content (e.g. that a table
    actually shows interior gridlines) by reading the PDF content
    stream, since no rasterizer is available in the main dependency
    group. Reuses this module's existing profile dir and conversion
    helper; it is not a fourth LibreOffice call path.

    Returns `(bytes, None)` on success, `(None, Issue)` otherwise.
    """
    return _convert(docx_path, extract_text=False, return_pdf_bytes=True)


def _convert(
    docx_path: Path, *, extract_text: bool, return_pdf_bytes: bool = False
) -> tuple[str | bytes | None, Issue | None]:
    if not docx_path.exists():
        return None, Issue(
            code="OPEN_PROBE_INPUT_MISSING",
            severity="error",
            node_id=None,
            path=str(docx_path),
            message=f"probe input {docx_path} does not exist.",
            details={},
        )

    profile_dir = _get_profile_dir()
    call_dir = _PROBE_ROOT / "calls" / uuid.uuid4().hex
    work_dir = call_dir / "work"
    out_dir = call_dir / "out"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        staged = work_dir / docx_path.name
        staged.write_bytes(docx_path.read_bytes())
        expected_pdf = out_dir / f"{staged.stem}.pdf"

        try:
            proc = _run_convert(staged, out_dir, profile_dir)
        except subprocess.TimeoutExpired:
            return None, Issue(
                code="OPEN_PROBE_TIMEOUT",
                severity="error",
                node_id=None,
                path=str(docx_path),
                message=f"libreoffice did not finish within {_TIMEOUT_SECONDS}s.",
                details={},
            )
        except FileNotFoundError:
            return None, Issue(
                code="OPEN_PROBE_UNAVAILABLE",
                severity="error",
                node_id=None,
                path=str(docx_path),
                message="libreoffice is not available on PATH.",
                details={},
            )

        if proc.returncode != 0 or not expected_pdf.exists() or expected_pdf.stat().st_size == 0:
            return None, Issue(
                code="OPEN_PROBE_FAILED",
                severity="error",
                node_id=None,
                path=str(docx_path),
                message=f"libreoffice failed to open/convert {docx_path.name} "
                f"(returncode={proc.returncode}).",
                details={"stdout": proc.stdout, "stderr": proc.stderr},
            )
        if return_pdf_bytes:
            return expected_pdf.read_bytes(), None
        if not extract_text:
            return None, None
        return _pdf_text(expected_pdf)
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)


def _pdf_text(pdf_path: Path) -> tuple[str | None, Issue | None]:
    try:
        proc = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return None, Issue(
            code="RENDER_PROBE_UNAVAILABLE",
            severity="error",
            node_id=None,
            path=str(pdf_path),
            message=f"pdftotext could not read the rendered PDF: {exc}",
            details={},
        )
    if proc.returncode != 0:
        return None, Issue(
            code="RENDER_PROBE_FAILED",
            severity="error",
            node_id=None,
            path=str(pdf_path),
            message=f"pdftotext failed (returncode={proc.returncode}).",
            details={"stderr": proc.stderr},
        )
    return proc.stdout, None
