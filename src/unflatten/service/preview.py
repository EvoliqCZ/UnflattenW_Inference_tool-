"""DEC-044 Dodatok D (C): side-by-side preview rendering.

`source_preview` is the raster the model actually saw, downsampled to a
manageable size. `docx_preview` is the produced `diagnostic_docx` rendered
through the SAME LibreOffice-headless-to-PDF path the renderer's own open
probe uses (`unflatten.libreoffice`'s per-process profile registry, so a
concurrent probe/synthetic/import-rasterize call can never collide with
this one), then `pdftoppm` PDF -> PNG per page -- the diagnostic docx is our
OWN trusted output, never untrusted input, so this runs directly on the
worker host with no admission sandbox.

LibreOffice/pdftoppm failure is never fatal to the job (design C): the
caller catches `PreviewRenderFailed`, records `PREVIEW_RENDER_FAILED`, and
the job still succeeds with its already-produced `diagnostic_docx`.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from PIL import Image

from unflatten.libreoffice import profile_dir

DEFAULT_MAX_LONG_SIDE = 1400
_CONVERT_TIMEOUT_SECONDS = 60
_RASTER_TIMEOUT_SECONDS = 30
_PREVIEW_DPI = 150
#: The snap-packaged LibreOffice on a dev machine can only read/write files
#: under a non-hidden directory directly beneath `$HOME` (CLAUDE.md,
#: `renderer/probe.py`'s own note) -- never `/tmp` or any dot-directory. A
#: worker's real `work_root`/attempt directory is not guaranteed to be
#: under `$HOME` either, so this module stages its OWN call-scoped work
#: directory here, independent of any caller-supplied path.
_PREVIEW_ROOT = Path.home() / "unflatten-diagnostic-preview"


class PreviewRenderFailed(RuntimeError):
    """LibreOffice or pdftoppm failed to produce a docx preview."""


def render_source_preview(png_bytes: bytes, *, max_long_side: int = DEFAULT_MAX_LONG_SIDE) -> bytes:
    """Downsample the page raster the model actually saw to `max_long_side`
    px on its longer side; never upsamples an already-smaller source."""
    with Image.open(io.BytesIO(png_bytes)) as image:
        width, height = image.size
        scale = min(1.0, max_long_side / max(width, height))
        rgb = image.convert("RGB")
        resized = (
            rgb
            if scale >= 1.0
            else rgb.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS
            )
        )
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        return buffer.getvalue()


def render_docx_previews(
    docx_bytes: bytes, *, max_long_side: int = DEFAULT_MAX_LONG_SIDE
) -> list[bytes]:
    """Render every page of `docx_bytes` to a PNG, `max_long_side` px on its
    longer side. Raises `PreviewRenderFailed` on any failure -- LibreOffice
    absent, conversion timeout, or a zero-page result. Always cleans up its
    own call-scoped work directory, success or failure."""
    call_dir = _PREVIEW_ROOT / "calls" / uuid.uuid4().hex
    call_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        docx_path = call_dir / "source.docx"
        docx_path.write_bytes(docx_bytes)
        profile = profile_dir("diagnostic_preview", process_id=os.getpid())
        profile.mkdir(parents=True, exist_ok=True)
        try:
            converted = subprocess.run(
                [
                    "libreoffice", "--headless",
                    f"-env:UserInstallation=file://{profile}",
                    "--convert-to", "pdf", "--outdir", str(call_dir), str(docx_path),
                ],
                capture_output=True, timeout=_CONVERT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PreviewRenderFailed(f"libreoffice convert-to pdf failed: {exc}") from exc
        pdf_path = docx_path.with_suffix(".pdf")
        if converted.returncode != 0 or not pdf_path.is_file():
            raise PreviewRenderFailed("libreoffice did not produce a PDF")
        page_prefix = call_dir / "page"
        try:
            rastered = subprocess.run(
                ["pdftoppm", "-png", "-r", str(_PREVIEW_DPI), str(pdf_path), str(page_prefix)],
                capture_output=True, timeout=_RASTER_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PreviewRenderFailed(f"pdftoppm failed: {exc}") from exc
        if rastered.returncode != 0:
            raise PreviewRenderFailed("pdftoppm returned a non-zero exit code")
        page_paths = sorted(call_dir.glob(f"{page_prefix.name}*.png"))
        if not page_paths:
            raise PreviewRenderFailed("pdftoppm produced no pages")
        return [
            render_source_preview(path.read_bytes(), max_long_side=max_long_side)
            for path in page_paths
        ]
    finally:
        shutil.rmtree(call_dir, ignore_errors=True)
