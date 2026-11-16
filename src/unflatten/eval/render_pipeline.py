"""Render one class of `/layout/spacing/after_mpt` per LibreOffice call.

The measurement D3 asks for is O(V) renders, not O(V²): render the
document once per vocabulary value, then compare the stored rasters for
all V² ordered pairs. The comparison is free relative to the render.

Determinism: the RFC002 compile is byte-deterministic, and the raster
depends on the recorded LibreOffice and poppler versions plus the DPI —
all three are written into `provenance`, because two matrices produced
under different versions are not comparable.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.dataset.migration.m2b import (
    _libreoffice_pdf,
    _libreoffice_version,
    _pdf_pages_png,
)
from unflatten.eval.render_cost import compare_class_rasters, substitute_spacing_after
from unflatten.eval.vocabulary import POINTER
from unflatten.renderer.api import RendererOptions
from unflatten.renderer.compiler import compile_docx
from unflatten.renderer.resources import DirResourceProvider
from unflatten.renderer.version import RENDERER_VERSION

#: LibreOffice cannot read `/tmp` or dot-directories under snap; the
#: whole repo stages conversions here (see `m2b._LO_ROOT`).
_STAGE_ROOT = Path.home() / "unflatten-probe" / "eval-cost"


class ClassRenderError(RuntimeError):
    """One class could not be rendered — compile, LibreOffice or raster."""


@dataclass(frozen=True)
class ClassRender:
    """The document rendered with `after_mpt` set to `value_mpt` everywhere."""

    value_mpt: int
    page_pngs: tuple[bytes, ...]
    docx_sha256: str
    render_seconds: float


def render_class(
    ast: dict[str, Any],
    value_mpt: int,
    *,
    dpi: int,
    resources: dict[str, bytes] | None = None,
    profile_slot: int = 0,
) -> ClassRender:
    """Compile and rasterize `ast` with every `after_mpt` set to `value_mpt`.

    Raises `ClassRenderError` on any leg failing: a partial render would
    silently become a cost of zero.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")
    substituted = substitute_spacing_after(ast, value_mpt)
    started = time.monotonic()
    _STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(dir=_STAGE_ROOT, prefix=f"v{value_mpt}-"))
    try:
        for path, data in (resources or {}).items():
            target = stage / "resources" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        docx_path = stage / "doc.docx"
        result = compile_docx(
            substituted,
            docx_path,
            RendererOptions(strict=True),
            resources=DirResourceProvider(stage / "resources") if resources else None,
        )
        if result.status == "failed":
            codes = ", ".join(issue.code for issue in result.issues)
            raise ClassRenderError(f"compile failed for after_mpt={value_mpt}: {codes}")
        docx_bytes = docx_path.read_bytes()
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    pdf_bytes = _libreoffice_pdf(docx_bytes, tag=f"cost-{value_mpt}", profile_slot=profile_slot)
    if pdf_bytes is None:
        raise ClassRenderError(f"LibreOffice produced no PDF for after_mpt={value_mpt}")
    pages = _pdf_pages_png(pdf_bytes, tag=f"cost-{value_mpt}", render_dpi=dpi)
    if not pages:
        raise ClassRenderError(f"pdftoppm produced no pages for after_mpt={value_mpt}")
    return ClassRender(
        value_mpt=value_mpt,
        page_pngs=tuple(pages),
        docx_sha256=f"sha256:{hashlib.sha256(docx_bytes).hexdigest()}",
        render_seconds=time.monotonic() - started,
    )


def render_classes(
    ast: dict[str, Any],
    values_mpt: Iterable[int],
    *,
    dpi: int,
    resources: dict[str, bytes] | None = None,
    profile_slot: int = 0,
) -> list[ClassRender]:
    """Render `ast` once per vocabulary value (V LibreOffice calls)."""
    return [
        render_class(ast, value, dpi=dpi, resources=resources, profile_slot=profile_slot)
        for value in values_mpt
    ]


def cost_matrix_from_renders(
    renders: Sequence[ClassRender], provenance: dict[str, Any]
) -> dict[str, Any]:
    """Compare every ordered pair of `renders` into a JSON-ready matrix."""
    if not renders:
        raise ValueError("no class renders to compare")
    values = [render.value_mpt for render in renders]
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate class renders for values {sorted(values)}")
    ordered = sorted(renders, key=lambda render: render.value_mpt)
    cells = []
    for truth in ordered:
        for predicted in ordered:
            diff = compare_class_rasters(truth.page_pngs, predicted.page_pngs)
            cells.append(
                {
                    "truth_mpt": truth.value_mpt,
                    "predicted_mpt": predicted.value_mpt,
                    **diff.to_dict(),
                }
            )
    return {
        "arm": "rendered",
        "pointer": POINTER,
        "vocabulary_mpt": [render.value_mpt for render in ordered],
        "provenance": {
            "renderer_version": RENDERER_VERSION,
            "libreoffice_version": _libreoffice_version(),
            **provenance,
        },
        "class_renders": [
            {
                "value_mpt": render.value_mpt,
                "page_count": len(render.page_pngs),
                "docx_sha256": render.docx_sha256,
                "render_seconds": render.render_seconds,
            }
            for render in ordered
        ],
        "cells": cells,
    }


def poppler_version() -> str:
    """`pdftoppm -v` first line, or "unknown" when poppler is absent."""
    try:
        proc = subprocess.run(  # noqa: S603
            ["pdftoppm", "-v"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    output = (proc.stderr or proc.stdout).strip().splitlines()
    return output[0] if output else "unknown"
