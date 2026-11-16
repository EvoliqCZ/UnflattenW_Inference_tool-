from __future__ import annotations

import json
import math
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

from scanner_content import classify_text_visibility as classify_text_visibility
from scanner_graph import c2pa_filespec_references, classify_graph_text, classify_object_graph

SCANNER_IDENTITY = (
    "unflatten-pdf-admission/qpdf-11.9.0-1.1ubuntu0.1-"
    "poppler-24.02.0-1ubuntu9.9/v13"
)
_PAGE_SIZE = re.compile(
    r"^Page\s+([0-9]+)\s+size:\s+([0-9]+(?:\.[0-9]+)?)\s+x\s+([0-9]+(?:\.[0-9]+)?)\s+pts"
    r"(?:\s+\([A-Za-z0-9 ]+\))?\s*$"
)


class InspectionError(RuntimeError):
    pass


def inspect_pdf(path: Path) -> dict[str, object]:
    limits = _limits()
    report = _empty_report()
    info = _run(["pdfinfo", "-f", "1", "-l", str(limits["max_pages"]), "-box", str(path)])
    info_text = info.stdout.decode("utf-8", "replace")
    encrypted = _pdfinfo_value(info_text, "Encrypted:").lower().startswith("yes")
    report["encrypted"] = encrypted
    if encrypted:
        return _reject(report, "PDF_ENCRYPTED")
    if info.returncode != 0:
        return _reject(report, "PDF_MALFORMED")
    try:
        page_count = int(_pdfinfo_value(info_text, "Pages:"))
    except (ValueError, InspectionError):
        return _reject(report, "PDF_MALFORMED")
    report["page_count"] = page_count
    if page_count < 1 or page_count > limits["max_pages"]:
        return _reject(report, "UNSUPPORTED_PAGE_COUNT")
    try:
        geometries = _page_geometries(info_text, page_count, limits["dpi"])
    except InspectionError:
        return _reject(report, "PDF_MALFORMED")
    report["page_geometries"] = geometries
    report.update(geometries[0])
    if any(
        p["page_width_points"] > limits["max_page_width"]
        or p["page_height_points"] > limits["max_page_height"]
        or p["raster_width_pixels"] > limits["max_raster_width"]
        or p["raster_height_pixels"] > limits["max_raster_height"]
        or p["raster_pixels"] > limits["max_raster_pixels"]
        for p in geometries
    ):
        return _reject(report, "PDF_PAGE_DIMENSIONS_EXCEEDED")

    if _run(["qpdf", "--check", str(path)]).returncode != 0:
        return _reject(report, "PDF_MALFORMED")
    graph_result = _run(
        ["qpdf", "--json", "--json-stream-data=inline", "--decode-level=all", str(path)]
    )
    if graph_result.returncode not in {0, 3}:
        return _reject(report, "PDF_MALFORMED")
    try:
        graph = json.loads(graph_result.stdout)
        counts = classify_object_graph(
            graph,
            max_image_width=limits["max_raster_width"],
            max_image_height=limits["max_raster_height"],
        )
        visibility = classify_graph_text(graph)
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return _reject(report, "PDF_MALFORMED")
    report.update(counts)
    (
        report["text_visibility"],
        report["hidden_text_objects"],
        report["ambiguous_text_objects"],
    ) = visibility
    reason = _feature_reason(report)
    if (graph_result.returncode == 3 and reason is None
            and not _c2pa_spelling_warning(graph_result.stderr, path, graph)):
        reason = "PDF_MALFORMED"
    return _reject(report, reason) if reason else report


def _feature_reason(report: Mapping[str, object]) -> str | None:
    # DEC-062: annotations and text visibility are diagnostics; input is pixels.
    checks = (
        (int(report["javascript_action_objects"]) > 0, "PDF_ACTIVE_CONTENT"),
        (int(report["attachment_objects"]) > int(report["c2pa_metadata_objects"]),
         "PDF_EMBEDDED_FILE"),
        (int(report["extreme_image_objects"]) > 0, "PDF_PAGE_DIMENSIONS_EXCEEDED"),
    )
    return next((reason for rejected, reason in checks if rejected), None)


def _c2pa_spelling_warning(stderr: bytes | None, path: Path, graph: object) -> bool:
    if not isinstance(stderr, bytes) or not stderr or len(stderr) > 65_536:
        return False
    try:
        lines = stderr.decode("utf-8", "strict").splitlines()
    except UnicodeDecodeError:
        return False
    if len(lines) < 2 or lines[-1] != "qpdf: operation succeeded with warnings":
        return False
    known = c2pa_filespec_references(graph)
    pattern = re.compile(
        rf"WARNING: {re.escape(str(path))}, object ([0-9]+) ([0-9]+) at offset [0-9]+: "
        r"Embedded file object's type is not /Filespec"
    )
    for line in lines[:-1]:
        matched = pattern.fullmatch(line)
        if not matched or f"{matched[1]} {matched[2]} R" not in known:
            return False
    return True


def _classify_graph_text(graph: object) -> tuple[str, int, int]:
    """Preserve the private graph-classifier test surface across the module split."""
    return classify_graph_text(graph)


def _run(command: list[str]) -> subprocess.CompletedProcess[bytes]:
    environment = {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C", "LANG": "C"}
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
        env=environment,
    )


def _pdfinfo_value(output: str, prefix: str) -> str:
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise InspectionError("required pdfinfo field absent")


def _page_geometries(output: str, page_count: int, dpi: int) -> list[dict[str, float | int]]:
    sizes = {}
    rotations = {}
    for line in output.splitlines():
        size = _PAGE_SIZE.fullmatch(line)
        rotation = re.fullmatch(r"Page\s+([0-9]+)\s+rot:\s+(-?[0-9]+)\s*", line)
        if size:
            index = int(size[1])
            if index in sizes:
                raise InspectionError("duplicate page size")
            sizes[index] = (float(size[2]), float(size[3]))
        if rotation:
            index = int(rotation[1])
            if index in rotations:
                raise InspectionError("duplicate page rotation")
            rotations[index] = int(rotation[2])
    expected = set(range(1, page_count + 1))
    if set(sizes) != expected or set(rotations) != expected:
        raise InspectionError("page geometries are incomplete")
    result = []
    for index in range(1, page_count + 1):
        width, height = sizes[index]
        rotation = rotations[index]
        if not all(math.isfinite(v) and v > 0 for v in (width, height)) or rotation % 90:
            raise InspectionError("page geometry is invalid")
        if rotation % 180:
            width, height = height, width
        pixels_w, pixels_h = math.ceil(width * dpi / 72), math.ceil(height * dpi / 72)
        result.append({"page_width_points": width, "page_height_points": height,
                       "raster_width_pixels": pixels_w, "raster_height_pixels": pixels_h,
                       "raster_pixels": pixels_w * pixels_h})
    return result


def _limits() -> dict[str, int]:
    names = {
        "dpi": "RASTER_DPI",
        "max_page_width": "MAX_PAGE_WIDTH_POINTS",
        "max_page_height": "MAX_PAGE_HEIGHT_POINTS",
        "max_raster_width": "MAX_RASTER_WIDTH_PIXELS",
        "max_raster_height": "MAX_RASTER_HEIGHT_PIXELS",
        "max_raster_pixels": "MAX_RASTER_PIXELS",
        # DEC-044 Dodatok D §1 (B1): real multi-page ceiling, default 20;
        # was hardcoded to exactly 1 page before this policy field existed.
        "max_pages": "MAX_PDF_PAGES",
    }
    result: dict[str, int] = {}
    for key, environment_name in names.items():
        raw = os.environ.get(environment_name)
        if raw is None:
            raise InspectionError("required scanner limit absent")
        try:
            result[key] = int(raw)
        except ValueError as exc:
            raise InspectionError("scanner limit invalid") from exc
        if result[key] <= 0:
            raise InspectionError("scanner limit invalid")
    if result["dpi"] != 300:
        raise InspectionError("scanner DPI is not pinned")
    return result


def _empty_report() -> dict[str, object]:
    return {
        "schema_version": "1.2",
        "scanner_identity": SCANNER_IDENTITY,
        "status": "accepted",
        "reason": None,
        "page_count": 0,
        "page_geometries": [],
        "page_width_points": 0.0,
        "page_height_points": 0.0,
        "raster_width_pixels": 0,
        "raster_height_pixels": 0,
        "raster_pixels": 0,
        "encrypted": False,
        "javascript_action_objects": 0,
        "attachment_objects": 0,
        "c2pa_metadata_objects": 0,
        "annotation_objects": 0,
        "text_visibility": "absent",
        "hidden_text_objects": 0,
        "ambiguous_text_objects": 0,
        "extreme_image_objects": 0,
        "object_count": 0,
    }


def _reject(report: dict[str, object], reason: str) -> dict[str, object]:
    report["status"] = "rejected"
    report["reason"] = reason
    return report
