"""Visual evidence of the builder chain's Word output, as data -- report-only.

Three sources, each read where it lives:

* the DOCX package XML (`docx_facts`): table width declarations against
  their grid, image extents and `a:srcRect` crops, counted over EVERY story
  part (`word/document.xml` plus every `word/header*.xml` / `word/footer*.xml`
  -- the ad-hoc census this replaces missed the header parts);
* the LibreOffice PDF (`pdf_facts`, `render_facts`): page count against the
  standalone-page expectation of 1 (DEC-033) and the table border span
  read back from the content stream (same stroke parser as
  `tests/eval/builder_chain/pdf_table_width.py`). When the open probe
  cannot run here the row says `render: unavailable` -- never a fake 0;
* the sealed artefact (`sealed_page_facts`): node-type counts, crop
  magnitudes, size-over-region ratios, table width coherence, and image
  region IoU against a gold regions file when one is given, otherwise
  `insufficient_evidence`.

Every quantity is an integer or an integer ratio with its denominator
(`{num, den, ratio_ppm}`); spans are centipoints, IoU is ppm. Nothing here
is hashed and nothing is a gate: RFC005 §12.2 requires a frozen policy
before a floor can decide anything, and no floor is frozen for these
(docs/plans/spec-visual-chain-metrics-design.md)."""

from __future__ import annotations

import json
import re
import statistics
import zipfile
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lxml import etree

from unflatten.renderer.probe import libreoffice_pdf_bytes, probe_available

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_NS = {"w": _W, "wp": _WP, "a": _A}

_STORY_PART = re.compile(r"^word/(document|header\d*|footer\d*)\.xml$")
_STREAM = re.compile(rb"\bstream\r?\n(.*?)endstream\b", re.DOTALL)
_SEGMENT = re.compile(rb"([\-\d.]+)\s+([\-\d.]+)\s+m\s+([\-\d.]+)\s+([\-\d.]+)\s+l")
_PAGE = re.compile(rb"/Type\s*/Page[^s]")

IMAGE_NODE_TYPES = frozenset({"inline_image", "floating_image"})
PPM = 1_000_000
#: |tblW / grid sum − 1| above this is "incoherent" (the census threshold).
INCOHERENT_TOLERANCE_PPM = 200_000


def _ratio(num: int, den: int) -> dict[str, int] | None:
    if den <= 0:
        return None
    return {"num": num, "den": den, "ratio_ppm": num * PPM // den}


def _median_int(values: Sequence[int]) -> int | None:
    return int(statistics.median(values)) if values else None


# --- DOCX package XML -----------------------------------------------------------


def _story_parts(archive: zipfile.ZipFile) -> list[str]:
    return sorted(name for name in archive.namelist() if _STORY_PART.match(name))


def _table_facts(part: str, tbl: etree._Element) -> dict[str, Any]:
    tblw = tbl.find("w:tblPr/w:tblW", _NS)
    layout = tbl.find("w:tblPr/w:tblLayout", _NS)
    grid = [int(col.get(f"{{{_W}}}w")) for col in tbl.findall("w:tblGrid/w:gridCol", _NS)]
    tblw_type = tblw.get(f"{{{_W}}}type") if tblw is not None else None
    tblw_value = int(tblw.get(f"{{{_W}}}w")) if tblw is not None else None
    grid_sum = sum(grid)
    ratio = None
    if tblw_type == "dxa" and tblw_value is not None:
        ratio = _ratio(tblw_value, grid_sum)
    return {
        "part": part,
        "tblw_type": tblw_type,
        "tblw_value": tblw_value,
        "grid": grid,
        "grid_sum": grid_sum,
        "layout": layout.get(f"{{{_W}}}type") if layout is not None else None,
        "tblw_over_grid": ratio,
    }


def docx_facts(path: Path) -> dict[str, Any]:
    """Tables, images and story parts of one DOCX, read from every story
    part's XML (no rendering)."""
    tables: list[dict[str, Any]] = []
    per_part: dict[str, int] = {}
    srcrect_per_part: dict[str, int] = {}
    extents: list[list[int]] = []
    inline = anchor = srcrect = 0
    headers = footers = 0
    with zipfile.ZipFile(path) as archive:
        for part in _story_parts(archive):
            if "/header" in part:
                headers += 1
            elif "/footer" in part:
                footers += 1
            root = etree.fromstring(archive.read(part))
            tables.extend(_table_facts(part, tbl) for tbl in root.iter(f"{{{_W}}}tbl"))
            part_extents = [
                [int(e.get("cx")), int(e.get("cy"))] for e in root.iter(f"{{{_WP}}}extent")
            ]
            if part_extents:
                per_part[part] = len(part_extents)
            extents.extend(part_extents)
            inline += sum(1 for _ in root.iter(f"{{{_WP}}}inline"))
            anchor += sum(1 for _ in root.iter(f"{{{_WP}}}anchor"))
            part_srcrect = sum(1 for _ in root.iter(f"{{{_A}}}srcRect"))
            if part_srcrect:
                srcrect_per_part[part] = part_srcrect
            srcrect += part_srcrect
    return {
        "tables": tables,
        "images": {
            "count": len(extents),
            "per_part": per_part,
            "inline": inline,
            "anchor": anchor,
            "srcrect_count": srcrect,
            "srcrect_per_part": srcrect_per_part,
            "extents_emu": extents,
        },
        "stories": {"header_parts": headers, "footer_parts": footers},
    }


# --- LibreOffice PDF -------------------------------------------------------------


def pdf_facts(pdf: bytes, min_height_pt: float = 20.0) -> dict[str, Any]:
    """Page count and the distinct x positions (centipoints) of vertical
    strokes at least `min_height_pt` tall -- LibreOffice's table borders."""
    xs: set[int] = set()
    for stream in _STREAM.finditer(pdf):
        try:
            content = zlib.decompress(stream.group(1))
        except zlib.error:
            continue
        for segment in _SEGMENT.finditer(content):
            x1, y1, x2, y2 = map(float, segment.groups())
            if abs(x1 - x2) < 0.05 and abs(y2 - y1) > min_height_pt:
                xs.add(round(x1 * 100))
    boundaries = sorted(xs)
    return {
        "pages": len(_PAGE.findall(pdf)),
        "table_boundaries_cpt": boundaries,
        "table_span_cpt": boundaries[-1] - boundaries[0] if len(boundaries) >= 2 else None,
    }


def render_facts(docx_path: Path, *, reuse_pdf: bool = True) -> dict[str, Any]:
    """`pdf_facts` of the DOCX rendered by LibreOffice. A `.pdf` next to the
    DOCX that is not older than it is reused when `reuse_pdf` (and the
    source says so); a fresh render is written there for the next run.
    Without the open probe the result is `{"status": "unavailable"}` -- the
    page count is absent, not 0."""
    pdf_path = docx_path.with_suffix(".pdf")
    if (
        reuse_pdf
        and pdf_path.exists()
        and pdf_path.stat().st_mtime >= docx_path.stat().st_mtime
    ):
        return {"status": "rendered", "source": "cached_pdf", **pdf_facts(pdf_path.read_bytes())}
    if not probe_available():
        return {"status": "unavailable", "reason": "libreoffice_open_probe_unavailable"}
    pdf, issue = libreoffice_pdf_bytes(docx_path)
    if pdf is None:
        reason = f"{issue.code}: {issue.message}" if issue is not None else "no_pdf_bytes"
        return {"status": "failed", "reason": reason}
    pdf_path.write_bytes(pdf)
    return {"status": "rendered", "source": "libreoffice", **pdf_facts(pdf)}


# --- sealed artefact -------------------------------------------------------------


def load_gold_regions(path: Path) -> dict[str, dict[str, Any]]:
    """`cohort_regions.jsonl` (one row per page: `bundle`, `region_targets`
    with `node_type` and `regions` as `[x1, y1, x2, y2]` ppm boxes), keyed
    by bundle."""
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["bundle"]] = row
    return rows


def _iou_ppm(a: Sequence[int], b: Sequence[int]) -> int:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * PPM // union if union > 0 else 0


def _node_states(page: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    states = page.get("node_states")
    if not states:
        states = (page.get("unterminated_prefix") or {}).get("node_states") or []
    return list(states)


def _image_facts(
    node_id: str,
    typed: Mapping[tuple[str, str], Any],
    region: Mapping[str, Any] | None,
    page_size: tuple[int | None, int | None],
    gold_boxes: list[Sequence[int]] | None,
) -> dict[str, Any]:
    crop = [
        abs(typed[(node_id, f"/crop/{side}_ppm")])
        for side in ("left", "right", "top", "bottom")
        if isinstance(typed.get((node_id, f"/crop/{side}_ppm")), int)
    ]
    width = typed.get((node_id, "/size/width_mpt"))
    height = typed.get((node_id, "/size/height_mpt"))
    page_w, page_h = page_size
    box = None
    if region and region.get("alignment_status") == "region" and region.get("regions"):
        first = region["regions"][0]
        if first.get("page_ordinal", 0) == 0:
            box = first["bbox_ppm"]
    size_ratio = None
    if box is not None and isinstance(width, int) and isinstance(height, int) and page_w and page_h:
        region_w = (box[2] - box[0]) * page_w  # ppm × mpt: divide by PPM for mpt
        region_h = (box[3] - box[1]) * page_h
        if region_w > 0 and region_h > 0:
            size_ratio = {
                "w": width * PPM * PPM // region_w,
                "h": height * PPM * PPM // region_h,
            }
    if gold_boxes is None:
        iou: int | str = "insufficient_evidence"
    elif not gold_boxes or box is None:
        iou = "no_gold_boxes" if not gold_boxes else "no_predicted_box"
    else:
        iou = max(_iou_ppm(box, gold) for gold in gold_boxes)
    return {
        "node_id": node_id,
        "crop_abs_ppm": crop,
        "rotation_mdeg": typed.get((node_id, "/rotation_mdeg")),
        "size_over_region_ppm": size_ratio,
        "iou_ppm": iou,
    }


def sealed_page_facts(
    page: Mapping[str, Any], *, gold_regions: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Node-type counts, per-image crop/size/IoU and per-table width
    coherence of one sealed page (JSON shape of `sealed.artefact` pages).
    `gold_regions=None` marks every IoU `insufficient_evidence`."""
    states = _node_states(page)
    typed = {
        (t["local_node_id"], t["pointer"]): t["value"] for t in page.get("typed_predictions") or []
    }
    regions = {r["local_node_id"]: r for r in page.get("region_predictions") or []}
    children: dict[str, list[Mapping[str, Any]]] = {}
    for state in states:
        parent = state.get("parent_local_node_id")
        if parent is not None:
            children.setdefault(parent, []).append(state)
    page_size = (typed.get(("n0", "/page/width_mpt")), typed.get(("n0", "/page/height_mpt")))

    gold_boxes: list[Sequence[int]] | None = None
    if gold_regions is not None:
        gold_row = gold_regions.get(page.get("document_bundle_id", ""))
        gold_boxes = [
            box
            for target in (gold_row or {}).get("region_targets", [])
            if target["node_type"] in IMAGE_NODE_TYPES
            for box in target["regions"]
        ]

    images = [
        _image_facts(s["local_node_id"], typed, regions.get(s["local_node_id"]), page_size,
                     gold_boxes)
        for s in states
        if s["node_type"] in IMAGE_NODE_TYPES
    ]
    tables = []
    for state in states:
        if state["node_type"] != "table":
            continue
        node_id = state["local_node_id"]
        rows = [c for c in children.get(node_id, []) if c["node_type"] == "row"]
        cell_count = max(
            (len([c for c in children.get(r["local_node_id"], []) if c["node_type"] == "cell"])
             for r in rows),
            default=0,
        )
        grid_sum = sum(
            w for w in (typed.get((node_id, f"/columns/{i}/width_mpt")) for i in range(cell_count))
            if isinstance(w, int)
        )
        mode = typed.get((node_id, "/layout/width/mode"))
        value = typed.get((node_id, "/layout/width/value_mpt"))
        ratio = _ratio(value, grid_sum) if mode == "absolute" and isinstance(value, int) else None
        tables.append({
            "node_id": node_id,
            "mode": mode,
            "autofit": typed.get((node_id, "/layout/autofit")),
            "tblw_over_grid": ratio,
        })
    crops = [c for image in images for c in image["crop_abs_ppm"]]
    return {
        "outcome": page.get("outcome"),
        "node_types": dict(Counter(s["node_type"] for s in states)),
        "images": images,
        "tables": tables,
        "crop_abs_ppm_median": _median_int(crops),
    }


# --- rows and aggregate ------------------------------------------------------------


def annotate_rows(
    rows: Iterable[Mapping[str, Any]],
    docx_dir: Path,
    *,
    sealed: Mapping[str, Any] | None = None,
    gold_regions: Mapping[str, Mapping[str, Any]] | None = None,
    render: bool = True,
    reuse_pdf: bool = True,
) -> list[dict[str, Any]]:
    """One record per row (`document_bundle_id`, or `item_id` for the
    chain report's rows): the DOCX facts, the render facts and the sealed
    page facts, each `None` when its source is absent. `sealed` is the
    JSON-loaded artefact (`{"pages": [...]}`)."""
    sealed_pages = {
        p["document_bundle_id"]: p for p in (sealed or {}).get("pages", [])
    }
    out = []
    for row in rows:
        bundle = row.get("document_bundle_id") or row.get("item_id")
        docx_path = docx_dir / f"{bundle}.docx"
        record: dict[str, Any] = {"document_bundle_id": bundle}
        if docx_path.exists():
            record["docx"] = "present"
            record["docx_facts"] = docx_facts(docx_path)
            if render:
                record["render"] = render_facts(docx_path, reuse_pdf=reuse_pdf)
            else:
                record["render"] = {"status": "skipped", "reason": "render_disabled"}
        else:
            record.update({"docx": "missing", "docx_facts": None, "render": None})
        page = sealed_pages.get(bundle)
        record["sealed"] = (
            sealed_page_facts(page, gold_regions=gold_regions) if page is not None else None
        )
        out.append(record)
    return out


def _part_kind(part: str) -> str:
    return "header" if "/header" in part else "footer" if "/footer" in part else "document"


def _incoherent(ratios: Sequence[dict[str, int] | None]) -> dict[str, int]:
    present = [r for r in ratios if r is not None]
    return {
        "num": sum(1 for r in present if abs(r["ratio_ppm"] - PPM) > INCOHERENT_TOLERANCE_PPM),
        "den": len(present),
    }


def _aggregate_sealed(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not facts:
        return None
    node_types: Counter[str] = Counter()
    for f in facts:
        node_types.update(f["node_types"])
    images = [i for f in facts for i in f["images"]]
    tables = [t for f in facts for t in f["tables"]]
    crops = [c for i in images for c in i["crop_abs_ppm"]]
    sizes = [i["size_over_region_ppm"] for i in images if i["size_over_region_ppm"]]
    ious = [i["iou_ppm"] for i in images]
    numeric = [v for v in ious if isinstance(v, int)]
    if all(v == "insufficient_evidence" for v in ious):
        iou_summary: dict[str, Any] | str = "insufficient_evidence"
    else:
        iou_summary = {
            "n": len(numeric),
            "median": _median_int(numeric),
            "ge_500000": sum(1 for v in numeric if v >= 500_000),
            "no_gold_boxes": sum(1 for v in ious if v == "no_gold_boxes"),
        }
    ratios = [t["tblw_over_grid"] for t in tables]
    return {
        "pages": len(facts),
        "outcomes": dict(Counter(f["outcome"] for f in facts)),
        "node_types_total": dict(sorted(node_types.items())),
        "images": len(images),
        "crop_abs_ppm_median": _median_int(crops),
        "crop_over_50000": {"num": sum(1 for c in crops if c > 50_000), "den": len(crops)},
        "size_over_region_ppm_median": (
            {"w": _median_int([s["w"] for s in sizes]), "h": _median_int([s["h"] for s in sizes])}
            if sizes else None
        ),
        "iou_ppm": iou_summary,
        "tables": {
            "count": len(tables),
            "modes": dict(Counter(str(t["mode"]) for t in tables)),
            "autofit": dict(Counter(str(t["autofit"]) for t in tables)),
            "incoherent_gt20pct": _incoherent(ratios),
            "tblw_over_grid_ppm_median": _median_int(
                [r["ratio_ppm"] for r in ratios if r is not None]
            ),
        },
    }


def aggregate(per_row: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Directory-level census of `annotate_rows` records: page histogram
    and one-page share over rendered rows, table layouts / width types /
    coherence / border span median, image and srcRect totals by part kind,
    story-part presence, and the sealed summary (or `None`)."""
    renders = [r["render"] for r in per_row if r["render"] is not None]
    rendered = [r for r in renders if r["status"] == "rendered"]
    docx = [r["docx_facts"] for r in per_row if r["docx_facts"] is not None]
    tables = [t for d in docx for t in d["tables"]]
    spans = [r["table_span_cpt"] for r in rendered if r["table_span_cpt"] is not None]
    per_part_kind = Counter({"document": 0, "header": 0, "footer": 0})
    srcrect_per_part_kind = Counter({"document": 0, "header": 0, "footer": 0})
    for d in docx:
        for part, count in d["images"]["per_part"].items():
            per_part_kind[_part_kind(part)] += count
        for part, count in d["images"]["srcrect_per_part"].items():
            srcrect_per_part_kind[_part_kind(part)] += count
    return {
        "rows": len(per_row),
        "docx_present": sum(1 for r in per_row if r["docx"] == "present"),
        "render": {
            "rendered": len(rendered),
            "unavailable": sum(1 for r in renders if r["status"] == "unavailable"),
            "failed": sum(1 for r in renders if r["status"] == "failed"),
            "skipped": sum(1 for r in renders if r["status"] == "skipped"),
            "cached_pdf": sum(1 for r in rendered if r["source"] == "cached_pdf"),
        },
        "pages_histogram": dict(sorted(Counter(str(r["pages"]) for r in rendered).items())),
        "one_page": {"num": sum(1 for r in rendered if r["pages"] == 1), "den": len(rendered)},
        "tables": {
            "count": len(tables),
            "layouts": dict(Counter(str(t["layout"]) for t in tables)),
            "tblw_types": dict(Counter(str(t["tblw_type"]) for t in tables)),
            "incoherent_gt20pct": _incoherent([t["tblw_over_grid"] for t in tables]),
            "span_cpt_median": _median_int(spans),
            "span_n": len(spans),
        },
        "images": {
            "count": sum(d["images"]["count"] for d in docx),
            "inline": sum(d["images"]["inline"] for d in docx),
            "anchor": sum(d["images"]["anchor"] for d in docx),
            "srcrect_total": sum(d["images"]["srcrect_count"] for d in docx),
            "per_part_kind": dict(per_part_kind),
            "srcrect_per_part_kind": dict(srcrect_per_part_kind),
        },
        "stories": {
            "rows_with_header_part": sum(1 for d in docx if d["stories"]["header_parts"]),
            "rows_with_footer_part": sum(1 for d in docx if d["stories"]["footer_parts"]),
        },
        "sealed": _aggregate_sealed([r["sealed"] for r in per_row if r["sealed"] is not None]),
    }
