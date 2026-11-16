"""Conservative PDF image-object extraction and AST image observation."""

from __future__ import annotations

import math
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any

from lxml import etree

from unflatten.ast.issues import Issue
from unflatten.dataset.observations.pdf_text import PdfPage
from unflatten.renderer.units import round_half_away_from_zero

IMAGE_SIZE_TOLERANCE_PT = Fraction(3, 5)
PAGE_FRAME_TOLERANCE_PT = Fraction(1)
_TIMEOUT_SECONDS = 120
_IMAGE_TYPES = frozenset(("inline_image", "floating_image"))


class PdfImageExtractionError(RuntimeError):
    """pdftohtml is unavailable, failed, or produced invalid XML."""


@dataclass(frozen=True)
class PdfImagePage:
    page_index: int
    width_pt_text: str
    height_pt_text: str
    placements: tuple[tuple[str, str, str, str], ...]


@dataclass(frozen=True)
class ImageFragment:
    node_id: str
    page_index: int
    bbox_ppm: tuple[int, int, int, int]
    fragment_index: int = 0
    continuation_prev: bool = False
    continuation_next: bool = False


@dataclass
class ImageObservation:
    observations: list[ImageFragment]
    issues: list[Issue]
    counters: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class _Image:
    node: dict[str, Any]
    paragraph_id: str
    story_id: str | None
    ancestors: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Placement:
    page_index: int
    index: int
    x: Fraction
    y: Fraction
    width: Fraction
    height: Fraction


def parse_pdf_images_xml(xml_bytes: bytes) -> list[PdfImagePage]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    try:
        root = etree.fromstring(xml_bytes, parser=parser)
    except etree.XMLSyntaxError as exc:
        raise PdfImageExtractionError(f"unparseable pdftohtml XML: {exc}") from exc
    pages: list[PdfImagePage] = []
    for page_index, page in enumerate(root.xpath("//*[local-name()='page']")):
        width = page.get("width")
        height = page.get("height")
        if width is None or height is None:
            raise PdfImageExtractionError(f"page {page_index} lacks width/height")
        placements = []
        for image in page.xpath(".//*[local-name()='image']"):
            values = tuple(image.get(k) for k in ("left", "top", "width", "height"))
            if any(value is None for value in values):
                raise PdfImageExtractionError(f"page {page_index} image lacks geometry")
            try:
                for value in values:
                    Fraction(value)
            except ValueError as exc:
                raise PdfImageExtractionError(
                    f"page {page_index} image has malformed geometry"
                ) from exc
            placements.append(values)
        pages.append(PdfImagePage(page_index, width, height, tuple(placements)))
    return pages


def extract_pdf_images(pdf_bytes: bytes) -> list[PdfImagePage]:
    """Run poppler without writing extracted image files into the workspace."""
    with tempfile.TemporaryDirectory(prefix="unflatten-images-") as tmp:
        pdf_path = Path(tmp) / "doc.pdf"
        pdf_path.write_bytes(pdf_bytes)
        try:
            proc = subprocess.run(
                [
                    "pdftohtml",
                    "-xml",
                    "-stdout",
                    "-zoom",
                    "1",
                    "-noroundcoord",
                    str(pdf_path),
                ],
                capture_output=True,
                timeout=_TIMEOUT_SECONDS,
                cwd=tmp,
            )
        except FileNotFoundError as exc:
            raise PdfImageExtractionError("pdftohtml (poppler-utils) not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise PdfImageExtractionError("pdftohtml timed out") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", "replace").strip()
        raise PdfImageExtractionError(f"pdftohtml failed (rc={proc.returncode}): {stderr}")
    return parse_pdf_images_xml(proc.stdout)


@lru_cache(maxsize=2)
def poppler_version(tool: str) -> str:
    """Return poppler's numeric version for a tool already used successfully."""
    proc = subprocess.run([tool, "-v"], capture_output=True, timeout=10, text=True)
    output = f"{proc.stdout}\n{proc.stderr}"
    match = re.search(r"\bversion\s+([0-9][0-9.]*)", output)
    return match.group(1) if proc.returncode == 0 and match else "unknown"


def _children(node: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in ("children", "items", "rows", "cells"):
        result.extend(x for x in (node.get(key) or []) if isinstance(x, dict))
    return result


def _collect_images(ast: dict[str, Any]) -> list[_Image]:
    result: list[_Image] = []

    def walk(node: dict[str, Any], ancestors: tuple[dict[str, Any], ...], story_id: str | None):
        node_type = node.get("type")
        if node_type in ("header", "footer"):
            story_id = node.get("id")
        if node_type in _IMAGE_TYPES:
            paragraph = next((x for x in reversed(ancestors) if x.get("type") == "paragraph"), None)
            if paragraph is not None:
                result.append(_Image(node, paragraph["id"], story_id, ancestors))
            return
        for child in _children(node):
            walk(child, (*ancestors, node), story_id)

    for section in ast.get("sections", []) or []:
        walk(section, (), None)
        for key in ("headers", "footers"):
            for story in section.get(key, []) or []:
                walk(story, (section,), story.get("id"))
    for key in ("footnotes", "endnotes"):
        for story in ast.get(key, []) or []:
            walk(story, (), None)
    return result


def header_footer_image_node_ids(ast: dict[str, Any]) -> frozenset[str]:
    return frozenset(img.node["id"] for img in _collect_images(ast) if img.story_id is not None)


def image_node_count(ast: dict[str, Any]) -> int:
    """Number of observable image-node candidates in canonical preorder."""
    return len(_collect_images(ast))


def story_text_pages(
    ast: dict[str, Any], fragments: list[Any]
) -> dict[str, frozenset[int]]:
    owners: dict[str, str] = {}

    def walk(node: dict[str, Any], story_id: str | None):
        if node.get("type") in ("header", "footer"):
            story_id = node["id"]
        if story_id is not None and node.get("type") not in _IMAGE_TYPES and "id" in node:
            owners[node["id"]] = story_id
        for child in _children(node):
            walk(child, story_id)

    for section in ast.get("sections", []) or []:
        for key in ("headers", "footers"):
            for story in section.get(key, []) or []:
                walk(story, None)
    pages: dict[str, set[int]] = {}
    for fragment in fragments:
        if owner := owners.get(fragment.node_id):
            pages.setdefault(owner, set()).add(fragment.page_index)
    return {owner: frozenset(value) for owner, value in pages.items()}


def _expected_size(image: _Image) -> tuple[float, float] | None:
    node = image.node
    crop = node["crop"]
    horizontal = crop["left_ppm"] + crop["right_ppm"]
    vertical = crop["top_ppm"] + crop["bottom_ppm"]
    if horizontal >= 1_000_000 or vertical >= 1_000_000:
        return None
    width = node["size"]["width_mpt"] / 1000 / (1 - horizontal / 1_000_000)
    height = node["size"]["height_mpt"] / 1000 / (1 - vertical / 1_000_000)
    angle = math.radians(node["rotation_mdeg"] / 1000)
    return (
        abs(width * math.cos(angle)) + abs(height * math.sin(angle)),
        abs(width * math.sin(angle)) + abs(height * math.cos(angle)),
    )


def _eligible(image: _Image) -> bool:
    crop = image.node["crop"]
    return (
        image.node["rotation_mdeg"] == 0
        and all(value >= 0 for value in crop.values())
        and crop["left_ppm"] + crop["right_ppm"] < 1_000_000
        and crop["top_ppm"] + crop["bottom_ppm"] < 1_000_000
    )


def _ppm(value: Fraction, extent: Fraction) -> int:
    ratio = value / extent
    result = round_half_away_from_zero(ratio.numerator * 1_000_000, ratio.denominator)
    return min(max(result, 0), 1_000_000)


def _bbox(placement: _Placement, page: PdfPage, image: _Image) -> tuple[int, int, int, int]:
    width = Fraction(page.width_pt_text)
    height = Fraction(page.height_pt_text)
    x1, y1 = placement.x, placement.y
    x2, y2 = x1 + placement.width, y1 + placement.height
    if _eligible(image):
        # Poppler reports the entire embedded bitmap, including the parts
        # outside the source crop frame. Crop before page clipping and
        # candidate geometry filters; the full size still verifies identity.
        crop = image.node["crop"]
        x1 += placement.width * Fraction(crop["left_ppm"], 1_000_000)
        y1 += placement.height * Fraction(crop["top_ppm"], 1_000_000)
        x2 -= placement.width * Fraction(crop["right_ppm"], 1_000_000)
        y2 -= placement.height * Fraction(crop["bottom_ppm"], 1_000_000)
    return (
        _ppm(max(x1, 0), width),
        _ppm(max(y1, 0), height),
        _ppm(min(x2, width), width),
        _ppm(min(y2, height), height),
    )


def _issue(code: str, node_id: str | None, message: str, path: str | None = None) -> Issue:
    return Issue(
        code=code,
        severity="warning",
        node_id=node_id,
        path=path or f"/nodes/{node_id}",
        message=message,
    )


def _subtree_ids(node: dict[str, Any]) -> set[str]:
    ids = {node["id"]} if "id" in node else set()
    for child in _children(node):
        ids.update(_subtree_ids(child))
    return ids


def _corridor_violates(
    corridor: tuple[Any, Any], page_index: int, box: tuple[int, int, int, int],
    page_height_pt: Fraction,
) -> bool:
    """Does ``box`` on ``page_index`` fall on the wrong side of a resolved
    ``_corridor`` result? A ``None`` side carries no constraint (T0 path
    2: a one-sided corridor bounds the vertical position only on the side
    that has text). The text glyph bounds and image-object geometry use
    the same existing 0.6 pt verification uncertainty. This admits tiny
    boundary overlaps without moving either observed box; size and
    two-sided uniqueness still have to verify independently."""
    prev, nxt = corridor
    tolerance_ppm = IMAGE_SIZE_TOLERANCE_PT * 1_000_000 / page_height_pt
    if prev is not None and page_index == prev[0] and prev[1][3] - box[1] > tolerance_ppm:
        return True
    if nxt is not None and page_index == nxt[0] and box[3] - nxt[1][1] > tolerance_ppm:
        return True
    return False


def _size_collides(
    node_id: str, expected: tuple[float, float], eligible_images: list[_Image]
) -> bool:
    """Does any OTHER eligible image node's expected size fall within
    ``IMAGE_SIZE_TOLERANCE_PT`` of ``expected`` (T0 path 1's global
    size-uniqueness guard)?"""
    return any(
        other.node["id"] != node_id
        and (other_expected := _expected_size(other)) is not None
        and abs(other_expected[0] - expected[0]) <= float(IMAGE_SIZE_TOLERANCE_PT)
        and abs(other_expected[1] - expected[1]) <= float(IMAGE_SIZE_TOLERANCE_PT)
        for other in eligible_images
    )


def _section_body_pages(
    image: _Image, text_boxes: dict[str, list[tuple[int, tuple[int, int, int, int]]]]
) -> frozenset[int] | None:
    """The pages ``image``'s own section body renders on, derived from the
    already-aligned body text fragments -- used only when a header/footer
    image's own story has no text of its own (a text-less logo, DEC-035
    §12/§14 measurement): the section is ``image.ancestors[0]`` for every
    image (`_collect_images` always pushes the section first, for both
    body and story walks); ``_subtree_ids`` walks only ``children``/
    ``items``/``rows``/``cells``, never the ``headers``/``footers`` keys,
    so this never picks up story text as if it were body text. ``None``
    when the section is not derivable (a footnote/endnote image) or no
    body fragment of that section landed on any page."""
    if not image.ancestors or image.ancestors[0].get("type") != "section":
        return None
    body_ids = _subtree_ids(image.ancestors[0])
    pages = {page for node_id in body_ids for page, _box in text_boxes.get(node_id, [])}
    return frozenset(pages) if pages else None


def _corridor(image: _Image, boxes: dict[str, list[tuple[int, tuple[int, int, int, int]]]]):
    """The nearest before/after sibling text boxes bounding ``image``.

    Returns ``(prev, nxt)`` where either element may be ``None`` when
    text exists on only one side (the paragraph is at the very start or
    end of the document, per DEC-035 §12/§14 measurement) -- the caller
    then bounds the region and the vertical filter on that side only.
    Both-sided matches are preferred and returned as soon as found,
    walking outward; a one-sided match is remembered as a fallback (the
    nearest one found) and returned only if no two-sided match exists at
    any ancestor level.
    """
    current = next(x for x in reversed(image.ancestors) if x.get("type") == "paragraph")
    ancestors = list(image.ancestors)
    one_sided_fallback = None
    while current in ancestors:
        index = ancestors.index(current)
        if index == 0:
            break
        parent = ancestors[index - 1]
        siblings = _children(parent)
        try:
            position = siblings.index(current)
        except ValueError:
            current = parent
            continue
        before = next(
            (s for s in reversed(siblings[:position]) if any(i in boxes for i in _subtree_ids(s))),
            None,
        )
        after = next(
            (s for s in siblings[position + 1 :] if any(i in boxes for i in _subtree_ids(s))),
            None,
        )
        before_boxes = (
            [v for i in _subtree_ids(before) for v in boxes.get(i, [])]
            if before is not None
            else []
        )
        after_boxes = (
            [v for i in _subtree_ids(after) for v in boxes.get(i, [])]
            if after is not None
            else []
        )
        prev = (
            max(before_boxes, key=lambda value: (value[0], value[1][3])) if before_boxes else None
        )
        nxt = min(after_boxes, key=lambda value: (value[0], value[1][1])) if after_boxes else None
        if prev is not None and nxt is not None:
            if 0 <= nxt[0] - prev[0] <= 1:
                return prev, nxt
        elif one_sided_fallback is None and (prev is not None or nxt is not None):
            one_sided_fallback = (prev, nxt)
        current = parent
    return one_sided_fallback


def observe_images(
    ast: dict[str, Any],
    pdf_pages: list[PdfPage],
    image_pages: list[PdfImagePage],
    text_fragments: list[Any],
    story_spans: dict[str, frozenset[int]],
) -> ImageObservation:
    """Observe only globally one-to-one, size-verified placements."""
    counters = {
        "image_nodes_total": 0,
        "image_nodes_observed": 0,
        "image_nodes_observed_by_size": 0,
        "image_nodes_observed_by_one_sided_corridor": 0,
        "image_nodes_observed_by_story_pages": 0,
        "image_nodes_transformed": 0,
        "image_nodes_ambiguous": 0,
        "image_nodes_unplaced": 0,
        "image_placements_unexplained": 0,
    }
    issues: list[Issue] = []
    images = _collect_images(ast)
    counters["image_nodes_total"] = len(images)
    pages = {page.page_index: page for page in pdf_pages}
    placements: dict[tuple[int, int], _Placement] = {}
    valid_pages: set[int] = set(pages)
    for image_page in image_pages:
        text_page = pages.get(image_page.page_index)
        if text_page is None:
            continue
        if (
            abs(Fraction(text_page.width_pt_text) - Fraction(image_page.width_pt_text))
            > PAGE_FRAME_TOLERANCE_PT
            or abs(Fraction(text_page.height_pt_text) - Fraction(image_page.height_pt_text))
            > PAGE_FRAME_TOLERANCE_PT
        ):
            valid_pages.discard(image_page.page_index)
            issues.append(
                _issue(
                    "OBSERVATION_IMAGE_PAGE_FRAME_MISMATCH",
                    None,
                    "pdftotext and pdftohtml page sizes differ by more than 1 pt",
                    f"/pages/{image_page.page_index}",
                )
            )
            continue
        for index, raw in enumerate(image_page.placements):
            x, y, width, height = map(Fraction, raw)
            placements[(image_page.page_index, index)] = _Placement(
                image_page.page_index, index, x, y, width, height
            )

    text_boxes: dict[str, list[tuple[int, tuple[int, int, int, int]]]] = {}
    for fragment in text_fragments:
        text_boxes.setdefault(fragment.node_id, []).append((fragment.page_index, fragment.bbox_ppm))

    relation: dict[str, list[tuple[int, int]]] = {}
    #: resolved per-page region for story (header/footer) images -- either
    #: the story's own text pages, or (T0 path 3) the pages its section's
    #: body renders on, when the story itself has no matchable text.
    resolved_story_pages: dict[str, frozenset[int]] = {}
    #: node ids whose region was derived from the section body fallback,
    #: not from the story's own text (for `image_nodes_observed_by_story_pages`)
    story_fallback_used: set[str] = set()
    #: node ids with NO derivable region at all (no story, no paragraph
    #: text, no one- or two-sided corridor) -- T0 path 1 candidates
    no_region_ids: set[str] = set()
    #: node ids whose accepted region came from a one-sided corridor
    one_sided_ids: set[str] = set()
    for image in images:
        node_id = image.node["id"]
        corridor = None
        if image.story_id is not None:
            region_pages = story_spans.get(image.story_id, frozenset())
            if not region_pages:
                derived = _section_body_pages(image, text_boxes)
                if derived:
                    region_pages = derived
                    story_fallback_used.add(node_id)
            resolved_story_pages[node_id] = region_pages
        elif paragraph_boxes := text_boxes.get(image.paragraph_id):
            region_pages = frozenset(page for page, _ in paragraph_boxes)
        else:
            corridor = _corridor(image, text_boxes)
            if corridor is None:
                region_pages = frozenset()
                no_region_ids.add(node_id)
            else:
                prev, nxt = corridor
                if prev is not None and nxt is not None:
                    region_pages = frozenset(range(prev[0], nxt[0] + 1))
                elif prev is not None:
                    region_pages = frozenset((prev[0], prev[0] + 1))
                    one_sided_ids.add(node_id)
                else:
                    region_pages = frozenset((nxt[0] - 1, nxt[0]))
                    one_sided_ids.add(node_id)
        expected = _expected_size(image)
        candidates = []
        for key, placement in placements.items():
            if placement.page_index not in valid_pages or placement.page_index not in region_pages:
                continue
            if (
                expected is None
                or abs(float(placement.width) - expected[0])
                > float(IMAGE_SIZE_TOLERANCE_PT)
            ):
                continue
            if abs(float(placement.height) - expected[1]) > float(IMAGE_SIZE_TOLERANCE_PT):
                continue
            box = _bbox(placement, pages[placement.page_index], image)
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            if image.node["type"] == "inline_image" and text_boxes.get(image.paragraph_id):
                paragraph_boxes = [
                    box
                    for page, box in text_boxes[image.paragraph_id]
                    if page == placement.page_index
                ]
                if not any(box[1] < b[3] and box[3] > b[1] for b in paragraph_boxes):
                    continue
            # The corridor's vertical bound is a property of an inline
            # image's line box (it sits between the neighbouring
            # paragraphs). A floating image's anchor offset moves it
            # anywhere on the page, over or past that text (design §5.3:
            # "a floating_image is page-only"), so for it the corridor
            # bounds the PAGE set only. Measured 2026-09-26 on val_iid:
            # 59 of 60 floating placements this bound refused were the
            # unique size-verified placement on their corridor pages.
            if (
                corridor
                and image.node["type"] == "inline_image"
                and _corridor_violates(
                    corridor, placement.page_index, box,
                    Fraction(pages[placement.page_index].height_pt_text),
                )
            ):
                continue
            candidates.append(key)
        relation[node_id] = candidates

    claimants: dict[tuple[int, int], list[str]] = {}
    for node_id, candidates in relation.items():
        for key in candidates:
            claimants.setdefault(key, []).append(node_id)

    # T0 path 1: an eligible image with no derivable region at all is
    # observed iff exactly one placement in the whole document matches its
    # expected size, no other eligible image node's size also matches it,
    # and that placement is not already claimed by another node's own
    # (region-bound) candidate list.
    eligible_images = [img for img in images if _eligible(img)]
    size_matches: dict[str, tuple[int, int]] = {}
    size_ambiguous: set[str] = set()
    for image in images:
        node_id = image.node["id"]
        if node_id not in no_region_ids or not _eligible(image):
            continue
        expected = _expected_size(image)
        if expected is None:
            continue
        matches = []
        for key, placement in placements.items():
            if placement.page_index not in valid_pages:
                continue
            if abs(float(placement.width) - expected[0]) > float(IMAGE_SIZE_TOLERANCE_PT):
                continue
            if abs(float(placement.height) - expected[1]) > float(IMAGE_SIZE_TOLERANCE_PT):
                continue
            box = _bbox(placement, pages[placement.page_index], image)
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            matches.append(key)
        if not matches:
            continue
        unresolved = (
            _size_collides(node_id, expected, eligible_images)
            or len(matches) != 1
            or claimants.get(matches[0])
        )
        if unresolved:
            size_ambiguous.add(node_id)
            continue
        size_matches[node_id] = matches[0]

    story_ids = {image.node["id"]: image.story_id for image in images}
    stories_by_page: dict[int, set[str]] = {}
    for node_id, candidates in relation.items():
        if story_id := story_ids[node_id]:
            for page_index, _placement_index in candidates:
                stories_by_page.setdefault(page_index, set()).add(story_id)
    contested_story_pages = {
        page_index for page_index, owners in stories_by_page.items() if len(owners) > 1
    }

    observations: list[ImageFragment] = []
    explained: set[tuple[int, int]] = set()
    for image in images:
        node_id = image.node["id"]
        candidates = relation[node_id]
        explained.update(candidates)
        if not _eligible(image):
            counters["image_nodes_transformed"] += 1
            issues.append(
                _issue(
                    "OBSERVATION_IMAGE_TRANSFORMED",
                    node_id,
                    "rotation or crop is unsupported",
                )
            )
            continue
        if image.story_id is not None and resolved_story_pages.get(node_id):
            selected: list[tuple[int, int]] = []
            refused = False
            refused_ambiguous = False
            for page_index in resolved_story_pages[node_id]:
                on_page = [key for key in candidates if key[0] == page_index]
                if page_index in contested_story_pages:
                    refused = True
                    refused_ambiguous = True
                elif len(on_page) == 1 and len(claimants[on_page[0]]) == 1:
                    selected.extend(on_page)
                else:
                    refused = True
            for key in selected:
                placement = placements[key]
                observations.append(
                    ImageFragment(
                        node_id,
                        placement.page_index,
                        _bbox(placement, pages[placement.page_index], image),
                    )
                )
            if selected:
                counters["image_nodes_observed"] += 1
                if node_id in story_fallback_used:
                    counters["image_nodes_observed_by_story_pages"] += 1
            if refused:
                issues.append(
                    _issue(
                        (
                            "OBSERVATION_IMAGE_AMBIGUOUS"
                            if candidates or refused_ambiguous
                            else "OBSERVATION_IMAGE_UNPLACED"
                        ),
                        node_id,
                        "a story page lacks one uncontested size-verified placement",
                    )
                )
                if not selected:
                    counters[
                        "image_nodes_ambiguous"
                        if candidates or refused_ambiguous
                        else "image_nodes_unplaced"
                    ] += 1
            continue
        unique = len(candidates) == 1 and len(claimants[candidates[0]]) == 1
        if unique:
            placement = placements[candidates[0]]
            observations.append(
                ImageFragment(
                    node_id,
                    placement.page_index,
                    _bbox(placement, pages[placement.page_index], image),
                )
            )
            counters["image_nodes_observed"] += 1
            if node_id in one_sided_ids:
                counters["image_nodes_observed_by_one_sided_corridor"] += 1
        elif candidates:
            counters["image_nodes_ambiguous"] += 1
            issues.append(
                _issue(
                    "OBSERVATION_IMAGE_AMBIGUOUS",
                    node_id,
                    "image match is not one-to-one",
                )
            )
        elif image.story_id is not None and not resolved_story_pages.get(node_id):
            counters["image_nodes_ambiguous"] += 1
            issues.append(
                _issue(
                    "OBSERVATION_IMAGE_AMBIGUOUS",
                    node_id,
                    "image-only story has no page evidence",
                )
            )
        elif node_id in size_matches:
            key = size_matches[node_id]
            placement = placements[key]
            observations.append(
                ImageFragment(
                    node_id,
                    placement.page_index,
                    _bbox(placement, pages[placement.page_index], image),
                )
            )
            explained.add(key)
            counters["image_nodes_observed"] += 1
            counters["image_nodes_observed_by_size"] += 1
        elif node_id in size_ambiguous:
            counters["image_nodes_ambiguous"] += 1
            issues.append(
                _issue(
                    "OBSERVATION_IMAGE_AMBIGUOUS",
                    node_id,
                    "global size match is not one-to-one",
                )
            )
        else:
            counters["image_nodes_unplaced"] += 1
            issues.append(
                _issue(
                    "OBSERVATION_IMAGE_UNPLACED",
                    node_id,
                    "no size-verified placement in region",
                )
            )
    counters["image_placements_unexplained"] = len(set(placements) - explained)
    return ImageObservation(observations, issues, counters)
