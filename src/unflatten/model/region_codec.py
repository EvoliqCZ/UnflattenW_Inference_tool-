"""RFC004 §21 region codec — the NodeVisualAligner's logical geometry.

Page-relative integer ``ppm`` coordinates (``0..1_000_000`` against the
post-transform page image, RFC003 §24.1), half-open ``[x1, y1, x2, y2)``
boxes, optional polygon with the RFC003 §24.1 canonical form, and the
§21.7 ownership point / comparison-region sort key.

The aligner's logical output (RFC004 §21.2) is a *hypothesis sidecar*:
zero/one/many regions per node, each with a ``continuation`` tag and a
``confidence_ppm``. The AST builder must never copy it into the Canonical
AST, and an abstained node is reported via ``alignment_status`` — it is
never masked as a high-confidence zero-area bbox (RFC004 §21.5), so every
region in this codec has strictly positive area.

This codec is pure Python on purpose: it is the contract the future
``NodeVisualAligner`` nn.Module (owner decisions R2–R6) and its supervision
targets must agree on, and it is fully testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: One part per million of the post-transform page dimension.
PPM_MAX = 1_000_000

#: The closed continuation tags (RFC004 §21.2 multi-region nodes).
CONTINUATIONS: tuple[str, ...] = ("start", "middle", "end", "complete")

_BBOX = tuple[int, int, int, int]
_Point = tuple[int, int]


def _require_ppm_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer ppm value, got {value!r}")
    if not 0 <= value <= PPM_MAX:
        raise ValueError(
            f"{name} must be within 0..{PPM_MAX} ppm, got {value!r}"
        )
    return value


def _validate_bbox(bbox_ppm: Any) -> _BBOX:
    if (
        isinstance(bbox_ppm, (str, bytes))
        or not isinstance(bbox_ppm, (list, tuple))
        or len(bbox_ppm) != 4
    ):
        raise ValueError(f"bbox_ppm must be [x1, y1, x2, y2], got {bbox_ppm!r}")
    x1, y1, x2, y2 = (_require_ppm_int(v, "bbox_ppm") for v in bbox_ppm)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"bbox_ppm must be half-open with positive area "
            f"(x1 < x2 and y1 < y2), got {bbox_ppm!r} — abstention is "
            "alignment_status, never a degenerate bbox (RFC004 §21.5)"
        )
    return (x1, y1, x2, y2)


def _signed_area2(polygon: tuple[_Point, ...]) -> int:
    total = 0
    for i in range(len(polygon)):
        x_i, y_i = polygon[i]
        x_next, y_next = polygon[(i + 1) % len(polygon)]
        total += x_i * y_next - x_next * y_i
    return total


def _orient(a: _Point, b: _Point, c: _Point) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(p: _Point, q: _Point, r: _Point) -> bool:
    return (
        min(p[0], q[0]) <= r[0] <= max(p[0], q[0])
        and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])
    )


def _segments_intersect(a: _Point, b: _Point, c: _Point, d: _Point) -> bool:
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _has_self_intersection(polygon: tuple[_Point, ...]) -> bool:
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _segments_intersect(a, b, polygon[j], polygon[(j + 1) % n]):
                return True
    return False


def canonicalize_polygon(vertices: Any) -> tuple[_Point, ...]:
    """Canonical integer polygon per RFC003 §24.1.

    1. ``A2 = Σ(x_i*y_(i+1) - x_(i+1)*y_i)``; in this y-down system
       ``A2 > 0`` is clockwise, ``A2 < 0`` is reversed, ``A2 == 0`` invalid.
    2. From the clockwise sequence the unique lexicographically lowest
       ``(y, x)`` vertex is selected.
    3. The sequence is rotated so that vertex is first; direction is not
       changed again.
    """
    if (
        isinstance(vertices, (str, bytes))
        or not isinstance(vertices, (list, tuple))
        or len(vertices) < 3
    ):
        raise ValueError(
            f"polygon_ppm needs at least 3 vertices, got {vertices!r}"
        )
    points: list[_Point] = []
    for vertex in vertices:
        if (
            isinstance(vertex, (str, bytes))
            or not isinstance(vertex, (list, tuple))
            or len(vertex) != 2
        ):
            raise ValueError(f"polygon vertex must be [x, y], got {vertex!r}")
        points.append(
            (_require_ppm_int(vertex[0], "polygon vertex x"),)
            + (_require_ppm_int(vertex[1], "polygon vertex y"),)
        )
    if len(set(points)) != len(points):
        raise ValueError(
            "polygon_ppm must not repeat a vertex (a duplicated last point "
            f"is a repeat), got {points!r}"
        )
    polygon = tuple(points)
    area2 = _signed_area2(polygon)
    if area2 == 0:
        raise ValueError(
            f"polygon_ppm has zero signed area (collinear/degenerate), "
            f"got {points!r}"
        )
    if area2 < 0:
        polygon = polygon[::-1]
    anchor = min(range(len(polygon)), key=lambda i: (polygon[i][1], polygon[i][0]))
    polygon = polygon[anchor:] + polygon[:anchor]
    if _has_self_intersection(polygon):
        raise ValueError(f"polygon_ppm is self-intersecting: {points!r}")
    return polygon


def bbox_envelope(polygon: tuple[_Point, ...]) -> _BBOX:
    """Exact integer axis-aligned bounding box of a canonical polygon."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class Region:
    """One page-relative region hypothesis (RFC004 §21.2).

    Construction is fail-closed: every RFC003 §24.1 / RFC004 §21 constraint
    is enforced here, and a stored polygon is always in canonical form.
    """

    page_ordinal: int
    bbox_ppm: _BBOX
    continuation: str
    confidence_ppm: int
    polygon_ppm: tuple[_Point, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.page_ordinal, bool) or not isinstance(
            self.page_ordinal, int
        ):
            raise ValueError(
                f"page_ordinal must be a non-negative integer, "
                f"got {self.page_ordinal!r}"
            )
        if self.page_ordinal < 0:
            raise ValueError(
                f"page_ordinal must be >= 0, got {self.page_ordinal!r}"
            )
        object.__setattr__(
            self, "bbox_ppm", _validate_bbox(tuple(self.bbox_ppm))
        )
        if self.continuation not in CONTINUATIONS:
            raise ValueError(
                f"continuation must be one of {list(CONTINUATIONS)}, "
                f"got {self.continuation!r}"
            )
        object.__setattr__(
            self, "confidence_ppm", _require_ppm_int(self.confidence_ppm, "confidence_ppm")
        )
        if self.polygon_ppm is not None:
            canonical = canonicalize_polygon(self.polygon_ppm)
            object.__setattr__(self, "polygon_ppm", canonical)
            if bbox_envelope(canonical) != self.bbox_ppm:
                raise ValueError(
                    "bbox_ppm must be the exact axis-aligned envelope of "
                    f"polygon_ppm (RFC004 §21.3): bbox {self.bbox_ppm} vs "
                    f"envelope {bbox_envelope(canonical)}"
                )


def region_ownership_point(region: Region) -> tuple[int, int, int]:
    """The §21.7 ownership point of a region.

    For half-open ``[left, top, right, bottom)``:
    ``(page_ordinal, floor((left + right - 1) / 2), floor((top + bottom - 1) / 2))``.
    """
    left, top, right, bottom = region.bbox_ppm
    return (
        region.page_ordinal,
        (left + right - 1) // 2,
        (top + bottom - 1) // 2,
    )


def comparison_sort_key(region: Region) -> tuple[int, int, int, int, int]:
    """The §21.7 canonical sort key for comparison regions."""
    left, top, right, bottom = region.bbox_ppm
    return (region.page_ordinal, top, left, bottom, right)
