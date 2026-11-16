from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Forward reference only (avoids scanner_state importing scanner_fonts): the
# provider maps a `Tf` font resource name to its resolved widths, or `None`
# when the font cannot be resolved (fail-closed one-em-per-glyph fallback).
FontMetricsProvider = Callable[[str], object]


@dataclass(frozen=True)
class Operand:
    kind: str
    value: object = None
    codes: bytes = b""  # decoded byte codes of a "string" operand, for width lookup


@dataclass
class GraphicsState:
    rendering_mode: int = 0
    font_size: float | None = None
    font_widths: object = None  # scanner_fonts.WidthTable | None, current Tf font
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 100.0
    leading: float = 0.0
    rise: float = 0.0
    line_width: float = 1.0
    miter_limit: float = 10.0
    ctm: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    ctm_ambiguous: bool = False
    clip: tuple[float, float, float, float] | None = None
    clip_ambiguous: bool = False
    page_box: tuple[float, float, float, float] | None = None
    external_state_ambiguous: bool = False
    fill_color_space: str = "/DeviceGray"
    stroke_color_space: str = "/DeviceGray"
    fill_color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    stroke_color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    background_color: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def copy(self) -> GraphicsState:
        return GraphicsState(**vars(self))

    def restore(self, saved: GraphicsState) -> None:
        for name, value in vars(saved).items():
            setattr(self, name, value)


@dataclass
class TextRun:
    """Device-space bound of one visible text run; `rect` None means anywhere."""

    rect: tuple[float, float, float, float] | None
    scale: float
    slivers: int = 0


@dataclass
class Visibility:
    hidden: bool = False
    ambiguous: bool = False
    visible: bool = False
    runs: list[TextRun] = field(default_factory=list)
    paints: list[tuple[float, float, float, float]] = field(default_factory=list)
    covers: list[tuple[float, float, float, float]] = field(default_factory=list)
    unbounded_paint: bool = False

    def merge(self, other: Visibility) -> None:
        """Combine verdicts only; scanner_paint.merge_nested relates runs and paints."""
        self.hidden |= other.hidden
        self.ambiguous |= other.ambiguous
        self.visible |= other.visible

    def result(self) -> tuple[str, int, int]:
        if self.ambiguous:
            return ("ambiguous", int(self.hidden), 1)
        if self.hidden:
            return ("hidden_proven", 1, 0)
        return ("visible_only" if self.visible else "absent", 0, 0)


@dataclass
class ContentState:
    graphics: GraphicsState
    inherited_optional: bool = False
    font_metrics: FontMetricsProvider | None = None
    graphics_stack: list[GraphicsState] = field(default_factory=list)
    optional_content: list[bool] = field(default_factory=list)
    operands: list[Operand] = field(default_factory=list)
    in_text: bool = False
    text_matrix: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line_matrix: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    path_open: bool = False
    path_nonrect: bool = False
    path_rects: list[tuple[float, float, float, float]] = field(default_factory=list)
    path_bbox: tuple[float, float, float, float] | None = None
    path_unbounded: bool = False
    path_segments: int = 0
    pending_clip: bool = False
    visibility: Visibility = field(default_factory=Visibility)
