"""RFC004 §11 model-safe input contract — the allowlist as code.

Only the channels in §11.2 may reach the neural network: pixel planes,
pixel validity masks, page ordinals, tile→page coordinate transforms,
pixel dimensions, declared physical scale, the active capability profile,
input mode, and an explicit locale hint. Everything in §11.3
(:data:`FORBIDDEN_CHANNELS`) stays host-side for audit/loss masking and is
physically excluded from :class:`ModelSafeInput` — the constructor takes
only the allowlisted fields and accepts no ``**kwargs`` side channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from unflatten.ast.issues import Issue

#: RFC004 §11.3 forbidden channels. These keys may exist in dataset-side
#: records but must never appear in anything handed to the model.
FORBIDDEN_CHANNELS: frozenset[str] = frozenset(
    {
        "document_bundle_id",
        "lineage_root_id",
        "task_view_id",
        "task_example_id",
        "split",
        "filename",
        "path",
        "url",
        "org_id",
        "customer_id",
        "label_origin",
        "verification_tier",
        "label_grade",
        "masks_as_features",
        "node_ids",
        "source_map",
        "hidden_pdf_text",
        "ooxml_source",
        "renderer_diagnostics",
        "pyramid_level",
    }
)

InputMode = Literal["single_page", "full_document", "bounded_multi_page"]
_INPUT_MODES = ("single_page", "full_document", "bounded_multi_page")


@dataclass(frozen=True, slots=True)
class PagePlane:
    """One decoded page: raw RGB8 pixel bytes, row-major, no padding."""

    page_ordinal: int
    width_px: int
    height_px: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.page_ordinal < 0:
            raise ValueError(f"page_ordinal must be >= 0, got {self.page_ordinal}")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError(f"pixel dimensions must be > 0, got {self.width_px}x{self.height_px}")
        expected = self.width_px * self.height_px * 3
        if len(self.pixels) != expected:
            raise ValueError(
                f"pixels must be {expected} RGB8 bytes for "
                f"{self.width_px}x{self.height_px}, got {len(self.pixels)}"
            )


@dataclass(frozen=True, slots=True)
class PixelValidityMask:
    """Optional per-page validity bitmap; ``None`` means all pixels valid."""

    bitmap: bytes | None = None


@dataclass(frozen=True, slots=True)
class CoordinateTransform:
    """Tile→page affine transform over a common integer denominator.

    ``page = (M @ tile + t) / denom`` with ``M = [[xx, xy], [yx, yy]]`` and
    ``t = (tx, ty)``. A1 uses only the identity transform (no tiling yet).
    """

    xx: int
    xy: int
    yx: int
    yy: int
    tx: int
    ty: int
    denom: int = 1

    def __post_init__(self) -> None:
        if self.denom <= 0:
            raise ValueError(f"denom must be > 0, got {self.denom}")

    @classmethod
    def identity(cls) -> CoordinateTransform:
        return cls(xx=1, xy=0, yx=0, yy=1, tx=0, ty=0, denom=1)


@dataclass(frozen=True, slots=True)
class PhysicalScale:
    """Declared physical scale (§11.6): ``known`` requires a positive dpi,
    ``unknown`` forbids one. Estimated DPI is never declared ``known``."""

    status: Literal["known", "unknown"]
    dpi: int | None

    def __post_init__(self) -> None:
        if self.status == "known":
            if not isinstance(self.dpi, int) or isinstance(self.dpi, bool) or self.dpi <= 0:
                raise ValueError(f"status 'known' requires a positive integer dpi, got {self.dpi}")
        elif self.status == "unknown":
            if self.dpi is not None:
                raise ValueError(f"status 'unknown' forbids dpi, got {self.dpi}")
        else:
            raise ValueError(f"status must be 'known' or 'unknown', got {self.status!r}")


@dataclass(frozen=True, slots=True)
class LocaleHint:
    """An explicitly provided locale hint. Absence is modeled as ``None`` on
    :class:`ModelSafeInput`, never as an empty string."""

    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("locale hint value must be a non-empty string")


@dataclass(frozen=True, slots=True)
class ModelSafeInput:
    """The complete §11.2 allowlisted model input. Nothing else exists here."""

    planes: tuple[PagePlane, ...]
    masks: tuple[PixelValidityMask, ...]
    transforms: tuple[CoordinateTransform, ...]
    scale: PhysicalScale
    profile: str
    input_mode: InputMode
    locale_hint: LocaleHint | None

    def __post_init__(self) -> None:
        if not self.planes:
            raise ValueError("at least one page plane is required")
        if len(self.masks) != len(self.planes):
            raise ValueError(f"masks ({len(self.masks)}) must align with planes")
        if len(self.transforms) != len(self.planes):
            raise ValueError(f"transforms ({len(self.transforms)}) must align with planes")
        ordinals = [plane.page_ordinal for plane in self.planes]
        if any(b <= a for a, b in zip(ordinals, ordinals[1:])):
            raise ValueError(f"page ordinals must be strictly increasing, got {ordinals}")
        if not self.profile:
            raise ValueError("profile must be a non-empty string")
        if self.input_mode not in _INPUT_MODES:
            raise ValueError(f"input_mode must be one of {_INPUT_MODES}, got {self.input_mode!r}")
        if self.input_mode == "single_page" and len(self.planes) != 1:
            raise ValueError(f"input_mode 'single_page' requires exactly 1 plane, got {ordinals}")


def to_payload_dict(msi: ModelSafeInput) -> dict[str, Any]:
    """Plain-dict view of a :class:`ModelSafeInput` (allowlisted keys only).

    Used for defense-in-depth forbidden-channel checks on the projector's
    output; carries raw pixel bytes, so it is not a serialization format.
    """
    return {
        "profile": msi.profile,
        "input_mode": msi.input_mode,
        "scale": {"status": msi.scale.status, "dpi": msi.scale.dpi},
        "planes": [
            {
                "page_ordinal": plane.page_ordinal,
                "width_px": plane.width_px,
                "height_px": plane.height_px,
                "pixels": plane.pixels,
            }
            for plane in msi.planes
        ],
        "masks": [mask.bitmap for mask in msi.masks],
        "transforms": [
            {"xx": t.xx, "xy": t.xy, "yx": t.yx, "yy": t.yy, "tx": t.tx, "ty": t.ty,
             "denom": t.denom}
            for t in msi.transforms
        ],
        "locale_hint": msi.locale_hint.value if msi.locale_hint is not None else None,
    }


_FORBIDDEN_COMPRESSED = frozenset(k.replace("_", "") for k in FORBIDDEN_CHANNELS)


def validate_no_forbidden(payload: Any) -> list[Issue]:
    """Recursively reject §11.3 forbidden channel keys anywhere in ``payload``.

    Returns one ``MODEL_INPUT_FORBIDDEN_CHANNEL`` (RFC004 §42.4) issue per hit (RFC001 §50 style:
    never raises); ``path`` is the RFC 6901 JSON Pointer of the offending key.
    """
    issues: list[Issue] = []
    _walk(payload, "", issues)
    return issues


def _walk(value: Any, pointer: str, issues: list[Issue]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{pointer}/{_escape(key)}" if isinstance(key, str) else f"{pointer}/{key}"
            normalized = key
            if isinstance(normalized, bytes):
                normalized = normalized.decode("utf-8", "replace")
            if isinstance(normalized, str):
                normalized = normalized.casefold().replace("-", "_")
            if normalized in FORBIDDEN_CHANNELS or (
                isinstance(normalized, str)
                and normalized.replace("_", "") in _FORBIDDEN_COMPRESSED
            ):
                issues.append(
                    Issue(
                        code="MODEL_INPUT_FORBIDDEN_CHANNEL",
                        severity="error",
                        node_id=None,
                        path=child,
                        message=f"forbidden input channel {key!r} (RFC004 §11.3)",
                    )
                )
            _walk(item, child, issues)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk(item, f"{pointer}/{index}", issues)


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")
