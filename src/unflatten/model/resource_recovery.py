"""Deterministic V0 exact-crop recovery from a page raster (RFC004 §28.4).

`masked_crop`/`rectified_crop` (RFC004 §28.2, DEC-047: dormant, opt-in,
additive) live here alongside `exact_crop`, sharing its private helpers and
its hash-after-bytes binding (`coordinate_transform_hash`,
`ENCODING_POLICY_HASH`) unchanged -- neither is wired into the builder
chain or the service. `OPEN-004-008` (`reconstructed_raster`) is out of
scope and keeps refusing.

Every crop function now opens `raster` through `visual_compression.
pre_encoder.guard_recovery_raster_source` (RFC004-ERRATA-005 E4 rule 5,
DEC-047) instead of directly -- a no-op type check for the plain
`bytes`/`bytearray` every existing caller already passes under the default
`pre_encoder_profile="bypass"`, so this is byte-identical for every
existing call site; it only starts refusing once a caller passes a
`ModelSafeInput`-shaped object under an active (non-`bypass`) profile."""

from __future__ import annotations

import io
from typing import Final

from PIL import Image, ImageDraw, UnidentifiedImageError

from unflatten.model.hashing import domain_hash
from unflatten.model.region_codec import PPM_MAX, Region
from unflatten.model.region_decode import RegionHypothesis
from unflatten.model.resource_binder import (
    RESOURCE_BYTES_INVALID,
    RESOURCE_CROP_BLANK,
    RESOURCE_CROP_LOW_CONFIDENCE,
    RESOURCE_CROP_TOO_SMALL,
    RESOURCE_SLOT_UNRESOLVED,
    RecoveredResource,
)
from unflatten.model.visual_compression.pre_encoder import guard_recovery_raster_source

MIN_SIDE_PX: Final = 16
MIN_INK_PPM: Final = 5_000
INK_GRAY_MAX: Final = 250
_IMAGE_NODE_TYPES: Final = frozenset({"inline_image", "floating_image"})

#: `rectified_crop`-only refusal: no quad to rectify from. Deliberately NOT
#: added to `model_issue_registry.py` -- that registry is scoped to codes
#: the builder chain can actually emit (module docstring there), and this
#: mode is not wired into the builder chain.
RESOURCE_RECTIFY_NO_QUAD: Final = "RESOURCE_RECTIFY_NO_QUAD"

#: Encoding policy versions (RFC004-ERRATA-004 decision 8 / DEC-049 point 6).
#: ``ENCODING_POLICY_VERSION_DEFAULT`` is today's behaviour -- no pHYs chunk,
#: byte-identical to before this parameter existed. ``ENCODING_POLICY_VERSION_DPI300``
#: writes a pHYs chunk pinning the recovered PNG at ``RECOVERED_PNG_DPI`` -- opt-in,
#: selected explicitly; it moves every `res_` id it touches (new bytes -> new sha256).
ENCODING_POLICY_VERSION_DEFAULT: Final = "v0"
ENCODING_POLICY_VERSION_DPI300: Final = "v1"
_ENCODING_POLICY_VERSIONS: Final = frozenset(
    {ENCODING_POLICY_VERSION_DEFAULT, ENCODING_POLICY_VERSION_DPI300}
)
RECOVERED_PNG_DPI: Final = 300


def _encoding_policy_payload(version: str) -> dict:
    payload: dict = {
        "format": "PNG",
        "ink_gray_max": INK_GRAY_MAX,
        "min_ink_ppm": MIN_INK_PPM,
        "min_side_px": MIN_SIDE_PX,
        "png": {"compress_level": 9, "optimize": False},
        "rgba_background": "white",
    }
    if version == ENCODING_POLICY_VERSION_DPI300:
        payload["png"] = {**payload["png"], "dpi": RECOVERED_PNG_DPI}
    return payload


def encoding_policy_hash(version: str = ENCODING_POLICY_VERSION_DEFAULT) -> str:
    """Pin the recovered-PNG encoding policy for one policy version.

    Each branch passes a literal domain string directly to `domain_hash`
    (never an f-string/lookup) on purpose: `test_m1_design_invariants.py
    ::test_each_hash_domain_has_exactly_one_owning_module` finds domains by
    walking the AST for a literal `domain_hash(...)` argument, so a
    dynamically-built domain string would silently stop being counted as an
    owned domain.

    Raises ``ValueError`` for any version outside ``_ENCODING_POLICY_VERSIONS``.
    """
    if version not in _ENCODING_POLICY_VERSIONS:
        raise ValueError(f"unknown encoding_policy_version: {version!r}")
    payload = _encoding_policy_payload(version)
    if version == ENCODING_POLICY_VERSION_DEFAULT:
        return domain_hash("rfc004.resource-recovery.encoding-policy/v0", payload)
    return domain_hash("rfc004.resource-recovery.encoding-policy/v1", payload)


#: Kept for existing callers/tests that pin the default policy's hash.
ENCODING_POLICY_HASH: Final = encoding_policy_hash(ENCODING_POLICY_VERSION_DEFAULT)


def coordinate_transform_hash(width: int, height: int) -> str:
    """Pin the floor-scaled ppm-to-raster transform for one raster shape."""
    return domain_hash(
        "rfc004.crop-transform/v0",
        {"height": height, "rule": "floor_scale", "source_px_delta_max": 1, "width": width},
    )


def exact_crop(
    raster: bytes,
    prediction: RegionHypothesis,
    *,
    confidence_threshold_ppm: int | None = None,
    pre_encoder_profile: str = "bypass",
    encoding_policy_version: str = ENCODING_POLICY_VERSION_DEFAULT,
) -> RecoveredResource:
    """Recover one image as a PNG or return a named, fail-closed refusal.

    ``None`` disables the confidence gate for report-only runs. Coordinates
    remain half-open after the pinned floor ppm-to-pixel transform.

    ``pre_encoder_profile`` (module docstring, RFC004-ERRATA-005 E4 rule 5):
    default ``"bypass"`` keeps this byte-identical to before the parameter
    existed.

    ``encoding_policy_version`` (RFC004-ERRATA-004 decision 8 / DEC-049 point 6):
    default ``ENCODING_POLICY_VERSION_DEFAULT`` keeps this byte-identical to
    before the parameter existed.
    """
    raster = guard_recovery_raster_source(raster, pre_encoder_profile=pre_encoder_profile)
    if confidence_threshold_ppm is not None and not 0 <= confidence_threshold_ppm <= PPM_MAX:
        raise ValueError("confidence_threshold_ppm must be within 0..1000000 or None")
    if not _can_resolve(prediction):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_SLOT_UNRESOLVED)
    if (
        confidence_threshold_ppm is not None
        and prediction.confidence_ppm < confidence_threshold_ppm
    ):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_LOW_CONFIDENCE)
    try:
        with Image.open(io.BytesIO(raster)) as opened:
            page = opened.convert("RGBA")
    except (OSError, UnidentifiedImageError):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_BYTES_INVALID)
    region = prediction.regions[0]
    box = _pixel_box(region, *page.size)
    width, height = box[2] - box[0], box[3] - box[1]
    if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_TOO_SMALL)
    crop = _flatten_on_white(page.crop(box))
    if not _has_minimum_ink(crop):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_BLANK)
    data = _encode_png(crop, version=encoding_policy_version)
    return RecoveredResource(
        prediction.local_node_id,
        "resolved",
        "image/png",
        data,
        width,
        height,
        fidelity="render_recovered",
        coordinate_transform_hash=coordinate_transform_hash(*page.size),
        encoding_policy_hash=encoding_policy_hash(encoding_policy_version),
    )


def masked_crop(
    raster: bytes,
    prediction: RegionHypothesis,
    *,
    confidence_threshold_ppm: int | None = None,
    pre_encoder_profile: str = "bypass",
    encoding_policy_version: str = ENCODING_POLICY_VERSION_DEFAULT,
) -> RecoveredResource:
    """RFC004 §28.2 `masked_crop`: `exact_crop`'s bbox crop, with every
    pixel OUTSIDE `prediction.regions[0].polygon_ppm` set to the same fixed
    background `exact_crop` already flattens transparency onto (white,
    `ENCODING_POLICY_HASH`'s `rgba_background`). A region with no polygon
    (bbox only) has nothing outside its own bbox to mask, so the result
    equals `exact_crop`'s crop rather than refusing.

    Opt-in (DEC-047): not wired into the builder chain or service.
    `pre_encoder_profile`: see `exact_crop`."""
    raster = guard_recovery_raster_source(raster, pre_encoder_profile=pre_encoder_profile)
    if confidence_threshold_ppm is not None and not 0 <= confidence_threshold_ppm <= PPM_MAX:
        raise ValueError("confidence_threshold_ppm must be within 0..1000000 or None")
    if not _can_resolve(prediction):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_SLOT_UNRESOLVED)
    if (
        confidence_threshold_ppm is not None
        and prediction.confidence_ppm < confidence_threshold_ppm
    ):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_LOW_CONFIDENCE)
    try:
        with Image.open(io.BytesIO(raster)) as opened:
            page = opened.convert("RGBA")
    except (OSError, UnidentifiedImageError):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_BYTES_INVALID)
    region = prediction.regions[0]
    box = _pixel_box(region, *page.size)
    width, height = box[2] - box[0], box[3] - box[1]
    if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_TOO_SMALL)
    crop = _flatten_on_white(page.crop(box))
    crop = _mask_outside_polygon(crop, region, box, page.size)
    if not _has_minimum_ink(crop):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_BLANK)
    data = _encode_png(crop, version=encoding_policy_version)
    return RecoveredResource(
        prediction.local_node_id,
        "resolved",
        "image/png",
        data,
        width,
        height,
        fidelity="render_recovered",
        coordinate_transform_hash=coordinate_transform_hash(*page.size),
        encoding_policy_hash=encoding_policy_hash(encoding_policy_version),
        mode="masked_crop",
    )


def rectified_crop(
    raster: bytes,
    prediction: RegionHypothesis,
    *,
    confidence_threshold_ppm: int | None = None,
    pre_encoder_profile: str = "bypass",
    encoding_policy_version: str = ENCODING_POLICY_VERSION_DEFAULT,
) -> RecoveredResource:
    """RFC004 §28.2 `rectified_crop`: a PIL `QUAD` perspective/affine
    rectification of the page raster from `prediction.regions[0]
    .polygon_ppm`'s four predicted corners onto `prediction.regions[0]
    .bbox_ppm`'s pixel-box size -- straightens a skewed quad into an
    upright rectangle instead of `exact_crop`'s plain axis-aligned crop.

    Corner order: `region_codec.canonicalize_polygon` always stores a
    quad clockwise, starting at its lexicographically-lowest (topmost,
    then leftmost) vertex `p0` -- i.e. `(p0, p1, p2, p3)` walks
    `(top, right-ish, bottom, left-ish)` for a roughly-upright quad. PIL's
    `QUAD` transform wants source corners in `(upper-left, lower-left,
    lower-right, upper-right)` order for its destination rectangle;
    `(p0, p3, p2, p1)` (keep the first vertex, reverse the rest) turns
    that clockwise cycle into exactly that order -- verified against a
    synthetic quadrant fixture (`test_resource_recovery.py`), not assumed
    from the `QUAD` docstring alone.

    Refuses with `RESOURCE_RECTIFY_NO_QUAD` (unregistered, see the module
    constant) when the prediction carries no polygon or the polygon is not
    exactly a quadrilateral -- there are only 3 predicted corners to
    rectify from otherwise, which `QUAD` cannot use.

    Opt-in (DEC-047): not wired into the builder chain or service.
    `pre_encoder_profile`: see `exact_crop`."""
    raster = guard_recovery_raster_source(raster, pre_encoder_profile=pre_encoder_profile)
    if confidence_threshold_ppm is not None and not 0 <= confidence_threshold_ppm <= PPM_MAX:
        raise ValueError("confidence_threshold_ppm must be within 0..1000000 or None")
    if not _can_resolve(prediction):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_SLOT_UNRESOLVED)
    if (
        confidence_threshold_ppm is not None
        and prediction.confidence_ppm < confidence_threshold_ppm
    ):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_LOW_CONFIDENCE)
    region = prediction.regions[0]
    if region.polygon_ppm is None or len(region.polygon_ppm) != 4:
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_RECTIFY_NO_QUAD)
    try:
        with Image.open(io.BytesIO(raster)) as opened:
            page = opened.convert("RGBA")
    except (OSError, UnidentifiedImageError):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_BYTES_INVALID)
    box = _pixel_box(region, *page.size)
    width, height = box[2] - box[0], box[3] - box[1]
    if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_TOO_SMALL)
    p0, p1, p2, p3 = region.polygon_ppm
    quad_data = _quad_pixel_data((p0, p3, p2, p1), page.size)
    rectified = page.transform((width, height), Image.QUAD, quad_data)
    crop = _flatten_on_white(rectified)
    if not _has_minimum_ink(crop):
        return RecoveredResource.unresolved(prediction.local_node_id, RESOURCE_CROP_BLANK)
    data = _encode_png(crop, version=encoding_policy_version)
    return RecoveredResource(
        prediction.local_node_id,
        "resolved",
        "image/png",
        data,
        width,
        height,
        fidelity="render_recovered",
        coordinate_transform_hash=coordinate_transform_hash(*page.size),
        encoding_policy_hash=encoding_policy_hash(encoding_policy_version),
        mode="rectified_crop",
    )


def _quad_pixel_data(
    points_ppm: tuple[tuple[int, int], ...], page_size: tuple[int, int]
) -> tuple[float, ...]:
    """`points_ppm` (page-relative ppm) -> flat `(x0, y0, x1, y1, ...)`
    page-pixel coordinates, the shape `Image.transform(..., Image.QUAD,
    data)` requires."""
    width, height = page_size
    return tuple(
        v
        for x, y in points_ppm
        for v in (x * width // PPM_MAX, y * height // PPM_MAX)
    )


def _mask_outside_polygon(
    crop: Image.Image,
    region: Region,
    box: tuple[int, int, int, int],
    page_size: tuple[int, int],
) -> Image.Image:
    """Set every `crop` pixel outside `region.polygon_ppm` to white.
    `crop` has no polygon to mask against (bbox-only region): returned
    unchanged."""
    if region.polygon_ppm is None:
        return crop
    width, height = page_size
    left, top = box[0], box[1]
    points_px = [
        (x * width // PPM_MAX - left, y * height // PPM_MAX - top) for x, y in region.polygon_ppm
    ]
    mask = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mask).polygon(points_px, fill=255)
    background = Image.new("RGB", crop.size, (255, 255, 255))
    return Image.composite(crop, background, mask)


def _can_resolve(prediction: RegionHypothesis) -> bool:
    return (
        prediction.node_type in _IMAGE_NODE_TYPES
        and prediction.alignment_status == "region"
        and len(prediction.regions) == 1
    )


def _pixel_box(region: Region, width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a validated ppm half-open box to an equally half-open pixel box."""
    x1, y1, x2, y2 = region.bbox_ppm
    return (
        x1 * width // PPM_MAX,
        y1 * height // PPM_MAX,
        x2 * width // PPM_MAX,
        y2 * height // PPM_MAX,
    )


def _flatten_on_white(image: Image.Image) -> Image.Image:
    white = Image.new("RGBA", image.size, (255, 255, 255, 255))
    white.alpha_composite(image)
    return white.convert("RGB")


def _has_minimum_ink(image: Image.Image) -> bool:
    grayscale = image.convert("L")
    ink = sum(pixel <= INK_GRAY_MAX for pixel in grayscale.get_flattened_data())
    return ink * PPM_MAX >= MIN_INK_PPM * image.width * image.height


def _encode_png(image: Image.Image, *, version: str = ENCODING_POLICY_VERSION_DEFAULT) -> bytes:
    if version not in _ENCODING_POLICY_VERSIONS:
        raise ValueError(f"unknown encoding_policy_version: {version!r}")
    output = io.BytesIO()
    save_kwargs = {"format": "PNG", "compress_level": 9, "optimize": False}
    if version == ENCODING_POLICY_VERSION_DPI300:
        save_kwargs["dpi"] = (RECOVERED_PNG_DPI, RECOVERED_PNG_DPI)
    image.save(output, **save_kwargs)
    return output.getvalue()
