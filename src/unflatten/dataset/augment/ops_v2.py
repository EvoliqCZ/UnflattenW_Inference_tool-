"""augmentation-v2 op registry (RFC003-ERRATA-004, design §1.3-§1.4).

``ops.py`` (the seven DEC-040 ops) stays untouched; this module adds the v2
signature ``(png_bytes, params, seed) -> OpResult`` and ONE table, ``OPS_V2``,
holding the seven v1 ops (wrapped by an adapter) plus the new ones.

The *declared* registry - :data:`OP_PARAMS_V2`, :data:`OP_PARAM_BOUNDS_V2`,
:data:`GEOMETRY_CLASS` - covers all 33 ops of design §1.3 from the start, so
policy validation and selection do not wait for every op to be ported.
:data:`OPS_V2` holds callables only for the ops that exist so far; an op is
*declared* here and *registered* when its implementation is added.

Rules (design §1.3): every op builds its own ``np.random.default_rng(seed)``
where it needs randomness - no ``random``, no global ``np.random``, no module
state; the canvas size never changes; hard parameter bounds are enforced by the
op itself (:func:`_check_params`), whatever the policy says.

Occluder-class ops receive the policy's ``ink_free`` mapping as a keyword
argument (``fn(png, params, seed, ink_free=...)``); every other op is called
with the three positional arguments only. :func:`apply_recipe_v2` does the
dispatch.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from unflatten.dataset.augment import ink as _ink
from unflatten.dataset.augment import ops as _v1
from unflatten.dataset.augment.geometry import (
    GeometryError,
    Matrix3x3,
    compose,
    homography_from_points,
    identity_matrix,
    invert_matrix,
)
from unflatten.dataset.augment.ops import OpError, _dump, _load

__all__ = [
    "GEOMETRY_CLASS",
    "OPS_V2",
    "OP_PARAMS_V2",
    "OP_PARAM_BOUNDS_V2",
    "OpError",
    "OpResult",
    "RecipeReport",
    "apply_recipe_v2",
    "is_int_param",
]

_FLOAT = "float"
_INT = "int"
_REFERENCE_DPI = 300.0

#: ``{op: (geometry class, {param: (kind, hard low, hard high)})}`` - design §1.3.
#: Bounds are inclusive and hold for every policy; kind ``"int"`` params are drawn
#: with ``rng.integers(low, high + 1)`` by the selection. Dict order is the
#: design table order (7 v1 ops first).
_SPEC: dict[str, tuple[str, dict[str, tuple[str, float, float]]]] = {
    # --- v1 ops (existing code, v2 ranges live in the policy file) ---
    "rotation": ("matrix", {"angle_deg": (_FLOAT, -10.0, 10.0)}),
    "noise": ("photometric", {"sigma": (_FLOAT, 0.0, 30.0)}),
    "jpeg": ("photometric", {"quality": (_INT, 1, 95)}),
    "blur": ("photometric", {"radius": (_FLOAT, 0.0, 6.0)}),
    "perspective": ("matrix", {"magnitude": (_FLOAT, 0.0, 0.12)}),
    "shadow": ("photometric", {"strength": (_FLOAT, 0.0, 0.8)}),
    "lighting": ("photometric", {"strength": (_FLOAT, 0.0, 0.6)}),
    # --- new in v2 ---
    "soft_resample": ("photometric", {"dpi": (_FLOAT, 50.0, 299.0)}),
    "page_offset": (
        "matrix",
        {
            "scale": (_FLOAT, 0.5, 1.0),
            "dx_frac": (_FLOAT, -0.1, 0.1),
            "dy_frac": (_FLOAT, -0.1, 0.1),
            "bg_gray": (_INT, 0, 255),
        },
    ),
    "toner_heavy": ("photometric", {"kernel_px": (_INT, 1, 5)}),
    "toner_faded": (
        "photometric",
        {"fade": (_FLOAT, 0.1, 0.9), "lowfreq_amp": (_FLOAT, 0.0, 0.3)},
    ),
    "paper_tint": (
        "photometric",
        {
            "tint_r": (_FLOAT, 150.0, 255.0),
            "tint_g": (_FLOAT, 150.0, 255.0),
            "tint_b": (_FLOAT, 140.0, 255.0),
            "noise_sigma": (_FLOAT, 0.0, 8.0),
        },
    ),
    "scanner_edge_shadow": (
        "photometric",
        {"width_frac": (_FLOAT, 0.02, 0.3), "depth": (_FLOAT, 0.1, 0.9)},
    ),
    "black_borders": (
        "occluder",
        {
            "top_frac": (_FLOAT, 0.0, 0.06),
            "side_frac": (_FLOAT, 0.0, 0.08),
            "gray": (_INT, 0, 60),
        },
    ),
    "halftone": ("photometric", {"period_px": (_FLOAT, 3.0, 16.0), "amp": (_FLOAT, 0.0, 30.0)}),
    "bleed_through": (
        "photometric",
        {"strength": (_FLOAT, 0.05, 0.5), "blur_px": (_FLOAT, 0.5, 4.0)},
    ),
    "binarize": (
        "photometric",
        {"threshold": (_FLOAT, 100.0, 220.0), "pre_blur": (_FLOAT, 0.0, 2.0)},
    ),
    "fax": ("photometric", {"threshold": (_FLOAT, 100.0, 200.0)}),
    "speckles": (
        "photometric",
        {"density_ppm": (_INT, 100, 3000), "radius_px": (_INT, 1, 3)},
    ),
    "roller_streaks": (
        "photometric",
        {
            "count": (_INT, 1, 8),
            "width_px": (_INT, 1, 16),
            "gain": (_FLOAT, 0.3, 0.95),
        },
    ),
    "copy_of_copy": (
        "photometric",
        {
            "passes": (_INT, 1, 3),
            "contrast": (_FLOAT, 1.0, 2.0),
            "offset": (_FLOAT, 30.0, 100.0),
        },
    ),
    "punch_holes": (
        "occluder",
        {
            "x_frac": (_FLOAT, 0.02, 0.08),
            "radius_frac": (_FLOAT, 0.005, 0.03),
            "gray": (_INT, 30, 120),
        },
    ),
    "staple_corner": (
        "occluder",
        {"length_frac": (_FLOAT, 0.02, 0.1), "width_px": (_INT, 3, 16)},
    ),
    "uneven_light": (
        "photometric",
        {
            "strength": (_FLOAT, 0.1, 0.8),
            "cx_frac": (_FLOAT, 0.0, 1.0),
            "cy_frac": (_FLOAT, 0.0, 1.0),
        },
    ),
    "glare": (
        "photometric",
        {
            "strength": (_FLOAT, 0.3, 0.95),
            "cx_frac": (_FLOAT, 0.0, 1.0),
            "cy_frac": (_FLOAT, 0.0, 1.0),
            "rx_frac": (_FLOAT, 0.05, 0.3),
            "ry_frac": (_FLOAT, 0.02, 0.15),
        },
    ),
    "motion_blur": (
        "photometric",
        {"length_px": (_INT, 3, 31), "angle_deg": (_FLOAT, 0.0, 180.0)},
    ),
    "desk_background": (
        "matrix",
        {
            "scale": (_FLOAT, 0.5, 0.95),
            "corner_jitter": (_FLOAT, 0.0, 0.06),
            "bg_r": (_FLOAT, 40.0, 220.0),
            "bg_g": (_FLOAT, 30.0, 200.0),
            "bg_b": (_FLOAT, 10.0, 160.0),
            "bg_noise_sigma": (_FLOAT, 0.0, 10.0),
        },
    ),
    "color_cast": ("photometric", {"cast": (_INT, 0, 1), "strength": (_FLOAT, 0.0, 1.5)}),
    "messenger_resize": (
        "photometric",
        {"long_edge_px": (_INT, 800, 2000), "jpeg_quality": (_INT, 30, 90)},
    ),
    "cropped_edge": (
        "occluder",
        {"side": (_INT, 0, 3), "frac": (_FLOAT, 0.01, 0.2), "gray": (_INT, 30, 200)},
    ),
    "coffee_stain": (
        "occluder",
        {
            "radius_frac": (_FLOAT, 0.03, 0.2),
            "ring_px": (_INT, 4, 30),
            "alpha": (_FLOAT, 0.1, 0.6),
        },
    ),
    "stamp": (
        "occluder",
        {
            "radius_frac": (_FLOAT, 0.03, 0.15),
            "color": (_INT, 0, 1),
            "alpha": (_FLOAT, 0.3, 0.95),
        },
    ),
    "pen_marks": (
        "occluder",
        {
            "count": (_INT, 1, 5),
            "width_px": (_INT, 2, 8),
            "len_frac": (_FLOAT, 0.05, 0.4),
        },
    ),
}

#: ``{op: {param: "float" | "int"}}`` - the exact parameter names every policy entry must carry.
OP_PARAMS_V2: dict[str, dict[str, str]] = {
    op: {name: kind for name, (kind, _lo, _hi) in params.items()}
    for op, (_cls, params) in _SPEC.items()
}
#: ``{op: {param: (low, high)}}`` - hard, inclusive bounds enforced by the op itself.
OP_PARAM_BOUNDS_V2: dict[str, dict[str, tuple[float, float]]] = {
    op: {name: (lo, hi) for name, (_kind, lo, hi) in params.items()}
    for op, (_cls, params) in _SPEC.items()
}
#: ``{op: "photometric" | "matrix" | "occluder" | "nonlinear"}``. No production op is nonlinear.
GEOMETRY_CLASS: dict[str, str] = {op: cls for op, (cls, _params) in _SPEC.items()}


def is_int_param(op: str, name: str) -> bool:
    """True when `op`'s parameter `name` is an integer (drawn with ``integers(low, high + 1)``)."""
    return OP_PARAMS_V2[op][name] == _INT


@dataclass(frozen=True)
class OpResult:
    """What a v2 op returns.

    `matrix` is the forward pixel map the op applied (identity for photometric
    ops and for a skipped op; ``None`` only for a nonlinear op, which
    :func:`apply_recipe_v2` refuses). `ink_overlap_ppm` is set by occluders: the
    share of ink (parts per million) the placed occluder covers.
    """

    png_bytes: bytes
    matrix: Matrix3x3 | None
    applied: bool
    skip_reason: str | None
    ink_overlap_ppm: int | None = None


class RecipeReport(TypedDict):
    applied: list[str]
    skipped: dict[str, str]
    ink_overlap_ppm: dict[str, int]


def _check_params(op: str, params: Mapping[str, Any]) -> None:
    """Refuse parameters that are not exactly the declared set or leave the hard bounds."""
    declared = OP_PARAMS_V2[op]
    if not isinstance(params, Mapping):
        raise OpError(f"{op}: params must be a mapping, got {type(params).__name__}")
    missing = sorted(set(declared) - set(params))
    unknown = sorted(set(params) - set(declared))
    if missing or unknown:
        raise OpError(
            f"{op}: params must be exactly {sorted(declared)}; "
            f"missing {missing}, unknown {unknown}"
        )
    for name, kind in declared.items():
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise OpError(f"{op}: param {name!r} must be a number, got {value!r}")
        if kind == _INT and float(value) != int(value):
            raise OpError(f"{op}: param {name!r} must be an integer, got {value!r}")
        low, high = OP_PARAM_BOUNDS_V2[op][name]
        if not (low <= value <= high):
            raise OpError(f"{op}: param {name!r}={value!r} outside the hard bound [{low}, {high}]")


# ------------------------------------------------------------------ helpers --


def _gray(array: np.ndarray) -> np.ndarray:
    """uint8 ``[H, W]`` gray of an RGB uint8 array (PIL's ITU-R 601 conversion)."""
    return np.asarray(Image.fromarray(array).convert("L"))


def _lowfreq(
    height: int, width: int, rng: np.random.Generator, scale: int, amp: float
) -> np.ndarray:
    """Smooth float32 ``[H, W]`` noise: a normal grid of ``~H/scale x W/scale`` cells, resized."""
    cells = (max(2, height // scale), max(2, width // scale))
    grid = rng.normal(0.0, 1.0, cells).astype(np.float32)
    field = Image.fromarray(grid, mode="F").resize((width, height), Image.BICUBIC)
    return np.asarray(field, dtype=np.float32) * np.float32(amp)


def _warp_fill(image: Image.Image, matrix: Matrix3x3, fill: Any) -> Image.Image:
    """Bicubic warp by the forward `matrix`, uncovered pixels `fill` (`ops._warp`, free fill)."""
    inverse = invert_matrix(matrix)
    scale = inverse[2][2]
    if scale == 0:
        raise OpError("inverse transform is singular (w-row collapsed to 0)")
    coeffs = (
        inverse[0][0] / scale,
        inverse[0][1] / scale,
        inverse[0][2] / scale,
        inverse[1][0] / scale,
        inverse[1][1] / scale,
        inverse[1][2] / scale,
        inverse[2][0] / scale,
        inverse[2][1] / scale,
    )
    return image.transform(
        image.size, Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor=fill
    )


# ------------------------------------------------------------------ v1 adapters --


def _adapt_v1(name: str) -> Callable[[bytes, Mapping[str, Any], int], OpResult]:
    v1_fn = _v1.OP_FUNCS[name]

    def adapter(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
        _check_params(name, params)
        out, matrix = v1_fn(png_bytes, dict(params), seed)
        return OpResult(out, matrix, True, None)

    adapter.__name__ = f"v1_{name}"
    return adapter


# ------------------------------------------------------------ photometric / matrix --


def _soft_resample(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Soft type: area-downsample to `dpi`/300 of the canvas, bicubic back."""
    _check_params("soft_resample", params)
    image = _load(png_bytes)
    width, height = image.size
    factor = float(params["dpi"]) / _REFERENCE_DPI
    small_size = (max(1, round(width * factor)), max(1, round(height * factor)))
    restored = image.resize(small_size, Image.BOX).resize((width, height), Image.BICUBIC)
    return OpResult(_dump(restored), identity_matrix(), True, None)


def _page_offset(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """The page, shrunk by `scale` and shifted, inside the unchanged canvas.

    The page is resized and pasted at an integer offset, and the returned matrix
    is derived from those exact integers, so box and pixels agree to sub-pixel.
    """
    _check_params("page_offset", params)
    image = _load(png_bytes)
    width, height = image.size
    new_w = max(1, round(width * float(params["scale"])))
    new_h = max(1, round(height * float(params["scale"])))
    off_x = round((width - new_w) / 2 + float(params["dx_frac"]) * width)
    off_y = round((height - new_h) / 2 + float(params["dy_frac"]) * height)
    gray = int(params["bg_gray"])
    canvas = Image.new("RGB", (width, height), (gray, gray, gray))
    canvas.paste(image.resize((new_w, new_h), Image.BICUBIC), (off_x, off_y))
    matrix: Matrix3x3 = (
        (new_w / width, 0.0, float(off_x)),
        (0.0, new_h / height, float(off_y)),
        (0.0, 0.0, 1.0),
    )
    return OpResult(_dump(canvas), matrix, True, None)


def _desk_background(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """The page, shrunk and corner-jittered by a homography, over a noisy desk colour."""
    _check_params("desk_background", params)
    rng = np.random.default_rng(seed)
    image = _load(png_bytes)
    width, height = image.size
    scale = float(params["scale"])
    jitter = float(params["corner_jitter"])
    x0, x1 = width * (1 - scale) / 2, width * (1 + scale) / 2
    y0, y1 = height * (1 - scale) / 2, height * (1 + scale) / 2
    base = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    shifts = rng.uniform(-jitter, jitter, (4, 2))
    dst = tuple(
        (base[i][0] + float(shifts[i, 0]) * width, base[i][1] + float(shifts[i, 1]) * height)
        for i in range(4)
    )
    src = ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))
    try:
        matrix = homography_from_points(src, dst)
    except GeometryError as exc:
        raise OpError(f"desk_background: {exc}") from exc
    colour = np.array([params["bg_r"], params["bg_g"], params["bg_b"]], dtype=np.float32)
    desk = colour + _lowfreq(height, width, rng, 40, 18.0)[..., None]
    desk += rng.standard_normal((height, width, 3), dtype=np.float32) * np.float32(
        params["bg_noise_sigma"]
    )
    background = Image.fromarray(np.clip(desk, 0, 255).astype(np.uint8), mode="RGB")
    page = _warp_fill(image, matrix, (255, 255, 255))
    mask = _warp_fill(Image.new("L", (width, height), 255), matrix, 0)
    return OpResult(_dump(Image.composite(page, background, mask)), matrix, True, None)


# ------------------------------------------------------------ photometric (scan, fax) --


def _planes(image: Image.Image) -> list[np.ndarray]:
    """The bands of `image` as contiguous float32 ``[H, W]`` arrays."""
    return [np.asarray(band, dtype=np.float32) for band in image.split()]


def _merge_planes(planes: list[np.ndarray]) -> bytes:
    """PNG bytes of float planes (rounded to nearest, clipped to uint8), merged to RGB."""
    bands = [
        Image.fromarray(np.clip(np.rint(plane), 0, 255).astype(np.uint8)) for plane in planes
    ]
    return _dump(Image.merge("RGB", bands))


def _lut_apply(image: Image.Image, table: np.ndarray) -> Image.Image:
    """Apply a 256-entry uint8 lookup table to every band of `image`."""
    return image.point(table.tolist() * len(image.getbands()))


def _toner_heavy(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Spilled toner: grey erosion (min filter) grows dark strokes by `kernel_px` per side."""
    _check_params("toner_heavy", params)
    radius = int(params["kernel_px"])
    eroded = np.asarray(_load(png_bytes)).copy()
    for axis in (0, 1):  # a square window's minimum is separable; edges use the in-bounds window
        source = eroded.copy()
        length = source.shape[axis]
        for shift in range(1, radius + 1):
            if shift >= length:
                break
            head = [slice(None)] * 3
            tail = [slice(None)] * 3
            head[axis], tail[axis] = slice(shift, None), slice(None, length - shift)
            np.minimum(eroded[tuple(head)], source[tuple(tail)], out=eroded[tuple(head)])
            np.minimum(eroded[tuple(tail)], source[tuple(head)], out=eroded[tuple(tail)])
    return OpResult(_dump(Image.fromarray(eroded)), identity_matrix(), True, None)


def _toner_faded(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Faded toner: ink density scaled by `fade` + a slowly varying field."""
    _check_params("toner_faded", params)
    rng = np.random.default_rng(seed)
    planes = _planes(_load(png_bytes))
    height, width = planes[0].shape
    wobble = _lowfreq(height, width, rng, 200, float(params["lowfreq_amp"]))
    density = np.clip(float(params["fade"]) + wobble, 0.05, 1.0)
    return OpResult(
        _merge_planes([255.0 - (255.0 - plane) * density for plane in planes]),
        identity_matrix(),
        True,
        None,
    )


def _paper_tint(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Grey / yellowish paper: multiplicative tint + low-frequency texture + grain."""
    _check_params("paper_tint", params)
    rng = np.random.default_rng(seed)
    planes = _planes(_load(png_bytes))
    height, width = planes[0].shape
    tints = (float(params["tint_r"]), float(params["tint_g"]), float(params["tint_b"]))
    texture = _lowfreq(height, width, rng, 16, 6.0)
    sigma = np.float32(params["noise_sigma"])
    out = []
    for plane, tint in zip(planes, tints):
        grain = rng.standard_normal((height, width), dtype=np.float32)
        out.append(plane * np.float32(tint / 255.0) + texture + grain * sigma)
    return OpResult(_merge_planes(out), identity_matrix(), True, None)


def _scanner_edge_shadow(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Dark gradient at the left or right page edge (side from the seed)."""
    _check_params("scanner_edge_shadow", params)
    rng = np.random.default_rng(seed)
    planes = _planes(_load(png_bytes))
    width = planes[0].shape[1]
    span = max(1, min(width, int(width * float(params["width_frac"]))))
    ramp = np.ones(width, dtype=np.float32)
    ramp[:span] = np.linspace(float(params["depth"]), 1.0, span, dtype=np.float32) ** 0.6
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    return OpResult(
        _merge_planes([plane * ramp[None, :] for plane in planes]), identity_matrix(), True, None
    )


def _halftone(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Printer raster: a sine grid at a seeded angle, added only where the page is not white."""
    _check_params("halftone", params)
    rng = np.random.default_rng(seed)
    planes = _planes(_load(png_bytes))
    height, width = planes[0].shape
    angle = float(rng.uniform(0.0, math.pi))
    xs = np.arange(width, dtype=np.float32) * np.float32(math.cos(angle))
    ys = np.arange(height, dtype=np.float32) * np.float32(math.sin(angle))
    phase = (ys[:, None] + xs[None, :]) * np.float32(2 * math.pi / float(params["period_px"]))
    pattern = np.sin(phase) * np.float32(params["amp"])
    out = [plane + pattern * (plane < 250) for plane in planes]
    return OpResult(_merge_planes(out), identity_matrix(), True, None)


def _bleed_through(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Show-through of the mirrored back side; only ever darkens."""
    _check_params("bleed_through", params)
    image = _load(png_bytes)
    back = image.transpose(Image.FLIP_LEFT_RIGHT).filter(
        ImageFilter.GaussianBlur(float(params["blur_px"]))
    )
    levels = np.arange(256, dtype=np.float32)
    table = np.rint(255.0 - (255.0 - levels) * float(params["strength"])).astype(np.uint8)
    ghost = _lut_apply(back, table)
    return OpResult(_dump(ImageChops.darker(image, ghost)), identity_matrix(), True, None)


def _binarize(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Black-and-white copy: slight blur, then a hard threshold."""
    _check_params("binarize", params)
    gray = _load(png_bytes).convert("L").filter(ImageFilter.GaussianBlur(float(params["pre_blur"])))
    threshold = float(params["threshold"])
    two_level = gray.point(lambda value: 255 if value > threshold else 0)
    return OpResult(_dump(two_level.convert("RGB")), identity_matrix(), True, None)


def _fax(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Fax: 200 x 100 dpi grid (from 300), 1-bit, stretched back with NEAREST."""
    _check_params("fax", params)
    gray = _load(png_bytes).convert("L")
    width, height = gray.size
    small = gray.resize(
        (max(1, round(width * 200 / _REFERENCE_DPI)), max(1, round(height * 100 / _REFERENCE_DPI))),
        Image.BOX,
    )
    threshold = float(params["threshold"])
    bits = small.point(lambda value: 255 if value > threshold else 0)
    restored = bits.resize((width, height), Image.NEAREST)
    return OpResult(_dump(restored.convert("RGB")), identity_matrix(), True, None)


def _speckles(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Salt-and-pepper dust: round dots, 70 % black / 30 % white (noise, not an occluder)."""
    _check_params("speckles", params)
    rng = np.random.default_rng(seed)
    image = _load(png_bytes)
    width, height = image.size
    count = int(height * width * float(params["density_ppm"]) / 1_000_000)
    ys = rng.integers(0, height, count)
    xs = rng.integers(0, width, count)
    radii = rng.integers(1, int(params["radius_px"]) + 1, count)
    black = rng.random(count) < 0.7
    draw = ImageDraw.Draw(image)
    for x, y, r, is_black in zip(xs.tolist(), ys.tolist(), radii.tolist(), black.tolist()):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 0) if is_black else (255, 255, 255))
    return OpResult(_dump(image), identity_matrix(), True, None)


def _roller_streaks(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Copier-roller streaks: `count` vertical bands, brightness times `gain`."""
    _check_params("roller_streaks", params)
    rng = np.random.default_rng(seed)
    array = np.asarray(_load(png_bytes)).copy()
    width = array.shape[1]
    band = int(params["width_px"])
    for x in rng.integers(0, width, int(params["count"])).tolist():
        array[:, x : x + band] = (
            array[:, x : x + band].astype(np.float32) * np.float32(params["gain"])
        ).astype(np.uint8)
    return OpResult(_dump(Image.fromarray(array)), identity_matrix(), True, None)


def _copy_of_copy(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Copy of a copy: `passes` x (blur 1.1, then clip((g - offset) * contrast)) in grey."""
    _check_params("copy_of_copy", params)
    levels = np.arange(256, dtype=np.float32)
    table = np.clip((levels - float(params["offset"])) * float(params["contrast"]), 0, 255)
    table = table.astype(np.uint8)
    gray = _load(png_bytes).convert("L")
    for _ in range(int(params["passes"])):
        gray = _lut_apply(gray.filter(ImageFilter.GaussianBlur(1.1)), table)
    return OpResult(_dump(gray.convert("RGB")), identity_matrix(), True, None)


# ------------------------------------------------------------------- photo --


def _uneven_light(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Uneven illumination: brightness falls off radially from (`cx_frac`, `cy_frac`)."""
    _check_params("uneven_light", params)
    planes = _planes(_load(png_bytes))
    height, width = planes[0].shape
    dx = ((np.arange(width, dtype=np.float32) - width * float(params["cx_frac"])) / width) ** 2
    dy = ((np.arange(height, dtype=np.float32) - height * float(params["cy_frac"])) / height) ** 2
    distance = np.sqrt(dy[:, None] + dx[None, :])
    factor = np.clip(1.0 - np.float32(params["strength"]) * distance, 0.0, 1.0)
    return OpResult(
        _merge_planes([plane * factor for plane in planes]), identity_matrix(), True, None
    )


def _glare(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Specular highlight: a blurred ellipse pushes pixels toward white by `strength`."""
    _check_params("glare", params)
    planes = _planes(_load(png_bytes))
    height, width = planes[0].shape
    cx, cy = width * float(params["cx_frac"]), height * float(params["cy_frac"])
    rx, ry = width * float(params["rx_frac"]), height * float(params["ry_frac"])
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(width * 0.04))
    weight = np.asarray(mask, dtype=np.float32) * np.float32(float(params["strength"]) / 255.0)
    out = [plane + (255.0 - plane) * weight for plane in planes]
    return OpResult(_merge_planes(out), identity_matrix(), True, None)


def _motion_blur(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Hand-shake blur: mean of `length_px` integer-shifted copies along `angle_deg`.

    Copy `i` is rolled by ``round((i - length // 2) * (sin, cos))`` (rows, cols);
    the uint16 sum is divided by `length` with round-half-up. Shifts wrap
    (``np.roll``), which only matters within `length` px of the canvas edge.
    """
    _check_params("motion_blur", params)
    array = np.asarray(_load(png_bytes))
    length = int(params["length_px"])
    angle = math.radians(float(params["angle_deg"]))
    cos, sin = math.cos(angle), math.sin(angle)
    total = np.zeros(array.shape, dtype=np.uint16)
    for index in range(length):
        offset = index - length // 2
        total += np.roll(array, (round(offset * sin), round(offset * cos)), axis=(0, 1))
    blurred = ((total + length // 2) // length).astype(np.uint8)
    return OpResult(_dump(Image.fromarray(blurred)), identity_matrix(), True, None)


#: (R, G, B) gain deltas at strength 1: a warm (bulb) and a cool (shade) cast.
_CASTS = ((0.0, -0.06, -0.16), (-0.14, -0.04, 0.04))


def _color_cast(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Warm (`cast` 0) or cool (`cast` 1) colour cast by per-channel gain."""
    _check_params("color_cast", params)
    image = _load(png_bytes)
    deltas = _CASTS[int(params["cast"])]
    levels = np.arange(256, dtype=np.float32)
    bands = [
        band.point(
            np.clip(levels * (1.0 + float(params["strength"]) * delta), 0, 255)
            .astype(np.uint8)
            .tolist()
        )
        for band, delta in zip(image.split(), deltas)
    ]
    return OpResult(_dump(Image.merge("RGB", bands)), identity_matrix(), True, None)


def _messenger_resize(png_bytes: bytes, params: Mapping[str, Any], seed: int) -> OpResult:
    """Chat-app re-encode: long edge to `long_edge_px` (BOX), JPEG, back to canvas (BICUBIC)."""
    _check_params("messenger_resize", params)
    image = _load(png_bytes)
    width, height = image.size
    factor = float(params["long_edge_px"]) / max(width, height)
    small = image.resize((max(1, round(width * factor)), max(1, round(height * factor))), Image.BOX)
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=int(params["jpeg_quality"]))
    decoded = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return OpResult(
        _dump(decoded.resize((width, height), Image.BICUBIC)), identity_matrix(), True, None
    )


# ------------------------------------------------------------ occluders (ink-free) --

_SKIP_INK_FREE = "ink_free_placement_failed"
_MIN_BAND_PX = 4
#: Mask level (0-255) from which a pixel counts as covered by an occluder footprint.
_FOOTPRINT_FLOOR = 16
_STAPLE_RGB = (110, 110, 120)
_STAMP_RGB = ((30, 50, 160), (170, 30, 40))
_COFFEE_RGB = (150, 105, 60)
_PEN_RGB = (25, 40, 140)


def _ink_context_of(image: Image.Image, ink_free: Mapping[str, Any]) -> _ink.InkContext:
    return _ink.ink_context(
        np.asarray(image),
        ink_threshold=int(ink_free["ink_threshold"]),
        dilate_px=int(ink_free["dilate_px"]),
    )


def _footprint_half(mask: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    """Occluder footprint at half resolution (2x2 max-pool, aligned with the ink mask)."""
    covered = np.asarray(mask) >= _FOOTPRINT_FLOOR
    height, width = covered.shape
    covered = np.pad(covered, ((0, height % 2), (0, width % 2)))
    pooled = covered.reshape(covered.shape[0] // 2, 2, covered.shape[1] // 2, 2).any(axis=(1, 3))
    return pooled[: shape[0], : shape[1]]


def _overlap_ppm(ctx: _ink.InkContext, mask: Image.Image) -> int:
    """Upper bound on full-resolution overlap, rounded up to parts per million.

    A touched half-resolution cell can cover up to ``scale**2`` pixels. Dividing
    by the original ink count preserves the bound even with uneven ink density
    or partial cells at odd canvas edges.
    """
    hit = int((_footprint_half(mask, ctx.mask.shape) & ctx.dilated).sum())
    if hit == 0:
        return 0
    return -(-hit * ctx.scale**2 * 1_000_000 // ctx.ink_pixels)


def _paint(image: Image.Image, mask: Image.Image, colour: tuple[int, int, int]) -> Image.Image:
    return Image.composite(Image.new("RGB", image.size, colour), image, mask)


def _place(
    png_bytes: bytes,
    image: Image.Image,
    ink_free: Mapping[str, Any],
    colour: tuple[int, int, int],
    make_mask: Callable[[int], Image.Image],
) -> OpResult:
    """Try ``placement_tries`` masks; paint the first within ``max_overlap_ppm`` of the ink."""
    ctx = _ink_context_of(image, ink_free)
    limit = int(ink_free["max_overlap_ppm"])
    best: int | None = None
    for attempt in range(max(1, int(ink_free["placement_tries"]))):
        mask = make_mask(attempt)
        ppm = _overlap_ppm(ctx, mask)
        if ppm <= limit:
            return OpResult(_dump(_paint(image, mask, colour)), identity_matrix(), True, None, ppm)
        best = ppm if best is None else min(best, ppm)
    return OpResult(png_bytes, identity_matrix(), False, _SKIP_INK_FREE, best)


def _band_place(
    png_bytes: bytes,
    image: Image.Image,
    ink_free: Mapping[str, Any],
    colour: tuple[int, int, int],
    bands: Mapping[str, tuple[int, int, int, int]],
) -> OpResult:
    """Paint edge bands (``{name: rect}``, rect already clipped to the ink-free margin).

    A band thinner than 4 px is dropped; with no band left the op skips.
    """
    kept = [
        rect
        for rect in bands.values()
        if min(rect[2] - rect[0], rect[3] - rect[1]) >= _MIN_BAND_PX
    ]
    if not kept:
        return OpResult(png_bytes, identity_matrix(), False, _SKIP_INK_FREE, None)
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in kept:
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=255)
    ppm = _overlap_ppm(_ink_context_of(image, ink_free), mask)
    if ppm > int(ink_free["max_overlap_ppm"]):
        return OpResult(png_bytes, identity_matrix(), False, _SKIP_INK_FREE, ppm)
    return OpResult(_dump(_paint(image, mask, colour)), identity_matrix(), True, None, ppm)


def _free_margins(image: Image.Image, ink_free: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """Ink-free distance (left, top, right, bottom) between the canvas edge and the ink box.

    The dilation radius is kept clear. A page without ink is free up to its centre.
    """
    width, height = image.size
    centre = (width // 2, height // 2, width // 2, height // 2)
    x1, y1, x2, y2 = _ink_context_of(image, ink_free).bbox or centre
    keep = int(ink_free["dilate_px"])
    return (
        max(0, x1 - keep),
        max(0, y1 - keep),
        max(0, width - min(x2, width) - keep),
        max(0, height - min(y2, height) - keep),
    )


def _black_borders(
    png_bytes: bytes, params: Mapping[str, Any], seed: int, *, ink_free: Mapping[str, Any]
) -> OpResult:
    """Scanner lid gaps: black bands on the top and both sides, never nearer than the ink box."""
    _check_params("black_borders", params)
    image = _load(png_bytes)
    width, height = image.size
    left, top, right, _bottom = _free_margins(image, ink_free)
    side = round(float(params["side_frac"]) * width)
    band_top = min(round(float(params["top_frac"]) * height), top)
    bands = {
        "top": (0, 0, width, band_top),
        "left": (0, 0, min(side, left), height),
        "right": (width - min(side, right), 0, width, height),
    }
    gray = int(params["gray"])
    return _band_place(png_bytes, image, ink_free, (gray, gray, gray), bands)


def _cropped_edge(
    png_bytes: bytes, params: Mapping[str, Any], seed: int, *, ink_free: Mapping[str, Any]
) -> OpResult:
    """A page-edge band hidden by the frame, on one side (0 top, 1 right, 2 bottom, 3 left)."""
    _check_params("cropped_edge", params)
    image = _load(png_bytes)
    width, height = image.size
    left, top, right, bottom = _free_margins(image, ink_free)
    side = int(params["side"])
    frac = float(params["frac"])
    if side in (0, 2):
        depth = min(round(frac * height), top if side == 0 else bottom)
        rect = (0, 0, width, depth) if side == 0 else (0, height - depth, width, height)
    else:
        depth = min(round(frac * width), right if side == 1 else left)
        rect = (width - depth, 0, width, height) if side == 1 else (0, 0, depth, height)
    gray = int(params["gray"])
    return _band_place(png_bytes, image, ink_free, (gray, gray, gray), {"band": rect})


def _punch_holes(
    png_bytes: bytes, params: Mapping[str, Any], seed: int, *, ink_free: Mapping[str, Any]
) -> OpResult:
    """Two punched holes at 35 % / 65 % of the page height (shifted together on retries)."""
    _check_params("punch_holes", params)
    rng = np.random.default_rng(seed)
    image = _load(png_bytes)
    width, height = image.size
    centre_x = float(params["x_frac"]) * width
    radius = max(2.0, float(params["radius_frac"]) * width)
    gray = int(params["gray"])

    def make_mask(attempt: int) -> Image.Image:
        shift = 0.0 if attempt == 0 else float(rng.uniform(-0.03, 0.03)) * height
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for frac in (0.35, 0.65):
            cy = frac * height + shift
            draw.ellipse([centre_x - radius, cy - radius, centre_x + radius, cy + radius], fill=255)
        return mask

    return _place(png_bytes, image, ink_free, (gray, gray, gray), make_mask)


def _staple_corner(
    png_bytes: bytes, params: Mapping[str, Any], seed: int, *, ink_free: Mapping[str, Any]
) -> OpResult:
    """A staple in the top-left corner: two short parallel diagonal strokes."""
    _check_params("staple_corner", params)
    rng = np.random.default_rng(seed)
    image = _load(png_bytes)
    width, _height = image.size
    length = max(float(_MIN_BAND_PX), float(params["length_frac"]) * width)
    stroke = int(params["width_px"])

    def make_mask(attempt: int) -> Image.Image:
        jitter = 0.0 if attempt == 0 else float(rng.uniform(0.0, 0.03)) * width
        origin = 0.02 * width + jitter
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for gap in (0.0, 3.0 * stroke):
            draw.line(
                [(origin, origin + gap), (origin + length, origin + gap + 0.35 * length)],
                fill=255,
                width=stroke,
            )
        return mask

    return _place(png_bytes, image, ink_free, _STAPLE_RGB, make_mask)


def _coffee_stain(
    png_bytes: bytes, params: Mapping[str, Any], seed: int, *, ink_free: Mapping[str, Any]
) -> OpResult:
    """A blurred ring with a faint fill, tinted coffee brown, at a seeded position."""
    _check_params("coffee_stain", params)
    rng = np.random.default_rng(seed)
    image = _load(png_bytes)
    width, height = image.size
    radius = max(4, round(float(params["radius_frac"]) * width))
    ring = int(params["ring_px"])
    alpha = float(params["alpha"])
    pad = ring + 24

    def make_mask(attempt: int) -> Image.Image:
        cx = int(rng.integers(0, width))
        cy = int(rng.integers(0, height))
        layer = Image.new("L", (2 * (radius + pad), 2 * (radius + pad)), 0)
        draw = ImageDraw.Draw(layer)
        local = [pad, pad, pad + 2 * radius, pad + 2 * radius]
        draw.ellipse(local, fill=round(alpha * 255 * 0.35))
        draw.ellipse(local, outline=round(alpha * 255), width=ring)
        layer = layer.filter(ImageFilter.GaussianBlur(6))
        mask = Image.new("L", image.size, 0)
        mask.paste(layer, (cx - radius - pad, cy - radius - pad))
        return mask

    return _place(png_bytes, image, ink_free, _COFFEE_RGB, make_mask)


def _stamp(
    png_bytes: bytes, params: Mapping[str, Any], seed: int, *, ink_free: Mapping[str, Any]
) -> OpResult:
    """A round stamp (shapes only, no text): two rings and three horizontal bars."""
    _check_params("stamp", params)
    rng = np.random.default_rng(seed)
    image = _load(png_bytes)
    width, height = image.size
    radius = max(8.0, float(params["radius_frac"]) * width)
    level = round(float(params["alpha"]) * 255)
    line = max(1, round(radius * 0.08))

    def make_mask(attempt: int) -> Image.Image:
        cx = float(rng.uniform(radius, max(radius + 1, width - radius)))
        cy = float(rng.uniform(radius, max(radius + 1, height - radius)))
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for scale in (1.0, 0.82):
            r = radius * scale
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=level, width=line)
        bar = max(1, round(radius * 0.14))
        for step in (-1, 0, 1):
            y = cy + step * radius * 0.32
            draw.rectangle(
                [cx - radius * 0.6, y - bar / 2, cx + radius * 0.6, y + bar / 2], fill=level
            )
        return mask

    return _place(png_bytes, image, ink_free, _STAMP_RGB[int(params["color"])], make_mask)


def _pen_marks(
    png_bytes: bytes, params: Mapping[str, Any], seed: int, *, ink_free: Mapping[str, Any]
) -> OpResult:
    """`count` hand-drawn sine-wave pen strokes at seeded positions."""
    _check_params("pen_marks", params)
    rng = np.random.default_rng(seed)
    image = _load(png_bytes)
    width, height = image.size
    length = max(8.0, float(params["len_frac"]) * width)
    stroke = int(params["width_px"])
    amplitude = 0.012 * height

    def make_mask(attempt: int) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for _ in range(int(params["count"])):
            x0 = float(rng.uniform(0, max(1.0, width - length)))
            y0 = float(rng.uniform(amplitude, max(amplitude + 1, height - amplitude)))
            phase = float(rng.uniform(0, 2 * math.pi))
            period = float(rng.uniform(0.15, 0.3)) * length
            points = [
                (x0 + t, y0 + amplitude * math.sin(2 * math.pi * t / period + phase))
                for t in np.arange(0.0, length + 1.0, 4.0)
            ]
            draw.line(points, fill=255, width=stroke, joint="curve")
        return mask

    return _place(png_bytes, image, ink_free, _PEN_RGB, make_mask)


#: The registered v2 ops. Tasks adding more ops extend this table.
OPS_V2: dict[str, Callable[..., OpResult]] = {
    **{name: _adapt_v1(name) for name in _v1.ALL_OPS},
    "soft_resample": _soft_resample,
    "page_offset": _page_offset,
    "desk_background": _desk_background,
    "toner_heavy": _toner_heavy,
    "toner_faded": _toner_faded,
    "paper_tint": _paper_tint,
    "scanner_edge_shadow": _scanner_edge_shadow,
    "halftone": _halftone,
    "bleed_through": _bleed_through,
    "binarize": _binarize,
    "fax": _fax,
    "speckles": _speckles,
    "roller_streaks": _roller_streaks,
    "copy_of_copy": _copy_of_copy,
    "uneven_light": _uneven_light,
    "glare": _glare,
    "motion_blur": _motion_blur,
    "color_cast": _color_cast,
    "messenger_resize": _messenger_resize,
    "black_borders": _black_borders,
    "punch_holes": _punch_holes,
    "staple_corner": _staple_corner,
    "cropped_edge": _cropped_edge,
    "coffee_stain": _coffee_stain,
    "stamp": _stamp,
    "pen_marks": _pen_marks,
}


# ------------------------------------------------------------------ recipe --


def apply_recipe_v2(
    png_bytes: bytes,
    ops: list[dict[str, Any]],
    seeds: list[int],
    *,
    ink_free: Mapping[str, Any],
) -> tuple[bytes, Matrix3x3, RecipeReport]:
    """Apply an ordered v2 recipe (``[{"op", "params"}, ...]``, one seed per op).

    Returns the PNG, the CUMULATIVE forward matrix (each step's matrix left-composed
    onto the previous ones, exactly like ``ops.apply_recipe``) and a report of what
    was applied or skipped. A step whose matrix is ``None`` (a nonlinear op, which
    cannot be mapped onto region boxes) raises :class:`OpError`.
    """
    if len(ops) != len(seeds):
        raise OpError(f"apply_recipe_v2: {len(ops)} ops but {len(seeds)} seeds")
    report: RecipeReport = {"applied": [], "skipped": {}, "ink_overlap_ppm": {}}
    current = png_bytes
    cumulative = identity_matrix()
    for entry, seed in zip(ops, seeds):
        name = entry["op"]
        if name not in OPS_V2:
            raise OpError(f"unknown v2 op {name!r}, expected one of {sorted(OPS_V2)}")
        fn = OPS_V2[name]
        if GEOMETRY_CLASS.get(name) == "occluder":
            result = fn(current, entry["params"], seed, ink_free=ink_free)
        else:
            result = fn(current, entry["params"], seed)
        if result.matrix is None:
            raise OpError(f"{name}: nonlinear op (matrix None) cannot be mapped onto region boxes")
        current = result.png_bytes
        cumulative = compose(result.matrix, cumulative)
        if result.applied:
            report["applied"].append(name)
        else:
            report["skipped"][name] = result.skip_reason or "skipped"
        if result.ink_overlap_ppm is not None:
            report["ink_overlap_ppm"][name] = result.ink_overlap_ppm
    return current, cumulative, report
