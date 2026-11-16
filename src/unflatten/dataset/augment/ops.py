"""The seven DEC-040 distortion ops, pure functions over image bytes.

Two families (`unflatten.dataset.enums.ViewKind`): `scan_simulation`
(rotation, noise, jpeg, blur) and `photo_simulation` (perspective,
shadow, lighting). Every op has the same signature —
``(png_bytes, params, seed) -> (png_bytes, Matrix3x3)`` — so
`migration.m2b_augment` can dispatch through one table
(:data:`OP_FUNCS`) regardless of family.

Determinism (RFC005 line ~68, no free unaudited RNG): every op that
needs randomness (`noise`'s pixel field, `perspective`'s corner jitter)
seeds a fresh `numpy.random.default_rng(seed)` from the caller-supplied
`seed` — never a module-level or global RNG. Same `(png_bytes, params,
seed)` in -> byte-identical PNG out, always.

Canvas size never changes: every op keeps the input's `width_px` /
`height_px`, so `ops.py` and `geometry.py` share one `(width_px,
height_px)` pair per page for the whole recipe.

Photometric ops (`noise`, `jpeg`, `blur`, `shadow`, `lighting`) return
`geometry.identity_matrix()` — they do not move anything, so a region box
under them needs no `transform_bbox_ppm` call at all (the build side may
still apply the identity matrix for uniformity, which is a no-op).
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from unflatten.dataset.augment.geometry import (
    GeometryError,
    Matrix3x3,
    homography_from_points,
    identity_matrix,
    invert_matrix,
    rotation_matrix,
)

#: The RFC003 ViewKind vocabulary these ops implement, RFC004... no —
#: DEC-040's own vocabulary, one list per family, applied in this order.
SCAN_SIMULATION_OPS: tuple[str, ...] = ("rotation", "noise", "jpeg", "blur")
PHOTO_SIMULATION_OPS: tuple[str, ...] = ("perspective", "shadow", "lighting")
ALL_OPS: tuple[str, ...] = SCAN_SIMULATION_OPS + PHOTO_SIMULATION_OPS

_FILL_WHITE = (255, 255, 255)


class OpError(ValueError):
    """A distortion op was called with an unsupported image or params."""


def _load(png_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(png_bytes))
    image.load()
    return image.convert("RGB")


def _dump(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _warp(image: Image.Image, matrix: Matrix3x3) -> Image.Image:
    """Resample `image` by the forward map `matrix` (src -> dst pixel).

    PIL's `Image.transform(..., PERSPECTIVE)` wants the INVERSE (dst ->
    src) mapping as 8 coefficients; a pure affine matrix is a perspective
    matrix with a zero bottom-left 1x2 block, so one PIL call covers
    both `rotation` and `perspective` — the only difference is which
    `geometry` matrix built the forward map.
    """
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
        image.size,
        Image.PERSPECTIVE,
        coeffs,
        resample=Image.BICUBIC,
        fillcolor=_FILL_WHITE,
    )


# ------------------------------------------------------------- scan_simulation --


def apply_rotation(
    png_bytes: bytes, params: dict[str, Any], seed: int
) -> tuple[bytes, Matrix3x3]:
    """Small rotation about the page center (`params["angle_deg"]`)."""
    image = _load(png_bytes)
    width, height = image.size
    matrix = rotation_matrix(float(params["angle_deg"]), width, height)
    return _dump(_warp(image, matrix)), matrix


def apply_noise(png_bytes: bytes, params: dict[str, Any], seed: int) -> tuple[bytes, Matrix3x3]:
    """Additive Gaussian pixel noise, `params["sigma"]` (0-255 scale)."""
    image = _load(png_bytes)
    rng = np.random.default_rng(seed)
    array = np.asarray(image).astype(np.float64)
    array += rng.normal(0.0, float(params["sigma"]), array.shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    return _dump(Image.fromarray(array, mode="RGB")), identity_matrix()


def apply_jpeg(png_bytes: bytes, params: dict[str, Any], seed: int) -> tuple[bytes, Matrix3x3]:
    """JPEG-compress-and-reload artifacts, `params["quality"]` (1-95)."""
    image = _load(png_bytes)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(params["quality"]))
    buf.seek(0)
    reloaded = Image.open(buf)
    reloaded.load()
    return _dump(reloaded.convert("RGB")), identity_matrix()


def apply_blur(png_bytes: bytes, params: dict[str, Any], seed: int) -> tuple[bytes, Matrix3x3]:
    """Gaussian blur, `params["radius"]` in pixels."""
    image = _load(png_bytes)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(params["radius"])))
    return _dump(blurred), identity_matrix()


# ------------------------------------------------------------ photo_simulation --


def apply_perspective(
    png_bytes: bytes, params: dict[str, Any], seed: int
) -> tuple[bytes, Matrix3x3]:
    """Perspective jitter of the four page corners, `params["magnitude"]`
    (fraction of width/height each corner may move)."""
    image = _load(png_bytes)
    width, height = image.size
    magnitude = float(params["magnitude"])
    rng = np.random.default_rng(seed)
    src = ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))
    jitter_x = rng.uniform(-magnitude * width, magnitude * width, size=4)
    jitter_y = rng.uniform(-magnitude * height, magnitude * height, size=4)
    dst = tuple(
        (src[i][0] + float(jitter_x[i]), src[i][1] + float(jitter_y[i])) for i in range(4)
    )
    try:
        matrix = homography_from_points(src, dst)
    except GeometryError as exc:  # pragma: no cover — needs a pathological magnitude
        raise OpError(f"perspective: {exc}") from exc
    return _dump(_warp(image, matrix)), matrix


def apply_shadow(png_bytes: bytes, params: dict[str, Any], seed: int) -> tuple[bytes, Matrix3x3]:
    """A linear darkening band across the page, `params["strength"]`
    (0-1, fraction of brightness removed at the darkest edge)."""
    image = _load(png_bytes)
    width, height = image.size
    strength = float(params["strength"])
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0.0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    ys, xs = np.mgrid[0:height, 0:width]
    projection = xs * direction[0] + ys * direction[1]
    projection -= projection.min()
    span = projection.max() or 1.0
    gradient = 1.0 - strength * (projection / span)
    array = np.asarray(image).astype(np.float64)
    array *= gradient[..., None]
    array = np.clip(array, 0, 255).astype(np.uint8)
    return _dump(Image.fromarray(array, mode="RGB")), identity_matrix()


def apply_lighting(
    png_bytes: bytes, params: dict[str, Any], seed: int
) -> tuple[bytes, Matrix3x3]:
    """A smooth radial brightness gradient, `params["strength"]` (0-1)."""
    image = _load(png_bytes)
    width, height = image.size
    strength = float(params["strength"])
    rng = np.random.default_rng(seed)
    center = (rng.uniform(0.0, width), rng.uniform(0.0, height))
    ys, xs = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xs - center[0]) ** 2 + (ys - center[1]) ** 2)
    max_dist = dist.max() or 1.0
    gradient = 1.0 + strength * (1.0 - 2.0 * dist / max_dist)
    array = np.asarray(image).astype(np.float64)
    array *= gradient[..., None]
    array = np.clip(array, 0, 255).astype(np.uint8)
    return _dump(Image.fromarray(array, mode="RGB")), identity_matrix()


OP_FUNCS = {
    "rotation": apply_rotation,
    "noise": apply_noise,
    "jpeg": apply_jpeg,
    "blur": apply_blur,
    "perspective": apply_perspective,
    "shadow": apply_shadow,
    "lighting": apply_lighting,
}


def apply_recipe(
    png_bytes: bytes, ops: list[dict[str, Any]], seeds: list[int]
) -> tuple[bytes, Matrix3x3]:
    """Apply an ordered recipe (`[{"op": name, "params": {...}}, ...]`),
    one seed per op, returning the final PNG bytes and the CUMULATIVE
    forward matrix (`geometry.compose`, each step's matrix left-composed
    onto the ones before it)."""
    if len(ops) != len(seeds):
        raise OpError(f"apply_recipe: {len(ops)} ops but {len(seeds)} seeds")
    from unflatten.dataset.augment.geometry import compose

    current_bytes = png_bytes
    cumulative = identity_matrix()
    for entry, seed in zip(ops, seeds):
        op_name = entry["op"]
        if op_name not in OP_FUNCS:
            raise OpError(f"unknown op {op_name!r}, expected one of {sorted(OP_FUNCS)}")
        current_bytes, step_matrix = OP_FUNCS[op_name](current_bytes, entry["params"], seed)
        cumulative = compose(step_matrix, cumulative)
    return current_bytes, cumulative
