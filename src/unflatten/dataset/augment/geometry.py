"""Pure pixel-space transform matrices + the region-target box mapping.

T5a (`docs/decisions/DEC-040-augmentacia-a-kontrolna-vzorka.md`): geometric
distortions (rotation, perspective) move node positions, so every region
box the sidecar carries (`bbox_ppm`, millionths of the page — RFC003
§24.1) must move by the SAME transform the pixels were warped with.
Photometric distortions (noise, JPEG, blur, shadow, lighting) keep
geometry and use :func:`identity_matrix`.

A ``Matrix3x3`` is a forward pixel-space homogeneous mapping: a source
pixel ``(x, y)`` lands at ``M @ (x, y, 1)`` (after dividing by the
homogeneous ``w``) in the OUTPUT image. `unflatten.dataset.augment.ops`
applies the same matrix to pixels (via its inverse, which is what an
image-resampling call needs) so a box computed here always agrees with
where the pixels actually moved.

``transform_bbox_ppm`` is a pure function: build cache T5b calls it with
the matrix + page pixel dimensions this module's op recorded, never
guessing at geometry the CPU side already computed.
"""

from __future__ import annotations

Matrix3x3 = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]

#: RFC003 §24.1 `bbox_ppm`: coordinates are millionths of the page.
PPM_SCALE = 1_000_000

#: `transform_bbox_ppm` outcomes. A box the transform pushed fully off
#: the page, or whose clipped area is zero, is `"collapsed"` — never
#: silently kept at its old position and never fabricated a nonzero size.
BBOX_STATUS_OK = "ok"
BBOX_STATUS_COLLAPSED = "collapsed"


class GeometryError(ValueError):
    """A matrix or bbox given to this module is not well-formed."""


def identity_matrix() -> Matrix3x3:
    """The transform photometric-only ops record: geometry is unchanged."""
    return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _matmul(a: Matrix3x3, b: Matrix3x3) -> Matrix3x3:
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3)
    )  # type: ignore[return-value]


def compose(outer: Matrix3x3, inner: Matrix3x3) -> Matrix3x3:
    """The matrix of applying `inner` first, then `outer` (`outer @ inner`).

    This is how a multi-op recipe's cumulative transform is built: each
    step's own matrix left-multiplies the transform accumulated so far.
    """
    return _matmul(outer, inner)


def translation_matrix(dx: float, dy: float) -> Matrix3x3:
    return ((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0))


def rotation_matrix(angle_deg: float, width_px: int, height_px: int) -> Matrix3x3:
    """Rotate `angle_deg` about the image center, canvas size unchanged."""
    import math

    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotate = ((cos_t, -sin_t, 0.0), (sin_t, cos_t, 0.0), (0.0, 0.0, 1.0))
    cx, cy = width_px / 2.0, height_px / 2.0
    return compose(compose(translation_matrix(cx, cy), rotate), translation_matrix(-cx, -cy))


def homography_from_points(
    src: tuple[tuple[float, float], ...], dst: tuple[tuple[float, float], ...]
) -> Matrix3x3:
    """The 3x3 homography mapping each `src[i]` to `dst[i]` (4 points, DLT).

    Solves the standard direct linear transform for a projective mapping
    (`ops.apply_perspective`'s corner-to-corner recipe); degenerates to an
    affine matrix when `dst` is an affine image of `src`.
    """
    if len(src) != 4 or len(dst) != 4:
        raise GeometryError(f"homography_from_points needs exactly 4 point pairs, got {len(src)}")
    # Build the 8x8 linear system A h = b for h = (a,b,c,d,e,f,g,h_).
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (xp, yp) in zip(src, dst):
        rows.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        rhs.append(xp)
        rows.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
        rhs.append(yp)
    coeffs = _solve_linear_system(rows, rhs)
    a, b, c, d, e, f, g, h = coeffs
    return ((a, b, c), (d, e, f), (g, h, 1.0))


def _solve_linear_system(rows: list[list[float]], rhs: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting — no numpy dependency
    for an 8x8 solve; `ops.py` keeps numpy for pixel arrays only."""
    n = len(rows)
    augmented = [list(rows[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot_row][col]) < 1e-12:
            raise GeometryError("homography_from_points: degenerate point configuration")
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        augmented[col] = [v / pivot for v in augmented[col]]
        for r in range(n):
            if r == col:
                continue
            factor = augmented[r][col]
            if factor:
                augmented[r] = [
                    augmented[r][k] - factor * augmented[col][k] for k in range(n + 1)
                ]
    return [augmented[i][n] for i in range(n)]


def invert_matrix(matrix: Matrix3x3) -> Matrix3x3:
    """3x3 inverse via the general solver — `ops._warp` needs the inverse
    (dst -> src) mapping for image resampling; this module's matrices are
    always forward (src -> dst)."""
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    columns = []
    for col in range(3):
        rhs = [identity[row][col] for row in range(3)]
        rows = [list(matrix[row]) for row in range(3)]
        columns.append(_solve_linear_system(rows, rhs))
    return tuple(tuple(columns[c][r] for c in range(3)) for r in range(3))  # type: ignore[return-value]


def _apply_point(matrix: Matrix3x3, x: float, y: float) -> tuple[float, float]:
    px = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    py = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    w = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if w == 0:
        raise GeometryError("transform sent a point to infinity (w == 0)")
    return px / w, py / w


def transform_bbox_ppm(
    bbox_ppm: tuple[int, int, int, int],
    matrix: Matrix3x3,
    width_px: int,
    height_px: int,
) -> tuple[str, tuple[int, int, int, int] | None]:
    """Map one `bbox_ppm` box through `matrix`, clip to the page.

    Returns `(status, bbox)`: `bbox` is a new `bbox_ppm` box on
    `BBOX_STATUS_OK`, `None` on `BBOX_STATUS_COLLAPSED` (the transformed,
    clipped box has zero or negative area — pushed off the page or
    squeezed to nothing). Never fabricates a box for the collapsed case.
    """
    x1, y1, x2, y2 = bbox_ppm
    if x2 <= x1 or y2 <= y1:
        raise GeometryError(f"bbox_ppm {bbox_ppm!r} is not a positive-area box")
    if width_px <= 0 or height_px <= 0:
        raise GeometryError(f"page dimensions must be positive, got {(width_px, height_px)!r}")

    corners_px = [
        (x1 / PPM_SCALE * width_px, y1 / PPM_SCALE * height_px),
        (x2 / PPM_SCALE * width_px, y1 / PPM_SCALE * height_px),
        (x2 / PPM_SCALE * width_px, y2 / PPM_SCALE * height_px),
        (x1 / PPM_SCALE * width_px, y2 / PPM_SCALE * height_px),
    ]
    transformed = [_apply_point(matrix, px, py) for px, py in corners_px]
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]

    clip_x1, clip_x2 = max(0.0, min(xs)), min(float(width_px), max(xs))
    clip_y1, clip_y2 = max(0.0, min(ys)), min(float(height_px), max(ys))
    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        return BBOX_STATUS_COLLAPSED, None

    return BBOX_STATUS_OK, (
        round(clip_x1 / width_px * PPM_SCALE),
        round(clip_y1 / height_px * PPM_SCALE),
        round(clip_x2 / width_px * PPM_SCALE),
        round(clip_y2 / height_px * PPM_SCALE),
    )
