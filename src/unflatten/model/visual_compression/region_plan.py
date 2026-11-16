"""RFC004-ERRATA-005 E4 variant 1: region resolution planner.

Low-res global view + full-res crops of regions that need them. This module
only plans coordinates from region proposals it is handed — it does not
produce those proposals (the scout model that proposes regions is out of
scope, per the task).

:class:`RegionPlanCoordinateTransform` is a *new* type. It intentionally does
not touch :class:`unflatten.model.input_contract.CoordinateTransform`, which
stays the identity-only tile→page transform documented there (§11, A1 has no
tiling yet) — extending it in place would silently change its behaviour for
every existing caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

RiskClass = Literal["protected", "high", "medium", "low", "uncertain"]
_RISK_CLASSES: tuple[RiskClass, ...] = ("protected", "high", "medium", "low", "uncertain")

#: E4 rules 6-7: uncertainty and the enumerated protected classes both get
#: full resolution. This planner only carries the risk *class*, not the
#: enumerated content types themselves (font size, table borders, ...) —
#: callers assign "protected" for those per rule 7.
FULL_RESOLUTION_RISK_CLASSES: frozenset[RiskClass] = frozenset({"protected", "uncertain"})

Resolution = Literal["full", "reduced"]


@dataclass(frozen=True)
class RegionPlanCoordinateTransform:
    """Invertible affine transform between plan-space and original-page pixels.

    Same rational-affine convention as
    :class:`unflatten.model.input_contract.CoordinateTransform`
    (``page = (M @ plan + t) / denom``), but this type additionally supports
    exact inversion via :meth:`to_plan`.
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
        if self._determinant() == 0:
            raise ValueError("transform is not invertible (determinant 0)")

    def _determinant(self) -> int:
        return self.xx * self.yy - self.xy * self.yx

    @classmethod
    def identity(cls) -> RegionPlanCoordinateTransform:
        return cls(xx=1, xy=0, yx=0, yy=1, tx=0, ty=0, denom=1)

    @classmethod
    def downsample(cls, factor: int) -> RegionPlanCoordinateTransform:
        """Global low-res view: one plan pixel covers ``factor`` page pixels."""
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")
        return cls(xx=factor, xy=0, yx=0, yy=factor, tx=0, ty=0, denom=1)

    @classmethod
    def crop(cls, x0: int, y0: int) -> RegionPlanCoordinateTransform:
        """Full-res crop: plan-space origin is the region's page-space origin."""
        return cls(xx=1, xy=0, yx=0, yy=1, tx=x0, ty=y0, denom=1)

    def to_page(self, plan_x: int, plan_y: int) -> tuple[int, int]:
        page_x = Fraction(self.xx * plan_x + self.xy * plan_y + self.tx, self.denom)
        page_y = Fraction(self.yx * plan_x + self.yy * plan_y + self.ty, self.denom)
        return _require_int(page_x), _require_int(page_y)

    def to_plan(self, page_x: int, page_y: int) -> tuple[int, int]:
        det = self._determinant()
        scaled_x = self.denom * page_x - self.tx
        scaled_y = self.denom * page_y - self.ty
        plan_x = Fraction(self.yy * scaled_x - self.xy * scaled_y, det)
        plan_y = Fraction(-self.yx * scaled_x + self.xx * scaled_y, det)
        return _require_int(plan_x), _require_int(plan_y)


def _require_int(value: Fraction) -> int:
    if value.denominator != 1:
        raise ValueError(f"coordinate {value} is not exactly invertible at this point")
    return value.numerator


@dataclass(frozen=True)
class RegionProposal:
    """A region to resolve, in original-page pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int
    risk_class: RiskClass

    def __post_init__(self) -> None:
        if self.risk_class not in _RISK_CLASSES:
            raise ValueError(f"risk_class must be one of {_RISK_CLASSES}, got {self.risk_class!r}")
        if self.x0 < 0 or self.y0 < 0:
            raise ValueError(f"region origin must be >= 0, got ({self.x0}, {self.y0})")
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError(
                f"region must have positive extent, got "
                f"({self.x0},{self.y0})-({self.x1},{self.y1})"
            )


@dataclass(frozen=True)
class ResolvedRegion:
    proposal: RegionProposal
    resolution: Resolution
    transform: RegionPlanCoordinateTransform


@dataclass(frozen=True)
class RegionResolutionPlan:
    page_width_px: int
    page_height_px: int
    global_downsample: int
    global_transform: RegionPlanCoordinateTransform
    regions: tuple[ResolvedRegion, ...]


def plan_region_resolution(
    page_width_px: int,
    page_height_px: int,
    regions: Sequence[RegionProposal],
    *,
    global_downsample: int = 4,
) -> RegionResolutionPlan:
    """E4 variant 1: low-res global view + full-res crops for regions that need them.

    ``regions`` are region *proposals*, taken as given — no scouting happens
    here. Regions whose ``risk_class`` is in :data:`FULL_RESOLUTION_RISK_CLASSES`
    (protected content per rule 7, or "uncertain" per rule 6) resolve to a
    full-resolution crop; everything else resolves to the shared downsampled
    global view.
    """
    if page_width_px <= 0 or page_height_px <= 0:
        raise ValueError(f"page dimensions must be > 0, got {page_width_px}x{page_height_px}")
    if global_downsample <= 0:
        raise ValueError(f"global_downsample must be > 0, got {global_downsample}")
    global_transform = RegionPlanCoordinateTransform.downsample(global_downsample)
    resolved: list[ResolvedRegion] = []
    for region in regions:
        if region.x1 > page_width_px or region.y1 > page_height_px:
            raise ValueError(
                f"region ({region.x0},{region.y0})-({region.x1},{region.y1}) "
                f"exceeds page {page_width_px}x{page_height_px}"
            )
        if region.risk_class in FULL_RESOLUTION_RISK_CLASSES:
            resolved.append(
                ResolvedRegion(
                    proposal=region,
                    resolution="full",
                    transform=RegionPlanCoordinateTransform.crop(region.x0, region.y0),
                )
            )
        else:
            resolved.append(
                ResolvedRegion(proposal=region, resolution="reduced", transform=global_transform)
            )
    return RegionResolutionPlan(
        page_width_px=page_width_px,
        page_height_px=page_height_px,
        global_downsample=global_downsample,
        global_transform=global_transform,
        regions=tuple(resolved),
    )
