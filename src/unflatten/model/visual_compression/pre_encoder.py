"""RFC004-ERRATA-005 E4: ``pre_encoder_compressor`` slot, rule 1.

``bypass`` is the product path and the control arm: it must produce a
byte-identical :class:`~unflatten.model.input_contract.ModelSafeInput` to
today's 300 DPI path. ``adaptive_v0`` is constructible only when an
explicit policy *and* scout weights are supplied (DEC-047: dormant code is
opt-in, never a silent default) -- calling
:func:`build_pre_encoder_compressor` with just ``"adaptive_v0"`` still
refuses at construction exactly as before, never silently falling back to
bypass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Protocol

import torch
from PIL import Image

from unflatten.model.input_contract import ModelSafeInput, PagePlane
from unflatten.model.visual_compression.region_plan import (
    RegionResolutionPlan,
    ResolvedRegion,
    plan_region_resolution,
)
from unflatten.model.visual_compression.scout import ScoutModel, scout_logits_to_region_proposals

#: E4: "Nový voliteľný komponent `pre_encoder_compressor` s konfiguráciou
#: `bypass | adaptive_v0 | experimental_*`". Only the two non-experimental
#: values are wired here.
PRE_ENCODER_COMPRESSOR_PROFILES: tuple[str, ...] = ("bypass", "adaptive_v0")

_POLICY_PATH: Final = Path(__file__).parent / "policies" / "pre_encoder_adaptive_v0_policy.json"


class PreEncoderCompressor(Protocol):
    """RFC004-ERRATA-005 E4 §12.1 boundary: acts before the vision encoder."""

    profile: str

    def compress(self, msi: ModelSafeInput) -> ModelSafeInput: ...


class PreEncoderCompressorUnimplementedError(NotImplementedError):
    """Raised when a declared-but-unimplemented profile is selected."""


class PreEncoderAdaptivePolicyError(ValueError):
    """A pre-encoder policy payload does not match the shape this module reads."""


class BypassPreEncoderCompressor:
    """E4 rule 1: byte-identical to today's 300 DPI model input."""

    profile = "bypass"

    def compress(self, msi: ModelSafeInput) -> ModelSafeInput:
        return msi


class AdaptivePreEncoderPolicy:
    """§12.1 policy fields for `adaptive_v0` -- data, never code literals
    (DEC-047 condition 4: "topológia a prahy sú parametre")."""

    __slots__ = ("policy_version", "scout_grid", "scout_hidden", "global_downsample")

    def __init__(
        self, *, policy_version: str, scout_grid: int, scout_hidden: int, global_downsample: int
    ) -> None:
        if scout_grid <= 0 or scout_hidden <= 0 or global_downsample <= 0:
            raise PreEncoderAdaptivePolicyError(
                "scout_grid, scout_hidden and global_downsample must all be > 0"
            )
        self.policy_version = policy_version
        self.scout_grid = scout_grid
        self.scout_hidden = scout_hidden
        self.global_downsample = global_downsample


def load_default_pre_encoder_policy() -> AdaptivePreEncoderPolicy:
    """Load the versioned starting policy from
    `policies/pre_encoder_adaptive_v0_policy.json`."""
    payload = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    return AdaptivePreEncoderPolicy(
        policy_version=payload["policy_version"],
        scout_grid=payload["scout_grid"],
        scout_hidden=payload["scout_hidden"],
        global_downsample=payload["global_downsample"],
    )


class AdaptivePreEncoderCompressor:
    """RFC004-ERRATA-005 E4 variant 1 (region resolution plan), constructed
    only via :func:`build_pre_encoder_compressor` with an explicit policy
    and scout weights.

    **Step 0 == bypass (E4 rule 2).** `scout`'s head is zero-initialized
    (see :class:`~unflatten.model.visual_compression.scout.ScoutModel`), so
    every cell resolves to `"protected"` -> full resolution -> every plane
    is returned unchanged, byte-identical to `bypass`
    (`test_pre_encoder.py::test_step_zero_matches_bypass_identity`).

    Pixel content changes only inside regions the plan marks `"reduced"`;
    plane dimensions, masks, transforms, scale and profile are never
    touched (E4 rule 4: "súradnice sú vždy pôvodnej strany" holds trivially
    since this profile never changes a plane's shape).
    """

    profile = "adaptive_v0"

    def __init__(self, *, policy: AdaptivePreEncoderPolicy, scout: ScoutModel) -> None:
        self.policy = policy
        self.scout = scout

    def compress(self, msi: ModelSafeInput) -> ModelSafeInput:
        new_planes = tuple(self._compress_plane(plane) for plane in msi.planes)
        return ModelSafeInput(
            planes=new_planes,
            masks=msi.masks,
            transforms=msi.transforms,
            scale=msi.scale,
            profile=msi.profile,
            input_mode=msi.input_mode,
            locale_hint=msi.locale_hint,
        )

    def _compress_plane(self, plane: PagePlane) -> PagePlane:
        low_res = _low_res_view(plane, self.policy.scout_grid)
        with torch.no_grad():
            logits = self.scout(low_res)
        proposals = scout_logits_to_region_proposals(logits, plane.width_px, plane.height_px)
        plan = plan_region_resolution(
            plane.width_px,
            plane.height_px,
            proposals,
            global_downsample=self.policy.global_downsample,
        )
        return apply_region_resolution_plan(plane, plan)


def _low_res_view(plane: PagePlane, grid: int) -> torch.Tensor:
    """`(1, 3, grid, grid)` float tensor the scout reads -- cheap and does
    not need the exact page raster path, only a coarse view of it."""
    image = Image.frombytes("RGB", (plane.width_px, plane.height_px), plane.pixels)
    small = image.resize((grid, grid), Image.BILINEAR)
    tensor = torch.frombuffer(bytearray(small.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(grid, grid, 3).permute(2, 0, 1).unsqueeze(0).to(torch.float32) / 255.0
    return tensor


def _pixelate_regions(
    plane: PagePlane, regions: list[ResolvedRegion], downsample: int
) -> PagePlane:
    """Box-filter down/up sample each `"reduced"` region in place. Plane
    dimensions are unchanged -- only pixel content inside each region."""
    image = Image.frombytes("RGB", (plane.width_px, plane.height_px), plane.pixels)
    for region in regions:
        box = (
            region.proposal.x0,
            region.proposal.y0,
            region.proposal.x1,
            region.proposal.y1,
        )
        width, height = box[2] - box[0], box[3] - box[1]
        crop = image.crop(box)
        degraded = crop.resize(
            (max(1, width // downsample), max(1, height // downsample)), Image.BILINEAR
        ).resize((width, height), Image.NEAREST)
        image.paste(degraded, box)
    return PagePlane(
        page_ordinal=plane.page_ordinal,
        width_px=plane.width_px,
        height_px=plane.height_px,
        pixels=image.tobytes(),
    )


def apply_region_resolution_plan(
    plane: PagePlane, plan: RegionResolutionPlan
) -> PagePlane:
    """Materialize a low-res global view plus full-res protected crops.

    The previous implementation only pixelated proposals explicitly marked
    ``reduced``. Unmentioned background therefore stayed full-resolution,
    which was not the plan encoded by :class:`RegionResolutionPlan` and made
    an ideal-map rehearsal unable to exercise PRE compression. This function
    first degrades the whole page according to ``global_downsample``, then
    restores every ``full`` proposal from the original raster.

    A plan whose full-resolution proposals tile the page remains byte-identical
    to bypass. That is the zero-initialized scout's step-0 contract.
    """
    if (plane.width_px, plane.height_px) != (plan.page_width_px, plan.page_height_px):
        raise ValueError(
            "region plan dimensions do not match the page plane: "
            f"{plan.page_width_px}x{plan.page_height_px} != "
            f"{plane.width_px}x{plane.height_px}"
        )

    original = Image.frombytes("RGB", (plane.width_px, plane.height_px), plane.pixels)
    reduced = original.resize(
        (
            max(1, plane.width_px // plan.global_downsample),
            max(1, plane.height_px // plan.global_downsample),
        ),
        Image.BILINEAR,
    ).resize((plane.width_px, plane.height_px), Image.NEAREST)

    for region in plan.regions:
        if region.resolution != "full":
            continue
        box = (
            region.proposal.x0,
            region.proposal.y0,
            region.proposal.x1,
            region.proposal.y1,
        )
        reduced.paste(original.crop(box), box)

    return PagePlane(
        page_ordinal=plane.page_ordinal,
        width_px=plane.width_px,
        height_px=plane.height_px,
        pixels=reduced.tobytes(),
    )


def build_pre_encoder_compressor(
    profile: str,
    *,
    policy: AdaptivePreEncoderPolicy | None = None,
    scout: ScoutModel | None = None,
) -> PreEncoderCompressor:
    """Construct the ``pre_encoder_compressor`` for ``profile``.

    Refuses at construction time for known-but-unimplemented and unknown
    profiles -- there is no silent fallback to ``bypass``. ``adaptive_v0``
    additionally refuses unless the caller supplies both ``policy`` and
    ``scout`` explicitly (DEC-047): a default-constructed ``adaptive_v0``
    is never usable by accident.
    """
    if profile == "bypass":
        return BypassPreEncoderCompressor()
    if profile == "adaptive_v0":
        if policy is None or scout is None:
            raise PreEncoderCompressorUnimplementedError(
                "pre_encoder_compressor profile 'adaptive_v0' requires an "
                "explicit policy and scout weights (RFC004-ERRATA-005 E4, "
                "DEC-047); construction without both refuses rather than "
                "silently falling back to bypass"
            )
        return AdaptivePreEncoderCompressor(policy=policy, scout=scout)
    raise ValueError(
        f"unknown pre_encoder_compressor profile {profile!r}, expected one of "
        f"{PRE_ENCODER_COMPRESSOR_PROFILES}"
    )


class ResourceRecoveryRasterSourceError(TypeError):
    """Resource Recovery was handed something other than raw raster bytes."""


def guard_recovery_raster_source(raster: object, *, pre_encoder_profile: str) -> bytes:
    """RFC004-ERRATA-005 E4 rule 5: "Resource Recovery číta pôvodný raster,
    nikdy komprimovaný vstup enkodéra."

    A boundary guard, not a compressor: it asserts `raster` is the raw
    bytes of the *original* page raster -- never a
    :class:`~unflatten.model.input_contract.ModelSafeInput` (the type any
    active `pre_encoder_compressor` profile rewrites) -- and returns it
    unchanged, regardless of `pre_encoder_profile`. Today's behaviour
    (`pre_encoder_profile == "bypass"`) is unaffected: `bypass` never
    changes the raster, so this call is a no-op type check either way; the
    check only starts mattering once a caller wires an active profile in.
    """
    if pre_encoder_profile not in PRE_ENCODER_COMPRESSOR_PROFILES:
        raise ValueError(
            f"unknown pre_encoder_compressor profile {pre_encoder_profile!r}, "
            f"expected one of {PRE_ENCODER_COMPRESSOR_PROFILES}"
        )
    if not isinstance(raster, (bytes, bytearray)):
        raise ResourceRecoveryRasterSourceError(
            "Resource Recovery must read the original page raster as bytes "
            f"(RFC004-ERRATA-005 E4 rule 5), got {type(raster).__name__}; never "
            "pass a pre_encoder_compressor's ModelSafeInput output here"
        )
    return bytes(raster)


__all__ = [
    "PRE_ENCODER_COMPRESSOR_PROFILES",
    "PreEncoderCompressor",
    "PreEncoderCompressorUnimplementedError",
    "PreEncoderAdaptivePolicyError",
    "BypassPreEncoderCompressor",
    "AdaptivePreEncoderPolicy",
    "load_default_pre_encoder_policy",
    "AdaptivePreEncoderCompressor",
    "apply_region_resolution_plan",
    "build_pre_encoder_compressor",
    "ResourceRecoveryRasterSourceError",
    "guard_recovery_raster_source",
]
