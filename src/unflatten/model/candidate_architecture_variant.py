"""`CandidateArchitectureVariant` -- DEC-047 dormant-code switch config.

DEC-047 (`docs/decisions/DEC-047-spiaci-kod-v-predstihu.md`) authorizes the
sparse Word Expert Adapter (`sparse_expert_adapter/`), the ADRF reasoning
path (`reasoning/`, `sparse_expert_adapter/reasoning_integration.py`) and
the post/pre-encoder visual compressors (`visual_compression/`) as dormant
code ahead of S6/G2 ratification, under five conditions -- default off, step
0 == dense, no hashed identity moved, topology as a parameter, no gradient
runs. Each module already satisfies those conditions on its own; this file
is the single config object that lets `QwenDenseCandidateModel` (and, later,
an S6/S6R/S7 trainer) select a combination of them WITHOUT importing any of
those modules directly -- see `candidate_model.py`'s `build_candidate_model`.

This module is data + validation only: it declares no `nn.Module`, imports
no trainable weights, and constructs nothing until a caller explicitly asks
`CandidateArchitectureVariant()` (or `build_candidate_model`) for one.

`expert_adapter`, `reasoning`, `post_encoder_compressor` and
`pre_encoder_compressor` mirror the vocabulary the RFC004 §38 architecture
manifest already uses for the first of these
(`expert_adapter.supported_modes: ["dense", "sparse_moe"]`,
`schemas/architecture_manifest.schema.json`) -- this config never edits that
schema or any manifest payload (DEC-047 condition 3: "žiadna zmena
hashovanej identity"), it only reuses the same two-mode vocabulary under a
shorter name (`"sparse"` here, `"sparse_moe"` there) for the pre-S6
in-process switch.

`pre_encoder_compressor` is carried here as data only -- unlike the other
three slots, it acts BEFORE the vision encoder
(`visual_compression.pre_encoder`, on a `ModelSafeInput`), a stage
`QwenDenseCandidateModel` never sees (it starts from an already-computed
`visual_features` tensor, per `candidate_model.py`'s own docstring
assumption 1). `build_pre_encoder_compressor()` lets a caller construct the
selected profile directly from this config without importing
`visual_compression.pre_encoder` itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from unflatten.model.level_router.policy import LevelRouterPolicy
    from unflatten.model.sparse_expert_adapter.manifest import SparseExpertAdapterManifest
    from unflatten.model.visual_compression.adaptive_v0 import AdaptiveVisualMemoryPolicy
    from unflatten.model.visual_compression.pre_encoder import (
        AdaptivePreEncoderPolicy,
        PreEncoderCompressor,
    )
    from unflatten.model.visual_compression.scout import ScoutModel

#: Bump on any field/semantics change to this config (never on the RFC004
#: architecture manifest -- this schema is separate, DEC-047 condition 3).
CANDIDATE_ARCHITECTURE_VARIANT_SCHEMA_VERSION = "0.2.0"  # 0.2.0: level_router (RFC007)

EXPERT_ADAPTER_MODES: tuple[str, ...] = ("dense", "sparse")
COMPRESSOR_PROFILES: tuple[str, ...] = ("bypass", "adaptive_v0")


class CandidateArchitectureVariantError(ValueError):
    """An invalid field or field combination on `CandidateArchitectureVariant`."""


@dataclass(frozen=True)
class ExpertAdapterVariant:
    """`"dense"` (default) needs no manifest; `"sparse"` requires one.

    RFC004 §22.3: dense mode has no router/shared/specialist bank, so a
    `sparse_manifest` on a `"dense"` variant would be a silent no-op field
    nothing reads -- refused here rather than accepted and ignored.
    """

    mode: str = "dense"
    sparse_manifest: SparseExpertAdapterManifest | None = None

    def __post_init__(self) -> None:
        if self.mode not in EXPERT_ADAPTER_MODES:
            raise CandidateArchitectureVariantError(
                f"expert_adapter.mode must be one of {EXPERT_ADAPTER_MODES}, got {self.mode!r}"
            )
        if self.mode == "sparse" and self.sparse_manifest is None:
            raise CandidateArchitectureVariantError(
                "expert_adapter.mode 'sparse' requires a sparse_manifest"
            )
        if self.mode == "dense" and self.sparse_manifest is not None:
            raise CandidateArchitectureVariantError(
                "expert_adapter.mode 'dense' must not carry a sparse_manifest"
            )


@dataclass(frozen=True)
class ReasoningVariant:
    """`enabled=False` (default) is RFC004-ERRATA-005 E1 rule 1's architectural
    bypass; `enabled=True` requires both `spec_id` and `attach_point` (the
    reasoning registry never falls back to a default specialization, E1
    rule 7 -- see `reasoning/registry.py`)."""

    enabled: bool = False
    spec_id: str | None = None
    attach_point: str | None = None

    def __post_init__(self) -> None:
        if self.enabled and (self.spec_id is None or self.attach_point is None):
            raise CandidateArchitectureVariantError(
                "reasoning.enabled requires both spec_id and attach_point"
            )
        if not self.enabled and (self.spec_id is not None or self.attach_point is not None):
            raise CandidateArchitectureVariantError(
                "reasoning.enabled is False but spec_id/attach_point were given; "
                "'off' must not carry a specialization identity"
            )


@dataclass(frozen=True)
class PostEncoderCompressorVariant:
    """RFC004-ERRATA-005 E5 slot, applied inside `QwenDenseCandidateModel`
    to the projected `VisualMemory` (after the backbone adapter, before the
    document context encoder)."""

    profile: str = "bypass"
    policy: AdaptiveVisualMemoryPolicy | None = None

    def __post_init__(self) -> None:
        _validate_profile("post_encoder_compressor", self.profile)
        if self.profile == "adaptive_v0" and self.policy is None:
            raise CandidateArchitectureVariantError(
                "post_encoder_compressor profile 'adaptive_v0' requires a policy"
            )
        if self.profile == "bypass" and self.policy is not None:
            raise CandidateArchitectureVariantError(
                "post_encoder_compressor profile 'bypass' must not carry a policy"
            )


@dataclass(frozen=True)
class PreEncoderCompressorVariant:
    """RFC004-ERRATA-005 E4 slot, applied BEFORE the vision encoder --
    `QwenDenseCandidateModel` never constructs or calls this; it is carried
    here as data for a caller (the input pipeline / a future trainer) to
    build via `build_pre_encoder_compressor()`."""

    profile: str = "bypass"
    policy: AdaptivePreEncoderPolicy | None = None
    scout: ScoutModel | None = None

    def __post_init__(self) -> None:
        _validate_profile("pre_encoder_compressor", self.profile)
        if self.profile == "adaptive_v0" and (self.policy is None or self.scout is None):
            raise CandidateArchitectureVariantError(
                "pre_encoder_compressor profile 'adaptive_v0' requires both a "
                "policy and scout weights (DEC-047: opt-in, never a silent default)"
            )
        if self.profile == "bypass" and (self.policy is not None or self.scout is not None):
            raise CandidateArchitectureVariantError(
                "pre_encoder_compressor profile 'bypass' must not carry a policy or scout"
            )

    def build(self) -> PreEncoderCompressor:
        """Construct the selected profile via `visual_compression.pre_encoder
        .build_pre_encoder_compressor` -- the model-input-side counterpart of
        `PostEncoderCompressorVariant`'s wiring inside the model."""
        from unflatten.model.visual_compression.pre_encoder import (
            build_pre_encoder_compressor,
        )

        return build_pre_encoder_compressor(self.profile, policy=self.policy, scout=self.scout)


@dataclass(frozen=True)
class LevelRouterVariant:
    """RFC007 variant B (PROPOSED, DEC-047 dormant): the grammar-conditioned
    query-side router over separate visual-level memories
    (`level_router.module.DocumentLevelRouter`). `enabled=False` (default)
    builds nothing; `enabled=True` requires the policy that fixes its
    topology (DEC-047 condition 4) -- and an "off" variant carrying a policy
    would be a silent no-op field, so it is refused."""

    enabled: bool = False
    policy: LevelRouterPolicy | None = None

    def __post_init__(self) -> None:
        if self.enabled and self.policy is None:
            raise CandidateArchitectureVariantError("level_router.enabled requires a policy")
        if not self.enabled and self.policy is not None:
            raise CandidateArchitectureVariantError(
                "level_router is disabled but a policy was given; 'off' must not carry one"
            )


def _validate_profile(field_name: str, profile: str) -> None:
    if profile not in COMPRESSOR_PROFILES:
        raise CandidateArchitectureVariantError(
            f"{field_name} must be one of {COMPRESSOR_PROFILES}, got {profile!r}"
        )


@dataclass(frozen=True)
class CandidateArchitectureVariant:
    """The full dormant-code switch: every field defaults to today's dense
    candidate exactly (`CandidateArchitectureVariant()` and `None` are
    interchangeable at every construction call site -- see
    `candidate_model.build_candidate_model`).

    RFC004-ERRATA-005 E1 rule 9: reasoning attaches only to the sparse
    branch, never dense -- enforced here, one level above
    `SparseExpertAdapterWithReasoning`'s own `TypeError`, so a misconfigured
    variant is refused before any module is built.
    """

    schema_version: str = CANDIDATE_ARCHITECTURE_VARIANT_SCHEMA_VERSION
    expert_adapter: ExpertAdapterVariant = field(default_factory=ExpertAdapterVariant)
    reasoning: ReasoningVariant = field(default_factory=ReasoningVariant)
    post_encoder_compressor: PostEncoderCompressorVariant = field(
        default_factory=PostEncoderCompressorVariant
    )
    pre_encoder_compressor: PreEncoderCompressorVariant = field(
        default_factory=PreEncoderCompressorVariant
    )
    level_router: LevelRouterVariant = field(default_factory=LevelRouterVariant)

    def __post_init__(self) -> None:
        if self.reasoning.enabled and self.expert_adapter.mode != "sparse":
            raise CandidateArchitectureVariantError(
                "reasoning requires expert_adapter.mode == 'sparse' "
                "(RFC004-ERRATA-005 E1 rule 9: dense has no reasoning path)"
            )
        # A merging compressor rewrites the composite range (fewer visual
        # placeholders) after the level memories were laid out on the full
        # page grid; RFC007 §8 point 5 (OPEN-007-6) leaves that combination OPEN.
        if self.level_router.enabled and self.post_encoder_compressor.profile != "bypass":
            raise CandidateArchitectureVariantError(
                "level_router is not supported together with a post_encoder_compressor "
                "other than 'bypass' (RFC007 OPEN-007-6)"
            )


def is_default(variant: CandidateArchitectureVariant | None) -> bool:
    """`True` for `None` and for a `CandidateArchitectureVariant` equal to
    the all-defaults instance -- the two are interchangeable everywhere this
    config is accepted."""
    return variant is None or variant == CandidateArchitectureVariant()


__all__ = [
    "CANDIDATE_ARCHITECTURE_VARIANT_SCHEMA_VERSION",
    "EXPERT_ADAPTER_MODES",
    "COMPRESSOR_PROFILES",
    "CandidateArchitectureVariantError",
    "ExpertAdapterVariant",
    "ReasoningVariant",
    "PostEncoderCompressorVariant",
    "PreEncoderCompressorVariant",
    "LevelRouterVariant",
    "CandidateArchitectureVariant",
    "is_default",
]
