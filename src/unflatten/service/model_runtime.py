"""Service-local projection and sealing for diagnostic model inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from unflatten.model.input_contract import (
    CoordinateTransform,
    ModelSafeInput,
    PagePlane,
    PhysicalScale,
    PixelValidityMask,
    validate_no_forbidden,
)
from unflatten.model.prediction_ir import (
    InferenceScope,
    PredictionIrInvalid,
    RawDocumentHypothesis,
    assemble,
    prediction_ir_projection,
    seal_fragment,
    validate_model_prediction_ir,
)


class ForbiddenModelChannel(ValueError):
    """Host-side data contains a channel forbidden from neural input."""


@dataclass(slots=True)
class DiagnosticModelRuntime:
    candidate_runtime: Any

    def extract(self, model_input: ModelSafeInput) -> Any:
        """Extract candidate-native visual features as an explicit worker stage."""
        return self.candidate_runtime.extract(model_input)

    def decode_features(self, prepared: Any) -> RawDocumentHypothesis:
        """Decode already-extracted features through the candidate grammar path."""
        return self.candidate_runtime.decode_features(prepared)

    @property
    def tokenizer(self) -> Any:
        """The DEC-007 D3-pinned tokenizer, delegated for docx_stage's `seal_page`."""
        return self.candidate_runtime.tokenizer

    @property
    def decode_policy(self) -> Any:
        """The frozen decode-time policy block (design D5), delegated for readiness."""
        return self.candidate_runtime.decode_policy

    @property
    def eval_cache_identity(self) -> Any:
        """The loaded checkpoint's cache identity (design G4), delegated for `seal_page`."""
        return self.candidate_runtime.eval_cache_identity

    @property
    def component_lock_hash(self) -> Any:
        """The verified component lock hash (design G4), delegated for `seal_page`."""
        return self.candidate_runtime.component_lock_hash

    @property
    def selection_policy_hash(self) -> Any:
        """The bundle's selection policy hash (design G4), delegated for `seal_page`."""
        return self.candidate_runtime.selection_policy_hash

    @property
    def vocabulary(self) -> Any:
        """The loaded composite vocabulary, delegated for `docx_stage.build_diagnostic_docx`'s
        `run.process_model_page(vocabulary=...)` call."""
        return self.candidate_runtime.vocabulary

    @staticmethod
    def seal(raw: RawDocumentHypothesis) -> Any:
        """Seal and serialize a terminated diagnostic hypothesis."""
        from unflatten.service.prediction_ir_artifact import build_prediction_ir_artifact

        return build_prediction_ir_artifact(seal_diagnostic_prediction(raw))

    def predict(self, model_input: ModelSafeInput) -> Any:
        if hasattr(self.candidate_runtime, "extract"):
            raw = self.decode_features(self.extract(model_input))
        else:
            raw = self.candidate_runtime.decode(model_input)
        return self.seal(raw)


def build_single_page_input(
    *,
    width_px: int,
    height_px: int,
    rgb_pixels: bytes,
    host_channels: Mapping[str, Any] | None = None,
) -> ModelSafeInput:
    """Build the sole authorized service input with every value explicit."""
    issues = validate_no_forbidden(host_channels or {})
    if issues:
        raise ForbiddenModelChannel(issues[0].message)
    return ModelSafeInput(
        planes=(PagePlane(0, width_px, height_px, rgb_pixels),),
        masks=(PixelValidityMask(bitmap=None),),
        transforms=(CoordinateTransform.identity(),),
        scale=PhysicalScale(status="known", dpi=300),
        profile="word-core-v0",
        input_mode="single_page",
        locale_hint=None,
    )


def seal_diagnostic_prediction(raw: RawDocumentHypothesis) -> dict[str, Any]:
    """Seal structure/text/typed output; resource predictions stay unresolved
    (D14: the S2-era policy that also refused typed predictions is superseded
    by DEC-021/§24 -- IR 0.3.0 carries typed predictions natively)."""
    if raw.resource_predictions:
        raise PredictionIrInvalid(
            "diagnostic runtime does not authorize resource predictions"
        )
    fragment = seal_fragment(raw, InferenceScope(kind="single_page"))
    payload = prediction_ir_projection(assemble((fragment,)))
    validate_model_prediction_ir(payload)
    return payload
