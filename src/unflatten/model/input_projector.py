"""RFC004 §11 input projection: dataset record → :class:`ModelSafeInput`.

The projector sits on the *non-learned* input boundary (§9): it consumes a
dataset-side record (factory index/run-manifest shape — envelope plus
artifact refs) together with a page-pixel loader callback, and emits ONLY
pixels + geometry + declared scale + profile. It never copies ids or
provenance into the output, and as defense in depth re-validates its own
emitted payload against the §11.3 forbidden-channel list.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from unflatten.ast.issues import Issue
from unflatten.model.input_contract import (
    CoordinateTransform,
    InputMode,
    LocaleHint,
    ModelSafeInput,
    PhysicalScale,
    PixelValidityMask,
    to_payload_dict,
    validate_no_forbidden,
)
from unflatten.model.preprocess import PagePreprocessor

#: Loader callback: artifact ref record (``{"ref": ..., "sha256": ...}``) →
#: stored PNG bytes. Storage layout stays outside the projector.
PageLoader = Callable[[dict[str, Any]], bytes]


class ForbiddenChannelError(ValueError):
    """The projector's own output contained a §11.3 forbidden channel."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        paths = ", ".join(issue.path for issue in issues) or "<none>"
        super().__init__(f"projected input carries forbidden channels at: {paths}")


def project(
    record_like: dict[str, Any],
    load_page_bytes: PageLoader,
    *,
    profile: str = "word-core-v0",
    scale: PhysicalScale | None = None,
    locale_hint: LocaleHint | None = None,
    input_mode: InputMode | None = None,
) -> ModelSafeInput:
    """Project a dataset record to the model-safe allowlisted input.

    ``scale`` defaults to ``unknown`` — a declared DPI must be passed
    explicitly by a caller that actually has it in production (§11.6).
    ``input_mode`` defaults to ``single_page`` for one page and
    ``bounded_multi_page`` otherwise.
    """
    page_refs = _page_refs(record_like)
    preprocessor = PagePreprocessor()
    planes = tuple(
        preprocessor.load_png(load_page_bytes(ref), page_ordinal=ordinal)
        for ordinal, ref in enumerate(page_refs)
    )
    if input_mode is None:
        input_mode = "single_page" if len(planes) == 1 else "bounded_multi_page"
    msi = ModelSafeInput(
        planes=planes,
        masks=tuple(PixelValidityMask(bitmap=None) for _ in planes),
        transforms=tuple(CoordinateTransform.identity() for _ in planes),
        scale=scale if scale is not None else PhysicalScale(status="unknown", dpi=None),
        profile=profile,
        input_mode=input_mode,
        locale_hint=locale_hint,
    )
    issues = validate_no_forbidden(to_payload_dict(msi))
    if issues:  # pragma: no cover — the dataclass cannot carry these keys
        raise ForbiddenChannelError(issues)
    return msi


def _page_refs(record_like: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = record_like.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError(f"record 'artifacts' must be an object, got {type(artifacts).__name__}")
    refs = artifacts.get("pages") or record_like.get("pages")
    if not refs:
        raise ValueError("record carries no rendered page artifacts ('artifacts.pages')")
    return list(refs)
