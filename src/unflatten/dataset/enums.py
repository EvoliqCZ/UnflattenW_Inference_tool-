"""RFC003 normative enums (v0.1) as Literal aliases + frozenset constants.

Values are exactly the RFC003 normative vocabularies (§11, §22, §24, §26,
§29–§30, §33, §41, §50 of docs/rfc/RFC003-dataset-architecture-v0.1-1.md).
The frozensets are the runtime validation source; the Literal aliases give
static type checking the same vocabulary.
"""

from __future__ import annotations

from typing import Literal

SourceOrigin = Literal[
    "synthetic_ast", "native_docx", "digital_pdf", "scanned_pdf", "raster_image"
]
SOURCE_ORIGIN: frozenset[str] = frozenset(
    {"synthetic_ast", "native_docx", "digital_pdf", "scanned_pdf", "raster_image"}
)

LabelMethod = Literal[
    "synthetic_generator",
    "docx_importer",
    "human_authored",
    "model_assisted_human",
    "pseudo_model",
    "none",
]
LABEL_METHOD: frozenset[str] = frozenset(
    {
        "synthetic_generator",
        "docx_importer",
        "human_authored",
        "model_assisted_human",
        "pseudo_model",
        "none",
    }
)

LabelGrade = Literal[
    "synthetic_exact",
    "imported_exact",
    "imported_normalized",
    "human_verified",
    "human_gold",
    "pseudo_verified_partial",
    "pseudo_unverified",
    "unknown",
]
LABEL_GRADE: frozenset[str] = frozenset(
    {
        "synthetic_exact",
        "imported_exact",
        "imported_normalized",
        "human_verified",
        "human_gold",
        "pseudo_verified_partial",
        "pseudo_unverified",
        "unknown",
    }
)

ProcessingState = Literal[
    "ingested",
    "canonicalized",
    "ast_validated",
    "compiled",
    "package_validated",
    "rendered",
    "grounded",
    "released",
]
PROCESSING_STATE: frozenset[str] = frozenset(
    {
        "ingested",
        "canonicalized",
        "ast_validated",
        "compiled",
        "package_validated",
        "rendered",
        "grounded",
        "released",
    }
)

VerificationTier = Literal["unverified", "machine_verified", "human_verified", "human_gold"]
VERIFICATION_TIER: frozenset[str] = frozenset(
    {"unverified", "machine_verified", "human_verified", "human_gold"}
)

Disposition = Literal["active", "quarantine", "rejected", "superseded"]
DISPOSITION: frozenset[str] = frozenset({"active", "quarantine", "rejected", "superseded"})

Eligibility = Literal["train_ready", "eval_ready", "partial_ready", "ineligible"]
ELIGIBILITY: frozenset[str] = frozenset(
    {"train_ready", "eval_ready", "partial_ready", "ineligible"}
)

ViewKind = Literal[
    "source_render",
    "canonical_render",
    "hypothesis_preview",
    "augmentation_render",
    "scan_simulation",
    "photo_simulation",
    "human_reference",
]
VIEW_KIND: frozenset[str] = frozenset(
    {
        "source_render",
        "canonical_render",
        "hypothesis_preview",
        "augmentation_render",
        "scan_simulation",
        "photo_simulation",
        "human_reference",
    }
)

TaskKind = Literal[
    "layout_v0",
    "paragraph_run_v0",
    "table_v0",
    "image_v0",
    "structure_v0",
    "full_ast_exact_v0",
    "partial_ast_v0",
    "renderer_negative_v0",
    "reasoning_constructive_v0",
    "reasoning_corrective_v0",
]
TASK_KIND: frozenset[str] = frozenset(
    {
        "layout_v0",
        "paragraph_run_v0",
        "table_v0",
        "image_v0",
        "structure_v0",
        "full_ast_exact_v0",
        "partial_ast_v0",
        "renderer_negative_v0",
        "reasoning_constructive_v0",
        "reasoning_corrective_v0",
    }
)

Observability = Literal[
    "direct_visible",
    "context_visible",
    "derived_deterministic",
    "source_known_hidden",
    "ambiguous",
    "not_applicable",
]
OBSERVABILITY: frozenset[str] = frozenset(
    {
        "direct_visible",
        "context_visible",
        "derived_deterministic",
        "source_known_hidden",
        "ambiguous",
        "not_applicable",
    }
)

LabelOrigin = Literal[
    "synthetic_generator",
    "docx_importer_resolved",
    "renderer_derived",
    "mapping_deterministic",
    "human_authored",
    "human_verified",
    "model_proposed",
    "imputed_for_render",
]
LABEL_ORIGIN: frozenset[str] = frozenset(
    {
        "synthetic_generator",
        "docx_importer_resolved",
        "renderer_derived",
        "mapping_deterministic",
        "human_authored",
        "human_verified",
        "model_proposed",
        "imputed_for_render",
    }
)

MappingMethod = Literal[
    "renderer_native",
    "pdf_text_alignment",
    "source_map_alignment",
    "image_object_match",
    "geometry_inferred",
    "human_verified",
    "ocr_derived_weak",
]
MAPPING_METHOD: frozenset[str] = frozenset(
    {
        "renderer_native",
        "pdf_text_alignment",
        "source_map_alignment",
        "image_object_match",
        "geometry_inferred",
        "human_verified",
        "ocr_derived_weak",
    }
)

Continuation = Literal["complete", "start", "middle", "end"]
CONTINUATION: frozenset[str] = frozenset({"complete", "start", "middle", "end"})

ResourceOrigin = Literal[
    "source_native", "synthetic_resource", "render_recovered", "human_reconstructed"
]
RESOURCE_ORIGIN: frozenset[str] = frozenset(
    {"source_native", "synthetic_resource", "render_recovered", "human_reconstructed"}
)

PrivacyClass = Literal[
    "synthetic",
    "public_non_personal",
    "public_personal",
    "internal_non_sensitive",
    "internal_restricted",
    "sensitive_personal",
    "prohibited",
]
PRIVACY_CLASS: frozenset[str] = frozenset(
    {
        "synthetic",
        "public_non_personal",
        "public_personal",
        "internal_non_sensitive",
        "internal_restricted",
        "sensitive_personal",
        "prohibited",
    }
)

LeakageClass = Literal[
    "instance_leakage",
    "template_leakage",
    "content_leakage",
    "render_leakage",
    "resource_leakage",
]
LEAKAGE_CLASS: frozenset[str] = frozenset(
    {
        "instance_leakage",
        "template_leakage",
        "content_leakage",
        "render_leakage",
        "resource_leakage",
    }
)

WindowPolicy = Literal["single_page", "two_page_overlap", "bounded_multi_page", "full_document"]
WINDOW_POLICY: frozenset[str] = frozenset(
    {"single_page", "two_page_overlap", "bounded_multi_page", "full_document"}
)
