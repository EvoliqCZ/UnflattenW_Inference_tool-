"""Failure taxonomy report (RFC005-ERRATA-007 E1).

Model side: a `sealed.SealedCohortArtefact` (free-running decode, the same
frozen decode block the selection uses). Gold side: `cohort_gold.GoldRow`
per page of the same cohort. Optional pipeline side: per-page strict-build
outcomes from `scripts/render_model_docx_with_defaults.py`'s `summary.json`
or from a `builder_chain_report.json`.

Every class is COUNTED: `frequency` carries `count` and `denominator` and
the unit they are in; a denominator of 0 is reported as
`insufficient_evidence` (RFC005 §39.5), never as a rate. Only `terminated`
pages with a gold row ("scored pages") enter a hypothesis-based
denominator; a `max_steps_exceeded` page is counted once, as
`page_unterminated`, and a page whose gold row is missing is accounted and
excluded.

Cause assignment is rule-based and the rule text travels with the class
(`cause_evidence.rule`). The register is closed (E1); `unknown` is used
wherever the cache row and the chain output available to this tool do not
separate the remaining causes -- the report is a podklad, the owner
decides (`selected_for_experiment` stays `null`).

Hashing: `domain_hash("rfc005.failure-taxonomy-report/v0", core)` over a
float-free core (the RFC001 CBOR profile forbids floats); `rate`, the
Wilson interval and its method note are derived from hashed counts and
merged in after hashing, exactly like `builder_chain.report` does with
`strict_valid_share`.
"""

from __future__ import annotations

import copy
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from unflatten.eval.cohort_gold import CROP_SIDES, IMAGE_TYPES, GoldRow
from unflatten.eval.s2_gate import sealed
from unflatten.model.composite_vocab import STRUCTURAL_COUNT
from unflatten.model.hashing import domain_hash

REPORT_DOMAIN = "rfc005.failure-taxonomy-report/v0"
REPORT_SCHEMA_VERSION = "0.1.0"

#: E1 closed register of suspected causes.
CAUSE_REGISTER: tuple[str, ...] = (
    "data_or_target",
    "pipeline_or_projection",
    "visual_representation",
    "decoder_prior",
    "relational_structure",
    "long_context",
    "unknown",
)
#: E1: reasoning answers only these two causes.
REASONING_CAUSES: frozenset[str] = frozenset({"relational_structure", "decoder_prior"})

#: E1 record field order (the YAML block), plus nothing.
FIELD_ORDER: tuple[str, ...] = (
    "failure_class",
    "frequency",
    "severity",
    "affected_slice",
    "dense_baseline_metric",
    "suspected_cause",
    "cause_evidence",
    "reasoning_candidate",
    "eligible",
    "reasoning_supervision_available",
    "estimated_training_samples",
    "selected_for_experiment",
    "decision_reason",
)

#: E1: only development validation cohorts; never locked test, never P6.
FORBIDDEN_ROLE_MARKERS: tuple[str, ...] = ("test", "locked", "p6")

IOU_THRESHOLD_PPM = 500_000
WIDTH_COHERENCE_PPM: tuple[int, int] = (800_000, 1_200_000)
TEXT_APPROX_RATIO_PPM = 900_000
MAX_EXAMPLES = 5
MAX_EXAMPLE_CODES = 10
#: `CODE|/sections/<i>/headers|footers/<j>...` -- a chain refusal at a story path.
_STORY_POINTER = re.compile(r"\|/sections/\d+/(headers|footers)/\d+")
SCALING_NOTE = (
    "lineárne škálovanie počtu dotknutých strán kohorty na počet riadkov splitu; "
    "odhad rozsahu, nie meranie"
)
WILSON_METHOD = (
    "Wilson 95 % na stranách kohorty (jeden dokument = jedna strana); nie párový "
    "bootstrap podľa RFC005 §39.2"
)


class ArtefactRoleRefused(ValueError):
    """The sealed artefact is not a development validation cohort."""


def check_role(role: str) -> None:
    lowered = role.lower()
    for marker in FORBIDDEN_ROLE_MARKERS:
        if marker in lowered:
            raise ArtefactRoleRefused(
                f"sealed artefact role {role!r} contains {marker!r}: the failure taxonomy "
                "report runs only over development validation cohorts (RFC005-ERRATA-007 "
                "E1: never locked test, never P6)"
            )


@dataclass(frozen=True, slots=True)
class ClassRule:
    cause: str
    reasoning_candidate: str
    rule: str


#: E1: causes checked against the evidence this tool has (cache row + chain
#: output) and left `unknown` because nothing here separates the remaining
#: candidates. Re-checked 2026-09-26 (W1-14): `image_missing`/`image_spurious`
#: (data_or_target vs decoder_prior -- no per-node chain-refusal pointer like
#: `_STORY_POINTER` exists for images), `image_localization`/`table_missing`/
#: `table_spurious` (visual_representation vs decoder_prior; no comparable
#: paragraph-box measurement is computed here to confirm the rule's own
#: "paragraph boxes are as low" claim) and `text_exact_mismatch`/
#: `text_approx_mismatch` (a text difference alone never locates a cause) all
#: still lack a fact split in `_ClassCount.facts` that would decide them; a
#: real reduction needs new evidence-gathering code, not just a rule.
UNKNOWN_CAUSE_CLASSES: frozenset[str] = frozenset({
    "image_missing", "image_spurious", "image_localization",
    "table_missing", "table_spurious",
    "text_exact_mismatch", "text_approx_mismatch",
})

CLASS_RULES: Mapping[str, ClassRule] = {
    "story_wrapper_dropped": ClassRule(
        "decoder_prior",
        "reasoning.section",
        "gold has HEADER_OPEN/FOOTER_OPEN as the first section child (one token decision right "
        "after SECTION_OPEN; the grammar forbids a story later), the target is supervised and "
        "the chain never removes a story: the decoder prior drops it",
    ),
    "story_kind_duplicate": ClassRule(
        "decoder_prior",
        "reasoning.section",
        "two header (or footer) nodes of the same /kind under one section; gold never carries a "
        "duplicate (semantic validator DUPLICATE_STORY_KIND), so the second one is a prior",
    ),
    "text_frame_missing": ClassRule(
        "decoder_prior",
        "reasoning.layout",
        "gold carries a supervised TEXT_FRAME_OPEN and the chain has no text_frame policy; the "
        "wrapper decision is the same kind as the story wrapper's",
    ),
    "image_missing": ClassRule(
        "unknown",
        "none",
        "gold image node with an observed region, no image node in the hypothesis; the row and "
        "the chain do not separate visual_representation from decoder_prior",
    ),
    "image_spurious": ClassRule(
        "unknown",
        "none",
        "image node in the hypothesis, no image node in gold; the row cannot tell a drawn shape "
        "the importer did not carry (data_or_target) from a prior",
    ),
    "image_localization": ClassRule(
        "unknown",
        "none",
        "best IoU of any predicted image box against the gold region below the threshold; "
        "the region head's cause is not separable here (paragraph boxes are as low)",
    ),
    "floating_as_inline": ClassRule(
        "relational_structure",
        "reasoning.layout",
        "gold floating_image (anchor/wrap relation to a paragraph) emitted as inline_image: the "
        "object is kept, the relation between objects is lost",
    ),
    "table_missing": ClassRule(
        "unknown",
        "none",
        "gold table, no table in the hypothesis; not separable by the row or the chain",
    ),
    "table_spurious": ClassRule(
        "unknown",
        "none",
        "table in the hypothesis, no table in gold; not separable by the row or the chain",
    ),
    "table_column_count_mismatch": ClassRule(
        "relational_structure",
        "reasoning.table",
        "grid column count (max grid_column + col_span over cells) differs from gold /columns "
        "cardinality on a page with exactly one table each side: the grid is a relation among "
        "cells",
    ),
    "table_width_incoherent": ClassRule(
        "data_or_target",
        "none",
        "tblW / sum(grid) outside the coherence band on an absolute-mode table; T1 (2026-09-26, "
        "val_iid 5 201 tables): the value_mpt head's supervised targets are ~50 % zeros from the "
        "auto arm (importer default, un-masked union arm) -- verified again on this cohort's gold "
        "tables in `facts`",
    ),
    "crop_nonzero": ClassRule(
        "pipeline_or_projection",
        "none",
        "predicted /crop/*_ppm != 0 while gold crop is 0 on every observed image (constant "
        "target at 1e6 scale); the recovered-image policy replaces it before the build",
    ),
    "page_unterminated": ClassRule(
        "long_context",
        "none",
        "decode hit max_steps without END; no hypothesis exists for the page",
    ),
    "pipeline_refusal": ClassRule(
        "pipeline_or_projection",
        "none",
        "strict build of the projected hypothesis failed; codes counted per page",
    ),
    "text_exact_mismatch": ClassRule(
        "unknown",
        "none",
        "text token sequence (ids >= STRUCTURAL_COUNT) differs from gold; a text difference "
        "alone does not locate its cause",
    ),
    "text_approx_mismatch": ClassRule(
        "unknown",
        "none",
        "difflib ratio of the text token sequences below the threshold; not separable here",
    ),
}


# ---------------------------------------------------------------------------
# Pipeline side
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    strict_valid: bool
    #: `CODE|pointer` strings as the chain writes them; empty when unknown.
    codes: tuple[str, ...] = ()


def pipeline_outcomes_from_summary(summary: Mapping[str, Any]) -> dict[str, PipelineOutcome]:
    """`summary.json` of `scripts/render_model_docx_with_defaults.py`."""
    out: dict[str, PipelineOutcome] = {}
    for row in summary["rows"]:
        out[str(row["document_bundle_id"])] = PipelineOutcome(
            strict_valid=row.get("ast_status") == "strict_valid",
            codes=tuple(str(c) for c in row.get("failure_codes") or ()),
        )
    return out


def pipeline_outcomes_from_chain_report(report: Mapping[str, Any]) -> dict[str, PipelineOutcome]:
    """`builder_chain_report.json`: per-row `strict_valid` only (its failure
    histogram is aggregate, so no per-page codes are invented)."""
    return {
        str(row["item_id"]): PipelineOutcome(strict_valid=bool(row["strict_valid"]))
        for row in report["rows"]
    }


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelImage:
    node_id: str
    node_type: str
    crop_ppm: tuple[int, ...]
    bbox_ppm: tuple[int, int, int, int] | None


@dataclass(frozen=True, slots=True)
class ModelTable:
    node_id: str
    width_mode: str | None
    width_value_mpt: int | None
    column_count: int
    grid_sum_mpt: int


@dataclass(slots=True)
class ModelPage:
    bundle: str
    outcome: str
    terminated: bool
    counts: Counter = field(default_factory=Counter)
    first_section_child: str | None = None
    stories: list[tuple[str, str, str | None]] = field(default_factory=list)
    images: list[ModelImage] = field(default_factory=list)
    tables: list[ModelTable] = field(default_factory=list)
    text_token_ids: tuple[int, ...] = ()
    max_steps: int | None = None
    prefix_tokens: int | None = None

    def count(self, node_type: str) -> int:
        return int(self.counts.get(node_type, 0))

    @property
    def image_count(self) -> int:
        return sum(self.count(t) for t in IMAGE_TYPES)


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def model_page(page: sealed.PageDecodeResult) -> ModelPage:
    terminated = page.outcome == "terminated" and page.node_states is not None
    out = ModelPage(bundle=page.document_bundle_id, outcome=page.outcome, terminated=terminated,
                    max_steps=page.max_steps)
    if not terminated:
        prefix = page.unterminated_prefix
        out.prefix_tokens = len(prefix.accepted_tokens) if prefix is not None else None
        return out
    states = list(page.node_states or ())
    out.counts = Counter(s.node_type for s in states)
    children: dict[str, list[sealed.NodeStateProjection]] = {}
    for s in states:
        if s.parent_local_node_id is not None:
            children.setdefault(s.parent_local_node_id, []).append(s)
    sections = [s for s in states if s.node_type == "section"]
    if sections and children.get(sections[0].local_node_id):
        out.first_section_child = children[sections[0].local_node_id][0].node_type
    typed: dict[tuple[str, str], Any] = {
        (t["local_node_id"], t["pointer"]): t["value"] for t in page.typed_predictions
    }
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for rp in page.region_predictions:
        if rp.alignment_status == "region" and rp.regions:
            boxes[rp.local_node_id] = rp.regions[0].bbox_ppm
    for s in states:
        if s.node_type in ("header", "footer"):
            kind = typed.get((s.local_node_id, "/kind"))
            out.stories.append(
                (s.parent_local_node_id or "", s.node_type, str(kind) if kind is not None else None)
            )
        elif s.node_type in IMAGE_TYPES:
            crop_values = (
                _int(typed.get((s.local_node_id, f"/crop/{side}_ppm"))) for side in CROP_SIDES
            )
            crop = tuple(v for v in crop_values if v is not None)
            out.images.append(
                ModelImage(s.local_node_id, s.node_type, crop, boxes.get(s.local_node_id))
            )
        elif s.node_type == "table":
            out.tables.append(_model_table(s, children, typed))
    out.text_token_ids = tuple(t for t in (page.accepted_tokens or ()) if t >= STRUCTURAL_COUNT)
    return out


def _model_table(
    table: sealed.NodeStateProjection,
    children: Mapping[str, list[sealed.NodeStateProjection]],
    typed: Mapping[tuple[str, str], Any],
) -> ModelTable:
    """Column count as `table_geometry.v0` reads it: max(grid_column +
    col_span) over the cells, falling back to the widest row; the grid sum
    covers only those columns (decoded heads carry more `/columns/<i>` slots
    than any grid has)."""
    rows = [c for c in children.get(table.local_node_id, []) if c.node_type == "row"]
    spans: list[int] = []
    widest = 0
    for row in rows:
        cells = [c for c in children.get(row.local_node_id, []) if c.node_type == "cell"]
        widest = max(widest, len(cells))
        for cell in cells:
            column = _int(typed.get((cell.local_node_id, "/grid_column")))
            span = _int(typed.get((cell.local_node_id, "/col_span")))
            if column is not None:
                spans.append(column + (span if span is not None and span > 0 else 1))
    ncols = max(spans) if spans else widest
    grid_sum = 0
    for i in range(ncols):
        width = _int(typed.get((table.local_node_id, f"/columns/{i}/width_mpt")))
        if width is not None:
            grid_sum += width
    mode = typed.get((table.local_node_id, "/layout/width/mode"))
    return ModelTable(
        node_id=table.local_node_id,
        width_mode=str(mode) if mode is not None else None,
        width_value_mpt=_int(typed.get((table.local_node_id, "/layout/width/value_mpt"))),
        column_count=ncols,
        grid_sum_mpt=grid_sum,
    )


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def iou_ppm(a: Sequence[int], b: Sequence[int]) -> int:
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return (inter * 1_000_000) // union if union > 0 else 0


def _upper_median(values: Sequence[int]) -> int | None:
    if not values:
        return None
    return sorted(values)[len(values) // 2]


def _ratio_ppm(a: Sequence[int], b: Sequence[int]) -> int:
    if not a and not b:
        return 1_000_000
    return int(round(SequenceMatcher(None, a, b, autojunk=False).ratio() * 1_000_000))


@dataclass(slots=True)
class _ClassCount:
    unit: str
    count: int = 0
    denominator: int = 0
    affected: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def unit_seen(self) -> None:
        self.denominator += 1

    def hit(self, bundle: str, *, gold: Mapping[str, Any] | None = None,
            model: Mapping[str, Any] | None = None) -> None:
        self.count += 1
        if bundle not in self.affected:
            self.affected.append(bundle)
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(
                {"bundle": bundle, "gold": dict(gold or {}), "model": dict(model or {})}
            )


def _count_classes(
    pages: Sequence[ModelPage],
    gold_rows: Mapping[str, GoldRow],
    pipeline: Mapping[str, PipelineOutcome] | None,
) -> dict[str, _ClassCount]:
    c = {
        "story_wrapper_dropped": _ClassCount("page_story_kinds"),
        "story_kind_duplicate": _ClassCount("pages_with_model_story"),
        "text_frame_missing": _ClassCount("pages_with_gold_text_frame"),
        "image_missing": _ClassCount("pages_with_gold_image"),
        "image_spurious": _ClassCount("pages_with_model_image"),
        "image_localization": _ClassCount("gold_images_with_model_box_on_page"),
        "floating_as_inline": _ClassCount("pages_with_gold_floating_and_model_image"),
        "table_missing": _ClassCount("pages_with_gold_table"),
        "table_spurious": _ClassCount("pages_with_model_table"),
        "table_column_count_mismatch": _ClassCount("pages_with_one_table_each_side"),
        "table_width_incoherent": _ClassCount("model_absolute_tables"),
        "crop_nonzero": _ClassCount("model_image_nodes"),
        "page_unterminated": _ClassCount("pages"),
        "pipeline_refusal": _ClassCount("terminated_pages_with_pipeline_outcome"),
        "text_exact_mismatch": _ClassCount("scored_pages"),
        "text_approx_mismatch": _ClassCount("scored_pages"),
    }
    ious: list[int] = []
    width_ratios: list[int] = []
    text_ratios: list[int] = []
    gold_tables = 0
    gold_auto_zero = 0
    gold_crop_targets = 0
    gold_crop_nonzero = 0
    pages_by_code: Counter[str] = Counter()
    story_path_refusals = 0

    for page in pages:
        b = page.bundle
        c["page_unterminated"].unit_seen()
        if not page.terminated:
            c["page_unterminated"].hit(
                b, model={"outcome": page.outcome, "max_steps": page.max_steps,
                          "prefix_tokens": page.prefix_tokens},
            )
        # a pipeline refusal needs a hypothesis to refuse: an unterminated
        # page has none and is already counted as page_unterminated
        if pipeline is not None and b in pipeline and page.terminated:
            c["pipeline_refusal"].unit_seen()
            outcome = pipeline[b]
            if not outcome.strict_valid:
                c["pipeline_refusal"].hit(
                    b,
                    model={"failure_codes": list(outcome.codes[:MAX_EXAMPLE_CODES]),
                           "failure_code_count": len(outcome.codes)},
                )
                for code in sorted({code.split("|", 1)[0] for code in outcome.codes}):
                    pages_by_code[code] += 1
                if any(_STORY_POINTER.search(code) for code in outcome.codes):
                    story_path_refusals += 1
        gold = gold_rows.get(b)
        if gold is None or not page.terminated:
            continue

        # gold-side facts that back the cause rules, over every scored page
        for table in gold.tables:
            gold_tables += 1
            if table.width_mode == "auto" and table.width_value_mpt == 0:
                gold_auto_zero += 1
        for image in gold.images:
            if image.crop_ppm:
                gold_crop_targets += 1
                if any(v != 0 for v in image.crop_ppm):
                    gold_crop_nonzero += 1

        # stories
        for kind in ("header", "footer"):
            if gold.count(kind) > 0:
                c["story_wrapper_dropped"].unit_seen()
                if page.count(kind) == 0:
                    c["story_wrapper_dropped"].hit(
                        b,
                        gold={"kind": kind, "count": gold.count(kind),
                              "first_section_child": gold.first_section_child},
                        model={"count": 0, "first_section_child": page.first_section_child},
                    )
        if page.stories:
            c["story_kind_duplicate"].unit_seen()
            seen: Counter[tuple[str, str, str | None]] = Counter(page.stories)
            duplicates = sorted(k for k, n in seen.items() if n > 1)
            if duplicates:
                c["story_kind_duplicate"].hit(
                    b, gold={"header": gold.count("header"), "footer": gold.count("footer")},
                    model={"duplicates": [list(d) for d in duplicates]},
                )
        if gold.count("text_frame") > 0:
            c["text_frame_missing"].unit_seen()
            if page.count("text_frame") == 0:
                c["text_frame_missing"].hit(
                    b, gold={"text_frame": gold.count("text_frame")}, model={"text_frame": 0}
                )

        # images
        gold_images = len(gold.images)
        gold_floating = gold.count("floating_image")
        model_images = page.image_count
        image_evidence = {
            "images": gold_images,
            "evidence": [img.page_evidence for img in gold.images],
            "types": [img.node_type for img in gold.images],
        }
        if gold_images > 0:
            c["image_missing"].unit_seen()
            if model_images == 0:
                c["image_missing"].hit(b, gold=image_evidence, model={"images": 0})
        if model_images > 0:
            c["image_spurious"].unit_seen()
            if gold_images == 0:
                c["image_spurious"].hit(
                    b, gold={"images": 0},
                    model={"images": model_images, "types": [i.node_type for i in page.images]},
                )
        model_boxes = [img.bbox_ppm for img in page.images if img.bbox_ppm is not None]
        if model_boxes:
            for image in gold.images:
                if image.bbox_ppm is None:
                    continue
                c["image_localization"].unit_seen()
                best = max(iou_ppm(image.bbox_ppm, box) for box in model_boxes)
                ious.append(best)
                if best < IOU_THRESHOLD_PPM:
                    c["image_localization"].hit(
                        b,
                        gold={"node_id": image.node_id, "type": image.node_type,
                              "bbox_ppm": list(image.bbox_ppm)},
                        model={"best_iou_ppm": best, "boxes": [list(x) for x in model_boxes]},
                    )
        if gold_floating > 0 and model_images > 0:
            c["floating_as_inline"].unit_seen()
            if page.count("floating_image") == 0:
                c["floating_as_inline"].hit(
                    b,
                    gold={"floating_image": gold_floating,
                          "inline_image": gold.count("inline_image")},
                    model={"floating_image": 0, "inline_image": page.count("inline_image")},
                )
        for image in page.images:
            if image.crop_ppm:
                c["crop_nonzero"].unit_seen()
                if any(v != 0 for v in image.crop_ppm):
                    c["crop_nonzero"].hit(
                        b, gold={"crop_ppm": [list(i.crop_ppm) for i in gold.images]},
                        model={"node_id": image.node_id, "crop_ppm": list(image.crop_ppm)},
                    )

        # tables
        if gold.tables:
            c["table_missing"].unit_seen()
            if not page.tables:
                c["table_missing"].hit(
                    b, gold={"tables": len(gold.tables),
                             "column_counts": [t.column_count for t in gold.tables]},
                    model={"tables": 0},
                )
        if page.tables:
            c["table_spurious"].unit_seen()
            if not gold.tables:
                c["table_spurious"].hit(
                    b, gold={"tables": 0},
                    model={"tables": len(page.tables),
                           "column_counts": [t.column_count for t in page.tables]},
                )
        if len(gold.tables) == 1 and len(page.tables) == 1:
            c["table_column_count_mismatch"].unit_seen()
            g, m = gold.tables[0], page.tables[0]
            if g.column_count != m.column_count:
                c["table_column_count_mismatch"].hit(
                    b, gold={"column_count": g.column_count, "width_mode": g.width_mode},
                    model={"column_count": m.column_count, "width_mode": m.width_mode},
                )
        for table in page.tables:
            if (
                table.width_mode != "absolute"
                or table.width_value_mpt is None
                or table.grid_sum_mpt <= 0
            ):
                continue
            c["table_width_incoherent"].unit_seen()
            ratio = (table.width_value_mpt * 1_000_000) // table.grid_sum_mpt
            width_ratios.append(ratio)
            if not WIDTH_COHERENCE_PPM[0] <= ratio <= WIDTH_COHERENCE_PPM[1]:
                c["table_width_incoherent"].hit(
                    b,
                    gold={"modes": [t.width_mode for t in gold.tables],
                          "tblw_mpt": [t.width_value_mpt for t in gold.tables],
                          "grid_sum_mpt": [sum(t.column_widths_mpt) for t in gold.tables]},
                    model={"node_id": table.node_id, "tblw_mpt": table.width_value_mpt,
                           "grid_sum_mpt": table.grid_sum_mpt, "tblw_over_grid_ppm": ratio},
                )

        # text
        c["text_exact_mismatch"].unit_seen()
        c["text_approx_mismatch"].unit_seen()
        ratio = _ratio_ppm(gold.text_token_ids, page.text_token_ids)
        text_ratios.append(ratio)
        text_facts_gold = {"text_tokens": len(gold.text_token_ids)}
        text_facts_model = {"text_tokens": len(page.text_token_ids), "ratio_ppm": ratio}
        if gold.text_token_ids != page.text_token_ids:
            c["text_exact_mismatch"].hit(b, gold=text_facts_gold, model=text_facts_model)
        if ratio < TEXT_APPROX_RATIO_PPM:
            c["text_approx_mismatch"].hit(b, gold=text_facts_gold, model=text_facts_model)

    c["image_localization"].facts = {
        "iou_threshold_ppm": IOU_THRESHOLD_PPM,
        "best_iou_ppm_median": _upper_median(ious),
    }
    c["table_width_incoherent"].facts = {
        "coherence_band_ppm": list(WIDTH_COHERENCE_PPM),
        "tblw_over_grid_ppm_median": _upper_median(width_ratios),
        "gold_tables": gold_tables,
        "gold_auto_arm_zero_value_targets": gold_auto_zero,
    }
    c["crop_nonzero"].facts = {
        "gold_images_with_crop_targets": gold_crop_targets,
        "gold_crop_nonzero_images": gold_crop_nonzero,
    }
    c["pipeline_refusal"].facts = {"pages_by_code": dict(sorted(pages_by_code.items()))}
    c["story_kind_duplicate"].facts = {
        # cross-evidence, not the same count: the chain refuses a story for
        # any semantic issue at its path, the class above counts only a
        # repeated /kind
        "pipeline_pages_refused_at_a_story_path": story_path_refusals,
    }
    c["text_approx_mismatch"].facts = {
        "ratio_threshold_ppm": TEXT_APPROX_RATIO_PPM,
        "ratio_ppm_median": _upper_median(text_ratios),
    }
    return c


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


#: E1: "severity" judges DEC-018 visual closeness, which is the owner's call,
#: not this tool's (RFC005-ERRATA-007 E1: "dopad na vizuálnu blízkosť
#: (DEC-018)"). The value stays `None` until the owner assigns it; the record
#: says explicitly who assigns it instead of a bare placeholder string.
NO_SEVERITY_ASSIGNED: dict[str, Any] = {
    "value": None,
    "assigned_by": "owner",
    "basis": "DEC-018 visual closeness to the original, not property-level exact match",
}

#: E1/E3: the reasoning dataset views (`reasoning_constructive_v0`,
#: `reasoning_corrective_v0`, RFC003-ERRATA-003) are ratified but no release
#: built from them exists yet (RFC005-ERRATA-007 DEC-045 item 4: "Predpoklad 3
#: z E2 tým nie je celý splnený"). This tool can only report what it can see;
#: it cannot see supervision that has not been built.
_REASONING_RELEASE_PENDING_REASON = (
    "RFC003-ERRATA-003 task_kinds (reasoning_constructive_v0/reasoning_corrective_v0) are "
    "ratified but no dataset release built from them exists yet (RFC005-ERRATA-007 E1 "
    "DEC-045 item 4); this tool cannot observe supervision that has not been built"
)


def _reasoning_supervision_available(reasoning: bool, cause: str) -> dict[str, Any]:
    """Whether targets exist to supervise this class's `reasoning_candidate`.

    `not reasoning` (the class's cause is not in `REASONING_CAUSES`) makes the
    question moot -- `not_applicable`, not `False`. A reasoning-eligible class
    is honestly `unknown` rather than a hardcoded `False`: nothing in the
    cache row or chain output this tool reads today tells it whether such
    targets exist, because the dataset release that would carry them has not
    been built (RFC005-ERRATA-007 E1)."""
    if not reasoning:
        return {
            "status": "not_applicable",
            "reason": f"suspected_cause {cause!r} has no reasoning_candidate "
            "(E1: reasoning answers only relational_structure and decoder_prior)",
        }
    return {"status": "unknown", "reason": _REASONING_RELEASE_PENDING_REASON}


def _wilson_95(count: int, n: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = count / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _record(
    name: str, counted: _ClassCount, *, scored_pages: int, split_entries: int | None
) -> dict[str, Any]:
    rule = CLASS_RULES[name]
    status = "counted" if counted.denominator > 0 else "insufficient_evidence"
    affected = len(counted.affected)
    scaled = (
        int(round(affected * split_entries / scored_pages))
        if split_entries is not None and scored_pages > 0
        else None
    )
    reasoning = rule.cause in REASONING_CAUSES
    record = {
        "failure_class": name,
        "frequency": {
            "count": counted.count,
            "denominator": counted.denominator,
            "unit": counted.unit,
            "status": status,
        },
        "severity": dict(NO_SEVERITY_ASSIGNED),
        "affected_slice": {
            "unit": "pages", "count": affected, "bundle_ids": list(counted.affected)
        },
        "dense_baseline_metric": {
            "count": counted.count,
            "denominator": counted.denominator,
            "unit": counted.unit,
        },
        "suspected_cause": rule.cause,
        "cause_evidence": {
            "rule": rule.rule,
            "examples": list(counted.examples),
            "facts": dict(counted.facts),
        },
        "reasoning_candidate": rule.reasoning_candidate if reasoning else "none",
        "eligible": reasoning,
        "reasoning_supervision_available": _reasoning_supervision_available(reasoning, rule.cause),
        "estimated_training_samples": {
            "cohort_affected_pages": affected,
            "cohort_pages_with_gold": scored_pages,
            "split_entries": split_entries,
            "scaled_estimate": scaled,
            "note": SCALING_NOTE,
        },
        "selected_for_experiment": None,
        "decision_reason": "",
    }
    assert tuple(record) == FIELD_ORDER
    return record


def build_report(
    artefact: sealed.SealedCohortArtefact,
    gold_rows: Mapping[str, GoldRow],
    *,
    pipeline: Mapping[str, PipelineOutcome] | None = None,
    split_entries: int | None = None,
    inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    check_role(artefact.role)
    pages = [model_page(p) for p in artefact.pages]
    bundles = [p.bundle for p in pages]
    gold_missing = [b for b in bundles if b not in gold_rows]
    scored = [p for p in pages if p.terminated and p.bundle in gold_rows]
    counts = _count_classes(pages, gold_rows, pipeline)
    identity = dict(artefact.eval_cache_identity)
    core: dict[str, Any] = {
        "failure_taxonomy_report_schema_version": REPORT_SCHEMA_VERSION,
        "run_class": "discovery",
        "inference_mode": "free_running",
        "role": artefact.role,
        "arm": artefact.arm,
        "checkpoint_id": identity.get("checkpoint_id"),
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "cohort_manifest_sha256": artefact.cohort_manifest_sha256,
        "eval_cache_identity": identity,
        "inputs": {**dict(inputs or {}), "split_entries": split_entries},
        "cause_register": list(CAUSE_REGISTER),
        "thresholds": {
            "iou_threshold_ppm": IOU_THRESHOLD_PPM,
            "width_coherence_band_ppm": list(WIDTH_COHERENCE_PPM),
            "text_approx_ratio_ppm": TEXT_APPROX_RATIO_PPM,
            "structural_token_count": STRUCTURAL_COUNT,
        },
        "accounting": {
            "pages_total": len(pages),
            "pages_terminated": sum(1 for p in pages if p.terminated),
            "pages_unterminated": sum(1 for p in pages if not p.terminated),
            "gold_found": len(pages) - len(gold_missing),
            "gold_missing": len(gold_missing),
            "gold_missing_bundle_ids": gold_missing,
            "scored_pages": len(scored),
            "pipeline_pages": (
                sum(1 for b in bundles if b in pipeline) if pipeline is not None else 0
            ),
            "sealed_accounting": artefact.accounting.as_dict(),
        },
        "classes": [
            _record(name, counts[name], scored_pages=len(scored), split_entries=split_entries)
            for name in CLASS_RULES
        ],
    }
    report_hash = domain_hash(REPORT_DOMAIN, core)
    doc = copy.deepcopy(core)
    for record in doc["classes"]:
        n, k = record["frequency"]["denominator"], record["frequency"]["count"]
        if n > 0:
            record["frequency"]["rate"] = k / n
            record["dense_baseline_metric"]["wilson_95"] = list(_wilson_95(k, n))
        else:
            record["dense_baseline_metric"]["wilson_95"] = None
        record["dense_baseline_metric"]["method"] = WILSON_METHOD
    doc["report_hash"] = report_hash
    return doc


_DERIVED_CLASS_FIELDS: tuple[tuple[str, str], ...] = (
    ("frequency", "rate"),
    ("dense_baseline_metric", "wilson_95"),
    ("dense_baseline_metric", "method"),
)


def hashed_core(report: Mapping[str, Any]) -> dict[str, Any]:
    """The float-free core `report_hash` was computed over."""
    core = copy.deepcopy(dict(report))
    core.pop("report_hash", None)
    for record in core["classes"]:
        for block, key in _DERIVED_CLASS_FIELDS:
            record[block].pop(key, None)
    return core


def hashed_report_hash(report: Mapping[str, Any]) -> str:
    return domain_hash(REPORT_DOMAIN, hashed_core(report))


# ---------------------------------------------------------------------------
# Diff mode (RFC005-ERRATA-007 E1 variant 1: report over S4, then over the
# G2 parent, to confirm or change the selection)
# ---------------------------------------------------------------------------


def diff_reports(
    report_a: Mapping[str, Any], report_b: Mapping[str, Any],
    *, label_a: str = "a", label_b: str = "b",
) -> dict[str, Any]:
    """Per-class frequency change between two failure-taxonomy reports (e.g.
    the S4 winner vs the future G2 parent). Both reports must carry the same
    `CLASS_RULES` register; this tool never invents a class. `rate_delta` is
    `None` whenever either side is `insufficient_evidence`."""
    by_class_a = {r["failure_class"]: r for r in report_a["classes"]}
    by_class_b = {r["failure_class"]: r for r in report_b["classes"]}
    missing = (set(CLASS_RULES) - by_class_a.keys()) | (set(CLASS_RULES) - by_class_b.keys())
    if missing:
        raise KeyError(f"report missing failure classes: {sorted(missing)}")
    rows = []
    for name in CLASS_RULES:
        fa, fb = by_class_a[name]["frequency"], by_class_b[name]["frequency"]
        rate_a = fa["count"] / fa["denominator"] if fa["status"] == "counted" else None
        rate_b = fb["count"] / fb["denominator"] if fb["status"] == "counted" else None
        rows.append({
            "failure_class": name,
            label_a: {"count": fa["count"], "denominator": fa["denominator"],
                      "status": fa["status"], "rate": rate_a},
            label_b: {"count": fb["count"], "denominator": fb["denominator"],
                      "status": fb["status"], "rate": rate_b},
            "rate_delta": (rate_b - rate_a) if rate_a is not None and rate_b is not None else None,
        })
    return {
        "label_a": label_a, "label_b": label_b,
        "report_hash_a": report_a.get("report_hash"), "report_hash_b": report_b.get("report_hash"),
        "classes": rows,
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _fmt_rate(record: Mapping[str, Any]) -> str:
    f = record["frequency"]
    if f["status"] != "counted":
        return "insufficient_evidence"
    lo, hi = record["dense_baseline_metric"]["wilson_95"]
    return f"{100 * f['rate']:.1f} % [{100 * lo:.1f}; {100 * hi:.1f}]"


def render_markdown(report: Mapping[str, Any]) -> str:
    acc = report["accounting"]
    lines = [
        "# Správa o zlyhaniach prvého modelu (RFC005-ERRATA-007 E1)",
        "",
        f"- trieda behu: `{report['run_class']}`, bez gradientu; "
        f"dekódovanie `{report['inference_mode']}`",
        f"- rola: `{report['role']}`, rameno: `{report['arm']}`",
        f"- checkpoint: `{report['checkpoint_id']}` ({report['checkpoint_sha256']})",
        f"- kohorta: `{report['cohort_manifest_sha256']}`",
        f"- hash správy: `{report['report_hash']}`",
        "",
        "## Účtovníctvo",
        "",
        "| položka | hodnota |",
        "|---|---:|",
    ]
    for key in ("pages_total", "pages_terminated", "pages_unterminated", "gold_found",
                "gold_missing", "scored_pages", "pipeline_pages"):
        lines.append(f"| {key} | {acc[key]} |")
    lines += [
        "",
        "## Triedy zlyhaní",
        "",
        "Závažnosť posudzuje vlastník podľa DEC-018 (vizuálna blízkosť), nástroj ju nevypĺňa. "
        "Interval: Wilson 95 % na stranách kohorty.",
        "",
        "| trieda | počet / menovateľ | jednotka | miera [95 %] | podozrivá príčina | "
        "kandidát uvažovania | dotknuté strany |",
        "|---|---:|---|---|---|---|---:|",
    ]
    for record in report["classes"]:
        f = record["frequency"]
        lines.append(
            f"| `{record['failure_class']}` | {f['count']} / {f['denominator']} | {f['unit']} | "
            f"{_fmt_rate(record)} | `{record['suspected_cause']}` | "
            f"`{record['reasoning_candidate']}` | {record['affected_slice']['count']} |"
        )
    lines += ["", "## Podrobnosti", ""]
    for record in report["classes"]:
        ev = record["cause_evidence"]
        lines += [
            f"### `{record['failure_class']}`",
            "",
            f"- pravidlo príčiny: {ev['rule']}",
            f"- fakty: `{ev['facts']}`" if ev["facts"] else "- fakty: —",
            f"- odhad tréningových vzoriek: `{record['estimated_training_samples']}`",
        ]
        if ev["examples"]:
            lines.append("- príklady:")
            for ex in ev["examples"]:
                lines.append(f"  - `{ex['bundle']}` gold `{ex['gold']}` model `{ex['model']}`")
        lines.append("")
    lines += [
        "## Rozhodnutie",
        "",
        "`selected_for_experiment` je pri každej triede `null`: výber špecializácie zapisuje "
        "vlastník do DEC (E1 „Brána“), nie tento nástroj.",
        "",
    ]
    return "\n".join(lines)
