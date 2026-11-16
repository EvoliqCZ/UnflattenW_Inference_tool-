"""The gold and model arm row-producing chain (design §2.6, plan P7):
`assemble -> build_strict -> promote -> build_review_candidate`, one
`ChainRow` per fixture (gold) or per artefact page (model).

U21: `PageStreamInvalid`/`AssemblyInvariantViolation` raised while parsing
or assembling one page are named page outcomes -- caught HERE, at the
single point they can occur, and turned into `ChainRow.page_outcome`, so a
malformed page never aborts the whole run. Any other exception propagates
unmodified: it is not a "this page failed" shape, it is a bug."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from unflatten.assembly import hypothesis as hy
from unflatten.assembly import policy as pol
from unflatten.assembly import stories as st
from unflatten.assembly.assemble import AssemblyInvariantViolation, assemble
from unflatten.assembly.hypothesis import PageStreamInvalid
from unflatten.ast.canonicalizer import canonicalize
from unflatten.eval.builder_chain import autolink as al
from unflatten.eval.builder_chain import gold
from unflatten.eval.builder_chain import headless_defaults as hd
from unflatten.eval.builder_chain import image_region_snap as irs
from unflatten.eval.builder_chain import language_detect as ld
from unflatten.eval.builder_chain import leader_fill as lf
from unflatten.eval.builder_chain import leader_fill_v1 as lf1
from unflatten.eval.builder_chain import page_geometry_from_raster as pg
from unflatten.eval.builder_chain import page_number_fields as pnf
from unflatten.eval.builder_chain import recovered_image_materialization as rim
from unflatten.eval.builder_chain import run_typography_snap as rts
from unflatten.eval.builder_chain import runaway_row_repair as rr
from unflatten.eval.builder_chain import sealed_adapter as sa
from unflatten.eval.builder_chain import story_dedup as sd
from unflatten.eval.builder_chain import tab_stops_from_regions as tsr
from unflatten.eval.builder_chain import table_geometry as tg
from unflatten.eval.builder_chain.raster_source import PinnedRasterSource
from unflatten.eval.s2_gate import sealed
from unflatten.model import ast_builder as ab
from unflatten.model import promotion
from unflatten.model import review_candidate as rc
from unflatten.model.resource_binder import RecoveredResource
from unflatten.model.resource_recovery import exact_crop

#: Model-arm base profile (design §2.6 deviation, P7): the sealed artefact
#: carries no per-page `profile`/`ast_version`. The base profile is used
#: whenever the page emitted only `word-core-v0` node types; a page that
#: emitted a story, a floating image, a field or a nested table is built
#: under `word-visual-v1`, one with a text frame under `word-visual-v1.1`
#: (`model_profile_for`). 26.9.: the earlier fixed pin refused every such
#: page as `V1_FEATURE_IN_V0_PROFILE` -- 9 pages at step 27 915 and 27 at
#: step 80 000, once the model had learnt to emit headers and footers.
MODEL_PROFILE = "word-core-v0"
MODEL_AST_VERSION = "0.1.0"
_VISUAL_PROFILE = "word-visual-v1"
_TEXT_FRAME_PROFILE = "word-visual-v1.1"
#: Mirrors `semantic_validator._V1_NODE_TYPES` (pinned equal by a test):
#: notes and note refs are v1 features too, so a page with a footnote is a
#: `word-visual-v1` page.
_V1_NODE_TYPES = frozenset({
    "header", "footer", "footnote", "endnote", "floating_image", "note_ref", "hyperlink", "field",
})


def model_profile_for(page: sealed.PageDecodeResult) -> str:
    """The smallest profile that admits every node type the page emitted."""
    states = page.node_states or ()
    types = {state.node_type for state in states}
    if "text_frame" in types:
        return _TEXT_FRAME_PROFILE
    if types & _V1_NODE_TYPES:
        return _VISUAL_PROFILE
    parent = {state.local_node_id: state.parent_local_node_id for state in states}
    kind = {state.local_node_id: state.node_type for state in states}
    for state in states:
        if state.node_type != "table":
            continue
        ancestor = parent.get(state.local_node_id)
        while ancestor is not None:
            if kind.get(ancestor) == "cell":
                return _VISUAL_PROFILE  # a nested table is a v1 feature
            ancestor = parent.get(ancestor)
    return MODEL_PROFILE


@dataclass(frozen=True, slots=True)
class ChainRow:
    item_id: str
    page_outcome: str
    envelope: promotion.PromotionEnvelope | None
    review: rc.ReviewCandidateResult | None
    #: `headless_defaults` edits on this page, by kind; `None` when the
    #: opt-in policy was not applied (never an empty dict standing for "off").
    policy_counts: dict[str, int] | None = None
    #: `table-geometry.v0` edits (RFC004-ERRATA-004, RATIFIED 26.9.), same contract:
    #: its own field so the two audits never mix; `None` when off.
    geometry_counts: dict[str, int] | None = None
    #: `recovered-image-materialization.v0` edits (RFC004-ERRATA-004,
    #: RATIFIED 26.9.); `None` when off.
    recovered_image_counts: dict[str, int] | None = None
    #: The profile the model page was built under (`model_profile_for`);
    #: `None` for gold rows (their fixture carries its own profile).
    profile: str | None = None
    #: `story-kind-dedup.v0` edits (RFC004-ERRATA-004 E5, RATIFIED 26.9.); `None` when off.
    story_dedup_counts: dict[str, int] | None = None
    #: `page-geometry-from-raster.v0` edits (PROPOSED, RFC004-ERRATA-008); `None` when off.
    page_geometry_counts: dict[str, int] | None = None
    #: `runaway-row-repair.v0` edits (PROPOSED, RFC004-ERRATA-008); `None` when off.
    runaway_row_repair_counts: dict[str, int] | None = None
    #: `language-detect.v0` edits (PROPOSED, RFC004-ERRATA-007); `None` when off.
    language_detect_counts: dict[str, int] | None = None
    #: `page-number-fields.v0` edits (RFC004-ERRATA-009, PROPOSED); `None` when off.
    page_number_fields_counts: dict[str, int] | None = None
    #: `autolink.v0` edits (RFC004-ERRATA-009, PROPOSED); `None` when off.
    autolink_counts: dict[str, int] | None = None
    #: `leader-fill.v0` edits (RFC004-ERRATA-010, RATIFIED); `None` when off.
    leader_fill_counts: dict[str, int] | None = None
    #: `image-region-snap.v0` outcomes (PROPOSED, RFC004-ERRATA-011 draft); `None` when off.
    image_region_snap_counts: dict[str, int] | None = None
    #: `run-typography-snap.v0` outcomes (PROPOSED, RFC004-ERRATA-011 E2); `None` when off.
    run_typography_snap_counts: dict[str, int] | None = None
    #: `tab-stops-from-regions.v0` outcomes (PROPOSED, RFC004-ERRATA-011 E3); `None` when off.
    tab_stops_counts: dict[str, int] | None = None
    #: `leader-fill.v1` outcomes (PROPOSED, RFC004-ERRATA-011 E4; alternative to v0);
    #: `None` when off.
    leader_fill_v1_counts: dict[str, int] | None = None
    #: DEC-044 Dodatok D §1 (B1): per-INPUT-page outcome for a multi-page
    #: `process_model_pages` row, in page order; `None` for a single-page
    #: `ChainRow` (`page_outcome` alone already says everything there is).
    page_outcomes: tuple[str, ...] | None = None
    #: DEC-044 Dodatok D §1 (B1): `assemble()`'s own seam decisions,
    #: summarized as `{"joins_accepted": int, "joins_refused": int,
    #: "refusal_codes": {rule: count}}`; `None` for a single-page row (a
    #: standalone page has no seam) or when assembly itself failed.
    assembler_summary: dict[str, Any] | None = None


def _render_request(render_dir: Path | None, item_id: str) -> promotion.RenderRequest | None:
    """Opt-in rendering (design §2.1): without a directory the renderer axis
    stays `not_run|not_requested`, which a renderer-ready metric must never
    read as a measurement."""
    if render_dir is None:
        return None
    name = item_id.replace("/", "__")
    return promotion.RenderRequest(output_path=Path(render_dir) / f"{name}.docx")


def process_gold_fixture(relative: str, *, render_dir: Path | None = None) -> ChainRow:
    canonical = gold.load(relative)
    page = gold.gold_page(canonical)
    document = assemble([page], pol.POLICY_B0, profile=canonical["profile"])
    kwargs: dict[str, Any] = {
        "profile": canonical["profile"],
        "ast_version": canonical["ast_version"],
        "document_id": canonical["document_id"],
    }
    result = ab.build_strict(document, **kwargs)
    envelope = promotion.promote(result, render=_render_request(render_dir, relative))
    review = rc.build_review_candidate(document, **kwargs)
    return ChainRow(item_id=relative, page_outcome="parsed", envelope=envelope, review=review)


def run_gold_arm(
    fixtures: Sequence[str] = gold.GOLD_FIXTURES, *, render_dir: Path | None = None
) -> list[ChainRow]:
    return [process_gold_fixture(relative, render_dir=render_dir) for relative in fixtures]


def process_model_page(
    page: sealed.PageDecodeResult,
    *,
    vocabulary: Any,
    tokenizer: Any,
    raster_source: PinnedRasterSource | None = None,
    render_dir: Path | None = None,
    headless_defaults: bool = False,
    table_geometry: bool = False,
    table_geometry_column_source: str = tg.DEFAULT_COLUMN_SOURCE,
    recovered_image_materialization: bool = False,
    story_dedup: bool = False,
    language_detect: bool = False,
    page_number_fields: bool = False,
    autolink: bool = False,
    page_geometry_from_raster: bool = False,
    runaway_row_repair: bool = False,
    leader_fill: bool = False,
    image_region_snap: bool = False,
    run_typography_snap: bool = False,
    tab_stops_from_regions: bool = False,
    leader_fill_v1: bool = False,
) -> ChainRow:
    """`headless_defaults`: apply the opt-in `headless_defaults` policy to the
    page's typed predictions first (RATIFIED, RFC004-ERRATA-003) and carry its
    counts on the row. `runaway_row_repair` (PROPOSED, RFC004-ERRATA-008)
    runs next so a runaway row's excess empty cells never reach the column
    head. `page_geometry_from_raster` (PROPOSED, RFC004-ERRATA-008) runs
    next, so a corrected section page size feeds `table_geometry`'s
    `available` width. `table_geometry` (RFC004-ERRATA-004 E2, RATIFIED) runs
    last of the pre-assembly (pre-parse) policies and needs the grid
    `headless_defaults` derived, so it is refused without it.
    `language_detect` (PROPOSED, RFC004-ERRATA-007) runs on the parsed page
    hypothesis, after `headless_defaults` set every run's `/language` to
    `null` and before assembly, and is refused without it for the same
    one-chain-order reason as `story_dedup`. `page_number_fields`/`autolink`
    (RFC004-ERRATA-009, PROPOSED) run on the assembled document, after
    `story_dedup` and before `recovered_image_materialization`, followed by
    `leader_fill` (RFC004-ERRATA-010, RATIFIED); both take
    the page's resolved `profile` and refuse (counted) on `word-core-v0`,
    since `field`/`hyperlink` are v1+ node types. `recovered_image_
    materialization` (E1, RATIFIED) runs after resource recovery, on the
    assembled document, so it sees the recovered pixels and their
    provenance; every policy counts its own edits in its own row field."""
    if table_geometry and not headless_defaults:
        raise ValueError(
            "table-geometry.v0 requires headless_defaults: it consumes the grid "
            "headless-defaults.v0 derives"
        )
    if story_dedup and not headless_defaults:
        raise ValueError(
            "story-kind-dedup.v0 requires headless_defaults: policies run in one chain order"
        )
    if language_detect and not headless_defaults:
        raise ValueError(
            "language-detect.v0 requires headless_defaults: policies run in one chain order"
        )
    if runaway_row_repair and not headless_defaults:
        raise ValueError(
            "runaway-row-repair.v0 requires headless_defaults: one chain order"
        )
    if leader_fill and leader_fill_v1:
        raise ValueError(
            "leader-fill.v1 is an alternative to leader-fill.v0: never both in one chain"
        )
    policy_counts: dict[str, int] | None = None
    geometry_counts: dict[str, int] | None = None
    page_geometry_counts: dict[str, int] | None = None
    runaway_row_repair_counts: dict[str, int] | None = None
    # `{}` from the start when the policy is on: a page that never reaches
    # recovery still says "on, nothing to edit", never "off".
    recovered_image_counts: dict[str, int] | None = {} if recovered_image_materialization else None
    # `{}` from the start when on: a page that never reaches assembly says
    # "on, nothing merged", never "off" (same contract as recovered images).
    story_dedup_counts: dict[str, int] | None = {} if story_dedup else None
    # `{}` from the start when on: a page that never reaches a parsed
    # hypothesis says "on, nothing detected", never "off".
    language_detect_counts: dict[str, int] | None = {} if language_detect else None
    page_number_fields_counts: dict[str, int] | None = {} if page_number_fields else None
    autolink_counts: dict[str, int] | None = {} if autolink else None
    leader_fill_counts: dict[str, int] | None = {} if leader_fill else None
    # `{}` from the start when on: a page without image predictions says "on,
    # nothing to snap", never "off" (same contract as recovered images).
    image_region_snap_counts: dict[str, int] | None = {} if image_region_snap else None
    run_typography_snap_counts: dict[str, int] | None = {} if run_typography_snap else None
    tab_stops_counts: dict[str, int] | None = {} if tab_stops_from_regions else None
    leader_fill_v1_counts: dict[str, int] | None = {} if leader_fill_v1 else None
    # Read once, reused by `page_geometry_from_raster` below and by
    # `recovered_image_materialization` after assembly.
    raster = raster_source.read(page.document_bundle_id, 0) if raster_source is not None else None
    if headless_defaults:
        page, counted = hd.apply(page)
        policy_counts = dict(sorted(counted.items()))
    if run_typography_snap:
        # RFC004-ERRATA-011 E2 (PROPOSED): run font size to the gold grid / one
        # size per paragraph, sub-point character spacing to zero.
        page, counted = rts.apply(page)
        run_typography_snap_counts = dict(sorted(counted.items()))
    if runaway_row_repair:
        page, counted = rr.apply(page)
        runaway_row_repair_counts = dict(sorted(counted.items()))
    if page_geometry_from_raster:
        page, counted = pg.apply(page, raster=raster)
        page_geometry_counts = dict(sorted(counted.items()))
    if image_region_snap:
        # RFC004-ERRATA-011 draft (PROPOSED): tighten the predicted image regions
        # onto the raster's ink before `_recover_resources` cuts the crop.
        page, counted = irs.apply(page, raster=raster)
        image_region_snap_counts = dict(sorted(counted.items()))
    if tab_stops_from_regions:
        # RFC004-ERRATA-011 E3 (PROPOSED): a TAB paragraph's stops from the regions of
        # the runs that follow its TAB groups; reads the (possibly corrected) section
        # geometry, so it runs after `page-geometry-from-raster`.
        page, counted = tsr.apply(page)
        tab_stops_counts = dict(sorted(counted.items()))
    if table_geometry:
        page, counted = tg.apply(page, column_source=table_geometry_column_source)
        geometry_counts = dict(sorted(counted.items()))
    profile = model_profile_for(page)
    if page.outcome != "terminated":
        return ChainRow(
            item_id=page.document_bundle_id, page_outcome=page.outcome, envelope=None,
            review=None, policy_counts=policy_counts, geometry_counts=geometry_counts,
            recovered_image_counts=recovered_image_counts, profile=profile,
            story_dedup_counts=story_dedup_counts, page_geometry_counts=page_geometry_counts,
            runaway_row_repair_counts=runaway_row_repair_counts,
            language_detect_counts=language_detect_counts,
            page_number_fields_counts=page_number_fields_counts, autolink_counts=autolink_counts,
            leader_fill_counts=leader_fill_counts,
            image_region_snap_counts=image_region_snap_counts,
            run_typography_snap_counts=run_typography_snap_counts,
            tab_stops_counts=tab_stops_counts,
            leader_fill_v1_counts=leader_fill_v1_counts,
        )
    try:
        hypothesis = sa.page_hypothesis(page, vocabulary=vocabulary, tokenizer=tokenizer)
        parser_ids = sa.parser_local_ids_by_model_id(page, vocabulary=vocabulary)
        if language_detect:
            hypothesis, counted = ld.apply(hypothesis)
            language_detect_counts = dict(sorted(counted.items()))
        document = assemble([hypothesis], pol.POLICY_B0, profile=profile)
    except (PageStreamInvalid, AssemblyInvariantViolation) as exc:
        return ChainRow(
            item_id=page.document_bundle_id,
            page_outcome=type(exc).__name__,
            envelope=None,
            review=None,
            policy_counts=policy_counts,
            geometry_counts=geometry_counts,
            recovered_image_counts=recovered_image_counts,
            profile=profile,
            story_dedup_counts=story_dedup_counts,
            page_geometry_counts=page_geometry_counts,
            runaway_row_repair_counts=runaway_row_repair_counts,
            language_detect_counts=language_detect_counts,
            page_number_fields_counts=page_number_fields_counts, autolink_counts=autolink_counts,
            leader_fill_counts=leader_fill_counts,
            image_region_snap_counts=image_region_snap_counts,
            run_typography_snap_counts=run_typography_snap_counts,
            tab_stops_counts=tab_stops_counts,
            leader_fill_v1_counts=leader_fill_v1_counts,
        )
    kwargs: dict[str, Any] = {
        "profile": profile,
        "ast_version": MODEL_AST_VERSION,
        "document_id": page.document_bundle_id,
    }
    recovered = _recover_resources(page, document, raster, parser_ids)
    if story_dedup:
        # post-assembly: the assembler lifts every same-type story of a page
        # into ONE derived story whose kind it writes onto each root, so the
        # duplicates are merged into the first root (RFC004-ERRATA-004 E5)
        document, counted = sd.apply(document)
        story_dedup_counts = dict(sorted(counted.items()))
    if page_number_fields:
        document, counted = pnf.apply(document, profile=profile)
        page_number_fields_counts = dict(sorted(counted.items()))
    if autolink:
        document, counted = al.apply(document, profile=profile)
        autolink_counts = dict(sorted(counted.items()))
    if leader_fill:
        document, counted = lf.apply(document)
        leader_fill_counts = dict(sorted(counted.items()))
    if leader_fill_v1:
        # RFC004-ERRATA-011 E4 (PROPOSED): v0's place in the chain, never with v0.
        document, counted = lf1.apply(document)
        leader_fill_v1_counts = dict(sorted(counted.items()))
    if recovered_image_materialization:
        raster_dpi = rim.read_raster_dpi(raster) if raster is not None else None
        document, counted = rim.apply(document, recovered, raster_dpi=raster_dpi)
        recovered_image_counts = dict(sorted(counted.items()))
    document = _refresh_story_resources(document, recovered)
    result = ab.build_strict(document, recovered=recovered, **kwargs)
    envelope = promotion.promote(
        result, render=_render_request(render_dir, page.document_bundle_id)
    )
    review = rc.build_review_candidate(document, recovered=recovered, **kwargs)
    return ChainRow(
        item_id=page.document_bundle_id, page_outcome="terminated", envelope=envelope,
        review=review, policy_counts=policy_counts, geometry_counts=geometry_counts,
        recovered_image_counts=recovered_image_counts, profile=profile,
        story_dedup_counts=story_dedup_counts, page_geometry_counts=page_geometry_counts,
        runaway_row_repair_counts=runaway_row_repair_counts,
        language_detect_counts=language_detect_counts,
        page_number_fields_counts=page_number_fields_counts, autolink_counts=autolink_counts,
            leader_fill_counts=leader_fill_counts,
            image_region_snap_counts=image_region_snap_counts,
            run_typography_snap_counts=run_typography_snap_counts,
            tab_stops_counts=tab_stops_counts,
            leader_fill_v1_counts=leader_fill_v1_counts,
    )


def run_model_arm(
    artefact: sealed.SealedCohortArtefact,
    *,
    vocabulary: Any,
    tokenizer: Any,
    raster_source: PinnedRasterSource | None = None,
    render_dir: Path | None = None,
    headless_defaults: bool = False,
    table_geometry: bool = False,
    table_geometry_column_source: str = tg.DEFAULT_COLUMN_SOURCE,
    recovered_image_materialization: bool = False,
    story_dedup: bool = False,
    language_detect: bool = False,
    page_number_fields: bool = False,
    autolink: bool = False,
    page_geometry_from_raster: bool = False,
    runaway_row_repair: bool = False,
    leader_fill: bool = False,
    image_region_snap: bool = False,
    run_typography_snap: bool = False,
    tab_stops_from_regions: bool = False,
    leader_fill_v1: bool = False,
) -> list[ChainRow]:
    if raster_source is not None:
        try:
            expected_release_root_hash = artefact.eval_cache_identity["dataset_release_root_hash"]
        except KeyError as exc:
            raise ValueError(
                "sealed artefact has no decoded input dataset_release_root_hash"
            ) from exc
        raster_source.require_decoded_input_provenance(expected_release_root_hash)
    return [
        process_model_page(
            page,
            vocabulary=vocabulary,
            tokenizer=tokenizer,
            raster_source=raster_source,
            render_dir=render_dir,
            headless_defaults=headless_defaults,
            table_geometry=table_geometry,
            table_geometry_column_source=table_geometry_column_source,
            recovered_image_materialization=recovered_image_materialization,
            story_dedup=story_dedup,
            language_detect=language_detect,
            page_number_fields=page_number_fields,
            autolink=autolink,
            page_geometry_from_raster=page_geometry_from_raster,
            runaway_row_repair=runaway_row_repair,
            leader_fill=leader_fill,
            image_region_snap=image_region_snap,
            run_typography_snap=run_typography_snap,
            tab_stops_from_regions=tab_stops_from_regions,
            leader_fill_v1=leader_fill_v1,
        )
        for page in artefact.pages
    ]


#: `word-core-v0` < `word-visual-v1` < `word-visual-v1.1` (`model_profile_for`'s
#: own three outputs): the profile a multi-page document builds under is the
#: WIDEST any one of its pages needs, never a per-page pin.
_PROFILE_RANK = {MODEL_PROFILE: 0, _VISUAL_PROFILE: 1, _TEXT_FRAME_PROFILE: 2}

#: DEC-044 Dodatok D §1 (B1): a minimal, known-valid `word-core-v0` page (P0
#: corpus `valid/minimal_paragraph.json`, inlined so a failed/unreadable page
#: never depends on a filesystem read in the service's hot path) -- the
#: template a failed page's placeholder paragraph is built from, its one run
#: text replaced by the page's own error/outcome code.
_PLACEHOLDER_CANONICAL: dict[str, Any] = {
    "ast_version": "0.1.0",
    "profile": "word-core-v0",
    "document_id": "placeholder",
    "metadata": {"language": "en-US", "title": None, "subject": None},
    "resources": [],
    "sections": [
        {
            "id": "s1",
            "type": "section",
            "break_type": "next_page",
            "page": {"width_mpt": 595276, "height_mpt": 841890, "orientation": "portrait"},
            "margins": {
                "top_mpt": 72000, "right_mpt": 72000, "bottom_mpt": 72000,
                "left_mpt": 72000, "gutter_mpt": 0,
            },
            "columns": {"count": 1, "equal_width": True, "gap_mpt": 18000, "widths_mpt": []},
            "children": [
                {
                    "id": "p1",
                    "type": "paragraph",
                    "semantic": {"role": "body", "heading_level": None, "outline_level": None},
                    "layout": {
                        "alignment": "left", "direction": "ltr",
                        "indent": {
                            "left_mpt": 0, "right_mpt": 0,
                            "first_line_mpt": 0, "hanging_mpt": 0,
                        },
                        "spacing": {
                            "before_mpt": 0, "after_mpt": 0, "line": {"mode": "single"},
                        },
                        "pagination": {
                            "keep_with_next": False, "keep_lines": False,
                            "page_break_before": False, "widow_control": True,
                        },
                        "tabs": [],
                    },
                    "style": {"background": None, "borders": None},
                    "children": [
                        {
                            "id": "r1",
                            "type": "run",
                            "text": "placeholder",
                            "language": "en-US",
                            "style": {
                                "font_family": "Arial", "font_size_mpt": 11000,
                                "bold": False, "italic": False, "underline": "none",
                                "strike": False, "color": "#000000", "highlight": None,
                                "superscript": False, "subscript": False,
                                "character_spacing_mpt": 0, "small_caps": False,
                                "all_caps": False,
                            },
                        }
                    ],
                }
            ],
        }
    ],
    "extensions": {},
}


def _placeholder_hypothesis(ordinal: int, error_code: str) -> hy.PageHypothesis:
    """A one-paragraph page carrying `error_code` as its only run's text
    (design B1: 'failed page -> placeholder paragraph with its error code,
    never drop other pages')."""
    canonical = copy.deepcopy(_PLACEHOLDER_CANONICAL)
    canonical["sections"][0]["children"][0]["children"][0]["text"] = error_code
    canonical["document_id"] = f"placeholder-{ordinal}"
    return gold.gold_page(canonicalize(canonical), ordinal=ordinal)


def process_model_pages(
    pages: Sequence[sealed.PageDecodeResult],
    *,
    vocabulary: Any,
    tokenizer: Any,
    document_id: str,
    render_dir: Path | None = None,
    headless_defaults: bool = True,
    table_geometry: bool = True,
    table_geometry_column_source: str = tg.DEFAULT_COLUMN_SOURCE,
    story_dedup: bool = True,
    leader_fill: bool = False,
    run_typography_snap: bool = False,
    tab_stops_from_regions: bool = False,
    leader_fill_v1: bool = False,
    page_rasters: Mapping[int, bytes] | None = None,
    image_region_snap: bool = False,
    recovered_image_materialization: bool = False,
) -> ChainRow:
    """DEC-044 Dodatok D §1 (B1): the multi-page counterpart of
    `process_model_page` -- the SAME pre-assembly chain policies applied to
    EACH page independently, then ONE `assemble()` call across every page's
    hypothesis, then ONE strict AST build and ONE render, so a joined seam
    produces ONE Word section and a refused seam produces two (the
    renderer's own page-break-on-new-section behaviour).

    A page whose decode never reached a parsed hypothesis (`page.outcome !=
    'terminated'`, or a `PageStreamInvalid` while parsing it) is replaced by
    `_placeholder_hypothesis` instead of raising -- 'never drop a page'.
    `page_outcomes` on the returned row carries each INPUT page's own
    outcome, in order, for status reporting.

    `page_rasters` carries the original raster for each input ordinal.
    Resource recovery joins each page's decoder ids to its parser ids and
    assembled owners using that ordinal, so repeated local ids never borrow
    another page's pixels. Missing rasters remain named resource failures.
    """
    if not pages:
        raise ValueError("process_model_pages requires at least one page")
    if leader_fill and leader_fill_v1:
        raise ValueError(
            "leader-fill.v1 is an alternative to leader-fill.v0: never both in one chain"
        )
    policy_counts: dict[str, int] | None = {} if headless_defaults else None
    geometry_counts: dict[str, int] | None = {} if table_geometry else None
    # RFC004-ERRATA-011 E2/E3 (DEC-077): the same per-page, pre-assembly policies as
    # `process_model_page`, in the same order (after headless-defaults, before
    # table-geometry), with image snapping against the same page's raster.
    run_typography_snap_counts: dict[str, int] | None = {} if run_typography_snap else None
    tab_stops_counts: dict[str, int] | None = {} if tab_stops_from_regions else None
    image_region_snap_counts: dict[str, int] | None = {} if image_region_snap else None
    recovered_image_counts: dict[str, int] | None = {} if recovered_image_materialization else None
    recovery_pages: list[tuple[int, sealed.PageDecodeResult, bytes, dict[str, str]]] = []
    page_outcomes: list[str] = []
    hypotheses: list[hy.PageHypothesis] = []
    profile = MODEL_PROFILE
    for ordinal, page in enumerate(pages):
        raster = page_rasters.get(ordinal) if page_rasters is not None else None
        if headless_defaults:
            page, counted = hd.apply(page)
            policy_counts = _merge_counts(policy_counts, counted)
        if run_typography_snap:
            page, counted = rts.apply(page)
            run_typography_snap_counts = _merge_counts(run_typography_snap_counts, counted)
        if image_region_snap:
            page, counted = irs.apply(page, raster=raster)
            image_region_snap_counts = _merge_counts(image_region_snap_counts, counted)
        if tab_stops_from_regions:
            page, counted = tsr.apply(page)
            tab_stops_counts = _merge_counts(tab_stops_counts, counted)
        if table_geometry:
            page, counted = tg.apply(page, column_source=table_geometry_column_source)
            geometry_counts = _merge_counts(geometry_counts, counted)
        page_profile = model_profile_for(page)
        if _PROFILE_RANK[page_profile] > _PROFILE_RANK[profile]:
            profile = page_profile
        if page.outcome != "terminated":
            page_outcomes.append(page.outcome)
            hypotheses.append(_placeholder_hypothesis(ordinal, page.outcome))
            continue
        try:
            hypotheses.append(
                sa.page_hypothesis(
                    page,
                    vocabulary=vocabulary,
                    tokenizer=tokenizer,
                    page_input=hy.PageInput(
                        page_ordinal=ordinal,
                        width_px=sa.PAGE_INPUT.width_px,
                        height_px=sa.PAGE_INPUT.height_px,
                    ),
                )
            )
            page_outcomes.append("terminated")
            if raster is not None and page.region_predictions:
                parser_ids = sa.parser_local_ids_by_model_id(page, vocabulary=vocabulary)
                recovery_pages.append((ordinal, page, raster, parser_ids))
        except PageStreamInvalid as exc:
            page_outcomes.append(type(exc).__name__)
            hypotheses.append(_placeholder_hypothesis(ordinal, type(exc).__name__))
    common: dict[str, Any] = {
        "policy_counts": policy_counts,
        "geometry_counts": geometry_counts,
        "run_typography_snap_counts": run_typography_snap_counts,
        "tab_stops_counts": tab_stops_counts,
        "image_region_snap_counts": image_region_snap_counts,
        "recovered_image_counts": recovered_image_counts,
        "profile": profile,
        "page_outcomes": tuple(page_outcomes),
    }
    try:
        document = assemble(hypotheses, pol.POLICY_B0, profile=profile)
    except AssemblyInvariantViolation as exc:
        return ChainRow(
            item_id=document_id, page_outcome=type(exc).__name__,
            envelope=None, review=None, **common,
        )
    assembler_summary = _assembler_summary(document.seam_decisions)
    recovered: dict[str, RecoveredResource] = {}
    raster_dpi_by_slot: dict[str, float | None] = {}
    for ordinal, page, raster, parser_ids in recovery_pages:
        resources = _recover_resources(
            page, document, raster, parser_ids, page_ordinal=ordinal,
        )
        if recovered.keys() & resources.keys():
            raise ValueError("multiple pages own the same assembled image resource")
        recovered.update(resources)
        dpi = rim.read_raster_dpi(raster)
        raster_dpi_by_slot.update(dict.fromkeys(resources, dpi))
    story_dedup_counts: dict[str, int] | None = None
    if story_dedup:
        document, counted = sd.apply(document)
        story_dedup_counts = dict(sorted(counted.items()))
    leader_fill_counts: dict[str, int] | None = None
    if leader_fill:
        document, counted = lf.apply(document)
        leader_fill_counts = dict(sorted(counted.items()))
    leader_fill_v1_counts: dict[str, int] | None = None
    if leader_fill_v1:
        document, counted = lf1.apply(document)
        leader_fill_v1_counts = dict(sorted(counted.items()))
    if recovered_image_materialization:
        document, counted = rim.apply(
            document, recovered, raster_dpi=None, raster_dpi_by_slot=raster_dpi_by_slot,
        )
        common["recovered_image_counts"] = dict(sorted(counted.items()))
    document = _refresh_story_resources(document, recovered)
    kwargs: dict[str, Any] = {
        "profile": profile, "ast_version": MODEL_AST_VERSION, "document_id": document_id,
    }
    result = ab.build_strict(document, recovered=recovered, **kwargs)
    envelope = promotion.promote(result, render=_render_request(render_dir, document_id))
    return ChainRow(
        item_id=document_id, page_outcome="terminated", envelope=envelope, review=None,
        story_dedup_counts=story_dedup_counts, leader_fill_counts=leader_fill_counts,
        leader_fill_v1_counts=leader_fill_v1_counts,
        assembler_summary=assembler_summary, **common,
    )


def _assembler_summary(seam_decisions: Sequence[Any]) -> dict[str, Any]:
    """DEC-044 Dodatok D §1 (B1): `joins_accepted`/`joins_refused` count
    every rule decision at every seam (a seam may see several rule attempts
    at different node levels); `refusal_codes` tallies refused decisions by
    their rule name (`SEC`, `PARA`, `TABLE`, ... -- `assemble.py`'s own
    `_RULE_BY_TYPE`), the closest thing this module has to a stable code for
    'why did these two pages not join'."""
    refusal_codes: dict[str, int] = {}
    accepted = refused = 0
    for decision in seam_decisions:
        if decision.joined:
            accepted += 1
        else:
            refused += 1
            refusal_codes[decision.rule] = refusal_codes.get(decision.rule, 0) + 1
    return {
        "joins_accepted": accepted,
        "joins_refused": refused,
        "refusal_codes": dict(sorted(refusal_codes.items())),
    }


def _merge_counts(total: dict[str, int], counted: Mapping[str, int]) -> dict[str, int]:
    merged = dict(total)
    for key, value in counted.items():
        merged[key] = merged.get(key, 0) + value
    return dict(sorted(merged.items()))


def _refresh_story_resources(document: Any, recovered: Mapping[str, RecoveredResource]) -> Any:
    """Derive repeating stories from the pixels that will actually be bound.

    Different logos must not compare equal just because their predicted
    geometry matches. Unresolved slots remain distinct so deduplication
    cannot hide a missing image on a later page.
    """
    if not document.stories:
        return document
    resource_keys = {
        slot: hashlib.sha256(resource.data).hexdigest()
        for slot, resource in recovered.items()
        if resource.status == "resolved" and resource.data is not None
    }
    return replace(document, story_sets=st.derive_story_sets(
        document.sections, document.stories, resource_keys=resource_keys,
    ))


def _recover_resources(
    page: sealed.PageDecodeResult,
    document: Any,
    raster: bytes | None,
    parser_ids: dict[str, str],
    *,
    page_ordinal: int = 0,
) -> dict[str, RecoveredResource]:
    """Join sealed image regions to assembled owners, never list positions.

    Model node ids and parser node ids are separate namespaces.  The sealed
    node's structural anchor identifies the parser node, whose `(page, id)`
    source ownership identifies the assembled image node. `raster` is the
    page's pinned PNG (read once by the caller, who also hands it to the
    recovered-image policy); `None` means no raster source at all.
    """
    if raster is None:
        return {}
    source_to_assembled = _assembled_source_owners(document)
    states = {state.local_node_id: state for state in page.node_states or ()}
    predictions: dict[str, RecoveredResource] = {}
    for prediction in page.region_predictions:
        if not isinstance(prediction, sealed.SealedRegionPrediction):
            raise ValueError("builder requires sealed, validated region predictions")
        state = states.get(prediction.local_node_id)
        if state is None or state.node_type != prediction.node_type:
            raise ValueError(
                f"region prediction has no matching sealed node {prediction.local_node_id!r}"
            )
        parser_id = parser_ids.get(prediction.local_node_id)
        if parser_id is None:
            raise ValueError(f"region prediction {prediction.local_node_id!r} has no parser source")
        owner = source_to_assembled.get((page_ordinal, parser_id))
        if owner is None:
            raise ValueError(
                f"region prediction {prediction.local_node_id!r} has no assembled source owner"
            )
        if owner[1] != prediction.node_type:
            raise ValueError(
                f"region prediction {prediction.local_node_id!r} has a foreign source owner"
            )
        assembled_id = owner[0]
        if assembled_id in predictions:
            raise ValueError(f"multiple region predictions own assembled node {assembled_id!r}")
        predictions[assembled_id] = replace(exact_crop(raster, prediction), slot=assembled_id)
    return predictions


def _assembled_source_owners(document: Any) -> dict[tuple[int, str], tuple[str, str]]:
    owners: dict[tuple[int, str], tuple[str, str]] = {}

    def walk(node: Any) -> None:
        for source in node.sources:
            if source in owners:
                raise ValueError(f"source {source!r} has multiple assembled owners")
            owners[source] = (node.assembled_id, node.node_type)
        for child in node.children:
            walk(child)

    for root in (*document.sections, *document.notes, *(story.node for story in document.stories)):
        walk(root)
    return owners
