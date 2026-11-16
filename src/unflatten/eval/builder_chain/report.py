"""`builder_chain_report.json` (design §2.6, plan P7): aggregates
`run.ChainRow`s into one report, hashed under `rfc004.builder-chain-report/v0`.

This domain lives in `eval/` (plan §0: "eval/ is not scanned by the m1
invariants test, so no module registration is needed") -- verified: `grep
-rn "eval" tests/model/test_m1_design_invariants.py` finds no reference to
`unflatten.eval` at all, only `unflatten.model`/`unflatten.ast` module
globs, so this domain needs no `_POST_M1_M2_MODULES`/§4.9 entry.

`report_hash` carries no path or timestamp field (there simply is none in
the payload) -- two runs over the same rows from different working
directories hash equal by construction, not by explicit exclusion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from unflatten.eval.builder_chain import autolink as al
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
from unflatten.eval.builder_chain import story_dedup as sd
from unflatten.eval.builder_chain import tab_stops_from_regions as tsr
from unflatten.eval.builder_chain import table_geometry as tg
from unflatten.eval.builder_chain.run import ChainRow
from unflatten.model import default_policy as dp
from unflatten.model import derivations as dv
from unflatten.model.hashing import domain_hash

#: Design §2.6: `domain_hash("rfc004.builder-chain-report/v0", ...)`.
_BUILDER_CHAIN_REPORT_DOMAIN = "rfc004.builder-chain-report/v0"

BUILDER_CHAIN_REPORT_SCHEMA_VERSION = "0.1.0"


def _failure_histogram(rows: Sequence[ChainRow]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for row in rows:
        if row.envelope is None or row.envelope.axes.ast_status != "failed":
            continue
        for issue in row.envelope.ast.failure_report["issues"]:
            pointer = issue.get("node_property_pointer") or ""
            key = f"{issue['code']}|{pointer}"
            hist[key] = hist.get(key, 0) + 1
    return hist


def _imputation_counts(rows: Sequence[ChainRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.review is None:
            continue
        for imputation in row.review.document["imputations"]:
            pointer = imputation["pointer"] or ""
            counts[pointer] = counts.get(pointer, 0) + 1
    return counts


def _derivation_counts(rows: Sequence[ChainRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.envelope is None:
            continue
        for derivation in row.envelope.ast.builder_report.get("derivations", ()):
            derivation_id = derivation["derivation_id"]
            counts[derivation_id] = counts.get(derivation_id, 0) + 1
    return counts


def _renderer_histogram(rows: Sequence[ChainRow]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for row in rows:
        if row.envelope is None:
            continue
        r = row.envelope.renderer
        key = f"{r.status}|{r.reason or ''}"
        hist[key] = hist.get(key, 0) + 1
    return hist


def _page_outcome_counts(rows: Sequence[ChainRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.page_outcome] = counts.get(row.page_outcome, 0) + 1
    return counts


def _per_row_records(rows: Sequence[ChainRow]) -> list[dict[str, Any]]:
    """Additive, per-row evidence (T2 point A, select_checkpoint's
    `per_lineage_metric_samples`): `item_id` (the model arm's
    `document_bundle_id`, gold's fixture path), whether the row is
    `strict_valid` (mirrors `strict_valid_count`'s own predicate) and
    whether the renderer reached `"render_ready"` (the histogram's
    `render_ready|...` keys). Not part of
    `core` -- merged into `doc` after `report_hash` is computed, exactly
    like `strict_valid_share`, so no existing artefact's hash changes."""
    def _strict_valid(row: ChainRow) -> bool:
        return row.envelope is not None and row.envelope.axes.ast_status == "strict_valid"

    def _renderer_ready(row: ChainRow) -> bool:
        return row.envelope is not None and row.envelope.renderer.status == "render_ready"

    return [
        {
            "item_id": row.item_id,
            "strict_valid": _strict_valid(row),
            "renderer_ready": _renderer_ready(row),
            "profile": row.profile,
        }
        for row in rows
    ]


def _review_candidate_count(rows: Sequence[ChainRow]) -> int:
    """Count of rows whose review candidate carries at least one
    imputation -- NOT every row with a `review_candidate.ReviewCandidateResult`
    (judgement call, verify should_fix: kept the existing name/report key
    rather than a breaking rename of the `builder_chain_report.json` schema
    field, documented here instead). A failed row with zero imputations
    (e.g. every failure was an abstention, not a default) still produces a
    `ReviewCandidateResult` but is NOT counted here -- see the gold-arm
    note in `docs/plans/spec-ast-builder-t5-promotion-design.md` ("Stav fix
    loop 1"), where all 5 failed rows produce a candidate but this count is
    still 0."""
    return sum(1 for row in rows if row.review is not None and row.review.document["imputations"])


def _merged_counts(per_row: Any) -> dict[str, int]:
    """Policy edit counts over the arm: summed per key, except the named
    per-page extrema (`table_geometry.EXTREMUM_KEYS`, e.g.
    `table_geometry:max_overflow_ppm`), which take the maximum -- an overflow
    factor summed over pages means nothing."""
    counts: dict[str, int] = {}
    for row_counts in per_row:
        for key, value in (row_counts or {}).items():
            if key in tg.EXTREMUM_KEYS:
                counts[key] = max(counts.get(key, value), value)
            else:
                counts[key] = counts.get(key, 0) + value
    return dict(sorted(counts.items()))


def build_report(
    rows: Sequence[ChainRow],
    *,
    arm: str,
    artefact_sha256: str | None = None,
    eval_cache_identity: Mapping[str, Any] | None = None,
    checkpoint_id: str | None = None,
    role: str | None = None,
    note: str | None = None,
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
) -> dict[str, Any]:
    """Assemble one `builder_chain_report.json`. `note` carries an
    explanatory string for a measured-but-degenerate arm (design §2.6:
    "the report must say why, and the test asserts that named reason
    rather than a bare zero") -- e.g. `typed_predictions were empty on
    every page (typed_decode=False)`."""
    total = len(rows)
    strict_valid_count = sum(
        1
        for row in rows
        if row.envelope is not None and row.envelope.axes.ast_status == "strict_valid"
    )
    review_candidate_count = _review_candidate_count(rows)
    # The RFC001 CBOR profile forbids floats (`cbor.py`, `_assert_no_floats`
    # precedent): the hashed core carries counts only, never a share --
    # `strict_valid_share`/`review_candidate_share` below are derived from
    # this same hashed data and merged in AFTER hashing, so they add no new
    # information the hash does not already cover (design §2.6).
    core: dict[str, Any] = {
        "builder_chain_report_schema_version": BUILDER_CHAIN_REPORT_SCHEMA_VERSION,
        "arm": arm,
        "role": role,
        "checkpoint_id": checkpoint_id,
        "artefact_sha256": artefact_sha256,
        "eval_cache_identity": (
            dict(eval_cache_identity) if eval_cache_identity is not None else None
        ),
        "note": note,
        "total": total,
        "strict_valid_count": strict_valid_count,
        "failure_histogram": dict(sorted(_failure_histogram(rows).items())),
        "review_candidate_count": review_candidate_count,
        "imputation_counts": dict(sorted(_imputation_counts(rows).items())),
        "derivation_counts": dict(sorted(_derivation_counts(rows).items())),
        "renderer_status_histogram": dict(sorted(_renderer_histogram(rows).items())),
        "page_outcome_counts": dict(sorted(_page_outcome_counts(rows).items())),
        "derivation_registry_hash": dv.derivation_registry_hash(),
        "default_policy_hash": dp.default_policy_hash(),
    }
    if headless_defaults:
        # Only when used, so a report without the policy keeps its hash. The
        # literal policy and every edit count enter the hashed core.
        core["headless_defaults"] = {
            "policy": hd.policy_record(),
            "counts": _merged_counts(row.policy_counts for row in rows),
        }
    if table_geometry:
        # RFC004-ERRATA-004 E2 (RATIFIED): same contract; the column source is
        # part of the literal record, so two runs under different sources
        # never hash equal.
        core["table_geometry"] = {
            "policy": tg.policy_record(table_geometry_column_source),
            "counts": _merged_counts(row.geometry_counts for row in rows),
        }
    if story_dedup:
        # RFC004-ERRATA-004 E5 (RATIFIED 2026-09-26): same contract.
        core["story_dedup"] = {
            "policy": sd.policy_record(),
            "counts": _merged_counts(row.story_dedup_counts for row in rows),
        }
    if recovered_image_materialization:
        # RFC004-ERRATA-004 E1 (PROPOSED).
        core["recovered_image_materialization"] = {
            "policy": rim.policy_record(),
            "counts": _merged_counts(row.recovered_image_counts for row in rows),
        }
    if runaway_row_repair:
        # RFC004-ERRATA-008 (PROPOSED, owner 2026-09-27): same contract; runs
        # before table_geometry, so its counts are merged first.
        core["runaway_row_repair"] = {
            "policy": rr.policy_record(),
            "counts": _merged_counts(row.runaway_row_repair_counts for row in rows),
        }
    if page_geometry_from_raster:
        # RFC004-ERRATA-008 (PROPOSED, owner 2026-09-27): same contract.
        core["page_geometry_from_raster"] = {
            "policy": pg.policy_record(),
            "counts": _merged_counts(row.page_geometry_counts for row in rows),
        }
    if language_detect:
        # RFC004-ERRATA-007 (PROPOSED, owner request 2026-09-27).
        core["language_detect"] = {
            "policy": ld.policy_record(),
            "counts": _merged_counts(row.language_detect_counts for row in rows),
        }
    if page_number_fields:
        # RFC004-ERRATA-009 (PROPOSED).
        core["page_number_fields"] = {
            "policy": pnf.policy_record(),
            "counts": _merged_counts(row.page_number_fields_counts for row in rows),
        }
    if autolink:
        # RFC004-ERRATA-009 (PROPOSED).
        core["autolink"] = {
            "policy": al.policy_record(),
            "counts": _merged_counts(row.autolink_counts for row in rows),
        }
    if leader_fill:
        # leader-fill.v0 (RFC004-ERRATA-010, RATIFIED).
        core["leader_fill"] = {
            "policy": lf.policy_record(),
            "counts": _merged_counts(row.leader_fill_counts for row in rows),
        }
    if image_region_snap:
        # RFC004-ERRATA-011 draft (PROPOSED, 2026-10-16): same contract.
        core["image_region_snap"] = {
            "policy": irs.policy_record(),
            "counts": _merged_counts(row.image_region_snap_counts for row in rows),
        }
    if run_typography_snap:
        # RFC004-ERRATA-011 E2 (PROPOSED, 2026-10-16): same contract.
        core["run_typography_snap"] = {
            "policy": rts.policy_record(),
            "counts": _merged_counts(row.run_typography_snap_counts for row in rows),
        }
    if tab_stops_from_regions:
        # RFC004-ERRATA-011 E3 (PROPOSED, 2026-10-16): same contract.
        core["tab_stops_from_regions"] = {
            "policy": tsr.policy_record(),
            "counts": _merged_counts(row.tab_stops_counts for row in rows),
        }
    if leader_fill_v1:
        # RFC004-ERRATA-011 E4 (PROPOSED, 2026-10-16): alternative to leader_fill.
        core["leader_fill_v1"] = {
            "policy": lf1.policy_record(),
            "counts": _merged_counts(row.leader_fill_v1_counts for row in rows),
        }
    report_hash = domain_hash(_BUILDER_CHAIN_REPORT_DOMAIN, core)
    doc = dict(core)
    doc["strict_valid_share"] = (strict_valid_count / total) if total else 0.0
    doc["review_candidate_share"] = (review_candidate_count / total) if total else 0.0
    doc["report_hash"] = report_hash
    doc["rows"] = _per_row_records(rows)
    return doc
