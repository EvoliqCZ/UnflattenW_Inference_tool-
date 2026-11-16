"""Cohort report for RFC005-ERRATA-006 B3 (text gaps/duplicates).

Consumes a `unflatten.eval.s2_gate.sealed.SealedCohortArtefact` and the
matching `unflatten.eval.cohort_gold.GoldRow`s, runs `unflatten.eval.
text_continuity.compute_page_continuity` per attempted page, and rolls the
result up into one verdict `unflatten.model.s4_stage_gate_report.
gate_b3_no_text_gaps_or_duplicates` can consume directly (path + sha256,
per this task's own instruction).

Only pages whose outcome is in `text_continuity.DECODED_OUTCOMES` are in
scope -- everything else is a pre-decode disposition with no tokens to
compare. A page that IS in scope but cannot actually be measured (its
bundle has no gold row, or its decode/gold token sequence is empty) is
counted `unmeasurable`, never silently skipped or scored clean.

RFC005-ERRATA-006 E6 B3 reads "zero text gaps and duplicates on every
strict-valid page output", not on every attempted page. When a caller
supplies the `unflatten.eval.builder_chain` report for the same sealed
artefact (`chain_report`), only pages the chain marked `strict_valid` are
measured; every other attempted page is counted `not_strict_valid`
(excluded, with its own denominator) and the report is tagged
`scope: "strict_valid_pages"`. Without a chain report, every attempted
page is measured as before, but the report is tagged
`scope: "all_attempted_pages"` so `s4_stage_gate_report.gate_b3_no_text_
gaps_or_duplicates` can recognise the wider scope and refuse to pass B3 on
it. A chain report whose own `artefact_sha256` does not match the sealed
artefact's `sha256` is refused outright -- it is evidence about a
different decode run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.eval import cohort_gold as cg
from unflatten.eval import text_continuity as tc
from unflatten.eval.s2_gate import sealed
from unflatten.eval.text_continuity_policy import (
    DEFAULT_POLICY_PATH,
    load_text_continuity_policy,
    validate_text_continuity_policy,
)

TEXT_CONTINUITY_REPORT_SCHEMA_VERSION = "1.0.0"

#: `build_text_continuity_report` scope tags (RFC005-ERRATA-006 E6 B3):
#: narrowed to the chain report's strict-valid pages, or every attempted
#: page when no chain report was supplied.
SCOPE_STRICT_VALID_PAGES = "strict_valid_pages"
SCOPE_ALL_ATTEMPTED_PAGES = "all_attempted_pages"


class TextContinuityReportError(ValueError):
    """The cohort report cannot be built from the artefacts given."""


def _page_row(page: sealed.PageDecodeResult, gold: cg.GoldRow | None) -> dict[str, Any]:
    if gold is None or not gold.text_token_ids:
        return {
            "document_bundle_id": page.document_bundle_id,
            "outcome": page.outcome,
            "measured": False,
            "reason": "no gold text tokens for this bundle",
        }
    try:
        decode_tokens = tc.decode_text_tokens_for_page(page)
    except tc.TextContinuityError as exc:
        return {
            "document_bundle_id": page.document_bundle_id,
            "outcome": page.outcome,
            "measured": False,
            "reason": str(exc),
        }
    return {
        "document_bundle_id": page.document_bundle_id,
        "outcome": page.outcome,
        "measured": True,
        "decode_tokens": decode_tokens,
        "gold_tokens": gold.text_token_ids,
    }


def _strict_valid_bundle_ids(chain_report: Mapping[str, Any]) -> set[str]:
    return {
        row["item_id"]
        for row in chain_report.get("rows", ())
        if row.get("strict_valid")
    }


def build_text_continuity_report(
    artefact: sealed.SealedCohortArtefact,
    gold_rows: Mapping[str, cg.GoldRow],
    *,
    policy: Mapping[str, Any],
    chain_report: Mapping[str, Any] | None = None,
    sealed_artefact_sha256: str | None = None,
) -> dict[str, Any]:
    validate_text_continuity_policy(policy)
    if chain_report is not None:
        chain_sha = chain_report.get("artefact_sha256")
        if sealed_artefact_sha256 is None or chain_sha != sealed_artefact_sha256:
            raise TextContinuityReportError(
                f"chain report artefact_sha256 {chain_sha!r} does not match the sealed "
                f"artefact's own sha256 {sealed_artefact_sha256!r} -- refusing to narrow B3 "
                "to strict-valid pages from a report about a different decode run"
            )

    attempted = [p for p in artefact.pages if p.outcome in tc.DECODED_OUTCOMES]
    if not attempted:
        raise TextContinuityReportError(
            "no attempted page (terminated/max_steps_exceeded/vocabulary_exhausted) in "
            "the sealed artefact -- refusing to build an empty report"
        )

    if chain_report is not None:
        scope = SCOPE_STRICT_VALID_PAGES
        strict_ids = _strict_valid_bundle_ids(chain_report)
        eligible = [p for p in attempted if p.document_bundle_id in strict_ids]
        not_strict_valid = len(attempted) - len(eligible)
        if not eligible:
            raise TextContinuityReportError(
                "no attempted page is strict_valid per the chain report -- refusing to "
                "build an empty report"
            )
    else:
        scope = SCOPE_ALL_ATTEMPTED_PAGES
        eligible = attempted
        not_strict_valid = 0

    rows: list[dict[str, Any]] = []
    unmeasurable = 0
    pages_with_gap = 0
    pages_with_duplicate = 0
    pages_bad = 0
    for page in eligible:
        prepared = _page_row(page, gold_rows.get(page.document_bundle_id))
        if not prepared["measured"]:
            unmeasurable += 1
            skip = ("decode_tokens", "gold_tokens")
            rows.append({k: v for k, v in prepared.items() if k not in skip})
            continue
        result = tc.compute_page_continuity(
            prepared["gold_tokens"],
            prepared["decode_tokens"],
            min_gap_tokens=policy["min_gap_tokens"],
            min_duplicate_tokens=policy["min_duplicate_tokens"],
        )
        if result.has_gap:
            pages_with_gap += 1
        if result.has_duplicate:
            pages_with_duplicate += 1
        if result.has_gap or result.has_duplicate:
            pages_bad += 1
        rows.append(
            {
                "document_bundle_id": prepared["document_bundle_id"],
                "outcome": prepared["outcome"],
                "measured": True,
                "gold_length": result.gold_length,
                "decode_length": result.decode_length,
                "gap_count": len(result.gaps),
                "duplicate_count": len(result.duplicates),
                "has_gap": result.has_gap,
                "has_duplicate": result.has_duplicate,
                "alignment_ratio_ppm": result.alignment_ratio_ppm,
            }
        )

    measured = len(eligible) - unmeasurable
    accounting = {
        "eligible": len(eligible),
        "measured": measured,
        "unmeasurable": unmeasurable,
        "not_strict_valid": not_strict_valid,
        "pages_with_gap": pages_with_gap,
        "pages_with_duplicate": pages_with_duplicate,
        "pages_bad": pages_bad,
    }
    verdict, reason = _verdict(policy, accounting)
    return {
        "text_continuity_report_schema_version": TEXT_CONTINUITY_REPORT_SCHEMA_VERSION,
        "policy": dict(policy),
        "is_placeholder": bool(policy["is_placeholder"]),
        "scope": scope,
        "accounting": accounting,
        "pages": rows,
        "verdict": verdict,
        "reason": reason,
    }


def _verdict(policy: Mapping[str, Any], accounting: Mapping[str, int]) -> tuple[str, str]:
    if policy["is_placeholder"]:
        return (
            "insufficient_evidence",
            "text continuity policy thresholds are still PLACEHOLDER values "
            "(RFC005-ERRATA-006 B3 not yet owner-frozen)",
        )
    measured = accounting["measured"]
    if measured == 0:
        return "insufficient_evidence", "no page in the cohort could be measured"
    if policy["verdict_mode"] == "zero_pages":
        if accounting["pages_bad"] == 0:
            return "pass", f"zero pages with a gap or duplicate across {measured} measured page(s)"
        return "fail", (
            f"{accounting['pages_bad']} of {measured} measured page(s) carry a gap or "
            f"duplicate ({accounting['pages_with_gap']} gap, "
            f"{accounting['pages_with_duplicate']} duplicate)"
        )
    rate_micro = round((measured - accounting["pages_bad"]) / measured * 1_000_000)
    if rate_micro >= policy["rate_floor_micro"]:
        return "pass", f"clean-page rate {rate_micro}/1000000 >= floor {policy['rate_floor_micro']}"
    return "fail", f"clean-page rate {rate_micro}/1000000 < floor {policy['rate_floor_micro']}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="text-continuity-report",
        description="Build the RFC005-ERRATA-006 B3 text-continuity cohort report.",
    )
    parser.add_argument("--sealed", type=Path, required=True, help="sealed cohort artefact JSON")
    parser.add_argument(
        "--cache-split", type=Path, required=True,
        help="exact-text cache split holding the cohort's gold rows (e.g. .../val_iid)",
    )
    parser.add_argument(
        "--policy", type=Path, default=None,
        help="text-continuity policy JSON; defaults to the shipped PLACEHOLDER policy",
    )
    parser.add_argument(
        "--chain-report", type=Path, default=None,
        help="unflatten.eval.builder_chain report JSON for this same sealed artefact -- "
        "narrows B3 to strict-valid pages (RFC005-ERRATA-006 E6); without it the report "
        "covers every attempted page and is tagged scope=all_attempted_pages",
    )
    parser.add_argument("--out", type=Path, required=True, help="output report JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw = args.sealed.read_bytes()
    artefact = sealed.artefact_from_json(raw.decode("utf-8"))
    policy = load_text_continuity_policy(args.policy or DEFAULT_POLICY_PATH)

    bundles = [p.document_bundle_id for p in artefact.pages if p.outcome in tc.DECODED_OUTCOMES]
    gold_rows = cg.load_gold_rows(args.cache_split, bundles)

    chain_report = None
    sealed_artefact_sha256 = None
    if args.chain_report is not None:
        chain_report = json.loads(args.chain_report.read_text(encoding="utf-8"))
        sealed_artefact_sha256 = hashlib.sha256(raw).hexdigest()

    report = build_text_continuity_report(
        artefact,
        gold_rows,
        policy=policy,
        chain_report=chain_report,
        sealed_artefact_sha256=sealed_artefact_sha256,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(
        json.dumps(
            {"verdict": report["verdict"], "accounting": report["accounting"],
             "sha256": f"sha256:{digest}"},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
