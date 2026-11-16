"""RFC003 §43 coverage exit criteria, recomputed from the release.

The node-type minima are recomputed from `coverage/report.json` rather
than quoted, and the node-type list is derived from the `word-core-v0`
profile alphabet (`unflatten.model.events_vocab`) rather than typed out —
a node type added to the profile then shows up as an uncovered one
instead of silently escaping the criterion.

It is deliberately NOT derived from the canonical AST schema: the schema
describes the RFC001 *language*, so its `$defs` is a superset of the
profile's mandatory nodes (RFC001 §42.1). Deriving from it made §43
criteria 1–3 fail on `hyperlink`, which RFC001 §42.2 and RFC002 §0.1 put
in `word-visual-v1` and which the model path cannot even represent
(`model/grammar.py`: "hyperlink ... has no tokens at all"). The
criterion was not violated, it was wrongly evaluated.

Failing coverage is NOT an admission blocker. §9.3 says an unmet minimum
restricts the run class to `research | smoke | coverage-building`; it
does not make the data inadmissible. The two verdicts are therefore
reported separately.

Criteria 4, 5, 6 and 7 stay `unmeasured`: per-categorical-value and
numeric-bin counts need the `ModelTargetCoverageReport` §9.3 requires and
which does not exist, and the generation yield needs the batch generation
log, which a release does not carry.
"""

from __future__ import annotations

from typing import Any

from unflatten.dataset.admission.evidence import ReleaseEvidence
from unflatten.model.events_vocab import node_types

#: §43.1–3: minimum instances per V0 node type, per split group.
TRAIN_MINIMUM = 1000
VALIDATION_MINIMUM = 100
TEST_MINIMUM = 100

#: Splits whose counts feed the validation minimum (§43.2).
_VALIDATION_SPLITS: tuple[str, ...] = ("val_iid", "val_template_ood")

#: RFC003 §43 criteria this report does not compute, with the reason.
_UNMEASURED: dict[int, str] = {
    4: (
        "per-categorical-value train counts need the ModelTargetCoverageReport "
        "(RFC005 §9.3); the release coverage report counts property PRESENCE, "
        "not values"
    ),
    5: "numeric bin coverage needs the same ModelTargetCoverageReport",
    6: "no pairwise-combination coverage target artifact exists",
    7: "batch generation yield is not carried by a release",
}


def v0_node_types() -> tuple[str, ...]:
    """The `word-core-v0` node types (RFC001 §42.1), from the V0 alphabet.

    `events_vocab.node_types("word-core-v0")` is the machine-readable
    profile set: exactly the §42.1 mandatory list (`document, section,
    paragraph, run, tab, line_break, page_break, column_break, list,
    list_item, table, row, cell, inline_image`). A node type added to
    THIS profile shows up here; the `word-visual-v1` types do have event
    tokens (DEC-003 §1) but are not eligible under v0, so they stay
    correctly absent — the criterion is keyed on the profile, not on the
    alphabet, which now spans both.
    """
    return tuple(sorted(node_types("word-core-v0")))


def story_context_counts(ev: ReleaseEvidence) -> dict[str, dict[str, int]]:
    """Train-split node counts split by story context — a REPORT axis.

    DEC-003 [OPEN] J is answered by reading (b): §43 criteria 1–3 stay
    keyed on the node type alone, because a header paragraph IS a
    paragraph and the cell precedent already counts a distributionally
    different population that way (at least 33 % of counted `paragraph`
    instances are cell paragraphs). What reading (b) adds is visibility:
    the criterion can be met with ZERO header paragraphs, and without
    this axis nobody could see that from the report.

    The cost is stated plainly: this does NOT make criterion 1 harder,
    so a release with no header coverage still passes it. Turning the
    axis into a criterion is a separate decision with a separate
    threshold, and it needs the regenerated store to set one on
    evidence. Until then, "1000 paragraphs" must not be read as "1000
    body paragraphs".
    """
    by_split = ev.coverage.get("by_split") or {}
    return {
        context: dict(counts)
        for context, counts in sorted(
            ((by_split.get("train") or {}).get("node_types_by_story") or {}).items()
        )
    }


def _counts(ev: ReleaseEvidence, node_type: str, splits: tuple[str, ...]) -> int:
    by_split = ev.coverage.get("by_split") or {}
    if node_type == "document":
        # The release counter walks `sections` (split/release.py
        # `_node_type_counts`), so the root node is never in
        # `node_types`. One canonical tree per bundle (RFC003 ERRATA-001
        # E1), so the document count IS the root-node count.
        return sum(int((by_split.get(split) or {}).get("documents", 0)) for split in splits)
    return sum(
        int(((by_split.get(split) or {}).get("node_types") or {}).get(node_type, 0))
        for split in splits
    )


def node_type_minima(ev: ReleaseEvidence) -> list[dict[str, Any]]:
    """Per node type: measured train/validation/test counts vs the minima."""
    rows = []
    for node_type in v0_node_types():
        train = _counts(ev, node_type, ("train",))
        validation = _counts(ev, node_type, _VALIDATION_SPLITS)
        test = _counts(ev, node_type, ("test",))
        rows.append(
            {
                "node_type": node_type,
                "train": train,
                "validation": validation,
                "test": test,
                "meets_minima": (
                    train >= TRAIN_MINIMUM
                    and validation >= VALIDATION_MINIMUM
                    and test >= TEST_MINIMUM
                ),
            }
        )
    return rows


def _criterion(number: int, text: str, status: str, measured: Any) -> dict[str, Any]:
    return {"id": number, "criterion": text, "status": status, "measured": measured}


def coverage_report(
    ev: ReleaseEvidence,
    store_index: dict[str, dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> dict[str, Any]:
    """The §43 block: per-node-type minima plus the fifteen criteria."""
    minima = node_type_minima(ev)
    by_row = {row["row"]: row for row in conditions}
    human_gold = sum(
        1
        for record in store_index.values()
        if ((record.get("envelope") or {}).get("verification_tier")) == "human_gold"
    )
    by_split = ev.coverage.get("by_split") or {}
    origins = sorted(
        {
            (record.get("envelope") or {}).get("source_origin")
            for record in store_index.values()
        }
        - {None}
    )

    def status_of(row: int) -> str:
        return by_row.get(row, {}).get("status", "unmeasured")

    criteria = [
        _criterion(
            1,
            "every V0 node type has >= 1000 train instances",
            "pass" if all(m["train"] >= TRAIN_MINIMUM for m in minima) else "fail",
            {"below_minimum": [m["node_type"] for m in minima if m["train"] < TRAIN_MINIMUM]},
        ),
        _criterion(
            2,
            "every V0 node type has >= 100 validation instances",
            "pass"
            if all(m["validation"] >= VALIDATION_MINIMUM for m in minima)
            else "fail",
            {
                "below_minimum": [
                    m["node_type"] for m in minima if m["validation"] < VALIDATION_MINIMUM
                ]
            },
        ),
        _criterion(
            3,
            "every V0 node type has >= 100 test instances",
            "pass" if all(m["test"] >= TEST_MINIMUM for m in minima) else "fail",
            {"below_minimum": [m["node_type"] for m in minima if m["test"] < TEST_MINIMUM]},
        ),
    ]
    for number, reason in _UNMEASURED.items():
        criteria.append(_criterion(number, "see RFC003 §43", "unmeasured", {"reason": reason}))
    criteria.extend(
        [
            _criterion(
                8,
                "100 % of train_ready TaskViews pass all hard gates",
                status_of(5),
                by_row.get(5, {}).get("measured"),
            ),
            _criterion(
                9,
                "100 % of full-AST train_ready TaskViews conserve text events",
                _text_conservation_status(ev),
                _text_conservation_measured(ev),
            ),
            _criterion(
                10,
                "0 required unknown properties in full_ast_exact_v0",
                status_of(6),
                by_row.get(6, {}).get("measured"),
            ),
            _criterion(
                11,
                "0 cross-split lineage collisions",
                status_of(4),
                by_row.get(4, {}).get("measured"),
            ),
            _criterion(
                12,
                "0 exact input/AST duplicates between train and eval",
                status_of(4),
                {
                    "exact_ast_duplicates_across_splits": by_row.get(4, {})
                    .get("measured", {})
                    .get("exact_ast_duplicates_across_splits")
                },
            ),
            _criterion(
                13,
                "0 policy-defined near-duplicate edges between train and eval",
                status_of(4),
                {
                    "derived_from_release": True,
                    "near_duplicate_lineages_quarantined": sum(
                        1
                        for row in ev.quarantine
                        if "NEAR_DUPLICATE_COLLISION" in (row.get("reason_codes") or [])
                    ),
                    "not_recomputed": "near_duplicate_fingerprint_graph",
                },
            ),
            _criterion(
                14,
                "a separate real human-gold test exists",
                "pass" if human_gold else "fail",
                {"human_gold_documents": human_gold},
            ),
            _criterion(
                15,
                "coverage is reported per split, source origin and domain",
                # Split and source origin ARE reported. The third
                # dimension has no field to report on: neither the
                # document bundle envelope nor the store record carries a
                # domain (grep `domain` over dataset/schemas returns only
                # hash-domain separation). Not a failure of the corpus —
                # an unmeasurable criterion.
                "unmeasured" if by_split and origins else "fail",
                {
                    "splits": sorted(by_split),
                    "documents_per_split": {
                        name: (by_split.get(name) or {}).get("documents", 0)
                        for name in sorted(by_split)
                    },
                    "empty_splits": sorted(
                        name
                        for name in by_split
                        if not (by_split.get(name) or {}).get("documents")
                    ),
                    "source_origins": origins,
                    "domain": None,
                    "domain_reason": (
                        "no domain dimension is recorded on the document bundle "
                        "envelope or the store record"
                    ),
                },
            ),
        ]
    )
    criteria.sort(key=lambda c: c["id"])
    failing = [c["id"] for c in criteria if c["status"] == "fail"]
    outcome = "fail" if failing else (
        "unmeasured" if any(c["status"] == "unmeasured" for c in criteria) else "pass"
    )
    return {
        "node_type_minima": minima,
        # DEC-003 [OPEN] J: reported, never a criterion — see
        # `story_context_counts`.
        "train_node_types_by_story": story_context_counts(ev),
        "criteria": criteria,
        "failing_criteria": failing,
        "outcome": outcome,
        # §9.3: an unmet minimum restricts the run class; it is not an
        # admission blocker, and the count of documents never substitutes
        # for these criteria.
        "permitted_run_classes": (
            ["research", "smoke", "coverage-building"]
            if outcome != "pass"
            else ["research", "smoke", "coverage-building", "full_product_candidate"]
        ),
    }


def _text_conservation_status(ev: ReleaseEvidence) -> str:
    offenders, scope = _text_conservation_counts(ev)
    if scope == 0:
        return "unmeasured"
    return "fail" if offenders else "pass"


def _text_conservation_measured(ev: ReleaseEvidence) -> dict[str, Any]:
    offenders, scope = _text_conservation_counts(ev)
    return {
        "full_ast_train_ready_documents": scope,
        "text_conservation_gate_not_pass": offenders,
    }


def _text_conservation_counts(ev: ReleaseEvidence) -> tuple[int, int]:
    offenders = scope = 0
    for row in ev.gate_rows:
        view = (row.get("task_eligibility") or {}).get("full_ast_exact_v0") or {}
        if view.get("eligibility") != "train_ready":
            continue
        scope += 1
        statuses = {c.get("gate_id"): c.get("status") for c in row.get("gate_report", [])}
        if statuses.get("text_conservation") != "pass":
            offenders += 1
    return offenders, scope
