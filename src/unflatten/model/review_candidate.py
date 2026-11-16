"""RFC004 §33.7 Review Candidate (design §2.5, T7b of
`spec-ast-builder-t5-promotion` P4).

`build_review_candidate` walks `ast_builder._build_raw` -- the same tree,
derivations, binder and required-property gate `build_strict` uses -- and
adds one thing `build_strict` never does: for every
`UNRESOLVED_REQUIRED_PROPERTY` issue whose `default_policy.lookup(...)` says
`may_impute`, it writes the §43 default at that pointer and records the
substitution. Everything else stays `unresolved`, including `/resource_ref`
and `/note_id` abstentions -- `default_policy` has no record for either
pointer at all (R8), so `lookup` raises `KeyError` for them and they fall
through to `unresolved` exactly like any other non-imputable gap.

The candidate is **never canonicalized**: no run merging, no id
renumbering, no `ast_hash`, and it never enters `compile_docx` (§33.7
DECIDED). `ast_builder.py` must not import `default_policy` (design §2.5 D1,
R1) -- this module is the only place a default value can reach an AST node,
and it only ever writes one after `_build_raw` has already finished.

`builder_report_hash` is the SAME raw build's `report_hash` (plan U1): a
strict and a review build of the same document read equal reports because
both call `_build_raw` before either the strict gate or this module's
post-pass runs, and the raw report never sees an imputation.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.assembly.assemble import AssembledDocumentHypothesis
from unflatten.model import ast_builder as ab
from unflatten.model import default_policy as dp
from unflatten.model import resource_binder as rb

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "review_candidate"

REVIEW_CANDIDATE_SCHEMA_VERSION = "0.1.0"


class ReviewCandidateInvalid(ValueError):
    """A review candidate wrapper fails its closed schema."""


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_review_candidate(doc: Mapping[str, Any]) -> None:
    """Validate `doc` against `review_candidate.schema.json`."""
    errors = list(_validator().iter_errors(doc))
    if errors:
        raise ReviewCandidateInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


@dataclass(frozen=True, slots=True)
class ReviewCandidateResult:
    #: The full wrapper document, already validated against
    #: `review_candidate.schema.json` (design §2.5).
    document: Mapping[str, Any]


def build_review_candidate(
    document: AssembledDocumentHypothesis,
    *,
    profile: str,
    ast_version: str,
    document_id: str,
    recovered: Mapping[str, rb.RecoveredResource] | None = None,
) -> ReviewCandidateResult:
    """`recovered` maps an image's assembled id to its recovered bytes, same
    as `ast_builder.build_strict` -- an image without one stays
    `RESOURCE_SLOT_UNRESOLVED` in `unresolved`, never imputed (R8)."""
    raw = ab._build_raw(
        document,
        profile=profile,
        ast_version=ast_version,
        document_id=document_id,
        recovered=recovered,
    )
    candidate = copy.deepcopy(raw.tree)
    by_id = _nodes_by_id(candidate)
    imputations: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for issue in raw.issues:
        record = None
        if issue.code == ab.UNRESOLVED_REQUIRED_PROPERTY and issue.node_id is not None:
            try:
                record = dp.lookup(issue.node_type, issue.pointer)
            except KeyError:
                record = None
        if record is not None and record.may_impute:
            ab._write(by_id[issue.node_id], issue.pointer, record.default_value)
            imputations.append({
                "node_id": issue.node_id,
                "pointer": issue.pointer,
                "default_value": record.default_value,
                "default_source": record.default_source,
                "reason_code": "imputed_profile_default",
            })
        else:
            unresolved.append({
                "node_id": issue.node_id,
                "pointer": issue.pointer,
                "code": issue.code,
            })

    doc: dict[str, Any] = {
        "review_candidate_schema_version": REVIEW_CANDIDATE_SCHEMA_VERSION,
        "profile": profile,
        "ast_version": ast_version,
        "document_id": document_id,
        "authoritative": False,
        "candidate": candidate,
        "imputations": _sorted_rows(imputations),
        "unresolved": _sorted_rows(unresolved),
        "builder_report_hash": raw.report["report_hash"],
    }
    validate_review_candidate(doc)
    return ReviewCandidateResult(document=doc)


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same order rule as `builder_report._sorted_rows`: deterministic
    regardless of the order `raw.issues` collected in."""
    return sorted(rows, key=lambda row: (row.get("node_id") or "", row.get("pointer") or ""))


def _nodes_by_id(tree: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "id" in value and "type" in value:
                out[value["id"]] = value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(tree)
    return out
