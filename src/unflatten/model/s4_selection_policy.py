"""RFC005 §40 checkpoint selection policy (DEC-041 A).

Like `s2_stage_gate_policy.py`, this policy carries no embedded self-hash --
its identity (`rfc005.s4-selection-policy/v1`) is a `domain_hash` over the
whole validated document, computed here, never stored inside it.

`gates` holds levels 1..7 of RFC005 §40.1's eight-level lexicographic
order; level 8 ("performance tie-break") is the separate `tie_break` field,
never a floored gate -- it only orders candidates that tie on every gate
above (see the schema's own `$comment` for the full level mapping this
module does not restate).

Floors are the OWNER's decision (DEC-035 §5). A gate's `is_placeholder`
flag marks a floor not yet ratified; `unflatten.model.select_checkpoint`
refuses to declare a `best` checkpoint while any gate is a placeholder,
unless run with `--report-only`.

Torch-free by construction: nothing here imports torch.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Design registry row `rfc005.s4-selection-policy/v1`. Owned here: no
#: other module computes an S4 selection policy's identity hash.
_POLICY_DOMAIN = "rfc005.s4-selection-policy/v1"

S4_SELECTION_POLICY_SCHEMA_VERSION = "1.0.0"
#: Fresh S4 (DEC-055 V5): the version a document declares when it carries
#: more than one gate per §40.1 level (list order decides inside a level).
MULTI_GATE_SCHEMA_VERSION = "1.1.0"

#: RFC005 §40.1 levels 1..7 (level 8 is the separate `tie_break` field).
_EXPECTED_GATE_LEVELS = frozenset(range(1, 8))


class S4SelectionPolicyInvalid(ValueError):
    """An S4 selection policy is malformed: the schema refuses it, or a
    semantic check the schema cannot express (distinct gate levels,
    distinct metric ids, the cohort's own `pages_sha256`) does not hold.

    M1-internal; not an RFC004/RFC005 registry code.
    """


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / "s4_selection_policy.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate_schema(doc: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        pointer = "/" + "/".join(str(p) for p in first.absolute_path)
        raise S4SelectionPolicyInvalid(
            f"s4 selection policy invalid at {pointer or '/'}: {first.message}"
        )


def pinned_pages_sha256(pinned_pages: list[list[Any]]) -> str:
    """`sha256:<hex>` over the pinned cohort's `(document_bundle_id,
    page_index)` pairs, sorted -- the same shape `s4_train.
    page_key_set_sha256` hashes for `s4_eval_pages.json`, so the two are
    directly comparable when checking disjointness."""
    import hashlib

    pairs = sorted((str(bundle), int(page)) for bundle, page in pinned_pages)
    canonical = json.dumps(pairs, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_s4_selection_policy(doc: dict[str, Any]) -> None:
    _validate_schema(doc)
    levels = {g["level"] for g in doc["gates"]}
    if levels != _EXPECTED_GATE_LEVELS:
        raise S4SelectionPolicyInvalid(
            f"s4 selection policy gate levels are {sorted(levels)}, expected exactly 1..7 distinct"
        )
    if (
        len(doc["gates"]) > len(_EXPECTED_GATE_LEVELS)
        and doc["s4_selection_policy_schema_version"] != MULTI_GATE_SCHEMA_VERSION
    ):
        raise S4SelectionPolicyInvalid(
            f"s4 selection policy has {len(doc['gates'])} gates: more than one gate per level "
            f"needs s4_selection_policy_schema_version {MULTI_GATE_SCHEMA_VERSION} (fresh S4, "
            "DEC-055 V5); a 1.0.0 document keeps exactly one gate per level"
        )
    metric_ids = [g["metric_id"] for g in doc["gates"]]
    if len(set(metric_ids)) != len(metric_ids):
        raise S4SelectionPolicyInvalid(
            f"s4 selection policy gates repeat a metric_id: {metric_ids} -- one gate per metric"
        )
    recomputed = pinned_pages_sha256(doc["cohort"]["pinned_pages"])
    if recomputed != doc["cohort"]["pages_sha256"]:
        raise S4SelectionPolicyInvalid(
            f"s4 selection policy cohort.pages_sha256 mismatch: recomputed {recomputed}, "
            f"policy claims {doc['cohort']['pages_sha256']!r} -- pinned_pages was edited "
            "after the cohort was sealed"
        )
    for gate in doc["gates"]:
        source = gate.get("floor_source")
        if source is None or source["kind"] != "reference":
            continue
        expected = max(0, source["reference_value_micro"] - source["margin_micro"])
        if gate["floor_micro"] != expected:
            raise S4SelectionPolicyInvalid(
                f"s4 selection policy gate {gate['metric_id']!r}: floor_micro "
                f"{gate['floor_micro']} does not match its own floor_source "
                f"(reference_value_micro {source['reference_value_micro']} - margin_micro "
                f"{source['margin_micro']}, clamped at 0, = {expected}) -- a floor derived "
                "from a reference checkpoint can never silently disagree with its own "
                "recorded provenance (RFC005-ERRATA-005 E3)"
            )


def s4_selection_policy_hash(policy: dict[str, Any]) -> str:
    """`selection_policy_hash` for `policy`: `domain_hash` over the whole
    validated document -- RFC005 §40.3's `best_under_selection_policy_hash`
    is `best_under_` this value."""
    validate_s4_selection_policy(policy)
    return domain_hash(_POLICY_DOMAIN, policy)


def load_s4_selection_policy(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise S4SelectionPolicyInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise S4SelectionPolicyInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_s4_selection_policy(payload)
    return payload


def has_placeholder_floor(policy: dict[str, Any]) -> bool:
    """`True` iff any gate's floor is still a DEC-035 §5 PROPOSED
    placeholder (owner has not ratified it). `select_checkpoint` reads
    this to refuse a declared `best` outside `--report-only`."""
    return any(g["is_placeholder"] for g in policy["gates"])
