"""Closed, self-hashing S4.5 RL artefacts: reward policy, rollout policy and
dataset manifest (RFC005-ERRATA-010 E3/E7, PROPOSED; dormant).

Every artefact is a plain dict validated against its closed JSON Schema in
``schemas/`` and carries its own hash field, derived with ``domain_hash`` over
the document minus that field, under a domain owned by ``rl/ids.py``. The CBOR
profile has no floats, so every quantity is an integer micro-unit and a float
anywhere in the input is refused by name rather than left to the encoder.

Thresholds are ``None`` (= ``insufficient_evidence``); there is no numeric
default for them, nor for ``group_size`` / ``rollout_budget``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any, Final

import jsonschema.validators

from unflatten.model.hashing import domain_hash
from unflatten.model.rl.ids import (
    DATASET_MANIFEST_DOMAIN,
    REWARD_POLICY_DOMAIN,
    ROLLOUT_POLICY_DOMAIN,
)

SCHEMA_VERSION: Final = "1.0.0"
_SCHEMA_DIR: Final = Path(__file__).parent / "schemas"

REWARD_COMPONENTS: Final = (
    "visible_text_control_events",
    "strict_ast_tree_semantics",
    "typed_layout_typography",
    "grid_span_cell_ownership",
    "alignment_fragment_continuity",
    "render_geometry_perceptual",
)
HARD_GATES: Final = (
    "missing_or_invented_text",
    "invalid_ast",
    "resource_fragment_integrity",
    "severe_structural_error",
    # appended last so the first four keep their positions and their precedence
    "rfc002_render",
)
#: Candidate<->gold alignment the scorer uses (RFC005-ERRATA-010 D1); hashed.
ALIGNMENT_POLICY: Final = "preorder-edit-script.v0"
#: Closed execution profile shared by rollout and update (D2/D9).
KV_CACHE_MODE: Final = "bf16_dynamic"
QUANTIZATION: Final = "none"
DEEPSTACK_MODES: Final = ("from_checkpoint",)
EXECUTION_PROFILE_KEYS: Final = (
    "kv_cache_mode",
    "quantization",
    "prefill_chunk_tokens",
    "deepstack",
)
#: LibreOffice open probe: never part of a hashed evaluation (D6).
RENDER_PROBE_MODES: Final = ("off", "status_only")
#: Boolean flags of ``eval.builder_chain.run.process_model_page``.
CHAIN_FLAGS: Final = (
    "headless_defaults",
    "table_geometry",
    "recovered_image_materialization",
    "story_dedup",
    "language_detect",
    "page_number_fields",
    "autolink",
    "page_geometry_from_raster",
    "runaway_row_repair",
    "leader_fill",
)
#: Flags `process_model_page` itself refuses without `headless_defaults`.
FLAGS_REQUIRING_HEADLESS: Final = (
    "table_geometry",
    "story_dedup",
    "language_detect",
    "runaway_row_repair",
)
THRESHOLD_KEYS: Final = (
    "min_group_reward_variance",
    "min_reward_quality_correlation",
    "min_independent_lineages",
    "max_zero_advantage_rate",
)
#: Per-arm floors of the severe-structural-error hard gate (integer micro or null).
SEVERE_KEYS: Final = ("tree_f1_floor_micro", "grid_f1_floor_micro")
SAMPLEABLE_HEAD_KINDS: Final = ("boolean", "enum", "cardinality", "palette")
#: Integer heads are argmax, never sampled (owner decision).
INTEGER_HEAD_KINDS: Final = (
    "integer",
    "integer_regression",
    "bins_plus_residual",
    "direct_regression",
)
SEED_SCHEME: Final = "H(run_seed, doc_id, rollout_index)"
FORBIDDEN_SOURCES: Final = (
    "calibration_fit",
    "calibration_check",
    "val_real_p5_dev",
    "test_real_gold",
    "P6_REAL_GOLD",
    "locked_test",
    "s5_control_sample",
)
STATUSES: Final = ("PROPOSED", "RATIFIED")

REWARD_HASH_FIELD: Final = "reward_policy_hash"
ROLLOUT_HASH_FIELD: Final = "rollout_policy_hash"
MANIFEST_HASH_FIELD: Final = "dataset_manifest_hash"


class RlPolicyInvalid(ValueError):
    """An S4.5 RL artefact is malformed, forbidden, or fails its hash."""


@cache
def _validator(name: str) -> Any:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def _reject_floats(value: Any, where: str = "$") -> None:
    if isinstance(value, float):
        raise RlPolicyInvalid(f"float at {where}: S4.5 artefacts carry integer micro units only")
    if isinstance(value, Mapping):
        for k, v in value.items():
            _reject_floats(v, f"{where}.{k}")
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _reject_floats(v, f"{where}[{i}]")


def _schema_check(name: str, doc: Mapping[str, Any]) -> None:
    _reject_floats(doc)
    errors = sorted(_validator(name).iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        pointer = "/" + "/".join(str(p) for p in errors[0].absolute_path)
        raise RlPolicyInvalid(f"{name} invalid at {pointer}: {errors[0].message}")


def _hash_of(domain: str, field: str, doc: Mapping[str, Any]) -> str:
    return domain_hash(domain, {k: v for k, v in doc.items() if k != field})


def _seal(name: str, domain: str, field: str, body: dict[str, Any]) -> dict[str, Any]:
    _reject_floats(body)
    doc = {**body, field: _hash_of(domain, field, body)}
    _schema_check(name, doc)
    return doc


def _verify(name: str, domain: str, field: str, doc: Mapping[str, Any]) -> None:
    _schema_check(name, doc)
    expected = _hash_of(domain, field, doc)
    if doc[field] != expected:
        raise RlPolicyInvalid(f"{name} {field} {doc[field]!r} != recomputed {expected!r}")


def _require_status(status: str) -> None:
    if status not in STATUSES:
        raise RlPolicyInvalid(f"status {status!r} not in {STATUSES}")


# --- execution profile -------------------------------------------------------


def build_execution_profile(
    *, prefill_chunk_tokens: int | None = None, deepstack: str = "from_checkpoint"
) -> dict[str, Any]:
    """The one sampler/update profile (D9): dynamic bf16 KV, no quantisation."""
    return validate_execution_profile(
        {
            "kv_cache_mode": KV_CACHE_MODE,
            "quantization": QUANTIZATION,
            "prefill_chunk_tokens": prefill_chunk_tokens,
            "deepstack": deepstack,
        }
    )


def validate_execution_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse any profile other than the closed one; returns a plain dict copy."""
    if set(profile) != set(EXECUTION_PROFILE_KEYS):
        raise RlPolicyInvalid(
            f"execution_profile keys must be exactly {EXECUTION_PROFILE_KEYS}, "
            f"got {sorted(profile)}"
        )
    if profile["kv_cache_mode"] != KV_CACHE_MODE:
        raise RlPolicyInvalid(
            f"kv_cache_mode must be {KV_CACHE_MODE!r}: {profile['kv_cache_mode']!r}"
        )
    if profile["quantization"] != QUANTIZATION:
        raise RlPolicyInvalid(f"quantization must be {QUANTIZATION!r}: {profile['quantization']!r}")
    chunk = profile["prefill_chunk_tokens"]
    if chunk is not None and (isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1):
        raise RlPolicyInvalid(f"prefill_chunk_tokens must be an int >= 1 or null, got {chunk!r}")
    if profile["deepstack"] not in DEEPSTACK_MODES:
        raise RlPolicyInvalid(
            f"deepstack must be one of {DEEPSTACK_MODES}: {profile['deepstack']!r}"
        )
    return dict(profile)


# --- reward policy -----------------------------------------------------------


def build_reward_policy(
    *,
    component_weights_micro: Mapping[str, int],
    chain_policy_flags: Mapping[str, bool],
    severe_structural_error: Mapping[str, int | None],
    hard_gates: Sequence[str] = HARD_GATES,
    pixel_component: str = "off",
    status: str = "PROPOSED",
    text_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """``text_gate`` (optional) replaces the exact-equality text gate by DEC-030 CER <=
    ``max_ppm`` plus an exact match of the named control events; absent, the key is not
    in the sealed body (old hashes stay valid) and the gate is exact."""
    if pixel_component != "off":
        raise RlPolicyInvalid(f"pixel_component must be 'off', got {pixel_component!r}")
    _require_status(status)
    _reject_floats(component_weights_micro, "component_weights_micro")
    severe = dict(severe_structural_error)
    if set(severe) != set(SEVERE_KEYS):
        raise RlPolicyInvalid(
            f"severe_structural_error keys must be exactly {SEVERE_KEYS}, got {sorted(severe)}"
        )
    _reject_floats(severe, "severe_structural_error")
    _check_ratified_floors(status, severe)
    flags = dict(chain_policy_flags)
    if set(flags) != set(CHAIN_FLAGS):
        raise RlPolicyInvalid(
            f"chain_policy_flags keys must be exactly {CHAIN_FLAGS}, got {sorted(flags)}"
        )
    if not flags["headless_defaults"]:
        needing = [f for f in FLAGS_REQUIRING_HEADLESS if flags[f]]
        if needing:
            raise RlPolicyInvalid(f"chain flags {needing} require headless_defaults")
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "component_weights_micro": dict(component_weights_micro),
        "hard_gates": list(hard_gates),
        "severe_structural_error": severe,
        "candidate_repair": "none_beyond_pinned_chain_policies",
        "chain_policy_flags": flags,
        "alignment_policy": ALIGNMENT_POLICY,
        "pixel_component": pixel_component,
        "thresholds": {k: None for k in THRESHOLD_KEYS},
    }
    if text_gate is not None:
        body["text_gate"] = _checked_text_gate(text_gate)
    return _seal("rl_reward_policy", REWARD_POLICY_DOMAIN, REWARD_HASH_FIELD, body)


def _checked_text_gate(text_gate: Mapping[str, Any]) -> dict[str, Any]:
    from unflatten.eval.s2_gate.slices import CONTROL_EVENT_TOKENS

    _reject_floats(text_gate, "text_gate")
    block = {**text_gate, "control_events_exact": list(text_gate.get("control_events_exact", ()))}
    unknown = sorted(set(block["control_events_exact"]) - set(CONTROL_EVENT_TOKENS))
    if unknown:
        raise RlPolicyInvalid(f"text_gate.control_events_exact names unknown events {unknown}")
    total = list(block.get("control_events_total", ()))
    unknown = sorted(set(total) - set(CONTROL_EVENT_TOKENS))
    if unknown:
        raise RlPolicyInvalid(f"text_gate.control_events_total names unknown events {unknown}")
    overlap = sorted(set(total) & set(block["control_events_exact"]))
    if overlap:
        raise RlPolicyInvalid(
            f"text_gate.control_events_total must be disjoint from control_events_exact, "
            f"both name {overlap}"
        )
    return block


def _check_ratified_floors(status: str, severe: Mapping[str, int | None]) -> None:
    """A RATIFIED policy may not carry an unfrozen (null) severe floor."""
    nulls = sorted(k for k, v in severe.items() if v is None)
    if status == "RATIFIED" and nulls:
        raise RlPolicyInvalid(f"RATIFIED reward policy carries null severe floors {nulls}")


def verify_reward_policy(doc: Mapping[str, Any]) -> None:
    _verify("rl_reward_policy", REWARD_POLICY_DOMAIN, REWARD_HASH_FIELD, doc)
    _check_ratified_floors(doc["status"], doc["severe_structural_error"])


# --- rollout policy ----------------------------------------------------------


def build_rollout_policy(
    *,
    temperature_micro: int,
    top_p_micro: int,
    group_size: int | None,
    rollout_budget: int | None,
    max_steps: int | None,
    loop_guard_arm: str | None,
    execution_profile: Mapping[str, Any],
    typed_head_sampling: Iterable[str] = (),
    seed_scheme: str | None = SEED_SCHEME,
    renormalise: str = "grammar_legal",
    loop_guards_in_policy: bool = True,
    render_probe: str = "off",
    status: str = "PROPOSED",
) -> dict[str, Any]:
    if group_size is None or rollout_budget is None or max_steps is None:
        raise RlPolicyInvalid(
            "group_size, rollout_budget and max_steps must be supplied (no default)"
        )
    if not isinstance(loop_guard_arm, str) or not loop_guard_arm:
        raise RlPolicyInvalid(f"loop_guard_arm must be a non-empty string, got {loop_guard_arm!r}")
    if render_probe not in RENDER_PROBE_MODES:
        raise RlPolicyInvalid(f"render_probe {render_probe!r} not in {RENDER_PROBE_MODES}")
    profile = validate_execution_profile(execution_profile)
    _require_status(status)
    if seed_scheme != SEED_SCHEME:
        raise RlPolicyInvalid(f"seed_scheme must be {SEED_SCHEME!r}, got {seed_scheme!r}")
    if renormalise != "grammar_legal":
        raise RlPolicyInvalid(f"renormalise must be 'grammar_legal', got {renormalise!r}")
    if temperature_micro == 0:
        raise RlPolicyInvalid("temperature 0 is greedy decoding, not a rollout")
    heads = list(typed_head_sampling)
    integer = [h for h in heads if h in INTEGER_HEAD_KINDS]
    if integer:
        raise RlPolicyInvalid(f"integer regression heads are argmax, never sampled: {integer}")
    return _seal(
        "rl_rollout_policy",
        ROLLOUT_POLICY_DOMAIN,
        ROLLOUT_HASH_FIELD,
        {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "temperature_micro": temperature_micro,
            "top_p_micro": top_p_micro,
            "seed_scheme": seed_scheme,
            "renormalise": renormalise,
            "typed_head_sampling": heads,
            "integer_regression_heads": "argmax",
            "loop_guards_in_policy": loop_guards_in_policy,
            "loop_guard_rewind_recorded_in_trajectory": True,
            "group_size": group_size,
            "rollout_budget": rollout_budget,
            "max_steps": max_steps,
            "loop_guard_arm": loop_guard_arm,
            "execution_profile": profile,
            "render_probe": render_probe,
        },
    )


def verify_rollout_policy(doc: Mapping[str, Any]) -> None:
    _verify("rl_rollout_policy", ROLLOUT_POLICY_DOMAIN, ROLLOUT_HASH_FIELD, doc)


# --- dataset manifest --------------------------------------------------------


def build_dataset_manifest(
    *,
    split: str,
    sources: Sequence[str],
    rights_privacy_refs: Sequence[str],
    lineage_ids: Sequence[str],
    selection_hash: str,
    exclude_lineage_ids: Iterable[str] = (),
) -> dict[str, Any]:
    if split != "train":
        raise RlPolicyInvalid(f"S4.5 trains on RFC003 split 'train' only, got {split!r}")
    forbidden = [s for s in sources if s in FORBIDDEN_SOURCES]
    if forbidden:
        raise RlPolicyInvalid(f"forbidden source(s) {forbidden} may not enter S4.5 rollouts")
    exclude = sorted(set(exclude_lineage_ids))
    if len(set(lineage_ids)) != len(lineage_ids):
        raise RlPolicyInvalid("lineage_ids contains duplicates")
    overlap = sorted(set(lineage_ids) & set(exclude))
    if overlap:
        raise RlPolicyInvalid(f"lineage_ids overlap calibration/dev/control lineages: {overlap}")
    return _seal(
        "rl_dataset_manifest",
        DATASET_MANIFEST_DOMAIN,
        MANIFEST_HASH_FIELD,
        {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "sources": list(sources),
            "forbidden_sources": list(FORBIDDEN_SOURCES),
            "rights_privacy_refs": list(rights_privacy_refs),
            "lineage_ids": list(lineage_ids),
            "exclude_lineage_ids": exclude,
            "selection_hash": selection_hash,
        },
    )


def verify_dataset_manifest(doc: Mapping[str, Any]) -> None:
    _verify("rl_dataset_manifest", DATASET_MANIFEST_DOMAIN, MANIFEST_HASH_FIELD, doc)
    overlap = sorted(set(doc["lineage_ids"]) & set(doc["exclude_lineage_ids"]))
    if overlap:
        raise RlPolicyInvalid(f"lineage_ids overlap exclude_lineage_ids: {overlap}")
