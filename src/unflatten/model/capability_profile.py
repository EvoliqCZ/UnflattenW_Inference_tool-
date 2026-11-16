"""M1 explicit model capability profile (design §4.4, §4.8).

Exact-text APIs never fall back to `targets.DEFAULT_PROFILE`
(`src/unflatten/model/targets.py:52`, currently `word-visual-v1`). This
module loads, validates and hashes model capability profile artifacts,
and exposes each one's declared capabilities and exclusion sets in a
typed, immutable form.

**Five ratified profiles today**, in `KNOWN_CAPABILITY_PROFILES`:
`PRIMARY_S1_CAPABILITY_PROFILE` (the one explicit S1 profile: a strict
subset of `word-visual-v1.1`, `typed_properties="not_applicable"`);
`WIDENED_V1_CAPABILITY_PROFILE` (excludes nothing, still
`typed_properties="not_applicable"`, not wired into any consumer);
`TYPED_FROZEN_V1_CAPABILITY_PROFILE` (primary S1's exact capability set,
`typed_properties="present_frozen"` -- the exact-text cache carries typed
targets, heads exist but are frozen until S3); and
`TYPED_WIDENED_V1_CAPABILITY_PROFILE` (DEC-024: excludes nothing,
`typed_properties="present_frozen"`, multi-page scope, for the large
build and S2); and `STANDALONE_V1_CAPABILITY_PROFILE` (DEC-033,
RFC004-ERRATA-002 E1: `TYPED_WIDENED_V1`'s capabilities plus the new
`excluded_structural_tokens` field -- schema 1.0.0 -> 1.1.0 -- naming all
28 RESUME/SUSPEND tokens, so a page under this profile is always a
standalone page). Every profile beyond primary S1 is additive: none of the
others' payloads are ever edited, since `PRIMARY_S1_CAPABILITY_PROFILE
.profile_hash` guards the S1 checkpoint identity and
`TYPED_FROZEN_V1_CAPABILITY_PROFILE`'s guards existing artefacts.

Two resolution surfaces, both fail-closed: `resolve_capability_profile`
(a caller-built registry, keyed by `profile_id`) and
`resolve_capability_profile_by_hash` / `resolve_capability_profile_by_id`
(against the fixed `KNOWN_CAPABILITY_PROFILES` registry -- the lookup an
artefact-bound consumer uses, since an `ExactTextCacheIdentity` or
component lock names the profile it was built under by hash, not by an
import-time global). An unknown, absent or ambiguous id or hash raises
rather than silently falling back to any default (design §4.4, task hard
constraint 1).

`model_capability_profile` is a **provisional** RFC004 §39.2 component-lock
kind pending ratification request RR-4 (design §4.8): this module may
require the artifact in validation, but must not claim that §39.2 already
mandates it.

This module authorizes no gradient update: it declares only dataclasses
and pure functions, no `nn.Module`, no weights, no optimizer.

Out of scope (explicitly left to other tasks, design §4.4/§4.5): rewiring
`composite_grammar.CompositeGrammarController`, which currently takes
`capability_excluded_local_ids` as a constructor argument and performs its
own fail-closed identity check in `fresh`. This module only derives that
exclusion set from a validated profile; it does not construct or modify a
controller.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import jsonschema.protocols
import jsonschema.validators

from unflatten.model import events_vocab, node_lifecycle
from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_PROFILE_DOMAIN = "rfc004.model-capability-profile/v1"

#: Bumped 1.0.0 -> 1.1.0 for the optional `excluded_structural_tokens` field
#: (DEC-033, RFC004-ERRATA-002 E1). The four pre-existing payloads below keep
#: their literal `"1.0.0"` version string rather than this constant, so their
#: `profile_hash` does not move: only a NEW payload (the standalone-page
#: profile) opts into 1.1.0.
MODEL_CAPABILITY_PROFILE_SCHEMA_VERSION = "1.1.0"

#: `typed_properties` closed domain (R3, `ZOSTAVA-typovany-head-modul.md`):
#: this field gated whether a profile admits typed properties
#: (`run_binding._verify_typed_property_profile`) via a denylist of the one
#: literal `"not_applicable"` against an otherwise-open string, so a fourth
#: profile with a typo (`"not-applicable"`, `"unset"`) would have passed a
#: gate meant to say "this profile admits typed properties". Closed here and
#: in the schema (`typed_properties.enum`) to the two values any profile in
#: this module legitimately declares today.
TYPED_PROPERTIES_VALUES = frozenset({"not_applicable", "present_frozen"})

#: The one explicit S1 capability profile id. Never used as a silent
#: fallback -- callers must always pass it (or another explicit id)
#: through `resolve_capability_profile`.
PRIMARY_S1_PROFILE_ID = "dense-v0-s1-exact-text-v1"


class ModelCapabilityProfileInvalid(ValueError):
    """A capability profile payload fails its closed schema or content rules
    (design §4.4; §16 criterion pinning packet R4-3's capability set).

    M1-internal (§9.3); not an RFC004 registry code.
    """


class CapabilityProfileUnresolved(ValueError):
    """A `profile_id` could not be resolved to exactly one capability profile.

    Raised for an unknown id, an absent/empty id, or when the registry
    itself could not be built unambiguously (see
    `CapabilityProfileAmbiguous`). Design §4.4: "one explicit S1 capability
    profile, never DEFAULT_PROFILE" -- this exception is the only way
    resolution fails; there is no fallback path.
    """


class CapabilityProfileAmbiguous(ValueError):
    """Two payloads share one `profile_id` with differing content.

    Raised while building a registry, before any resolution is attempted,
    so an ambiguous id can never be silently resolved by "last one wins".
    """


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """Concrete, validated, immutable capability profile (design §4.4)."""

    profile_id: str
    ast_profile: str
    scope: str
    allowed_node_types: frozenset[str]
    allowed_leaf_controls: frozenset[str]
    capability_excluded: frozenset[str]
    exact_text: bool
    typed_properties: str
    expert_adapter_mode: Literal["dense", "sparse_moe"]
    sparse_moe: str
    visual_compressor: str
    speculators: str
    profile_hash: str
    #: This profile's OWN `model_capability_profile_schema_version`, read
    #: from its payload -- never the module-level
    #: `MODEL_CAPABILITY_PROFILE_SCHEMA_VERSION` constant, which moves
    #: forward for new profiles while old payloads keep an earlier literal
    #: version. A consumer that stamped a lock's `model_capability_profile`
    #: component with the constant instead of this field would silently
    #: change that lock's hash whenever the constant moved, even for a
    #: profile whose own payload never changed.
    schema_version: str
    #: Structural local ids (`events_vocab.TOKEN_IDS` values) excluded by
    #: `capability_excluded`, derived here for the composite controller
    #: (design §3's `CompositeGrammarController.capability_excluded_local_ids`)
    #: without this module constructing or modifying that controller.
    excluded_structural_local_ids: frozenset[int]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


@cache
def _validator(name: str) -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def validate_model_capability_profile(payload: Mapping[str, Any]) -> None:
    """Validate `payload` against `model_capability_profile.schema.json`."""
    errors = list(_validator("model_capability_profile").iter_errors(payload))
    if errors:
        raise ModelCapabilityProfileInvalid(
            f"schema validation failed: {errors[0].message} at {list(errors[0].absolute_path)}"
        )


# ---------------------------------------------------------------------------
# Hashing (design §4.4: "Hash domain: rfc004.model-capability-profile/v1")
# ---------------------------------------------------------------------------


def capability_profile_hash(payload: Mapping[str, Any]) -> str:
    return domain_hash(_PROFILE_DOMAIN, payload)


# ---------------------------------------------------------------------------
# Structural local-id exclusion derivation
# ---------------------------------------------------------------------------


def _structural_local_ids_for(node_type: str) -> frozenset[int]:
    """Every structural token local id that names `node_type`.

    A container contributes its OPEN/CLOSE tokens, plus RESUME/SUSPEND if
    it is a resumable container type (`text_frame` is not, per
    ERRATA-004). A leaf control contributes its single leaf token.
    """
    if node_type in events_vocab.CONTAINER_TYPES:
        ids = {
            events_vocab.TOKEN_IDS[events_vocab.OPEN_TOKEN[node_type]],
            events_vocab.TOKEN_IDS[events_vocab.CLOSE_TOKEN[node_type]],
        }
        if node_type in events_vocab.RESUMABLE_CONTAINER_TYPES:
            ids.add(events_vocab.TOKEN_IDS[events_vocab.RESUME_TOKEN[node_type]])
            ids.add(events_vocab.TOKEN_IDS[events_vocab.SUSPEND_TOKEN[node_type]])
        return frozenset(ids)
    if node_type in events_vocab.LEAF_TYPES:
        return frozenset({events_vocab.TOKEN_IDS[events_vocab.LEAF_TOKEN[node_type]]})
    raise ModelCapabilityProfileInvalid(
        f"{node_type!r} is neither a container nor a leaf node type"
    )


def _excluded_structural_local_ids(capability_excluded: frozenset[str]) -> frozenset[int]:
    ids: set[int] = set()
    for node_type in capability_excluded:
        ids |= _structural_local_ids_for(node_type)
    return frozenset(ids)


def _validate_excluded_structural_tokens(names: frozenset[str]) -> None:
    """`excluded_structural_tokens` (schema 1.1.0) names RESUME/SUSPEND
    tokens directly -- it is not a second way to exclude a node type.

    A name outside `node_lifecycle.RESUME_OR_SUSPEND_TOKEN_NAMES` (e.g.
    `PARAGRAPH_OPEN`) must go through `capability_excluded` instead, so this
    refuses it rather than silently folding it in (design §6).
    """
    invalid = names - node_lifecycle.RESUME_OR_SUSPEND_TOKEN_NAMES
    if invalid:
        raise ModelCapabilityProfileInvalid(
            f"excluded_structural_tokens contains names outside "
            f"RESUME_OR_SUSPEND_TOKEN_NAMES: {sorted(invalid)}"
        )


# ---------------------------------------------------------------------------
# Building a validated CapabilityProfile
# ---------------------------------------------------------------------------


def build_capability_profile(payload: Mapping[str, Any]) -> CapabilityProfile:
    """Validate `payload` end to end and build the typed, immutable profile.

    Beyond the closed JSON schema (`validate_model_capability_profile`),
    this enforces the content rule the schema cannot express: under
    `payload["ast_profile"]`, `allowed_node_types`, `allowed_leaf_controls`
    and `capability_excluded` must exactly and disjointly partition every
    eligible node type (`events_vocab.node_types(ast_profile)`). A missing
    node type, an overlap, or a container/leaf name in the wrong set is
    rejected -- there is no default disposition for an unclassified
    capability.
    """
    validate_model_capability_profile(payload)

    ast_profile = payload["ast_profile"]
    try:
        eligible = events_vocab.node_types(ast_profile)
    except ValueError as exc:
        raise ModelCapabilityProfileInvalid(str(exc)) from exc

    typed_properties = payload["typed_properties"]
    if typed_properties not in TYPED_PROPERTIES_VALUES:
        raise ModelCapabilityProfileInvalid(
            f"typed_properties must be one of {sorted(TYPED_PROPERTIES_VALUES)}, "
            f"got {typed_properties!r}"
        )

    allowed_node_types = frozenset(payload["allowed_node_types"])
    allowed_leaf_controls = frozenset(payload["allowed_leaf_controls"])
    capability_excluded = frozenset(payload["capability_excluded"])
    excluded_structural_tokens = frozenset(payload.get("excluded_structural_tokens", ()))
    _validate_excluded_structural_tokens(excluded_structural_tokens)

    _require_kind(allowed_node_types, events_vocab.CONTAINER_TYPES, "allowed_node_types")
    _require_kind(allowed_leaf_controls, events_vocab.LEAF_TYPES, "allowed_leaf_controls")

    total = len(allowed_node_types) + len(allowed_leaf_controls) + len(capability_excluded)
    union = allowed_node_types | allowed_leaf_controls | capability_excluded
    if len(union) != total:
        raise ModelCapabilityProfileInvalid(
            "allowed_node_types, allowed_leaf_controls and capability_excluded "
            "must be pairwise disjoint"
        )
    if union != eligible:
        missing = eligible - union
        extra = union - eligible
        raise ModelCapabilityProfileInvalid(
            f"allowed_node_types + allowed_leaf_controls + capability_excluded must "
            f"exactly partition node_types({ast_profile!r}); missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    return CapabilityProfile(
        profile_id=payload["profile_id"],
        ast_profile=ast_profile,
        scope=payload["scope"],
        allowed_node_types=allowed_node_types,
        allowed_leaf_controls=allowed_leaf_controls,
        capability_excluded=capability_excluded,
        exact_text=payload["exact_text"],
        typed_properties=payload["typed_properties"],
        expert_adapter_mode=payload["expert_adapter_mode"],
        sparse_moe=payload["sparse_moe"],
        visual_compressor=payload["visual_compressor"],
        speculators=payload["speculators"],
        profile_hash=capability_profile_hash(payload),
        schema_version=payload["model_capability_profile_schema_version"],
        excluded_structural_local_ids=(
            _excluded_structural_local_ids(capability_excluded)
            | frozenset(events_vocab.TOKEN_IDS[name] for name in excluded_structural_tokens)
        ),
    )


def _require_kind(names: frozenset[str], valid_kind: Iterable[str], field: str) -> None:
    valid = frozenset(valid_kind)
    invalid = names - valid
    if invalid:
        raise ModelCapabilityProfileInvalid(
            f"{field} contains non-{field} names: {sorted(invalid)}"
        )


# ---------------------------------------------------------------------------
# Fail-closed registry and resolution (design §4.4: no DEFAULT_PROFILE)
# ---------------------------------------------------------------------------


def build_registry(payloads: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a `profile_id -> raw payload` registry.

    Every payload is schema-validated. Two payloads sharing one
    `profile_id` with differing content raise `CapabilityProfileAmbiguous`
    immediately -- an ambiguous id must never resolve to either candidate
    silently. An exact duplicate (same id, identical content) is not an
    ambiguity and is accepted idempotently.
    """
    registry: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        validate_model_capability_profile(payload)
        profile_id = payload["profile_id"]
        candidate = dict(payload)
        existing = registry.get(profile_id)
        if existing is not None and existing != candidate:
            raise CapabilityProfileAmbiguous(
                f"profile_id {profile_id!r} maps to two differing payloads"
            )
        registry[profile_id] = candidate
    return registry


def resolve_capability_profile(
    registry: Mapping[str, Mapping[str, Any]], profile_id: str | None
) -> CapabilityProfile:
    """Resolve `profile_id` in `registry`, fail-closed.

    Raises `CapabilityProfileUnresolved` for an absent/empty/`None` id or
    an id not present in `registry`. There is no default fallback: an
    unresolved profile is always a hard error, never
    `targets.DEFAULT_PROFILE` or any other implicit choice.
    """
    if not profile_id:
        raise CapabilityProfileUnresolved(
            "profile_id must be a non-empty string; no DEFAULT_PROFILE fallback is permitted"
        )
    if profile_id not in registry:
        raise CapabilityProfileUnresolved(
            f"unknown capability profile_id {profile_id!r}; "
            "no DEFAULT_PROFILE fallback is permitted"
        )
    return build_capability_profile(registry[profile_id])


# ---------------------------------------------------------------------------
# The one explicit S1 primary capability profile (design §4.4; packet R4-3,
# dense-v0-ratification-packet.md:139-160).
#
# Reconciled against a design/packet discrepancy (reported upstream): design
# §4.4's illustrative JSON omits "run" from allowed_node_types (which cannot
# be right -- TEXT_BEGIN is only legal inside a run, per grammar.py, and the
# omission also leaves the schema's node-type partition one type short) and
# the packet's capability-excluded table omits "inline_image" from its list,
# even though the packet's own prose right above the table excludes
# inline_image from primary S1 loss. This profile therefore adds "run" to
# allowed_node_types (packet's allowed list) and keeps "inline_image" in
# capability_excluded (design's set, matching the packet's prose) -- the
# only reading under which allowed_node_types + allowed_leaf_controls +
# capability_excluded exactly partitions every word-visual-v1.1 node type.
# ---------------------------------------------------------------------------

_PRIMARY_S1_PAYLOAD: dict[str, Any] = {
    # Literal, not MODEL_CAPABILITY_PROFILE_SCHEMA_VERSION: this payload
    # predates the 1.1.0 bump and its profile_hash must not move with it.
    "model_capability_profile_schema_version": "1.0.0",
    "profile_id": PRIMARY_S1_PROFILE_ID,
    "ast_profile": "word-visual-v1.1",
    "scope": "single_page_primary_s1",
    "allowed_node_types": [
        "cell",
        "document",
        "list",
        "list_item",
        "paragraph",
        "row",
        "run",
        "section",
        "table",
    ],
    "allowed_leaf_controls": ["line_break", "tab"],
    "capability_excluded": [
        "column_break",
        "endnote",
        "field",
        "floating_image",
        "footer",
        "footnote",
        "header",
        "hyperlink",
        "inline_image",
        "note_ref",
        "page_break",
        "text_frame",
    ],
    "exact_text": True,
    "typed_properties": "not_applicable",
    "expert_adapter_mode": "dense",
    "sparse_moe": "not_applicable",
    "visual_compressor": "not_applicable",
    "speculators": "not_applicable",
}

PRIMARY_S1_CAPABILITY_PROFILE: CapabilityProfile = build_capability_profile(_PRIMARY_S1_PAYLOAD)


# ---------------------------------------------------------------------------
# Widened profile artifact (spec-widened-capability-profile). A second,
# additive profile that excludes nothing -- built alongside
# `PRIMARY_S1_CAPABILITY_PROFILE` without modifying it or its payload in any
# way, since `profile_hash` flows into `composite_grammar_hash` and the
# exact-text cache identity that the current S1 checkpoint is bound to.
#
# Not wired into `s1_runner`, `build_cache` or the eval module: this is an
# artifact for a future run. Using it requires rebuilding both caches and
# retraining.
# ---------------------------------------------------------------------------

WIDENED_V1_PROFILE_ID = "dense-v0-widened-v1"

_WIDENED_V1_PAYLOAD: dict[str, Any] = {
    # Literal, same reason as _PRIMARY_S1_PAYLOAD above.
    "model_capability_profile_schema_version": "1.0.0",
    "profile_id": WIDENED_V1_PROFILE_ID,
    "ast_profile": "word-visual-v1.1",
    "scope": "single_page_primary_s1",
    "allowed_node_types": [
        "cell",
        "document",
        "endnote",
        "footer",
        "footnote",
        "header",
        "hyperlink",
        "list",
        "list_item",
        "paragraph",
        "row",
        "run",
        "section",
        "table",
        "text_frame",
    ],
    "allowed_leaf_controls": [
        "column_break",
        "field",
        "floating_image",
        "inline_image",
        "line_break",
        "note_ref",
        "page_break",
        "tab",
    ],
    "capability_excluded": [],
    "exact_text": True,
    "typed_properties": "not_applicable",
    "expert_adapter_mode": "dense",
    "sparse_moe": "not_applicable",
    "visual_compressor": "not_applicable",
    "speculators": "not_applicable",
}

WIDENED_V1_CAPABILITY_PROFILE: CapabilityProfile = build_capability_profile(_WIDENED_V1_PAYLOAD)


# ---------------------------------------------------------------------------
# Typed-frozen profile (spec-promotable-typed-head-module, O1 answered by the
# owner 2026-09-09). Additive in exactly the same sense as the widened
# profile above: neither payload above is touched, because
# `PRIMARY_S1_CAPABILITY_PROFILE.profile_hash` still guards the identity the
# current S1 checkpoint is bound to.
#
# Capabilities are primary S1's, unchanged -- the grammar this profile binds
# is behaviourally identical. The one difference is `typed_properties`: the
# exact-text cache now carries typed property targets, and a profile
# declaring `not_applicable` over such a cache would be a profile that lies
# about its own rows. `present_frozen` says what is true today: the typed
# heads exist and are frozen; no gradient reaches them until S3.
# ---------------------------------------------------------------------------

TYPED_FROZEN_V1_PROFILE_ID = "s1-typed-frozen-v1"

_TYPED_FROZEN_V1_PAYLOAD: dict[str, Any] = {
    **_PRIMARY_S1_PAYLOAD,
    "profile_id": TYPED_FROZEN_V1_PROFILE_ID,
    "typed_properties": "present_frozen",
}

TYPED_FROZEN_V1_CAPABILITY_PROFILE: CapabilityProfile = build_capability_profile(
    _TYPED_FROZEN_V1_PAYLOAD
)


# ---------------------------------------------------------------------------
# Typed-widened profile (DEC-024: "zapojiť všetko" for the large build and S2).
# Additive in exactly the same sense as the two profiles above: no existing
# payload is touched, because `PRIMARY_S1_CAPABILITY_PROFILE.profile_hash`
# still guards the S1 checkpoint identity and `TYPED_FROZEN_V1` is bound by
# existing artefacts.
#
# Capabilities are `WIDENED_V1`'s, unchanged -- every `word-visual-v1.1` node
# type is admitted, nothing is `capability_excluded`. `WIDENED_V1` itself
# cannot be reused as-is (DEC-024 §"Čo to znamená prakticky" point 1): it
# declares `typed_properties: "not_applicable"` and
# `scope: "single_page_primary_s1"`, which would be a profile lying about its
# own rows over an exact-text cache that carries typed targets, and a profile
# claiming single-page scope for a multi-page build -- exactly the reason
# `TYPED_FROZEN_V1` exists alongside `PRIMARY_S1`. This profile is the
# widened counterpart of that split: full node-type coverage, typed
# properties present (heads exist and are frozen; no gradient reaches them
# until S3, same as `TYPED_FROZEN_V1`), multi-page scope.
#
# `scope` has no closed domain in the schema (`{"type": "string", "minLength":
# 1}`) -- there was nothing to extend the "right way"; `multi_page_s2` is a
# new value in an already-open field.
# ---------------------------------------------------------------------------

TYPED_WIDENED_V1_PROFILE_ID = "s2-typed-widened-v1"

_TYPED_WIDENED_V1_PAYLOAD: dict[str, Any] = {
    **_WIDENED_V1_PAYLOAD,
    "profile_id": TYPED_WIDENED_V1_PROFILE_ID,
    "scope": "multi_page_s2",
    "typed_properties": "present_frozen",
}

TYPED_WIDENED_V1_CAPABILITY_PROFILE: CapabilityProfile = build_capability_profile(
    _TYPED_WIDENED_V1_PAYLOAD
)


# ---------------------------------------------------------------------------
# Standalone-page profile (DEC-033, RFC004-ERRATA-002 E1, design §6). Additive
# in the same sense as the three profiles above: no existing payload is
# touched. Capabilities are `TYPED_WIDENED_V1`'s, unchanged; the one addition
# is `excluded_structural_tokens`, naming all 28 RESUME/SUSPEND tokens
# (`node_lifecycle.RESUME_OR_SUSPEND_TOKEN_NAMES`) so a page decoded under
# this profile can never resume or suspend a node -- it is always a
# standalone page, never a fragment window into a larger stream. This is the
# new optional schema field (1.0.0 -> 1.1.0): the four profiles above keep
# their literal `"1.0.0"` and are unaffected by the bump.
# ---------------------------------------------------------------------------

STANDALONE_V1_PROFILE_ID = "s2-standalone-page-v1"

_STANDALONE_V1_PAYLOAD: dict[str, Any] = {
    **_TYPED_WIDENED_V1_PAYLOAD,
    "model_capability_profile_schema_version": MODEL_CAPABILITY_PROFILE_SCHEMA_VERSION,
    "profile_id": STANDALONE_V1_PROFILE_ID,
    "scope": "standalone_page",
    "excluded_structural_tokens": sorted(node_lifecycle.RESUME_OR_SUSPEND_TOKEN_NAMES),
}

STANDALONE_V1_CAPABILITY_PROFILE: CapabilityProfile = build_capability_profile(
    _STANDALONE_V1_PAYLOAD
)


#: The capability profile the exact-text-cache build defaults to (an
#: explicit build names its profile by id instead). Owned by this module --
#: the one that owns the profiles -- not by the build CLI module that
#: consumes it as its default (R4, `ZOSTAVA-typovany-head-modul.md`).
BOUND_CAPABILITY_PROFILE: CapabilityProfile = TYPED_FROZEN_V1_CAPABILITY_PROFILE


# ---------------------------------------------------------------------------
# Registry and artefact-driven resolution (DEC-024 wiring,
# `spec-viacstranove-dokumenty-plan.md` "DEC-024 -- plan zapojenia profilu").
#
# S1 and S2 consumers used to import `PRIMARY_S1_CAPABILITY_PROFILE` or
# `build_exact_text_cache.BOUND_CAPABILITY_PROFILE` as an import-time global
# (R4, `ZOSTAVA-typovany-head-modul.md`): a checkpoint or cache bound to a
# DIFFERENT profile than whichever one happened to be imported would still
# verify, silently, against the wrong one. Every consumer below instead
# resolves the profile the ARTEFACT ITSELF names -- a `capability_profile_
# hash` on a cache identity, or a `model_capability_profile` component on a
# component lock -- against this registry. An unresolvable hash is refused,
# never defaulted, mirroring `resolve_capability_profile`'s existing
# contract for `profile_id`.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v15 hyperlink-flattened profile (DEC-043 §1, cache rebuild v15). Additive
# in the same sense as every profile above: no existing payload is touched.
#
# Supersedes DEC-037 §2's "exclude hyperlink from the model profile at the
# next cache rebuild" plan -- this IS that next rebuild. `hyperlink` moves
# from `allowed_node_types` to `capability_excluded`, exactly as DEC-037 §2
# said, but that move only became safe once
# `unflatten.dataset.targets.events.derive_event_streams` started treating
# `hyperlink` as a transparent container (flattened to plain text, DEC-043
# §1): a hyperlink node never contributes a HYPERLINK_OPEN/CLOSE/RESUME/
# SUSPEND token to the derived stream any more, so excluding those tokens
# from the grammar's structural domain (`composite_grammar._structural_
# domain`) never refuses a page for carrying one -- the token the exclusion
# names is simply never asked for. Without that flattening, this exact
# payload would refuse every page with a hyperlink (measured 2026-09-23:
# 6,098/18,898 documents, 32.3%), which is why DEC-043 revises DEC-037 §2
# instead of just implementing it as written.
#
# Capabilities are `STANDALONE_V1`'s, with `hyperlink` moved across the one
# partition boundary; `excluded_structural_tokens` (the 28 RESUME/SUSPEND
# names) is unchanged, so a page under this profile is still always a
# standalone page in the same sense `STANDALONE_V1` already established.
# ---------------------------------------------------------------------------

V15_HYPERLINK_FLAT_PROFILE_ID = "v15-hyperlink-flat-v1"

_V15_HYPERLINK_FLAT_PAYLOAD: dict[str, Any] = {
    **_STANDALONE_V1_PAYLOAD,
    "profile_id": V15_HYPERLINK_FLAT_PROFILE_ID,
    "allowed_node_types": [
        n for n in _STANDALONE_V1_PAYLOAD["allowed_node_types"] if n != "hyperlink"
    ],
    "capability_excluded": [
        *_STANDALONE_V1_PAYLOAD["capability_excluded"],
        "hyperlink",
    ],
}

V15_HYPERLINK_FLAT_CAPABILITY_PROFILE: CapabilityProfile = build_capability_profile(
    _V15_HYPERLINK_FLAT_PAYLOAD
)


#: Every capability profile this module ships, keyed here for O(1)
#: fail-closed lookup by hash. Adding a new ratified profile means adding
#: it to this tuple; nothing else in this module treats profile discovery
#: as inference.
KNOWN_CAPABILITY_PROFILES: tuple[CapabilityProfile, ...] = (
    PRIMARY_S1_CAPABILITY_PROFILE,
    WIDENED_V1_CAPABILITY_PROFILE,
    TYPED_FROZEN_V1_CAPABILITY_PROFILE,
    TYPED_WIDENED_V1_CAPABILITY_PROFILE,
    STANDALONE_V1_CAPABILITY_PROFILE,
    V15_HYPERLINK_FLAT_CAPABILITY_PROFILE,
)

#: Ids of every shipped profile whose `scope` is `standalone_page` --
#: derived from the profiles themselves, never enumerated by hand.
#:
#: Consumers gate the standalone-page encoding (DEC-033: each page is
#: rewritten as its own document, so no RESUME/SUSPEND token is ever
#: emitted) on THIS set, not on a single profile id. Keyed on one literal
#: id, the gate silently stopped applying the moment a second standalone
#: profile shipped: `v15-hyperlink-flat-v1` inherits the whole standalone
#: payload -- same scope, same 28 excluded RESUME/SUSPEND names -- but
#: under its own id, so pages kept `RESUME_NODE(document)` while the
#: grammar still excluded it. Measured 2026-09-23 before the fix: 96 of
#: 98 probe pages `skipped_grammar`, 0 written.
STANDALONE_PAGE_PROFILE_IDS: frozenset[str] = frozenset(
    profile.profile_id
    for profile in KNOWN_CAPABILITY_PROFILES
    if profile.scope == "standalone_page"
)

_CAPABILITY_PROFILES_BY_HASH: dict[str, CapabilityProfile] = {
    profile.profile_hash: profile for profile in KNOWN_CAPABILITY_PROFILES
}

_CAPABILITY_PROFILES_BY_ID: dict[str, CapabilityProfile] = {
    profile.profile_id: profile for profile in KNOWN_CAPABILITY_PROFILES
}


class CapabilityProfileHashUnresolved(ValueError):
    """A `capability_profile_hash` or `capability_profile_id` named by an
    artefact or a caller is not one of `KNOWN_CAPABILITY_PROFILES`.

    Raised by both `resolve_capability_profile_by_hash` and
    `resolve_capability_profile_by_id` for an absent/empty value or one
    this module does not recognise -- there is no default profile an
    unresolved hash or id falls back to (mirrors
    `resolve_capability_profile`'s existing contract for `profile_id`).
    """


def resolve_capability_profile_by_hash(profile_hash: str | None) -> CapabilityProfile:
    """Resolve `profile_hash` against `KNOWN_CAPABILITY_PROFILES`, fail-closed.

    This is the lookup an artefact-bound consumer uses: the artefact names
    the hash it was built or bound under (an `ExactTextCacheIdentity`'s
    `capability_profile_hash`, or a component lock's
    `model_capability_profile` component `semantic_hash`), and this
    function is the one place that turns that hash back into the profile
    object the artefact meant -- never an import-time global that may name
    a different profile than the one the artefact actually carries.
    """
    if not profile_hash:
        raise CapabilityProfileHashUnresolved(
            "capability_profile_hash must be a non-empty string; no default "
            "capability profile is permitted"
        )
    profile = _CAPABILITY_PROFILES_BY_HASH.get(profile_hash)
    if profile is None:
        raise CapabilityProfileHashUnresolved(
            f"unknown capability_profile_hash {profile_hash!r}; no default "
            "capability profile is permitted"
        )
    return profile


def resolve_capability_profile_by_id(profile_id: str | None) -> CapabilityProfile:
    """Resolve `profile_id` against `KNOWN_CAPABILITY_PROFILES`, fail-closed.

    The ergonomic counterpart to `resolve_capability_profile_by_hash`: a
    CLI flag choosing a capability profile for a NEW build names it by id
    (e.g. `s2-typed-widened-v1`), not by the hash only the resulting
    artefact carries. Same fail-closed contract -- an absent/empty/unknown
    id is refused, never defaulted.
    """
    if not profile_id:
        raise CapabilityProfileHashUnresolved(
            "capability_profile_id must be a non-empty string; no default "
            "capability profile is permitted"
        )
    profile = _CAPABILITY_PROFILES_BY_ID.get(profile_id)
    if profile is None:
        raise CapabilityProfileHashUnresolved(
            f"unknown capability_profile_id {profile_id!r}; no default "
            "capability profile is permitted"
        )
    return profile


def capability_profile_hash_from_component_lock(component_lock: Mapping[str, Any]) -> str:
    """The `model_capability_profile` component's `semantic_hash` from a
    Model Component Lock document, fail-closed.

    Shared by every consumer that resolves a profile from a component lock
    (`candidate_runtime.load_candidate_runtime`,
    `freeze_candidate_track.freeze`) so this lookup exists in one place.
    """
    for entry in component_lock.get("components", ()):
        if entry.get("kind") == "model_capability_profile":
            semantic_hash = entry.get("semantic_hash")
            if not semantic_hash:
                raise CapabilityProfileHashUnresolved(
                    "component lock's model_capability_profile component has "
                    "no semantic_hash"
                )
            return semantic_hash
    raise CapabilityProfileHashUnresolved(
        "component lock has no model_capability_profile component"
    )


def resolve_capability_profile_from_component_lock(
    component_lock: Mapping[str, Any],
) -> CapabilityProfile:
    """Resolve the capability profile a component lock names, fail-closed."""
    return resolve_capability_profile_by_hash(
        capability_profile_hash_from_component_lock(component_lock)
    )
