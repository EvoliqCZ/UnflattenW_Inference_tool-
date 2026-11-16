"""RFC004 §38–§39 Architecture Manifest / Model Component Lock validators.

Each validator runs a JSON Schema pass (Draft 2020-12, closed envelopes)
and, only when that pass is clean, a semantic pass — mirroring the RFC003
D0 envelope validators. Validators never raise on invalid documents; they
return ``list[Issue]`` (empty means valid).

Model release identity (§39.6, A0 provisional projection)
---------------------------------------------------------
``model_release_id = "model_" + hex(SHA-256("rfc004.model-release.v0\\0"
|| CBOR(projection)))`` where the projection is the documented A0 subset of
the RFC §39.6 release-manifest projection::

    {
      "model_release_manifest_schema_version": "0.1.0",   # pinned
      "architecture_id":            lock.architecture_id,
      "model_version":              lock.model_version,
      "model_component_lock_hash":  raw SHA-256 of the lock's canonical
                                    UTF-8 bytes (RFC 8785 JCS via
                                    unflatten.ast.canonical_json),
      "model_capability_manifest_hash": null,   # capability manifest deferred
      "supported_profiles":         lock.profiles,
      "release_channel":            release_channel parameter,
    }

Deviation from full §39.6, tracked for A0 part 2: the standalone
``model-release-manifest.json`` schema and the Model Capability Manifest do
not exist yet, so the projection is built from the Component Lock and the
capability-manifest hash is pinned ``null``. The lock hash is a *raw* byte
hash of the lock's canonical JCS serialization (its on-disk canonical
form), not a §3.1 semantic hash.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import jsonschema.protocols
import jsonschema.validators

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.ast.issues import Issue
from unflatten.model.hashing import make_id

_SCHEMA_DIR = Path(__file__).parent / "schemas"

#: Closed A0 component-kind vocabulary (§39.2 subset; see the schema
#: ``$comment`` for the deferred kinds).
COMPONENT_KIND: frozenset[str] = frozenset(
    {
        "rfc001_schema_binding",
        "rfc001_hash_projection",
        "cbor_profile",
        "io_schemas",
        "architecture_manifest",
        "backbone_adapter",
        "base_model_weights",
        "image_preprocessor",
        "document_context_encoder",
        "structural_tokenizer",
        "text_tokenizer",
        "grammar",
        "expert_adapter",
        "typed_head_registry",
        "default_policy",
        "decode_config",
        "error_code_registry",
        "runtime_lock",
        "renderer_binding",
    }
)

#: MINOR extension (design §4.8): component kinds an exact-text S1 Component Lock adds
#: on top of the A0 :data:`COMPONENT_KIND` set above. Kept as a separate constant, not
#: folded into :data:`COMPONENT_KIND`, because that constant is documented and tested as
#: the closed A0 subset (`tests/model/test_manifests.py::TestComponentKinds`) and these
#: two kinds are M1 additions, not A0. The JSON Schema enum
#: (`schemas/component_lock.schema.json`) is the union of both sets.
EXACT_TEXT_S1_COMPONENT_KIND: frozenset[str] = frozenset(
    {"composite_vocabulary", "model_capability_profile"}
)

#: Of :data:`EXACT_TEXT_S1_COMPONENT_KIND`, the subset RFC004 §39.2 does **not** name
#: (design §4.8, RR-4 `[PROPOSED — owner decision required]`). `composite_vocabulary`
#: rides existing §39.2 authority ("structural tokenizer/vocabulary", RFC004:3827) and is
#: deliberately absent from this set. Nothing in this module may assert that §39.2
#: mandates a kind in :data:`PROVISIONAL_COMPONENT_KIND`.
PROVISIONAL_COMPONENT_KIND: frozenset[str] = frozenset({"model_capability_profile"})

#: Component kinds an exact-text S1 lock must carry (design §4.8 table). Distinct from
#: :data:`COMPONENT_KIND` | :data:`EXACT_TEXT_S1_COMPONENT_KIND`: those two describe what
#: the *schema* allows, this describes what the exact-text S1 *profile* requires.
#:
#: **Widened 2026-09-05 (review, gate 14 audit).** RFC005 §14.4 gate 14, verbatim:
#: "Component Lock viaže exact base weights, processor, tokenizer, vocabulary, grammar,
#: decoder additions a decode config" -- seven named bindings. This set previously named
#: five of the seven (tokenizer as two kinds, vocabulary, grammar, decode config) and
#: silently omitted "base weights" and "processor": a lock entirely missing
#: `base_model_weights`/`image_preprocessor` validated as a complete exact-text S1 lock
#: (measured: `tests/model/test_manifests.py::TestExactTextS1RequiredComponentKind
#: ::test_baseline_fixture_already_needed_base_weights_and_processor` -- the existing
#: "valid" fixture never carried either kind). "decoder additions" maps to
#: `document_context_encoder` and `expert_adapter`: RFC005's own §14.4's four
#: `trainable_primary_s1` groups (`trainable_group_registry.py`) are
#: `adapter.projection`/`document_context_encoder`/`tree_text_decoder`/`expert_adapter.dense`,
#: and of those, `document_context_encoder` and `expert_adapter` are the only ones that are
#: ALSO already-existing, unambiguous `COMPONENT_KIND` entries -- `tree_text_decoder` has no
#: single matching kind in the current A0 taxonomy, so this mapping is a documented
#: INTERPRETATION of "decoder additions", not a silent guess: if a future
#: `tree_text_decoder`-shaped kind is added to `COMPONENT_KIND`, it belongs here too.
#: This is a real gap, not just a naming question: until that kind exists, the
#: Component Lock CANNOT bind a component that genuinely exists in the model --
#: `tree_text_decoder` runs, unpinned, in every exact-text S1 run today.
#:
#: The resulting set below is all TWELVE kinds RFC005 §14.4 covers (seven named
#: bindings, several of which decompose into more than one existing `COMPONENT_KIND`
#: -- e.g. "tokenizer" into `structural_tokenizer` + `text_tokenizer`). The count and
#: membership come from §14.4's text, not from convenience or from what the fixtures
#: happened to already carry -- do not trim this set without re-reading §14.4.
EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND: frozenset[str] = frozenset(
    {
        "base_model_weights",
        "image_preprocessor",
        "structural_tokenizer",
        "text_tokenizer",
        "composite_vocabulary",
        "grammar",
        "document_context_encoder",
        "expert_adapter",
        "model_capability_profile",
        "architecture_manifest",
        "decode_config",
        "runtime_lock",
    }
)

_RELEASE_DOMAIN = "rfc004.model-release.v0"
_RELEASE_MANIFEST_SCHEMA_VERSION = "0.1.0"

#: Closed release-channel vocabulary — free text would mint distinct
#: release ids for typo'd channels (schema-style enum check).
RELEASE_CHANNEL: frozenset[str] = frozenset({"dev", "rc", "prod"})


@cache
def _validator(name: str) -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{name}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _pointer(path: list[Any]) -> str:
    # RFC 6901: the root pointer is the empty string.
    return "".join(f"/{p}" for p in path)


def _schema_issues(name: str, doc: Any) -> list[Issue]:
    return [
        Issue(
            code="MODEL_SCHEMA_INVALID",
            severity="error",
            node_id=None,
            path=_pointer(list(error.absolute_path)),
            message=error.message,
        )
        for error in _validator(name).iter_errors(doc)
    ]


def validate_architecture_manifest(doc: Any) -> list[Issue]:
    """Validate one Architecture Manifest (§38.1)."""
    issues = _schema_issues("architecture_manifest", doc)
    if issues:
        return issues
    # JSON Schema Draft 2020-12 accepts 1024.0 as an "integer"; the manifest
    # data model has no floats (§3.1 discipline), so require a true int.
    d_model = doc["d_model"]
    if isinstance(d_model, bool) or not isinstance(d_model, int):
        issues.append(
            Issue(
                code="MODEL_MANIFEST_D_MODEL_NOT_INT",
                severity="error",
                node_id=None,
                path="/d_model",
                message=f"d_model must be a true integer, got {d_model!r}",
            )
        )
    return issues


def validate_component_lock(doc: Any) -> list[Issue]:
    """Validate one Model Component Lock (§39, A0 subset)."""
    issues = _schema_issues("component_lock", doc)
    if issues:
        return issues
    components = doc["components"]
    kinds = [entry["kind"] for entry in components]
    seen: set[str] = set()
    for index, kind in enumerate(kinds):
        if kind in seen:
            issues.append(
                Issue(
                    code="MODEL_LOCK_DUPLICATE_COMPONENT_KIND",
                    severity="error",
                    node_id=None,
                    path=f"/components/{index}/kind",
                    message=f"duplicate component kind {kind!r}",
                )
            )
        seen.add(kind)
    # §39.3: the components array is canonicalized by UTF-8 byte order of its
    # sort key; any other order is a hard lock error.
    if kinds != sorted(kinds, key=lambda k: k.encode("utf-8")):
        issues.append(
            Issue(
                code="MODEL_LOCK_COMPONENTS_UNSORTED",
                severity="error",
                node_id=None,
                path="/components",
                message="components must be sorted by UTF-8 byte order of 'kind'",
            )
        )
    # §3.1: the deterministic CBOR profile implementation/test-vector hash is
    # a mandatory Component Lock field.
    if "cbor_profile" not in seen:
        issues.append(
            Issue(
                code="MODEL_LOCK_CBOR_PROFILE_MISSING",
                severity="error",
                node_id=None,
                path="/components",
                message="a 'cbor_profile' component record is mandatory (RFC004 §3.1)",
            )
        )
    return issues


def validate_exact_text_s1_component_lock(doc: Any) -> list[Issue]:
    """Validate one Component Lock against the exact-text S1 profile (design §4.8).

    Additive to :func:`validate_component_lock`: runs it first (structural/A0
    checks), then requires every kind in
    :data:`EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND` to be present. A lock that
    omits a required kind, or whose entry was mutated to a different `kind`
    (so the binding no longer exists under its required name), fails here
    while still passing the plain A0 :func:`validate_component_lock`.
    """
    issues = validate_component_lock(doc)
    if issues:
        return issues
    kinds = {entry["kind"] for entry in doc["components"]}
    missing = EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND - kinds
    return [
        Issue(
            code="MODEL_LOCK_EXACT_TEXT_S1_BINDING_MISSING",
            severity="error",
            node_id=None,
            path="/components",
            message=f"exact-text S1 lock is missing required component kind {kind!r}",
        )
        for kind in sorted(missing)
    ]


def validate_exact_text_s1_architecture_manifest(doc: Any) -> list[Issue]:
    """Validate one Architecture Manifest against the exact-text/composite-decoder
    contract (design §4.8): declares `composite_decoder_contract`, declares dense
    active, and marks any present optional slot (visual compressor / speculators)
    `not_applicable` since M1 carries no weight refs for them
    (RFC004 §39.5, `docs/rfc/RFC004-ai-model-architecture-v0.1-1.md:3905-3907`).
    """
    issues = validate_architecture_manifest(doc)
    if issues:
        return issues
    if "composite_decoder_contract" not in doc["components"]:
        issues.append(
            Issue(
                code="MODEL_MANIFEST_EXACT_TEXT_CONTRACT_MISSING",
                severity="error",
                node_id=None,
                path="/components",
                message="exact-text S1 manifest requires 'composite_decoder_contract'",
            )
        )
    if "dense" not in doc["expert_adapter"]["supported_modes"]:
        issues.append(
            Issue(
                code="MODEL_MANIFEST_EXACT_TEXT_DENSE_NOT_DECLARED",
                severity="error",
                node_id=None,
                path="/expert_adapter/supported_modes",
                message="exact-text S1 manifest requires 'dense' in supported_modes",
            )
        )
    for slot, enabled in doc["optional_slots"].items():
        if enabled is not False:
            issues.append(
                Issue(
                    code="MODEL_MANIFEST_EXACT_TEXT_OPTIONAL_SLOT_NOT_APPLICABLE",
                    severity="error",
                    node_id=None,
                    path=f"/optional_slots/{slot}",
                    message=(
                        f"exact-text S1 manifest declares no weight refs for {slot!r}; "
                        "it must be false (not_applicable)"
                    ),
                )
            )
    return issues


def component_lock_hash(component_lock: dict[str, Any]) -> str:
    """The raw SHA-256 of `component_lock`'s canonical JCS bytes
    (`sha256:<64hex>`) -- RFC005 §14.4/RFC004 §39.6's `model_component_lock_hash`
    ("Pre kazdy RFC004 Component Lock plati `model_component_lock_hash ==
    artifact.sha256`", RFC005-training-strategy-v0.1-2.md:3007).

    Public and reused by :func:`model_release_id` rather than duplicated,
    so there is exactly one place that decides how a lock's identity is
    computed. **This is the check gate 14 was missing before 2026-09-05**:
    nothing recomputed this hash from the real lock and compared it to the
    `component_lock_hash` a run manifest/cache identity CLAIMS -- so a
    component's pinned hash (or the whole component list) could change in
    the lock and a stale claim elsewhere would never be caught. Does NOT
    validate `component_lock` itself; call :func:`validate_component_lock`
    first if the document's own well-formedness has not already been
    checked (mirrors `model_release_id`'s own fail-closed order).
    """
    lock_bytes = to_canonical_json(component_lock)
    return f"sha256:{hashlib.sha256(lock_bytes).hexdigest()}"


def model_release_id(component_lock: dict[str, Any], *, release_channel: str = "dev") -> str:
    """Derive ``model_<64hex>`` from a valid Component Lock (§39.6, A0).

    Raises ``ValueError`` when the lock does not validate or the channel is
    not in :data:`RELEASE_CHANNEL`; the projection is documented in the
    module docstring.
    """
    if release_channel not in RELEASE_CHANNEL:
        raise ValueError(
            f"release_channel must be one of {sorted(RELEASE_CHANNEL)}, got {release_channel!r}"
        )
    issues = validate_component_lock(component_lock)
    if issues:
        raise ValueError(
            "component lock invalid: "
            + "; ".join(f"{issue.path}: {issue.message}" for issue in issues)
        )
    lock_hash = component_lock_hash(component_lock)
    projection = _release_projection(component_lock, lock_hash, release_channel)
    return make_id("model", _RELEASE_DOMAIN, projection)


def _release_projection(
    component_lock: dict[str, Any], lock_hash: str, release_channel: str
) -> dict[str, Any]:
    return {
        "model_release_manifest_schema_version": _RELEASE_MANIFEST_SCHEMA_VERSION,
        "architecture_id": component_lock["architecture_id"],
        "model_version": component_lock["model_version"],
        "model_component_lock_hash": lock_hash,
        "model_capability_manifest_hash": None,
        # sorted: supported_profiles is a set — the projection must not
        # depend on the declaration order in the lock. (The lock hash above
        # still covers the raw lock bytes, order included.)
        "supported_profiles": sorted(component_lock["profiles"]),
        "release_channel": release_channel,
    }
