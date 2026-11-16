"""The exact-text S1 Component Lock producer (design §4.8, RFC005 §14.4
gate 14).

`manifests.py` has validated Component Locks and computed
`component_lock_hash` since A0, but nothing built one -- so every
`component_lock_hash` in this repository was a placeholder, and gate 14
("Component Lock viaze exact base weights, processor, tokenizer,
vocabulary, grammar, decoder additions a decode config") had nothing real
to check. This module assembles the twelve
`manifests.EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND` bindings plus the
mandatory `cbor_profile` record from the live artifacts that define them,
and validates the result with
`manifests.validate_exact_text_s1_component_lock` before returning it.

**What is derived, and what cannot be.** Ten of the thirteen records are
computed here from something that exists at run time: the structural token
map, the composite vocabulary, the composite grammar descriptor, the
capability profile, the runtime limits policy, the architecture manifest
document, the tokenizer artifact set, this package's own CBOR/encoder/
adapter sources, and the installed tokenizer dependency versions. Two are
**not**: `base_model_weights` and `image_preprocessor` are the exact
checkpoint and processor hashes that design §18.1 item 8 lists as
unresolved, non-M1 DEC-005 D3 pins. They are required arguments with no
default and are checked for shape only -- this producer will not
manufacture a weight hash, and the local feature cache's 40-hex hub commit
and 16-hex processor digest are not one (they are neither SHA-256 nor over
the artifact bytes).

**Consistency is checked, not assumed.** A lock is a set of bindings that
must describe one configuration, so the producer refuses a composite
grammar descriptor whose `composite_vocabulary_hash` names a different
vocabulary than the one being locked, whose `capability_profile_hash`
names a different profile, whose `structural_token_map_hash` is not the
live token map, or whose `budget_policy_hash` is not the encoding-budget
subset of the runtime limits policy being locked (design §4.7's binding,
mechanized here rather than asserted in prose).

Ships no `nn.Module`, no optimizer, and authorizes no gradient update.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.model import events_vocab
from unflatten.model.capability_profile import CapabilityProfile
from unflatten.model.composite_grammar import composite_grammar_hash
from unflatten.model.composite_vocab import (
    COMPOSITE_VOCABULARY_SCHEMA_VERSION,
    CompositeVocabulary,
)
from unflatten.model.manifests import (
    validate_exact_text_s1_architecture_manifest,
    validate_exact_text_s1_component_lock,
)
from unflatten.model.runtime_limits import budget_policy_hash, runtime_limits_policy_hash

#: The schema revision that carries `composite_vocabulary` and
#: `model_capability_profile` -- a MINOR over the A0 19-kind enum (design
#: §4.8, "Add ... in a schema MINOR revision"). An exact-text lock uses
#: two kinds an A0 `0.1.0` document could not carry, so it does not claim
#: to be one.
MODEL_COMPONENT_LOCK_SCHEMA_VERSION = "0.2.0"

#: The `version` recorded for components that declare no version of their
#: own (the pinned weights, the processor, the tokenizer artifact set, the
#: CBOR profile implementation, the two decoder-addition modules, and the
#: runtime lock). Their *identity* is the hash in the same record, which is
#: derived from real bytes; this field is a label for the lock profile that
#: assembled them, and is stated once rather than guessed per component.
EXACT_TEXT_S1_COMPONENT_VERSION = "1.0.0"

#: Distributions whose versions make up the `runtime_lock` record: the
#: tokenizer implementation and its runtime dependencies (design §4.8,
#: "runtime_lock -> tokenizer implementation/dependency hashes").
RUNTIME_LOCK_DISTRIBUTIONS: tuple[str, ...] = ("tokenizers", "transformers", "torch")

#: Module sources bound as "decoder additions" (§4.8 via
#: `manifests.EXACT_TEXT_S1_REQUIRED_COMPONENT_KIND`). Hashed as raw file
#: bytes: these carry no version constant and no serialized artifact, so
#: their source IS the artifact, exactly as `grammar.controller_hash`
#: treats the controller modules.
_SOURCE_BOUND_COMPONENTS: dict[str, str] = {
    "cbor_profile": "cbor.py",
    "document_context_encoder": "document_context_encoder.py",
    "expert_adapter": "expert_adapter.py",
}

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ComponentLockInvalid(ValueError):
    """The inputs to a Component Lock disagree, are malformed, or the
    assembled lock fails its own validation (design §4.8).

    M1-internal (§9.3); not an RFC004 registry code.
    """


def text_tokenizer_identity(component_lock: Mapping[str, Any]) -> str:
    """The lock's text-tokenizer artifact-set hash, whichever key spells it.

    `build_exact_text_s1_component_lock` writes the entry with
    `semantic_hash`; older fixtures and one reader spelled it
    `artifact_sha256`. 26.9.: the first real service start on prod refused a
    correct v15 lock ("component lock tokenizer identity mismatch") because
    the reader looked only for the fixture spelling."""
    for entry in component_lock["components"]:
        if entry.get("kind") == "text_tokenizer":
            for key in ("semantic_hash", "artifact_sha256"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    return value
            raise KeyError("text_tokenizer entry carries neither semantic_hash nor artifact_sha256")
    raise KeyError("component lock has no text_tokenizer component")


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ComponentLockInvalid(
            f"{name} must be 'sha256:<64 hex>', got {value!r} -- this producer will "
            "not manufacture an artifact hash (design §18.1 item 8)"
        )
    return value


def _source_sha256(filename: str) -> str:
    path = Path(__file__).parent / filename
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ComponentLockInvalid(
            f"cannot hash the source of {filename}: {exc} -- a lock record whose "
            "artifact is unreadable would bind nothing"
        ) from exc
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _document_sha256(document: Mapping[str, Any]) -> str:
    """Raw SHA-256 of a document's canonical JCS bytes -- the same identity
    rule `manifests.component_lock_hash` uses for the lock itself."""
    return f"sha256:{hashlib.sha256(to_canonical_json(dict(document))).hexdigest()}"


def document_sha256(document: Mapping[str, Any]) -> str:
    """Public accessor for `_document_sha256` -- a Component Lock document
    never carries its own hash, so callers must compute it (design §4.8)."""
    return _document_sha256(document)


def runtime_lock_document() -> dict[str, Any]:
    """The tokenizer implementation/dependency versions this runtime pins.

    A missing distribution raises rather than being recorded absent: a
    runtime lock that silently omits the tokenizer implementation pins
    less than it claims.
    """
    distributions: dict[str, str] = {}
    for name in RUNTIME_LOCK_DISTRIBUTIONS:
        try:
            distributions[name] = distribution_version(name)
        except PackageNotFoundError as exc:
            raise ComponentLockInvalid(
                f"runtime_lock cannot be built: distribution {name!r} is not "
                f"installed ({exc}), so its version cannot be pinned"
            ) from exc
    return {"distributions": distributions}


def _verify_grammar_descriptor_agrees(
    composite_grammar: Mapping[str, Any],
    *,
    vocabulary: CompositeVocabulary,
    profile: CapabilityProfile,
    runtime_limits_policy: Mapping[str, Any],
) -> None:
    """Refuse a descriptor that describes a different configuration."""
    expected = {
        "composite_vocabulary_hash": vocabulary.artifact_hash,
        "capability_profile_hash": profile.profile_hash,
        "structural_token_map_hash": events_vocab.TOKEN_MAP_HASH,
        "budget_policy_hash": budget_policy_hash(runtime_limits_policy),
    }
    for field, want in expected.items():
        got = composite_grammar.get(field)
        if got != want:
            raise ComponentLockInvalid(
                f"composite_grammar.{field} is {got!r}, but the artifacts being "
                f"locked give {want!r} -- the descriptor and the lock would "
                "describe two different configurations"
            )


def _ast_version(architecture_manifest: Mapping[str, Any]) -> str:
    versions = architecture_manifest["compatible_ast_versions"]
    if len(versions) != 1:
        raise ComponentLockInvalid(
            f"the lock names one ast_version, but the architecture manifest declares "
            f"{len(versions)} compatible_ast_versions ({versions!r}) -- which one this "
            "release locks is a decision this producer will not make"
        )
    return versions[0]


def build_exact_text_s1_component_lock(
    *,
    model_name: str,
    model_version: str,
    architecture_manifest: Mapping[str, Any],
    runtime_limits_policy: Mapping[str, Any],
    composite_grammar: Mapping[str, Any],
    vocabulary: CompositeVocabulary,
    profile: CapabilityProfile,
    tokenizer_artifact_set_hash: str,
    base_model_weights_sha256: str,
    image_preprocessor_sha256: str,
) -> dict[str, Any]:
    """Assemble one exact-text S1 Component Lock, validated on the way out.

    `model_name`/`model_version` name the model release this lock belongs
    to (RFC004 §39.1); everything else is derived from the artifacts passed
    in, except the two §18.1 item 8 pins named in the module docstring.
    """
    issues = validate_exact_text_s1_architecture_manifest(dict(architecture_manifest))
    if issues:
        raise ComponentLockInvalid(
            "architecture manifest is not an exact-text S1 manifest: "
            + "; ".join(f"{issue.path or '/'}: {issue.message}" for issue in issues)
        )
    _verify_grammar_descriptor_agrees(
        composite_grammar,
        vocabulary=vocabulary,
        profile=profile,
        runtime_limits_policy=runtime_limits_policy,
    )
    weights = _require_sha256("base_model_weights_sha256", base_model_weights_sha256)
    preprocessor = _require_sha256("image_preprocessor_sha256", image_preprocessor_sha256)
    tokenizer_hash = _require_sha256(
        "tokenizer_artifact_set_hash", tokenizer_artifact_set_hash
    )

    components = [
        {
            "kind": "architecture_manifest",
            "version": architecture_manifest["architecture_version"],
            "artifact_sha256": _document_sha256(architecture_manifest),
        },
        {
            "kind": "base_model_weights",
            "version": EXACT_TEXT_S1_COMPONENT_VERSION,
            "artifact_sha256": weights,
        },
        {
            "kind": "composite_vocabulary",
            "version": COMPOSITE_VOCABULARY_SCHEMA_VERSION,
            "semantic_hash": vocabulary.artifact_hash,
        },
        {
            # §4.8: "decode_config -> runtime_limits_policy_hash +
            # lowest-packed-ID tie-break". The tie-break is a field of the
            # composite vocabulary payload (`greedy_tie_break`), so it is
            # already bound through `composite_vocabulary` above; binding
            # it twice would create two places that could disagree.
            "kind": "decode_config",
            "version": runtime_limits_policy["policy_version"],
            "semantic_hash": runtime_limits_policy_hash(runtime_limits_policy),
        },
        {
            "kind": "grammar",
            "version": composite_grammar["composite_grammar_schema_version"],
            "semantic_hash": composite_grammar_hash(composite_grammar),
        },
        {
            "kind": "image_preprocessor",
            "version": EXACT_TEXT_S1_COMPONENT_VERSION,
            "artifact_sha256": preprocessor,
        },
        {
            "kind": "model_capability_profile",
            "version": profile.schema_version,
            "semantic_hash": profile.profile_hash,
        },
        {
            "kind": "runtime_lock",
            "version": EXACT_TEXT_S1_COMPONENT_VERSION,
            "artifact_sha256": _document_sha256(runtime_lock_document()),
        },
        {
            "kind": "structural_tokenizer",
            "version": events_vocab.TOKEN_MAP_VERSION,
            "semantic_hash": events_vocab.TOKEN_MAP_HASH,
        },
        {
            "kind": "text_tokenizer",
            "version": EXACT_TEXT_S1_COMPONENT_VERSION,
            "semantic_hash": tokenizer_hash,
        },
    ]
    components += [
        {
            "kind": kind,
            "version": EXACT_TEXT_S1_COMPONENT_VERSION,
            "artifact_sha256": _source_sha256(filename),
        }
        for kind, filename in _SOURCE_BOUND_COMPONENTS.items()
    ]

    lock = {
        "model_component_lock_schema_version": MODEL_COMPONENT_LOCK_SCHEMA_VERSION,
        "model_name": model_name,
        "model_version": model_version,
        "architecture_id": architecture_manifest["architecture_id"],
        "ast_version": _ast_version(architecture_manifest),
        "profiles": list(architecture_manifest["profiles"]),
        # §39.3: canonical order is UTF-8 byte order of `kind`.
        "components": sorted(components, key=lambda c: c["kind"].encode("utf-8")),
    }
    issues = validate_exact_text_s1_component_lock(lock)
    if issues:
        raise ComponentLockInvalid(
            "assembled component lock is invalid: "
            + "; ".join(f"{issue.path or '/'}: {issue.message}" for issue in issues)
        )
    return lock
