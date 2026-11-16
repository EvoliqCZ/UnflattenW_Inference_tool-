"""Component Lock kinds for ADRF/VC modules -- RFC004-ERRATA-005 E3 (§39.2).

Adds four closed-enum kinds to `schemas/component_lock.schema.json`:
`reasoning_module`, `reasoning_gate`, `pre_encoder_compressor`,
`visual_compressor`. Nothing in Wave 1 produces a lock carrying any of
these -- no specialization exists yet -- so this module only adds the enum
values and the load-time check E3 requires ("Chýbajúca váha deklarovaného
aktívneho modulu je tvrdá chyba načítania", §23): a lock that *declares* one
of these components but has no matching weight artifact available must
refuse to load, not silently skip the component.

Adding enum members to an already-emitted schema file does not change the
identity of any existing lock: `component_lock.component_lock_hash` hashes
the canonical JSON of the *lock document*, never the schema file's bytes
(verified against `manifests.component_lock_hash` and
`component_lock._document_sha256` -- both take a document, not a schema
path). A lock that used none of the four kinds before this change still
canonicalizes to the same bytes after it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

#: The four kinds this erratum adds to the closed `component.kind` enum.
REASONING_COMPONENT_KINDS = frozenset(
    {
        "reasoning_module",
        "reasoning_gate",
        "pre_encoder_compressor",
        "visual_compressor",
    }
)


class MissingReasoningComponentWeightError(LookupError):
    """A lock declares an active ADRF/VC component with no matching weight.

    E3, §23: this must be a hard load error, never a silent skip.
    """


def _component_hash(component: Mapping[str, Any]) -> str | None:
    for key in ("semantic_hash", "artifact_sha256"):
        value = component.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def verify_reasoning_component_weights(
    component_lock: Mapping[str, Any], available_hashes: Iterable[str]
) -> None:
    """Refuse to proceed if a declared `REASONING_COMPONENT_KINDS` component
    has no artifact among `available_hashes`.

    `available_hashes` is whatever the caller's weight store can actually
    produce bytes for (e.g. checkpoint shard hashes on disk); this function
    does no I/O itself, so it stays testable without a real weight store.
    """
    available = frozenset(available_hashes)
    for component in component_lock.get("components", []):
        kind = component.get("kind")
        if kind not in REASONING_COMPONENT_KINDS:
            continue
        component_hash = _component_hash(component)
        if component_hash is None or component_hash not in available:
            raise MissingReasoningComponentWeightError(
                f"component lock declares an active {kind!r} component "
                f"(hash={component_hash!r}) but no matching weight artifact "
                "is available -- refusing to load rather than skip it silently"
            )
