"""DEC-036 loop guard v2: the frozen caps file, its lazy loader, its
canonical hash and the staleness check (workflow `spec-loop-guard-v2`,
design §3).

The caps are measured on the full train cache by `scripts/loop_guard_v2_
caps.py measure` (design §4) and committed as `loop_guard_v2_caps.json`
next to this module. Nothing here touches that file at import time:
`load_loop_guard_v2_caps` is called only when a decode asks for
`loop_guard == "v2"`, so the `None`/`"v1"` paths run even when the file is
absent (design §3, T1).

The hash is over the canonical JSON of the whole object
(`unflatten.ast.canonical_json`), so whitespace and key order do not move
it, while `semantics` and `close_recovery_variant` -- both inside the
hashed object -- do: the hash names the rules as well as the numbers
(DEC-031 point 4).
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json

#: Read at call time (never at import), so tests can point it elsewhere.
CAPS_PATH = Path(__file__).with_name("loop_guard_v2_caps.json")

CAPS_SCHEMA = "loop-guard-v2-caps/1"
#: Bumped on any change to the counting/anchor/veto/net-depth rules; the
#: code implements exactly this one value and refuses any other.
CAPS_SEMANTICS = "loop-guard-v2-semantics/1"
CLOSE_RECOVERY_VARIANTS = frozenset({"A", "B", "C"})
PERIOD_MAX = 32

#: The eval cache identity keys a v2 decode must share with the train
#: cache the caps were measured on (design §3). A leader-canonicalisation
#: rebuild (DEC-036 point 2) moves the release root and/or materializer
#: hash, so stale caps are refused by mechanism.
CAPS_BINDING_KEYS = (
    "dataset_release_root_hash",
    "training_target_materializer_hash",
    "capability_profile_hash",
    "composite_grammar_hash",
    "composite_vocabulary_hash",
    "qwen_text_allowlist_hash",
    "target_projector_version",
    "target_bundle_schema_version",
)

#: Paths this process has tried to read -- the import-laziness test (M1c)
#: asserts this stays empty after importing the decode module.
_LOADED: list[str] = []


class LoopGuardCapsStale(ValueError):
    """The caps were measured on a train cache whose target identity
    differs from the cache being decoded (design §3)."""


@dataclass(frozen=True)
class LoopGuardV2Caps:
    path: str
    sha256: str
    capability_profile_id: str
    close_recovery_variant: str
    period_max: int
    text_run_cap: int
    #: `{p: cap}` for every p in 1..period_max.
    periodic_cap: Mapping[int, int]
    #: The train manifest's `identity` minus `split`.
    target_identity: Mapping[str, Any]


def loop_guard_v2_caps_sha256(obj: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(to_canonical_json(obj)).hexdigest()


def _cap(value: Any, *, field: str, origin: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{origin}: {field} must be a non-negative integer, got {value!r}")
    return value


def _expect(obj: Mapping[str, Any], field: str, expected: Any, *, origin: str) -> None:
    if obj.get(field) != expected:
        raise ValueError(f"{origin}: {field} must be {expected!r}, got {obj.get(field)!r}")


def parse_loop_guard_v2_caps(obj: Mapping[str, Any], *, origin: str) -> LoopGuardV2Caps:
    """Validate a caps object and return it typed; `ValueError` naming
    `origin` and the offending field on any deviation."""
    _expect(obj, "schema", CAPS_SCHEMA, origin=origin)
    _expect(obj, "guard", "v2", origin=origin)
    _expect(obj, "semantics", CAPS_SEMANTICS, origin=origin)
    _expect(obj, "period_max", PERIOD_MAX, origin=origin)
    variant = obj.get("close_recovery_variant")
    if variant not in CLOSE_RECOVERY_VARIANTS:
        raise ValueError(
            f"{origin}: close_recovery_variant must be one of "
            f"{sorted(CLOSE_RECOVERY_VARIANTS)}, got {variant!r}"
        )
    profile = obj.get("capability_profile_id")
    if not isinstance(profile, str) or not profile:
        raise ValueError(f"{origin}: capability_profile_id must be a non-empty string")

    raw_periodic = obj.get("periodic_cap")
    if not isinstance(raw_periodic, Mapping):
        raise ValueError(f"{origin}: periodic_cap must be an object")
    expected_keys = {str(p) for p in range(1, PERIOD_MAX + 1)}
    missing = sorted(expected_keys - set(raw_periodic), key=int)
    extra = sorted(set(raw_periodic) - expected_keys)
    if missing or extra:
        raise ValueError(
            f"{origin}: periodic_cap must name exactly periods 1..{PERIOD_MAX}; "
            f"missing {missing}, extra {extra}"
        )
    periodic = {
        int(p): _cap(v, field=f"periodic_cap[{p}]", origin=origin) for p, v in raw_periodic.items()
    }

    source = obj.get("source")
    target_identity = source.get("target_identity") if isinstance(source, Mapping) else None
    if not isinstance(target_identity, Mapping):
        raise ValueError(f"{origin}: source.target_identity must be an object")

    return LoopGuardV2Caps(
        path=origin,
        sha256=loop_guard_v2_caps_sha256(obj),
        capability_profile_id=profile,
        close_recovery_variant=variant,
        period_max=PERIOD_MAX,
        text_run_cap=_cap(obj.get("text_run_cap"), field="text_run_cap", origin=origin),
        periodic_cap=periodic,
        target_identity=dict(target_identity),
    )


@functools.lru_cache(maxsize=8)
def _load_caps_file(path: str) -> LoopGuardV2Caps:
    _LOADED.append(path)
    try:
        raw = Path(path).read_text()
    except OSError as exc:
        raise ValueError(f"loop guard v2 caps file {path} cannot be read: {exc}") from exc
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"loop guard v2 caps file {path} is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"loop guard v2 caps file {path} must hold a JSON object")
    return parse_loop_guard_v2_caps(obj, origin=path)


def load_loop_guard_v2_caps(path: Path | str | None = None) -> LoopGuardV2Caps:
    """Load (memoised per path) `path`, or `CAPS_PATH` as it is at call
    time. Raises `ValueError` naming the file on a missing/invalid file."""
    return _load_caps_file(str(CAPS_PATH if path is None else path))


def refuse_unless_caps_bind(caps: LoopGuardV2Caps, eval_cache_identity: Mapping[str, Any]) -> None:
    """Raise `LoopGuardCapsStale` unless `eval_cache_identity` agrees with
    the caps' `source.target_identity` on every `CAPS_BINDING_KEYS` key.
    A key missing on either side is a mismatch, never a pass."""
    mismatched = [
        f"{key}: caps {caps.target_identity.get(key)!r} != eval {eval_cache_identity.get(key)!r}"
        for key in CAPS_BINDING_KEYS
        if key not in caps.target_identity
        or key not in eval_cache_identity
        or caps.target_identity[key] != eval_cache_identity[key]
    ]
    if mismatched:
        raise LoopGuardCapsStale(
            f"loop guard v2 caps {caps.path} were measured on another train cache "
            f"(re-run measure + replay, design §7 P1/P2): {'; '.join(mismatched)}"
        )
