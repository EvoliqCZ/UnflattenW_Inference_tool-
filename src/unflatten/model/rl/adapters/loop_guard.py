"""Loop-guard factory for the S4.5 real run (design P5, strategy §4 variant A).

``rollout._build_guard`` builds the default v3 guard only for the
``s2-standalone-page-v1`` profile; any other profile needs a ``loop_guard_factory``
port. This module builds that port over the v16 caps file, refusing by name when the
caps do not bind the run's cache identity or capability profile. Torch-free import:
``loop_guard_v3`` is imported inside the factory.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from unflatten.model.loop_guard_v2_caps import (
    LoopGuardCapsStale,
    LoopGuardV2Caps,
    load_loop_guard_v2_caps,
    refuse_unless_caps_bind,
)


class LoopGuardCapsRefused(ValueError):
    """The caps file / policy cannot back a loop guard for this run."""


def verify_caps_binding(
    *,
    caps_path: Path | str,
    caps_sha256: str,
    cache_identity: Mapping[str, Any],
    capability_profile_id: str,
) -> LoopGuardV2Caps:
    """sha256 of the file bytes, cache binding, then capability profile; each refusal named."""
    try:
        raw = Path(caps_path).read_bytes()
    except OSError as exc:
        raise LoopGuardCapsRefused(f"loop guard caps {caps_path} cannot be read: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != caps_sha256.removeprefix("sha256:"):
        raise LoopGuardCapsRefused(
            f"loop guard caps {caps_path} sha256 {actual} != expected {caps_sha256}"
        )
    try:
        caps = load_loop_guard_v2_caps(caps_path)
        refuse_unless_caps_bind(caps, cache_identity)
    except LoopGuardCapsStale as exc:
        raise LoopGuardCapsRefused(str(exc)) from exc
    except ValueError as exc:
        raise LoopGuardCapsRefused(f"loop guard caps {caps_path} invalid: {exc}") from exc
    if caps.capability_profile_id != capability_profile_id:
        raise LoopGuardCapsRefused(
            f"loop guard caps are for profile {caps.capability_profile_id!r}, "
            f"candidate runs {capability_profile_id!r}"
        )
    return caps


def make_loop_guard_factory(
    *,
    caps_path: Path | str,
    caps_sha256: str,
    cache_identity: Mapping[str, Any],
    capability_profile_id: str,
    rollout_policy: Mapping[str, Any],
    ranges: Any,
) -> Callable[[], Any]:
    """Verify now (torch-free); return a closure building a fresh stateful guard per call."""
    if rollout_policy.get("loop_guards_in_policy") is not True:
        raise LoopGuardCapsRefused("rollout policy has loop_guards_in_policy != true")
    caps = verify_caps_binding(
        caps_path=caps_path,
        caps_sha256=caps_sha256,
        cache_identity=cache_identity,
        capability_profile_id=capability_profile_id,
    )
    from unflatten.model.loop_guard_v3 import LoopGuardV3, load_loop_guard_v3_policy, resolve_arm

    policy = load_loop_guard_v3_policy()
    try:
        arm = resolve_arm(policy, rollout_policy["loop_guard_arm"])
    except ValueError as exc:
        raise LoopGuardCapsRefused(f"loop guard arm refused: {exc}") from exc

    def factory() -> Any:
        return LoopGuardV3(ranges=ranges, policy=policy, arm=arm, v2_caps=caps)

    return factory
