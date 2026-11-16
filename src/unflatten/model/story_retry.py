"""`story-retry.v0` (PROPOSED, default off, torch-free).

A header/footer the model opens and then fills with the page body (`body_in_story`) is a
wrong decision at the story's *open*, not at its close: with the story kind banned the same
model writes the body correctly (docs/runs/header-length-cap-analysis-2026-10-11.md, the
five-page test of 2026-10-11). This policy detects an over-long story in a finished first
decode, drops that kind from a `page_start_policy.StoryPattern` and asks for a second decode
under it. Kinds whose stories were short in the first decode are kept, so a real header or
footer survives. The pattern comes from the model's own tokens -- never from the reference.

Nothing is read unless a decode asks for it (`load_story_retry_policy` is lazy and memoised).
Not part of `composite_grammar_hash`; no new token; no gradient.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.model import events_vocab
from unflatten.model.composite_vocab import CompositeVocabulary, StructuralRef
from unflatten.model.page_start_policy import (
    StoryPattern,
    page_start_policy_sha256,
)

POLICY_PATH = Path(__file__).with_name("story_retry_policy.json")
POLICY_SCHEMA = "story-retry-policy/1"
POLICY_ID = "story-retry.v0"
POLICY_SEMANTICS = "story-retry-semantics/1"
PATTERN_SOURCE = "retry_from_first_decode"
_UNIT = "tokens_in_story"

_NAME_BY_ID = {local_id: name for name, local_id in events_vocab.TOKEN_IDS.items()}
_OPEN = {"HEADER_OPEN": "header", "FOOTER_OPEN": "footer"}
_CLOSE = {"HEADER_CLOSE": "header", "FOOTER_CLOSE": "footer"}


@dataclass(frozen=True)
class StoryRetryPolicy:
    path: str
    sha256: str
    status: str
    policy_id: str
    semantics: str
    threshold_tokens: int | None


def parse_story_retry_policy(obj: Mapping[str, Any], *, origin: str) -> StoryRetryPolicy:
    for field, expected in (
        ("schema", POLICY_SCHEMA),
        ("policy_id", POLICY_ID),
        ("semantics", POLICY_SEMANTICS),
        ("unit", _UNIT),
        ("pattern_source", PATTERN_SOURCE),
    ):
        if obj.get(field) != expected:
            raise ValueError(f"{origin}: {field} must be {expected!r}, got {obj.get(field)!r}")
    status = obj.get("status")
    if status not in ("PROPOSED", "RATIFIED"):
        raise ValueError(f"{origin}: status must be PROPOSED or RATIFIED, got {status!r}")
    threshold = obj.get("threshold_tokens")
    if threshold is not None and (type(threshold) is not int or threshold <= 0):
        raise ValueError(
            f"{origin}: threshold_tokens must be null or a positive integer, got {threshold!r}"
        )
    return StoryRetryPolicy(
        path=origin,
        sha256=page_start_policy_sha256(obj),
        status=status,
        policy_id=POLICY_ID,
        semantics=POLICY_SEMANTICS,
        threshold_tokens=threshold,
    )


@functools.lru_cache(maxsize=8)
def _load_policy_file(path: str) -> StoryRetryPolicy:
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"story retry policy {path} cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"story retry policy {path} must hold a JSON object")
    return parse_story_retry_policy(obj, origin=path)


def load_story_retry_policy(path: Path | str | None = None) -> StoryRetryPolicy:
    """Load (memoised per path) `path`, or `POLICY_PATH` as it is at call time."""
    return _load_policy_file(str(POLICY_PATH if path is None else path))


def refuse_unless_policy_usable(policy: StoryRetryPolicy) -> None:
    if policy.threshold_tokens is None:
        raise ValueError(f"{policy.path}: no threshold is frozen (threshold_tokens is null)")


def story_lengths(
    accepted_tokens: Sequence[int], vocabulary: CompositeVocabulary
) -> list[tuple[str, int]]:
    """`(kind, accepted tokens strictly inside the story)` for every header/footer of a stream,
    in order; a story never closed counts to the end of the stream."""
    out: list[tuple[str, int]] = []
    open_kind: str | None = None
    start = 0
    for index, packed in enumerate(accepted_tokens):
        ref = vocabulary.unpack(packed)
        name = _NAME_BY_ID.get(ref.local_id) if isinstance(ref, StructuralRef) else None
        if name in _OPEN:
            open_kind, start = _OPEN[name], index
        elif name in _CLOSE and open_kind is not None:
            out.append((open_kind, index - start - 1))
            open_kind = None
    if open_kind is not None:
        out.append((open_kind, len(accepted_tokens) - start - 1))
    return out


def retry_pattern(stories: Sequence[tuple[str, int]], threshold: int) -> StoryPattern | None:
    """`None` when no story exceeds `threshold` (no retry); otherwise the pattern that keeps
    exactly the kinds whose every story was within it."""
    over = {kind for kind, length in stories if length > threshold}
    if not over:
        return None
    kept = {kind for kind, _ in stories} - over
    return StoryPattern(
        has_header="header" in kept, has_footer="footer" in kept, source=PATTERN_SOURCE
    )


def retry_record(
    policy: StoryRetryPolicy,
    stories: Sequence[tuple[str, int]],
    pattern: StoryPattern | None,
    outcome: str,
) -> dict[str, Any]:
    """The `story_retry` issue entry. `outcome`: `not_needed`, `retried` or `retry_failed:<type>`
    (the second decode raised; the first hypothesis is returned unchanged)."""
    return {
        "kind": "story_retry",
        "policy_id": policy.policy_id,
        "policy_status": policy.status,
        "policy_sha256": policy.sha256,
        "threshold_tokens": policy.threshold_tokens,
        "first_decode_stories": [[kind, length] for kind, length in stories],
        "pattern": (
            None
            if pattern is None
            else {"has_header": pattern.has_header, "has_footer": pattern.has_footer}
        ),
        "outcome": outcome,
    }
