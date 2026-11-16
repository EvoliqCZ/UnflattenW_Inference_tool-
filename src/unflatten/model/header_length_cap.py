"""`header-length-cap.v0` (PROPOSED, default off, torch-free).

A header/footer the model opens and does not close until ~98 % of the stream
swallows the whole page body (`body_in_story`). This policy closes an open
story once it holds `cap` accepted tokens: from then on every step's legal set
is narrowed to the one token that moves towards the story's close (the
innermost close, `TEXT_END` in text mode, `PARAGRAPH_OPEN` when the innermost
close is not yet legal). It only ever removes tokens from the grammar's own legal
set -- never adds -- so the stream stays grammar-legal, and a story shorter than
the cap is untouched (decode == dense, step for step).

Not part of `composite_grammar_hash`, no new token, no gradient. The numbers live
in `header_length_cap_policy.json` and are `null` until the owner freezes them;
a decode with both caps `null` is refused. Evidence of the analysis:
`docs/runs/header-length-cap-analysis-2026-10-11.md`.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.model import events_vocab
from unflatten.model.composite_vocab import CompositeVocabulary, StructuralRef
from unflatten.model.page_start_policy import page_start_policy_sha256

POLICY_PATH = Path(__file__).with_name("header_length_cap_policy.json")
POLICY_SCHEMA = "header-length-cap-policy/1"
POLICY_ID = "header-length-cap.v0"
POLICY_SEMANTICS = "header-length-cap-semantics/1"
_UNIT = "tokens_in_story"
_BYTE_MODE = "defer"
_CLOSING_RULE = "innermost_close_else_paragraph_open"

_STORY_NODE_TYPES = {"header": "header", "footer": "footer"}
_PARAGRAPH_OPEN_ID = events_vocab.TOKEN_IDS["PARAGRAPH_OPEN"]
_TEXT_END_ID = events_vocab.TOKEN_IDS[events_vocab.TEXT_END]


@dataclass(frozen=True)
class HeaderLengthCapPolicy:
    path: str
    sha256: str
    status: str
    policy_id: str
    semantics: str
    cap_header_tokens: int | None
    cap_footer_tokens: int | None


def _cap(obj: Mapping[str, Any], field: str, origin: str) -> int | None:
    value = obj.get(field)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{origin}: {field} must be null or a positive integer, got {value!r}")
    return value


def parse_header_length_cap_policy(obj: Mapping[str, Any], *, origin: str) -> HeaderLengthCapPolicy:
    """Validate a policy object; `ValueError` naming `origin` and the field."""
    for field, expected in (
        ("schema", POLICY_SCHEMA),
        ("policy_id", POLICY_ID),
        ("semantics", POLICY_SEMANTICS),
        ("unit", _UNIT),
        ("byte_mode", _BYTE_MODE),
        ("closing_rule", _CLOSING_RULE),
    ):
        if obj.get(field) != expected:
            raise ValueError(f"{origin}: {field} must be {expected!r}, got {obj.get(field)!r}")
    status = obj.get("status")
    if status not in ("PROPOSED", "RATIFIED"):
        raise ValueError(f"{origin}: status must be PROPOSED or RATIFIED, got {status!r}")
    return HeaderLengthCapPolicy(
        path=origin,
        sha256=page_start_policy_sha256(obj),
        status=status,
        policy_id=POLICY_ID,
        semantics=POLICY_SEMANTICS,
        cap_header_tokens=_cap(obj, "cap_header_tokens", origin),
        cap_footer_tokens=_cap(obj, "cap_footer_tokens", origin),
    )


@functools.lru_cache(maxsize=8)
def _load_policy_file(path: str) -> HeaderLengthCapPolicy:
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"header length cap policy {path} cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"header length cap policy {path} must hold a JSON object")
    return parse_header_length_cap_policy(obj, origin=path)


def load_header_length_cap_policy(path: Path | str | None = None) -> HeaderLengthCapPolicy:
    """Load (memoised per path) `path`, or `POLICY_PATH` as it is at call time."""
    return _load_policy_file(str(POLICY_PATH if path is None else path))


def refuse_unless_policy_usable(policy: HeaderLengthCapPolicy) -> None:
    if policy.cap_header_tokens is None and policy.cap_footer_tokens is None:
        raise ValueError(
            f"{policy.path}: no cap is frozen (cap_header_tokens and cap_footer_tokens are null)"
        )


def _open_story_kind(state: Any) -> str | None:
    """`"header"`/`"footer"` when a story frame is on the grammar stack."""
    structural = getattr(state, "structural_state", None)
    for frame in getattr(structural, "stack", ()):
        kind = _STORY_NODE_TYPES.get(frame.node_type)
        if kind is not None:
            return kind
    return None


class StoryLengthCap:
    """Per-decode state of the policy: for every accepted prefix length, which
    story is open and at which length it opened (`_marks[n]`), so a loop-guard
    rewind restores it with `rewind`, exactly as the grammar state list does."""

    def __init__(self, policy: HeaderLengthCapPolicy, vocabulary: CompositeVocabulary) -> None:
        self.policy = policy
        self._caps = {"header": policy.cap_header_tokens, "footer": policy.cap_footer_tokens}
        self._pack = lambda local_id: vocabulary.pack(StructuralRef(local_id))
        self._close_ids = {
            node_type: self._pack(events_vocab.TOKEN_IDS[name])
            for node_type, name in events_vocab.CLOSE_TOKEN.items()
        }
        self._text_end = self._pack(_TEXT_END_ID)
        self._paragraph_open = self._pack(_PARAGRAPH_OPEN_ID)
        self._marks: list[tuple[str | None, int]] = [(None, 0)]
        self._run = 0
        self.counters: dict[str, Any] = {
            "forced_steps": 0,
            "capped_stories": 0,
            "forced_by_kind": {"header": 0, "footer": 0},
            "deferred_byte_steps": 0,
            "fallback_empty": 0,
            "max_forced_path_len": 0,
        }

    def observe(self, state: Any, n_accepted: int) -> None:
        """Call after every accepted token, with the new state and `len(accepted)`."""
        kind = _open_story_kind(state)
        prev_kind, prev_open = self._marks[-1]
        if kind is None:
            self._marks.append((None, 0))
            self._run = 0
        else:
            self._marks.append((kind, prev_open if prev_kind == kind else n_accepted))

    def rewind(self, position: int) -> None:
        """The accepted stream was cut back to `position` tokens."""
        del self._marks[position + 1 :]
        self._run = 0

    def open_story_tokens(self, n_accepted: int) -> int | None:
        """Accepted tokens inside the currently open story, `None` outside one."""
        kind, opened = self._marks[-1]
        return None if kind is None else n_accepted - opened

    def narrow(self, legal_ids: list[int], state: Any, n_accepted: int) -> list[int]:
        kind, opened = self._marks[-1]
        cap = self._caps.get(kind) if kind is not None else None
        if cap is None or n_accepted - opened < cap:
            return legal_ids
        if state.mode == "byte":
            self.counters["deferred_byte_steps"] += 1
            return legal_ids
        forced = self._forced_id(legal_ids, state)
        if forced is None:
            self.counters["fallback_empty"] += 1
            return legal_ids
        c = self.counters
        c["forced_steps"] += 1
        c["forced_by_kind"][kind] += 1
        self._run += 1
        if self._run == 1:
            c["capped_stories"] += 1
        c["max_forced_path_len"] = max(c["max_forced_path_len"], self._run)
        return [forced]

    def _forced_id(self, legal_ids: list[int], state: Any) -> int | None:
        legal = set(legal_ids)
        if state.mode == "text":
            return self._text_end if self._text_end in legal else None
        stack = state.structural_state.stack
        if stack:
            close = self._close_ids.get(stack[-1].node_type)
            if close in legal:
                return close
        return self._paragraph_open if self._paragraph_open in legal else None

    def record(self) -> dict[str, Any]:
        """The `header_length_cap` issue entry."""
        p = self.policy
        return {
            "kind": "header_length_cap",
            "policy_id": p.policy_id,
            "policy_status": p.status,
            "policy_sha256": p.sha256,
            "cap_header_tokens": p.cap_header_tokens,
            "cap_footer_tokens": p.cap_footer_tokens,
            "counters": {
                **self.counters,
                "forced_by_kind": dict(self.counters["forced_by_kind"]),
            },
        }
