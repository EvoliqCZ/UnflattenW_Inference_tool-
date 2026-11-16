"""DEC-061 page-start story pattern (PROPOSED, default off, torch-free).

A standalone page starts `FRAGMENT_BEGIN SECTION_OPEN` and then, at the
section level, decides irreversibly whether the section's header/footer
stories come first (the grammar's canonical order is `headers -> footers ->
body`). This module holds the three things that decision needs:

- `StoryPattern`: which of header/footer the first section carries, and where
  the claim comes from (`gold_oracle` for the diagnostic oracle, `head:<sha>`
  for a trained head). `story_pattern_from_structural_names` is the one
  definition of that label, shared by census, oracle and report.
- `PageStartPolicy` / `page_start_story_policy.json`: the numbers (feature
  bands, confidence threshold) and the status. Nothing is read unless a decode
  asks for it (`load_page_start_policy` is lazy and memoised).
- `StoryPatternController` (in `page_start_controller`, re-exported lazily so
  this module imports without torch): a wrapper that narrows the structural domain at the
  first section's story decision points to the tokens that keep the pattern
  satisfiable, and is a pass-through everywhere else. It only ever narrows
  (empty keep set -> the unnarrowed domain, `fallback_empty`), exactly like the
  empty-paragraph bound in `composite_grammar`. It is never part of
  `composite_grammar_hash` and must not be added to `grammar.CONTROLLER_MODULES`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.model import events_vocab


def _canonical_bytes(obj: Mapping[str, Any]) -> bytes:
    """Sorted-key compact UTF-8 JSON. `ast.canonical_json` refuses floats and
    this policy's band fractions / threshold are floats (Python's float repr is
    deterministic, so the bytes are stable)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


POLICY_PATH = Path(__file__).with_name("page_start_story_policy.json")
POLICY_SCHEMA = "page-start-story-policy/1"
POLICY_ID = "page-start-story.v0"
POLICY_SEMANTICS = "page-start-story-semantics/1"
PATTERN_SOURCE_GOLD = "gold_oracle"  # refused by the seal check; so is any `gold*`
HEAD_SOURCE_PREFIX = "head:"  # `head:<sha256 of the head bundle>`

_PHASE_STORIES = "stories"
_PHASE_DONE = "done"

_HEADER_OPEN = "HEADER_OPEN"
_FOOTER_OPEN = "FOOTER_OPEN"
_HEADER_OPEN_ID = events_vocab.TOKEN_IDS[_HEADER_OPEN]
_FOOTER_OPEN_ID = events_vocab.TOKEN_IDS[_FOOTER_OPEN]
_STORY_OPEN_IDS = frozenset({_HEADER_OPEN_ID, _FOOTER_OPEN_ID})
_STORY_CLOSE_BY_OPEN = {_HEADER_OPEN: "HEADER_CLOSE", _FOOTER_OPEN: "FOOTER_CLOSE"}
_NAME_BY_ID = {local_id: name for name, local_id in events_vocab.TOKEN_IDS.items()}


# -- pattern ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryPattern:
    has_header: bool
    has_footer: bool
    source: str  # "gold_oracle" | "head:<sha256>"
    confidence: float | None = None  # None = always applied (gold)

    def kinds(self) -> frozenset[str]:
        return frozenset(
            kind for kind, present in (("H", self.has_header), ("F", self.has_footer)) if present
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_header": self.has_header,
            "has_footer": self.has_footer,
            "source": self.source,
            "confidence": self.confidence,
        }


def story_pattern_from_structural_names(names: Iterable[str], *, source: str) -> StoryPattern:
    """Header/footer kinds of the FIRST section of a structural token stream.

    Scans at section level: a header/footer open adds its kind and its whole
    content (nested paragraphs, tables, ...) is skipped up to the matching
    close; the first other section-level token (body open, `PAGE_BREAK`,
    `SECTION_CLOSE`) ends the scan, so a second section is never read. A stream
    with no `SECTION_OPEN` before its first non-framing token has pattern none.
    """
    in_section = False
    awaiting_close: str | None = None
    kinds: set[str] = set()
    for name in names:
        if awaiting_close is not None:
            if name == awaiting_close:
                awaiting_close = None
            continue
        if not in_section:
            if name == "SECTION_OPEN":
                in_section = True
            elif name not in ("FRAGMENT_BEGIN", "DOCUMENT_OPEN"):
                break
            continue
        if name in _STORY_CLOSE_BY_OPEN:
            kinds.add("H" if name == _HEADER_OPEN else "F")
            awaiting_close = _STORY_CLOSE_BY_OPEN[name]
            continue
        break
    return StoryPattern(has_header="H" in kinds, has_footer="F" in kinds, source=source)


# -- policy ----------------------------------------------------------------------


@dataclass(frozen=True)
class PageStartPolicy:
    path: str
    sha256: str
    status: str
    policy_id: str
    semantics: str
    band_fraction_top: float
    band_fraction_bottom: float
    confidence_threshold: float

    def feature_identity_sha256(
        self, *, merge_size: int, visual_input_policy_hash: str, feature_tap: str
    ) -> str:
        """Identity of the moments / head features. Excludes `status` and
        `confidence_threshold` on purpose: freezing the threshold or ratifying
        the policy must not invalidate moments that took hours to extract."""
        return (
            "sha256:"
            + hashlib.sha256(
                _canonical_bytes(
                    {
                        "band_fraction_top": self.band_fraction_top,
                        "band_fraction_bottom": self.band_fraction_bottom,
                        "merge_size": merge_size,
                        "visual_input_policy_hash": visual_input_policy_hash,
                        "feature_tap": feature_tap,
                        "semantics": self.semantics,
                    }
                )
            ).hexdigest()
        )


def page_start_policy_sha256(obj: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(obj)).hexdigest()


def _fraction(obj: Mapping[str, Any], field: str, origin: str, *, closed_top: bool) -> float:
    value = obj.get(field)
    ok = type(value) in (int, float) and 0 < value and (value <= 1 if closed_top else value < 1)
    if not ok:
        interval = "(0, 1]" if closed_top else "(0, 1)"
        raise ValueError(f"{origin}: {field} must be a number in {interval}, got {value!r}")
    return float(value)


def parse_page_start_policy(obj: Mapping[str, Any], *, origin: str) -> PageStartPolicy:
    """Validate a policy object; `ValueError` naming `origin` and the field."""
    for field, expected in (
        ("schema", POLICY_SCHEMA),
        ("policy_id", POLICY_ID),
        ("semantics", POLICY_SEMANTICS),
    ):
        if obj.get(field) != expected:
            raise ValueError(f"{origin}: {field} must be {expected!r}, got {obj.get(field)!r}")
    status = obj.get("status")
    if status not in ("PROPOSED", "RATIFIED"):
        raise ValueError(f"{origin}: status must be PROPOSED or RATIFIED, got {status!r}")
    return PageStartPolicy(
        path=origin,
        sha256=page_start_policy_sha256(obj),
        status=status,
        policy_id=POLICY_ID,
        semantics=POLICY_SEMANTICS,
        band_fraction_top=_fraction(obj, "band_fraction_top", origin, closed_top=False),
        band_fraction_bottom=_fraction(obj, "band_fraction_bottom", origin, closed_top=False),
        confidence_threshold=_fraction(obj, "confidence_threshold", origin, closed_top=True),
    )


@functools.lru_cache(maxsize=8)
def _load_policy_file(path: str) -> PageStartPolicy:
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"page start policy {path} cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"page start policy {path} must hold a JSON object")
    return parse_page_start_policy(obj, origin=path)


def load_page_start_policy(path: Path | str | None = None) -> PageStartPolicy:
    """Load (memoised per path) `path`, or `POLICY_PATH` as it is at call time."""
    return _load_policy_file(str(POLICY_PATH if path is None else path))


# -- decision record (pure) -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoryDecision:
    banned_local_ids: frozenset[int]  # what the mask removed (empty = pass-through)
    fallback_empty: bool  # keep set was empty -> domain left unnarrowed (D-3)
    unexpected_state: bool  # footers open although the pattern has none
    #: True when the mask removed the body although a story is already closed,
    #: i.e. a model that wants to leave the story block is forced to open
    #: another story (A-2b). The decoder counts these steps as
    #: `forced_story_reopen`, to tell them apart from `fallback_empty`.
    forced_story_reopen: bool = False


#: Names defined in `page_start_controller`, which imports the composite grammar
#: and through it torch (`capability_profile -> node_lifecycle`). They resolve
#: lazily so this module stays torch-free for `score.verify_seal` and the census.
_CONTROLLER_EXPORTS = frozenset({"PageStartGrammarState", "StoryPatternController"})


def __getattr__(name: str) -> Any:
    if name in _CONTROLLER_EXPORTS:
        from unflatten.model import page_start_controller

        return getattr(page_start_controller, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
