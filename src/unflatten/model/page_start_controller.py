"""DEC-061 `StoryPatternController` (PROPOSED, default off): the mask half of
`page_start_policy`, split out because the composite grammar imports torch."""

from __future__ import annotations

import functools
from dataclasses import dataclass, fields, replace
from typing import Any

from unflatten.model import grammar
from unflatten.model.composite_grammar import (
    CompositeGrammarController,
    CompositeGrammarState,
    CompositeGrammarViolation,
)
from unflatten.model.composite_vocab import StructuralRef, TokenRef
from unflatten.model.page_start_policy import (
    _FOOTER_OPEN_ID,
    _HEADER_OPEN_ID,
    _NAME_BY_ID,
    _PHASE_DONE,
    _PHASE_STORIES,
    _STORY_OPEN_IDS,
    StoryDecision,
    StoryPattern,
)


@dataclass(frozen=True, slots=True)
class PageStartGrammarState(CompositeGrammarState):
    """`CompositeGrammarState` plus whether the first section's story block is
    still open. A fragment-mode stack has no document frame, so the grammar
    state alone cannot say "first section"; this field can."""

    page_start_phase: str = _PHASE_STORIES  # "stories" | "done"


@functools.cache
def _section_layout(profile: str) -> tuple[int | None, int | None, int | None] | None:
    """Group indices `(header, footer, body)` of the section frame, or None."""
    groups = grammar._child_groups(profile).get("section")
    if groups is None:
        return None

    def index(node_type: str) -> int | None:
        return next((i for i, g in enumerate(groups) if node_type in g), None)

    return index("header"), index("footer"), index("paragraph")


@dataclass(frozen=True, slots=True, kw_only=True)
class StoryPatternController(CompositeGrammarController):
    """See the module docstring. Delegates to `inner` (so it composes with
    `ForceContentController`); narrows only the structure-mode domain."""

    inner: CompositeGrammarController
    pattern: StoryPattern

    @classmethod
    def wrap(
        cls, controller: CompositeGrammarController, *, pattern: StoryPattern
    ) -> StoryPatternController:
        return cls(
            vocabulary=controller.vocabulary,
            capability_profile_id=controller.capability_profile_id,
            structural_profile=controller.structural_profile,
            capability_excluded_local_ids=controller.capability_excluded_local_ids,
            inner=controller,
            pattern=pattern,
        )

    def fresh(self, capability_profile_id: str) -> PageStartGrammarState:
        base = self.inner.fresh(capability_profile_id)
        values = {f.name: getattr(base, f.name) for f in fields(base)}
        return PageStartGrammarState(**values, page_start_phase=_PHASE_STORIES)

    # -- decision points ------------------------------------------------------

    def _section_top(self, state: CompositeGrammarState) -> Any | None:
        """The section frame when `state` is a story decision point (first
        section, section level, body not started), else None."""
        if (
            state.mode != "structure"
            or getattr(state, "page_start_phase", _PHASE_DONE) != _PHASE_STORIES
        ):
            return None
        layout = _section_layout(self.structural_profile)
        stack = state.structural_state.stack
        if layout is None or layout[2] is None or not stack:
            return None
        top = stack[-1]
        return top if top.node_type == "section" and top.group < layout[2] else None

    def _keep(
        self,
        *,
        kinds: frozenset[str],
        at_header_group: bool,
        child_count: int,
        ids: frozenset[int],
    ) -> frozenset[int]:
        """Mask table of design §3.2: the legal tokens the pattern keeps."""
        other = ids - _STORY_OPEN_IDS
        if not at_header_group:  # footer group: footers already opened
            return ids
        header, footer = "H" in kinds, "F" in kinds
        if child_count == 0:
            if header:
                return frozenset({_HEADER_OPEN_ID})
            return frozenset({_FOOTER_OPEN_ID}) if footer else other
        if header and footer:
            return frozenset({_HEADER_OPEN_ID, _FOOTER_OPEN_ID})
        if header:
            return frozenset({_HEADER_OPEN_ID}) | other
        return frozenset({_FOOTER_OPEN_ID}) if footer else other

    def _decide(self, state: CompositeGrammarState, ids: frozenset[int]) -> StoryDecision | None:
        top = self._section_top(state)
        if top is None:
            return None
        g_header, g_footer, _ = _section_layout(self.structural_profile) or (None,) * 3
        kinds = self.pattern.kinds()
        if g_header is not None and top.group == g_header:
            at_header_group = True
        elif g_footer is not None and top.group == g_footer:
            at_header_group = False
        else:
            return None
        if not at_header_group and "F" not in kinds:
            return StoryDecision(frozenset(), False, True)
        keep = (
            self._keep(
                kinds=kinds, at_header_group=at_header_group, child_count=top.child_count, ids=ids
            )
            & ids
        )
        if not keep:
            return StoryDecision(frozenset(), True, False)
        banned = ids - keep
        reopen = at_header_group and top.child_count >= 1 and bool((ids - _STORY_OPEN_IDS) & banned)
        return StoryDecision(banned, False, False, reopen)

    def decision(self, state: CompositeGrammarState) -> StoryDecision | None:
        """None outside a decision point; otherwise what the mask does here."""
        if self._section_top(state) is None:
            return None
        return self._decide(state, frozenset(self.inner._structural_domain(state)))

    def _structural_domain(self, state: CompositeGrammarState) -> tuple[int, ...]:
        ids = self.inner._structural_domain(state)
        decision = self._decide(state, frozenset(ids))
        if decision is None or not decision.banned_local_ids:
            return ids
        return tuple(i for i in ids if i not in decision.banned_local_ids)

    def step(self, state: CompositeGrammarState, token: TokenRef) -> CompositeGrammarState:
        if not isinstance(state, PageStartGrammarState):
            raise CompositeGrammarViolation(
                f"StoryPatternController steps only PageStartGrammarState (from its own "
                f"fresh()), got {type(state).__name__}"
            )
        at_point = self._section_top(state) is not None
        if (
            state.mode == "structure"
            and isinstance(token, StructuralRef)
            and token.local_id not in self._structural_domain(state)
        ):
            name = _NAME_BY_ID.get(token.local_id, token.local_id)
            raise CompositeGrammarViolation(
                f"structural token {name!r} is not legal here under page-start pattern "
                f"{sorted(self.pattern.kinds())} (mode={state.mode!r})"
            )
        new = self.inner.step(state, token)
        opens_story = isinstance(token, StructuralRef) and token.local_id in _STORY_OPEN_IDS
        if at_point and not opens_story:
            new = _with_phase(new, _PHASE_DONE)
        return new


def _with_phase(state: CompositeGrammarState, phase: str) -> CompositeGrammarState:
    return replace(state, page_start_phase=phase)  # type: ignore[call-arg]
