"""DEC-055 loop guard v3 (PROPOSED, default off): runtime fixes for runaway
decodes on top of loop guard v2 (`docs/runs/loop-autopsy-2026-09-29.md`).

v3 never replaces v2. It wraps one unmodified `LoopGuardV2` (same caps file,
same text-run/periodic/close-recovery rules, same evidence) and adds four
optional rules, selected per decode by an *arm* (`"r1"`, `"r2+r4"`, ...):

- **R1** (`unit_cap`, `cardinality`): a text-free (no text, no image)
  net-depth-0 unit of period <= `unit_period_max` may repeat back-to-back at
  most `text_free_unit_cap` times (the invisible empty run `RUN_OPEN
  TEXT_BEGIN TEXT_END RUN_CLOSE`: `empty_run_unit_cap`, until DEC-055 V1
  canonicalises it); a row may hold at most `max_cells_per_row` cells and a
  table at most `max_rows_per_table` rows. The action is a veto of the token
  that would start the next repetition (the next cell/row); decoding
  continues. Unlike
  v2 the count is per consecutive run, not a per-page total, so one outlier
  train page cannot set it.
- **R2** (`rewind`): the same detectors, but instead of vetoing, decoding
  goes back to the start of the detected repetition -- extended over the
  table/row/paragraph/run openers immediately before it, so a hallucinated
  empty table is undone from its `TABLE_OPEN` -- and bans the token that
  opened it at that position. The KV cache is cropped with the same
  `IncrementalDecodeState.rollback_to` the speculative verify path uses.
  After `max_rewinds` rewinds on one page R2 falls back to the R1 veto.
  Cardinality stays a veto under R2 (a too-wide row has no repetition start).
- **R3** (`long_period`): exact periodicity with period `long_period_min`..
  `long_period_max` and at least `long_period_reps` back-to-back copies. The
  comparison is over every token, text included, so a repeated block counts
  only when it adds no new content; legitimately repeated table rows with
  different text are never exactly periodic. Only blocks that carry text and
  structure count (text-free wide rows repeat legitimately in forms; R1's
  cardinality bounds them). Acts as R1 (veto) or R2 (rewind).
- **R4** (`force_content`): DEC-016 C the other way round. Once the grammar's
  empty-paragraph bound is reached, `PARAGRAPH_OPEN` stays legal but the
  empty close is forbidden (`ForceContentController`), so the next paragraph
  must carry content instead of the decode escaping into another container.

Every number is a field of the frozen policy file `loop_guard_v3_policy.json`
(status PROPOSED); its canonical-JSON sha256 names the rules and the numbers.
Nothing here is imported or read unless a decode asks for `loop_guard="v3"`.
"""

from __future__ import annotations

import functools
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.model import events_vocab
from unflatten.model.composite_grammar import (
    MAX_CONSECUTIVE_EMPTY_PARAGRAPHS,
    CompositeGrammarController,
    CompositeGrammarState,
)
from unflatten.model.composite_vocab import (
    ByteValueRef,
    PackedRanges,
    QwenRef,
    StructuralRef,
    pack,
)
from unflatten.model.loop_guard_v2_caps import LoopGuardV2Caps

POLICY_PATH = Path(__file__).with_name("loop_guard_v3_policy.json")
POLICY_SCHEMA = "loop-guard-v3-policy/1"
POLICY_ID = "loop-guard-v3"
POLICY_SEMANTICS = "loop-guard-v3-semantics/1"

_NAME_BY_ID = {local_id: name for name, local_id in events_vocab.TOKEN_IDS.items()}
_PARAGRAPH_OPEN_ID = events_vocab.TOKEN_IDS["PARAGRAPH_OPEN"]
_PARAGRAPH_CLOSE_ID = events_vocab.TOKEN_IDS["PARAGRAPH_CLOSE"]
_IMAGE_LEAVES = frozenset(events_vocab.TOKEN_IDS[n] for n in ("INLINE_IMAGE", "FLOATING_IMAGE"))

#: Openers a rewind may extend over, backwards from the first copy of the
#: repetition: the containers a text-free loop is typically opened with. A
#: cell, section, header/footer or frame boundary stops it, so a loop inside
#: an outer table's cell is undone from its own `TABLE_OPEN`, never from the
#: outer cell.
_REWIND_OPENERS = frozenset(
    events_vocab.TOKEN_IDS[n] for n in ("TABLE_OPEN", "ROW_OPEN", "PARAGRAPH_OPEN", "RUN_OPEN")
)


# -- policy -----------------------------------------------------------------


@dataclass(frozen=True)
class LoopGuardV3Policy:
    path: str
    sha256: str
    status: str
    arms: tuple[str, ...]
    default_arm: str
    text_free_unit_cap: int
    empty_run_unit_cap: int
    unit_period_max: int
    max_cells_per_row: int
    max_rows_per_table: int
    long_period_min: int
    long_period_max: int
    long_period_reps: int
    max_rewinds: int


@dataclass(frozen=True)
class Arm:
    """Which v3 rules one decode runs. Exactly one of R1 (veto) / R2
    (rewind) is the action; R3 adds a detector, R4 changes the grammar."""

    name: str
    rewind: bool
    long_period: bool
    force_content: bool


def parse_arm(name: str) -> Arm:
    """`"r1"`, `"r2+r4"`, `"r1+r3+r4"`, ... -> `Arm`; `ValueError` otherwise."""
    parts = name.split("+") if isinstance(name, str) else []
    allowed = {"r1", "r2", "r3", "r4"}
    if (
        not parts
        or len(set(parts)) != len(parts)
        or set(parts) - allowed
        or ("r1" in parts) == ("r2" in parts)
        or parts != sorted(parts)
    ):
        raise ValueError(
            f"loop guard v3 arm must be 'r1' or 'r2' plus optional '+r3'/'+r4' "
            f"in that order, got {name!r}"
        )
    return Arm(
        name=name, rewind="r2" in parts, long_period="r3" in parts, force_content="r4" in parts
    )


def loop_guard_v3_policy_sha256(obj: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(to_canonical_json(obj)).hexdigest()


def _positive(obj: Mapping[str, Any], field: str, origin: str) -> int:
    value = obj.get(field)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{origin}: {field} must be a positive integer, got {value!r}")
    return value


def parse_loop_guard_v3_policy(obj: Mapping[str, Any], *, origin: str) -> LoopGuardV3Policy:
    """Validate a policy object; `ValueError` naming `origin` and the field."""
    for field, expected in (
        ("schema", POLICY_SCHEMA),
        ("policy_id", POLICY_ID),
        ("semantics", POLICY_SEMANTICS),
        ("base_guard", "v2"),
    ):
        if obj.get(field) != expected:
            raise ValueError(f"{origin}: {field} must be {expected!r}, got {obj.get(field)!r}")
    status = obj.get("status")
    if status not in ("PROPOSED", "RATIFIED"):
        raise ValueError(f"{origin}: status must be PROPOSED or RATIFIED, got {status!r}")
    arms = obj.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"{origin}: arms must be a non-empty list")
    for arm in arms:
        parse_arm(arm)
    if obj.get("default_arm") not in arms:
        raise ValueError(f"{origin}: default_arm {obj.get('default_arm')!r} is not in arms")
    ints = {
        field: _positive(obj, field, origin)
        for field in (
            "text_free_unit_cap",
            "empty_run_unit_cap",
            "unit_period_max",
            "max_cells_per_row",
            "max_rows_per_table",
            "long_period_min",
            "long_period_max",
            "long_period_reps",
            "max_rewinds",
        )
    }
    if not ints["unit_period_max"] < ints["long_period_min"] <= ints["long_period_max"]:
        raise ValueError(
            f"{origin}: need unit_period_max < long_period_min <= long_period_max, got "
            f"{ints['unit_period_max']}, {ints['long_period_min']}, {ints['long_period_max']}"
        )
    if min(ints["text_free_unit_cap"], ints["empty_run_unit_cap"], ints["long_period_reps"]) < 2:
        raise ValueError(
            f"{origin}: text_free_unit_cap, empty_run_unit_cap and long_period_reps must be >= 2"
        )
    return LoopGuardV3Policy(
        path=origin,
        sha256=loop_guard_v3_policy_sha256(obj),
        status=status,
        arms=tuple(arms),
        default_arm=obj["default_arm"],
        **ints,
    )


@functools.lru_cache(maxsize=8)
def _load_policy_file(path: str) -> LoopGuardV3Policy:
    try:
        obj = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"loop guard v3 policy {path} cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"loop guard v3 policy {path} must hold a JSON object")
    return parse_loop_guard_v3_policy(obj, origin=path)


def load_loop_guard_v3_policy(path: Path | str | None = None) -> LoopGuardV3Policy:
    """Load (memoised per path) `path`, or `POLICY_PATH` as it is at call time."""
    return _load_policy_file(str(POLICY_PATH if path is None else path))


def resolve_arm(policy: LoopGuardV3Policy, arm: str | None) -> Arm:
    """`arm` (or the policy's `default_arm` when `None`), refused unless the
    policy lists it."""
    name = policy.default_arm if arm is None else arm
    if name not in policy.arms:
        raise ValueError(f"loop guard v3 arm {name!r} is not one of the policy's {policy.arms}")
    return parse_arm(name)


# -- R4: the grammar variant ---------------------------------------------------


def force_content_active(state: CompositeGrammarState) -> bool:
    """Whether R4 changes the structural domain at `state`: the DEC-033 empty
    paragraph bound has been reached."""
    return (
        state.mode == "structure" and state.empty_paragraph_run >= MAX_CONSECUTIVE_EMPTY_PARAGRAPHS
    )


@dataclass(frozen=True, slots=True)
class ForceContentController(CompositeGrammarController):
    """R4: at the empty-paragraph bound keep `PARAGRAPH_OPEN` legal (where the
    structural grammar allows it) and forbid closing a paragraph that is
    still empty, so the paragraph must carry content. Used only inside a v3
    decode; `composite_grammar_hash` is never computed from it, and every
    state below the bound gets exactly the parent's domain."""

    def _structural_domain(self, state: CompositeGrammarState) -> tuple[int, ...]:
        ids = CompositeGrammarController._structural_domain(self, state)
        if (
            not self._dec033_bounds_active
            or state.empty_paragraph_run < MAX_CONSECUTIVE_EMPTY_PARAGRAPHS
        ):
            return ids
        widened = set(ids)
        if (
            _PARAGRAPH_OPEN_ID not in widened
            and "PARAGRAPH_OPEN" in _grammar_names(state)
            and _PARAGRAPH_OPEN_ID not in self.capability_excluded_local_ids
        ):
            widened.add(_PARAGRAPH_OPEN_ID)
        if state.paragraph_pending_empty and _PARAGRAPH_CLOSE_ID in widened:
            narrowed = widened - {_PARAGRAPH_CLOSE_ID}
            if narrowed:
                widened = narrowed
        return tuple(sorted(widened))

    @classmethod
    def wrap(cls, controller: CompositeGrammarController) -> ForceContentController:
        return cls(
            vocabulary=controller.vocabulary,
            capability_profile_id=controller.capability_profile_id,
            structural_profile=controller.structural_profile,
            capability_excluded_local_ids=controller.capability_excluded_local_ids,
        )


def _grammar_names(state: CompositeGrammarState) -> frozenset[str]:
    from unflatten.model import grammar

    return frozenset(grammar.allowed_events(state.structural_state))


# -- the guard -------------------------------------------------------------------


@dataclass(frozen=True)
class Rewind:
    """R2's decision: truncate the page to `position` tokens and never emit
    `banned` at that position again."""

    position: int
    banned: int
    rule: str
    period: int
    count: int


class LoopGuardV3:
    """See the module docstring. `decide` is called once per structure/text
    step with the post-grammar legal set and returns a packed id, or a
    `Rewind` the decode loop must carry out (then call `rewind_to`)."""

    def __init__(
        self,
        *,
        ranges: PackedRanges,
        policy: LoopGuardV3Policy,
        arm: Arm,
        v2_caps: LoopGuardV2Caps,
    ) -> None:
        from unflatten.model.free_running_decode import LoopGuardV2

        self._ranges = ranges
        self._policy = policy
        self._arm = arm
        self._v2_caps = v2_caps
        self.v2 = LoopGuardV2(ranges=ranges, caps=v2_caps)
        self._max_period = policy.long_period_max if arm.long_period else policy.unit_period_max
        self._min_unit_cap = min(policy.text_free_unit_cap, policy.empty_run_unit_cap)
        #: `RUN_OPEN TEXT_BEGIN TEXT_END RUN_CLOSE`: an invisible empty run,
        #: capped separately (`empty_run_unit_cap`) until DEC-055 V1
        #: canonicalises chains of them in the v16 targets.
        self._empty_run = tuple(
            pack(StructuralRef(events_vocab.TOKEN_IDS[n]), ranges)
            for n in ("RUN_OPEN", "TEXT_BEGIN", "TEXT_END", "RUN_CLOSE")
        )
        self._reset_detection()
        self.bans: dict[int, set[int]] = {}
        self.rewinds = 0
        self.counters: dict[str, int] = {
            "unit_cap": 0,
            "cardinality": 0,
            "long_period": 0,
            "rewind": 0,
            "rewind_exhausted": 0,
            "force_content": 0,
            "noop": 0,
        }
        self.interventions: list[dict[str, Any]] = []

    def _reset_detection(self) -> None:
        self._hist: list[int] = []
        self._runs = [0] * (self._max_period + 1)
        #: per open table: [rows so far, cells in the current row]
        self._tables: list[list[int]] = []

    # -- classification (memoised by the wrapped v2 guard) --------------------

    def _name(self, packed_id: int) -> str | None:
        token = self.v2._token(packed_id)
        return _NAME_BY_ID[token.local_id] if isinstance(token, StructuralRef) else None

    def _is_text(self, packed_id: int) -> bool:
        return isinstance(self.v2._token(packed_id), QwenRef | ByteValueRef)

    def _is_content(self, packed_id: int) -> bool:
        """Text or an image leaf: a unit holding either is not "text-free"
        for R1 (25 back-to-back floating images on one genuine train page)."""
        if self._is_text(packed_id):
            return True
        token = self.v2._token(packed_id)
        return isinstance(token, StructuralRef) and token.local_id in _IMAGE_LEAVES

    # -- state ----------------------------------------------------------------------

    def observe(self, packed_id: int) -> None:
        """Accept `packed_id` (once per accepted token, after `controller.step`)."""
        self.v2.observe(packed_id)
        hist, runs = self._hist, self._runs
        n = len(hist)
        for p in range(1, min(self._max_period, n) + 1):
            runs[p] = runs[p] + 1 if packed_id == hist[n - p] else 0
        hist.append(packed_id)
        name = self._name(packed_id)
        if name == "TABLE_OPEN":
            self._tables.append([0, 0])
        elif name == "TABLE_CLOSE" and self._tables:
            self._tables.pop()
        elif name == "ROW_OPEN" and self._tables:
            self._tables[-1][0] += 1
            self._tables[-1][1] = 0
        elif name == "CELL_OPEN" and self._tables:
            self._tables[-1][1] += 1

    def rewind_to(self, rewind: Rewind) -> None:
        """Carry out `rewind` on the guard's own state: detection state is
        rebuilt by replaying the kept prefix; evidence (counters,
        interventions of both v3 and the wrapped v2) is kept."""
        from unflatten.model.free_running_decode import LoopGuardV2

        kept = self._hist[: rewind.position]
        old = self.v2
        fresh = LoopGuardV2(ranges=self._ranges, caps=self._v2_caps)
        fresh._tokens, fresh._windows, fresh._anchors = old._tokens, old._windows, old._anchors
        fresh.counters, fresh.interventions = old.counters, old.interventions
        self.v2 = fresh
        self._reset_detection()
        for packed_id in kept:
            self.observe(packed_id)
        self.bans = {p: b for p, b in self.bans.items() if p <= rewind.position}
        self.bans.setdefault(rewind.position, set()).add(rewind.banned)
        self.rewinds += 1

    # -- detectors -------------------------------------------------------------------

    def _repeat_event(self, x: int) -> tuple[str, int, int, int] | None:
        """`(rule, period, copies, start)` when `x` would start one more copy
        of a unit that already repeats back-to-back at the cap; `start` is the
        first token of the first counted copy."""
        hist, runs = self._hist, self._runs
        n = len(hist)
        policy = self._policy
        for p in range(1, min(self._max_period, n) + 1):
            if hist[n - p] != x:
                continue
            long = p > policy.unit_period_max
            copies = (runs[p] + p) // p
            if copies < (policy.long_period_reps if long else self._min_unit_cap):
                continue  # cheap pre-check before any window is built
            window = tuple(hist[n - p :])
            if long:
                need = policy.long_period_reps
                text = [self._is_text(t) for t in window]
                if not any(text) or all(text):
                    continue  # R3 is about blocks that carry content
            elif any(self._is_content(t) for t in window):
                continue
            elif window == self._empty_run * (p // 4):
                if p != len(self._empty_run):
                    continue  # counted in single empty runs at p == 4
                need = policy.empty_run_unit_cap
            else:
                need = policy.text_free_unit_cap
            if copies < need:
                continue
            _canon, net = self.v2._window(window)
            if net != 0:
                continue
            if self.v2._anchor(window) != 0:
                continue
            return ("long_period" if long else "unit_cap", p, copies, n - copies * p)
        return None

    def _cardinality_event(self, x: int) -> tuple[str, int, int] | None:
        if not self._tables:
            return None
        name = self._name(x)
        rows, cells = self._tables[-1]
        if name == "CELL_OPEN" and cells >= self._policy.max_cells_per_row:
            return ("cardinality", 0, cells)
        if name == "ROW_OPEN" and rows >= self._policy.max_rows_per_table:
            return ("cardinality", 0, rows)
        return None

    def _rewind_position(self, start: int) -> int:
        b = start
        while b > 0:
            token = self.v2._token(self._hist[b - 1])
            if not (isinstance(token, StructuralRef) and token.local_id in _REWIND_OPENERS):
                break
            b -= 1
        return b

    # -- the step -------------------------------------------------------------------

    def decide(
        self,
        logits_row: torch.Tensor,
        legal_ids: list[int],
        *,
        domain: Any,
        state: CompositeGrammarState,
        step: int,
    ) -> int | Rewind:
        from unflatten.model.free_running_decode import _select_greedy

        position = len(self._hist)
        banned = self.bans.get(position)
        if banned:
            narrowed = [i for i in legal_ids if i not in banned]
            if narrowed:
                legal_ids = narrowed
        if self._arm.force_content and force_content_active(state):
            self.counters["force_content"] += 1
            self.interventions.append({"step": step, "rule": "force_content"})

        greedy = _select_greedy(logits_row, legal_ids)
        recovered = self.v2._close_recovery(
            logits_row, legal_ids, domain=domain, state=state, step=step, greedy=greedy
        )
        if recovered is not None:
            return recovered

        chosen = greedy
        vetoed: list[int] = []
        first: tuple[str, str, int | None, int] | None = None  # (source, rule, period, count)
        while (event := self._event(chosen)) is not None:
            source, rule, period, count, start = event
            if start is not None and self._arm.rewind:
                if self.rewinds < self._policy.max_rewinds:
                    return self._rewind(step, rule, period, count, start)
                self.counters["rewind_exhausted"] += 1
            first = first or (source, rule, period, count)
            vetoed.append(chosen)
            excluded = set(vetoed)
            remaining = [i for i in legal_ids if i not in excluded]
            if not remaining:
                self.counters["noop"] += 1
                self.interventions.append({"step": step, "rule": "noop", "vetoed": vetoed})
                return greedy
            chosen = _select_greedy(logits_row, remaining)
        if first is not None:
            source, rule, period, count = first
            if source == "v2":
                self.v2._record(step, rule, greedy, chosen, vetoed, period=period, count=count)
            else:
                self.counters[rule] += 1
                self.interventions.append(
                    {
                        "step": step,
                        "rule": rule,
                        "greedy": greedy,
                        "chosen": chosen,
                        "vetoed": list(vetoed),
                        "period": period,
                        "count": count,
                    }
                )
        return chosen

    def _rewind(self, step: int, rule: str, period: int, count: int, start: int) -> Rewind:
        position = self._rewind_position(start)
        decision = Rewind(
            position=position, banned=self._hist[position], rule=rule, period=period, count=count
        )
        self.counters[rule] += 1
        self.counters["rewind"] += 1
        self.interventions.append(
            {
                "step": step,
                "rule": "rewind",
                "detector": rule,
                "period": period,
                "count": count,
                "to": position,
                "banned": decision.banned,
            }
        )
        return decision

    def _event(self, x: int) -> tuple[str, str, int | None, int, int | None] | None:
        """`(source, rule, period, count, repetition start or None)` for the
        first rule that refuses `x`: v2 first (unchanged), then R1/R3's
        repetition detectors, then R1's cardinality."""
        event_v2 = self.v2.veto(x)
        if event_v2 is not None:
            return ("v2", *event_v2, None)
        repeat = self._repeat_event(x)
        if repeat is not None:
            return ("v3", *repeat)
        card = self._cardinality_event(x)
        if card is not None:
            return ("v3", *card, None)
        return None

    def veto(self, packed_id: int) -> tuple[str, int | None, int] | None:
        """Would any rule refuse `packed_id` next? `(rule, period, count)` or
        `None` -- the gold-replay question (DEC-055 criterion 2)."""
        event = self._event(packed_id)
        return None if event is None else event[1:4]

    # -- evidence -------------------------------------------------------------------

    def loop_interventions(self) -> int:
        """Interventions that mean the page looped (§4.2: such a page is never
        a clean success): every v3 rule plus v2's text-run/periodic veto."""
        v2_loop = sum(1 for i in self.v2.interventions if i["rule"] in ("text_run", "periodic"))
        v3_loop = sum(1 for i in self.interventions if i["rule"] != "force_content")
        return v2_loop + v3_loop

    def record(self) -> dict[str, Any]:
        """The `loop_guard_v3` issue entry; v2's own entry is nested whole."""
        return {
            "kind": "loop_guard_v3",
            "policy_id": POLICY_ID,
            "policy_status": self._policy.status,
            "policy_sha256": self._policy.sha256,
            "arm": self._arm.name,
            "caps_sha256": self._v2_caps.sha256,
            "counters": dict(self.counters),
            "interventions": [dict(i) for i in self.interventions],
            "loop_interventions": self.loop_interventions(),
            "v2": self.v2.record({}),
        }
