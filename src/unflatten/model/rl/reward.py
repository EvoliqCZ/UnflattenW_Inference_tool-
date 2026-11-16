"""S4.5 reward engine: lexicographic, hard gates first (RFC005-ERRATA-010 E3.4).

Order is the contract, not a convenience:

1. Every hard gate is evaluated and recorded -- missing/invented visible text
   and control events (against the GOLD text authority, never against the
   candidate's own assembly), invalid/partial/failed strict AST,
   resource/fragment integrity, severe structural error.
2. Only when all four pass is a scalar computed, as an integer-micro weighted
   mean over the *observable* components. An unobservable component is
   recorded and excluded; observed and total weight are both stored so the
   effective denominator is never silent. No observable component => ``None``
   (``insufficient_evidence``), never ``0``.

A failed candidate is never repaired and never retried. The pixel component
stays off: there is no pinned renderer/font lock (RFC006-ERRATA-001 E2), and a
missing render is ``render_missing``, never a perfect match.

The deterministic RFC002 render (AST -> DOCX) is a fifth hard gate, appended
after the four above so their order and precedence do not move: a candidate the
renderer refuses is ``gate_failed:rfc002_render`` (never repaired, never a
match); ``not_run`` is a missing observation like an unfrozen floor
(``insufficient_evidence``, null scalar) unless another hard gate failed. The
LibreOffice open probe is NOT part of the sealed evaluation (D6): its result
is a sibling record keyed by ``candidate_evaluation_hash``.

No float is returned or stored; nothing here renders, decodes or loads a model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Final

from unflatten.eval.s2_gate.distance import levenshtein
from unflatten.eval.s2_gate.score import normalize_for_approx_match
from unflatten.model.rl.ids import CANDIDATE_EVALUATION_DOMAIN
from unflatten.model.rl.policies import (
    HARD_GATES,
    REWARD_COMPONENTS,
    RlPolicyInvalid,
    _seal,
    _verify,
    verify_reward_policy,
)
from unflatten.model.rl.reward_inputs import (
    RFC002_RENDER_STATES,
    CandidateInputs,
    GoldReference,
    ast_run_text,
    grid_cells_from_ast,
    preorder_from_ast,
)
from unflatten.model.rl.scorers import (
    MICRO,
    table_grid_f1_micro,
    tree_edit_distance,
    tree_node_f1_micro,
    typed_value_score_micro,
)

SCORER_VERSION: Final = "rl-reward-engine/0.1.0"
SCHEMA_VERSION: Final = "1.0.0"
EVALUATION_HASH_FIELD: Final = "candidate_evaluation_hash"
_SCHEMA_NAME: Final = "rl_candidate_evaluation"
_BUILD_STATUSES: Final = ("built", "failed")
_TEXT, _TREE, _TYPED, _GRID, _CONT, _RENDER = REWARD_COMPONENTS


class PixelComponentLocked(RuntimeError):
    """The pixel/perceptual reward component cannot be enabled."""


class ChainPolicyMismatch(ValueError):
    """The candidate was produced under a chain-flag set the policy does not pin."""


def enable_pixel_component(*_args: Any, **_kwargs: Any) -> None:
    """Always refuses: no renderer/font lock is pinned (RFC006-ERRATA-001 E2)."""
    raise PixelComponentLocked(
        "pixel_component stays 'off': the renderer/font lock (RFC005 s9.5) is not pinned, "
        "so a pixel reward would be an uncertain reward (RFC006-ERRATA-001 E2)"
    )


# --- hard gates --------------------------------------------------------------

_SEVERE_ARMS: Final = ("tree", "grid")
_EVIDENCE_KEYS: Final = {
    "missing_or_invented_text": frozenset(
        {"missing_count", "invented_count", "control_missing", "control_invented"}
    ),
    "invalid_ast": frozenset(),
    "resource_fragment_integrity": frozenset({"violations"}),
    "rfc002_render": frozenset({"render_failed", "render_not_run"}),
}


_TOLERANT_TEXT_KEYS: Final = frozenset(
    {"text_cer_ppm", "text_max_ppm", "control_missing", "control_invented"}
)


def derive_gate_status(gate: str, evidence: Mapping[str, int], *, built: bool = True) -> str:
    """The one place a gate's status is derived from its recorded evidence.

    The engine calls it when it builds a gate record and the verifier calls it
    again on a stored one, so the two cannot drift. ``built`` only matters for
    ``invalid_ast`` (whose evidence is empty by design).
    """
    if gate == "invalid_ast":
        return "pass" if built else "fail"
    if gate == "rfc002_render":
        if evidence["render_failed"]:
            return "fail"
        return "unmeasured" if evidence["render_not_run"] else "pass"
    if gate == "severe_structural_error":
        if not evidence:  # no AST to measure
            return "unmeasured"
        below = sum(v for k, v in evidence.items() if k.endswith("below_floor"))
        unset = sum(v for k, v in evidence.items() if k.endswith("floor_unset"))
        return "fail" if below else "unmeasured" if unset else "pass"
    if gate == "missing_or_invented_text" and "text_max_ppm" in evidence:
        controls = evidence["control_missing"] + evidence["control_invented"]
        return "fail" if controls or evidence["text_cer_ppm"] > evidence["text_max_ppm"] else "pass"
    return "fail" if any(evidence.values()) else "pass"


def _multiset_diff(reference: Mapping[Any, int], candidate: Mapping[Any, int]) -> tuple[int, int]:
    ref, cand = Counter(reference), Counter(candidate)
    return sum((ref - cand).values()), sum((cand - ref).values())


def _text_gate(
    gold: GoldReference,
    cand: CandidateInputs,
    text: str,
    block: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate = "missing_or_invented_text"
    if block is not None:
        evidence = _tolerant_text_evidence(gold, cand, text, block)
        return {"gate": gate, "status": derive_gate_status(gate, evidence), "evidence": evidence}
    missing, invented = _multiset_diff(Counter(gold.visible_text), Counter(text))
    c_missing, c_invented = _multiset_diff(gold.control_event_counts, cand.control_event_counts)
    evidence = {
        "missing_count": missing,
        "invented_count": invented,
        "control_missing": c_missing,
        "control_invented": c_invented,
    }
    gate = "missing_or_invented_text"
    return {"gate": gate, "status": derive_gate_status(gate, evidence), "evidence": evidence}


def _tolerant_text_evidence(
    gold: GoldReference, cand: CandidateInputs, text: str, block: Mapping[str, Any]
) -> dict[str, int]:
    """DEC-030 CER (integer ppm) of the text plus an exact match of the named control events."""
    ref, hyp = normalize_for_approx_match(gold.visible_text), normalize_for_approx_match(text)
    distance = levenshtein(hyp, ref)
    ppm = distance * MICRO // len(ref) if ref else (0 if distance == 0 else MICRO)
    exact = set(block["control_events_exact"])
    c_missing, c_invented = _multiset_diff(
        {k: v for k, v in gold.control_event_counts.items() if k in exact},
        {k: v for k, v in cand.control_event_counts.items() if k in exact},
    )
    total = set(block.get("control_events_total", ()))  # compared by summed count
    gold_sum = sum(v for k, v in gold.control_event_counts.items() if k in total)
    cand_sum = sum(v for k, v in cand.control_event_counts.items() if k in total)
    c_missing += max(0, gold_sum - cand_sum)
    c_invented += max(0, cand_sum - gold_sum)
    return {
        "text_cer_ppm": ppm,
        "text_max_ppm": block["max_ppm"],
        "control_missing": c_missing,
        "control_invented": c_invented,
    }


def _ast_gate(built: bool) -> dict[str, Any]:
    return {
        "gate": "invalid_ast",
        "status": derive_gate_status("invalid_ast", {}, built=built),
        "evidence": {},
    }


def _integrity_gate(cand: CandidateInputs) -> dict[str, Any]:
    gate = "resource_fragment_integrity"
    evidence = {"violations": len(cand.fragment_violations)}
    return {"gate": gate, "status": derive_gate_status(gate, evidence), "evidence": evidence}


def _render_gate(state: str) -> dict[str, Any]:
    """``rfc002_render``: exactly one of rendered / failed / not_run is recorded."""
    gate = "rfc002_render"
    evidence = {"render_failed": int(state == "failed"), "render_not_run": int(state == "not_run")}
    return {"gate": gate, "status": derive_gate_status(gate, evidence), "evidence": evidence}


def _severe_gate(
    gold: GoldReference,
    preorder: tuple[tuple[int, str], ...] | None,
    grid_micro: int | None,
    floors: Mapping[str, int | None],
) -> dict[str, Any]:
    """``severe_structural_error``: a structural score below a policy-owned floor.

    Each arm (tree node F1, grid F1) fails when its measured score is strictly
    below the policy floor. A floor of ``None`` means the threshold is not
    frozen: that arm is ``unmeasured`` (non-pass, never a scalar), and a failing
    arm outranks an unmeasured one (the engine then reports
    ``insufficient_evidence``, not ``gate_failed``). An arm with no observable reference
    (``None`` score) has nothing to be severe about and passes.
    """
    if preorder is None:
        gate = "severe_structural_error"
        return {"gate": gate, "status": derive_gate_status(gate, {}), "evidence": {}}
    arms = {
        "tree": (tree_node_f1_micro(gold.structure, preorder), floors["tree_f1_floor_micro"]),
        "grid": (grid_micro, floors["grid_f1_floor_micro"]),
    }
    evidence: dict[str, int] = {}
    for name, (score, floor) in arms.items():
        severe = floor is not None and score is not None and score < floor
        evidence[f"{name}_f1_below_floor"] = int(severe)
        evidence[f"{name}_floor_unset"] = int(floor is None)
    gate = "severe_structural_error"
    return {"gate": gate, "status": derive_gate_status(gate, evidence), "evidence": evidence}


# --- components --------------------------------------------------------------


def _text_micro(gold_text: str, text: str) -> int | None:
    if not gold_text:
        return None
    if gold_text == text:
        return MICRO
    return max(0, MICRO - levenshtein(text, gold_text) * MICRO // len(gold_text))


def _tree_micro(gold: GoldReference, preorder: tuple[tuple[int, str], ...]) -> int | None:
    f1 = tree_node_f1_micro(gold.structure, preorder)
    distance = tree_edit_distance(gold.structure, preorder)
    if f1 is None or distance is None:
        return None
    span = max(len(gold.structure), len(preorder))
    similarity = max(0, MICRO - distance * MICRO // span)
    return (f1 + similarity) // 2


def _checked_micro(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int in micro units or None, got {type(value).__name__}")
    if not 0 <= value <= MICRO:
        raise ValueError(f"{name} must be within 0..{MICRO}, got {value}")
    return value


def _components(
    policy: Mapping[str, Any],
    micros: Mapping[str, int | None] | None,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    weights = policy["component_weights_micro"]
    out: dict[str, dict[str, Any]] = {}
    observed = 0
    for name in REWARD_COMPONENTS:
        weight = weights[name]
        if micros is None:
            state, micro = "not_computed", None
        elif name == _RENDER:
            state, micro = "off", None
        elif micros[name] is None:
            state, micro = "unobservable", None
        else:
            state, micro = "scored", micros[name]
            observed += weight
        out[name] = {"state": state, "micro": micro, "weight_micro": weight}
    return out, observed, sum(weights.values())


def _scalar(components: Mapping[str, Mapping[str, Any]], observed: int) -> int | None:
    if observed == 0:
        return None
    total = sum(
        c["micro"] * c["weight_micro"] for c in components.values() if c["state"] == "scored"
    )
    return total // observed


# --- engine ------------------------------------------------------------------


def score_candidate(
    cand: CandidateInputs,
    gold: GoldReference,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one candidate; returns a sealed ``rl_candidate_evaluation``."""
    verify_reward_policy(policy)
    # defence in depth: the schema const already pins pixel_component to "off"
    if policy["pixel_component"] != "off":
        raise RlPolicyInvalid("pixel_component must be 'off'")
    if dict(cand.chain_flags) != dict(policy["chain_policy_flags"]):
        diff = sorted(
            k
            for k in set(cand.chain_flags) | set(policy["chain_policy_flags"])
            if cand.chain_flags.get(k) != policy["chain_policy_flags"].get(k)
        )
        raise ChainPolicyMismatch(
            f"candidate chain flags differ from the policy's pinned set at {diff}: "
            "not comparable to the baseline"
        )
    continuity = _checked_micro(cand.continuity_micro, "continuity_micro")
    if cand.rfc002_render not in RFC002_RENDER_STATES:
        raise ValueError(f"rfc002_render {cand.rfc002_render!r} not in {RFC002_RENDER_STATES}")

    build = cand.build_candidate()
    if build.status not in _BUILD_STATUSES:
        raise ValueError(f"build status {build.status!r} not in {_BUILD_STATUSES}")
    built = build.status == "built" and build.ast is not None
    ast = build.ast if built else None
    if not built and cand.rfc002_render != "not_run":
        raise ValueError(
            f"candidate did not build, so rfc002_render must be 'not_run', "
            f"got {cand.rfc002_render!r}"
        )
    if cand.visible_text is not None:
        text = cand.visible_text
    else:
        text = ast_run_text(ast) if ast is not None else ""
    preorder = preorder_from_ast(ast) if ast is not None else None
    grid_micro = (
        table_grid_f1_micro(gold.grid_cells, grid_cells_from_ast(ast)) if ast is not None else None
    )

    gates = [
        _text_gate(gold, cand, text, policy.get("text_gate")),
        _ast_gate(built),
        _integrity_gate(cand),
        _severe_gate(gold, preorder, grid_micro, policy["severe_structural_error"]),
        _render_gate(cand.rfc002_render),
    ]
    assert [g["gate"] for g in gates] == list(HARD_GATES)
    failed = [g["gate"] for g in gates if g["status"] == "fail"]
    # a gate that could not be measured (threshold not frozen) is a missing
    # observation, never a failure: it must not become a numeric reward
    unmeasured = [g["gate"] for g in gates if g["status"] == "unmeasured"]

    micros: dict[str, int | None] | None = None
    if not failed and not unmeasured:
        assert preorder is not None
        micros = {
            _TEXT: _text_micro(gold.visible_text, text),
            _TREE: _tree_micro(gold, preorder),
            _TYPED: typed_value_score_micro(gold.typed, cand.typed, gold.typed_observable),
            _GRID: grid_micro,
            _CONT: continuity,
            _RENDER: None,
        }
    components, observed, total = _components(policy, micros)
    scalar = _scalar(components, observed) if micros is not None else None
    if failed:
        status = f"gate_failed:{failed[0]}"
    elif unmeasured:
        status = "insufficient_evidence"
    else:
        status = "scored" if scalar is not None else "insufficient_evidence"
    return build_candidate_evaluation(
        {
            "schema_version": SCHEMA_VERSION,
            "scorer_version": SCORER_VERSION,
            "document_id": gold.document_id,
            "candidate_id": cand.candidate_id,
            "reward_policy_hash": policy["reward_policy_hash"],
            "chain_policy_flags": dict(cand.chain_flags),
            "gates": gates,
            "status": status,
            "scalar_micro": scalar,
            "components": components,
            "weight_observed_micro": observed,
            "weight_total_micro": total,
        }
    )


# --- artefact ----------------------------------------------------------------


def build_candidate_evaluation(body: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(_SCHEMA_NAME, CANDIDATE_EVALUATION_DOMAIN, EVALUATION_HASH_FIELD, dict(body))


def _verify_gate_statuses(gates: list[Mapping[str, Any]]) -> None:
    """Each recorded status must equal the one derived from its own evidence."""
    by_name = {g["gate"]: g for g in gates}
    severe_evidence = by_name["severe_structural_error"]["evidence"]
    for g in gates:
        name, evidence = g["gate"], g["evidence"]
        expected_keys = _EVIDENCE_KEYS.get(name)
        if name == "missing_or_invented_text" and "text_max_ppm" in evidence:
            expected_keys = _TOLERANT_TEXT_KEYS
        if expected_keys is None:  # severe: empty (no AST) or one pair per arm
            expected_keys = frozenset(
                f"{arm}_{suffix}"
                for arm in _SEVERE_ARMS
                for suffix in ("f1_below_floor", "floor_unset")
            )
            keys_ok = not evidence or set(evidence) == expected_keys
        else:
            keys_ok = set(evidence) == expected_keys
        if not keys_ok:
            raise RlPolicyInvalid(f"gate {name!r} evidence keys {sorted(evidence)} are not valid")
        # the severe gate records evidence only when an AST was built
        derived = derive_gate_status(name, evidence, built=bool(severe_evidence))
        if g["status"] != derived:
            raise RlPolicyInvalid(
                f"gate {name!r}: derived {derived!r} from its evidence {dict(evidence)}, "
                f"recorded {g['status']!r}"
            )
        if name == "rfc002_render" and sum(evidence.values()) > 1:
            raise RlPolicyInvalid(
                f"gate {name!r}: render is exactly one of rendered/failed/not_run, "
                f"got evidence {dict(evidence)}"
            )
    if (
        by_name["invalid_ast"]["status"] == "fail"
        and not (by_name["rfc002_render"]["evidence"]["render_not_run"])
    ):
        raise RlPolicyInvalid(
            "an AST that did not build cannot have been rendered: not_run expected"
        )


def verify_candidate_evaluation(doc: Mapping[str, Any]) -> None:
    _verify(_SCHEMA_NAME, CANDIDATE_EVALUATION_DOMAIN, EVALUATION_HASH_FIELD, doc)
    if [g["gate"] for g in doc["gates"]] != list(HARD_GATES):
        raise RlPolicyInvalid(f"gates must be recorded in order {HARD_GATES}")
    _verify_gate_statuses(doc["gates"])
    failed = [g["gate"] for g in doc["gates"] if g["status"] == "fail"]
    unmeasured = [g["gate"] for g in doc["gates"] if g["status"] == "unmeasured"]
    status, scalar = doc["status"], doc["scalar_micro"]
    if failed:
        if status != f"gate_failed:{failed[0]}":
            raise RlPolicyInvalid(f"status {status!r} does not name failed gate {failed[0]!r}")
        if scalar is not None:
            raise RlPolicyInvalid("scalar must be null when a gate failed")
    elif unmeasured:
        if status != "insufficient_evidence" or scalar is not None:
            raise RlPolicyInvalid(
                f"gates {unmeasured} are unmeasured: status must be insufficient_evidence "
                "with a null scalar"
            )
    elif status == "scored":
        if scalar is None:
            raise RlPolicyInvalid("scored evaluation needs an integer scalar")
    elif status != "insufficient_evidence" or scalar is not None:
        raise RlPolicyInvalid(f"status {status!r} inconsistent with passed gates and scalar")
