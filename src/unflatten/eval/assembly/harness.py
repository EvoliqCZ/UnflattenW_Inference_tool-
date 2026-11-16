"""Per-document evaluation harness for the S4 readiness gate, Track A
(assembler design §8, §10; RFC005-ERRATA-006 gate A rows A1–A3).

For one multi-page document of an RFC003 release the harness

1. builds every page through the PRODUCTION standalone-page path
   (`build_exact_text_cache.row_for_page`, stub features, floor 0.0) and
   turns the row's packed token ids into a `PageHypothesis` with the ONE
   adapter a model decode uses (`assembly.hypothesis.from_token_ids`),
   typed values = the page's supervised typed targets by anchor; a page the
   row path skips is a hole with its skip reason;
2. builds the gold reference `R(doc)` (`reference.build`) from the same
   fragments, projection drops, text authorities and typed targets;
3. assembles the pages under EVERY policy in the pass (B0 and B00, design
   §10 "always scored in the same pass") and scores each policy's seam
   decisions against the gold seam labels per rule class -- the primary
   G-A1 number. Matching is by gold node id on both sides of the seam; the
   page-local -> gold id map is built here from the row's
   `source_event_index`, so nothing gold reaches the hypothesis;
4. records `AssemblyInvariantViolation` (text conservation = gate A3,
   ownership, grammar replay) as the policy's status, never repairs it.

Output is one JSONL row per document (append-only, resumable); the summary
(`compute_summary`) is computed from the rows, so `summary --ledger` and
the end of `eval` print the same thing, with a lineage-cluster bootstrap
lower bound per class (design §10 "confidence bound").

The T5 harness this replaces read a store layout that does not exist
(top-level `document_bundle_id`/`pages` rows in `index.jsonl`), passed
`typed_targets=None` for every page (which `reference._validate` refuses)
and compared the reference against an empty hypothesis; see
`docs/plans/spec-s4-assembler-m1-results.md` §1.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from unflatten.assembly.assemble import (
    AssembledDocumentHypothesis,
    AssemblyInvariantViolation,
    SeamDecision,
    assemble,
)
from unflatten.assembly.hypothesis import (
    PageHypothesis,
    PageInput,
    PageStreamInvalid,
    TypedValue,
    from_token_ids,
)
from unflatten.assembly.policy import POLICY_B0, POLICY_B00, JoinPolicy
from unflatten.assembly.stories import STORY_UNRESOLVED
from unflatten.eval.assembly.reference import (
    ATTRIBUTION_KEYS,
    EXCLUDED_KEYS,
    ReferenceError,
    Seam,
    SeamLabel,
    build,
)
from unflatten.model import composite_vocab as cv
from unflatten.model import events_vocab as ev

#: JSONL statuses that are final on resume; a last row with any other status
#: is re-attempted.
_RESUME_FINAL_STATUSES: frozenset[str] = frozenset({"done", "failed"})

#: The two fixed baseline policies (design §10). B0 is also the de facto
#: frozen policy until T6 adopts a variant B.
DEFAULT_POLICIES: dict[str, JoinPolicy] = {"B0": POLICY_B0, "B00": POLICY_B00}

#: G-A1 reporting classes (design §8); `mixed` is a block seam whose two
#: edge blocks have different types (gold never joins them, no policy can).
RULE_CLASSES: tuple[str, ...] = (
    "section", "paragraph", "table", "row", "list", "list_item", "run", "mixed",
)
_RULE_CLASS_BY_RULE: dict[str, str] = {
    "SEC": "section", "PARA": "paragraph", "TABLE": "table", "ROW": "row",
    "LIST": "list", "ITEM": "list_item", "RUN": "run",
}
_BLOCK_CLASSES = frozenset({"paragraph", "table", "list"})

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260926
_ALPHA = 0.05

_TOKEN_NAME_BY_ID: dict[int, str] = {i: name for name, i in ev.TOKEN_IDS.items()}
_NODE_TOKEN_NAMES = frozenset(ev.OPEN_TOKEN.values()) | frozenset(ev.LEAF_TOKEN.values())
_HYPHENS = frozenset("-‐‑­‒–")


class HarnessError(RuntimeError):
    """A run-level refusal -- never a per-document failure."""


class DocumentError(RuntimeError):
    """A per-document processing failure -- lands as a typed JSONL record."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredDecision:
    """One gold seam label with the assembler's answer at the same pair.
    `decided` is False when the assembler never reached this pair (a parent
    it did not join), which counts as "no join"."""

    seam: tuple[int, int]
    rule_class: str
    gold_a: str
    gold_b: str
    gold_join: bool
    predicted: bool
    decided: bool


@dataclass(frozen=True)
class PolicyOutcome:
    policy: str
    status: str  # assembled | invariant_violation | error
    detail: str | None
    conflicts: int
    joins: int
    unlabelled_decisions: int
    labels_at_dropped_nodes: int
    stories: dict[str, int]
    decisions: tuple[ScoredDecision, ...]


@dataclass(frozen=True)
class DocumentResult:
    document_bundle_id: str
    lineage_root_id: str | None
    n_pages: int
    holes: dict[int, str]
    status: str  # done | failed
    error: str | None
    seams_scored: int
    seams_hole: int
    reference_excluded: dict[str, int] | None
    reference_attribution: dict[str, int] | None
    split_run_left_ends: dict[str, int]
    policies: tuple[PolicyOutcome, ...]


@dataclass(frozen=True)
class GoldPages:
    """Everything one document's pages give the harness (design §8 steps
    1–2). Parallel per page; a hole is `None` in every per-page sequence."""

    targets: dict[str, Any]
    ast: dict[str, Any]
    fragments: list[dict[str, Any] | None]
    dropped: list[frozenset[str] | None]
    text_spans: list[Any]
    typed_targets: list[Any]
    hole_reasons: list[str | None]
    hypotheses: list[PageHypothesis | None]
    gold_ids: list[dict[str, str] | None]


# ---------------------------------------------------------------------------
# Store -> gold pages (production row path)
# ---------------------------------------------------------------------------


def walk(nodes: Sequence[Any]) -> Iterator[Any]:
    for node in nodes:
        yield node
        yield from walk(getattr(node, "children", ()))


def gold_ids_by_local(
    hypothesis: PageHypothesis,
    token_ids: Sequence[int],
    source_event_index: Sequence[int],
    events: Sequence[Mapping[str, Any]],
    *,
    vocabulary: cv.CompositeVocabulary,
) -> dict[str, str]:
    """Page-local id -> gold node id, by pairing the hypothesis preorder
    with the open/leaf tokens of the stream (the parser assigns ids in token
    order). Lives in the harness so the hypothesis never carries an id."""
    node_ids: list[str] = []
    for position, token_id in enumerate(token_ids):
        ref = vocabulary.unpack(int(token_id))
        if not isinstance(ref, cv.StructuralRef):
            continue
        if _TOKEN_NAME_BY_ID.get(ref.local_id) not in _NODE_TOKEN_NAMES:
            continue
        index = int(source_event_index[position])
        if index < 0:
            raise DocumentError(f"node token at position {position} carries no source event")
        node_ids.append(events[index]["node_id"])
    local_ids = [node.local_id for node in walk([*hypothesis.sections, *hypothesis.notes])]
    if len(local_ids) != len(node_ids):
        raise DocumentError(
            f"{len(local_ids)} parsed nodes but {len(node_ids)} node tokens with an origin"
        )
    return dict(zip(local_ids, node_ids, strict=True))


def typed_by_anchor(
    targets: Iterable[Any], cardinalities: Iterable[Any]
) -> dict[int, dict[str, TypedValue]]:
    """Supervised typed targets keyed by their anchor token position, plus
    each supervised repeated container's element count at the container
    pointer (the cardinality channel, as `builder_chain.gold._gold_typed`
    writes it)."""
    out: dict[int, dict[str, TypedValue]] = {}
    for target in targets:
        if target.supervision_mask:
            out.setdefault(target.anchor_token_index, {})[target.node_property_pointer] = (
                target.value
            )
    for card in cardinalities:
        if card.supervision_mask:
            out.setdefault(card.anchor_token_index, {})[card.container_pointer] = card.count
    return out


def build_bound(profile: Any) -> tuple[Any, Any]:
    """`(MaterializerBundle, tokenizer)` for `profile`: the real DEC-007
    tokenizer/allowlist lineage and the same diagnostic bindings
    `scripts/census/multipage_admission_repeat.py` replays under (every
    hash from its in-repo producer; `component_lock_hash` a placeholder no
    row-path reader consumes; `head_registry_hash` null -- a diagnostic
    materialization never derives it, because a derived non-null value
    would be a binding site `test_policy_producers` forbids outside
    `verify_bindings_freshness`)."""
    from unflatten.model import build_exact_text_cache as bxc
    from unflatten.model import composite_grammar as cg
    from unflatten.model.conditioning_policy import (
        conditioning_policy_hash,
        load_conditioning_policy,
    )
    from unflatten.model.loss_unit import loss_unit_schema_hash
    from unflatten.model.runtime_limits import (
        load_runtime_limits_policy,
        runtime_limits_policy_hash,
    )
    from unflatten.model.stage_policy import load_stage_policy, stage_policy_hash

    binding, tokenizer, _width = bxc._real_binding(None)
    vocabulary = cv.make_composite_vocabulary(
        tokenizer_artifact_set_hash=binding.tokenizer_artifact_set_hash,
        allowlist_payload=binding.allowlist,
    )
    controller = cg.CompositeGrammarController(
        vocabulary=vocabulary,
        capability_profile_id=profile.profile_id,
        structural_profile=profile.ast_profile,
        capability_excluded_local_ids=profile.excluded_structural_local_ids,
    )
    policies_dir = Path(bxc.__file__).parent / "policies"
    limits = load_runtime_limits_policy(bxc.RUNTIME_LIMITS_POLICY_PATH)
    bindings = {
        "component_lock_hash": "sha256:" + "0" * 64,
        "runtime_limits_policy_hash": runtime_limits_policy_hash(limits),
        "composite_grammar_hash": cg.composite_grammar_hash(
            cg.build_composite_grammar_descriptor(
                controller=controller, profile=profile, runtime_limits_policy=limits
            )
        ),
        "loss_unit_schema_hash": loss_unit_schema_hash(),
        "stage_policy_hash": stage_policy_hash(
            load_stage_policy(policies_dir / "s1_stage_policy.json")
        ),
        "conditioning_policy_hash": conditioning_policy_hash(
            load_conditioning_policy(policies_dir / "s1_conditioning_policy.json")
        ),
        "head_registry_hash": None,
    }
    bound = bxc.build_materializer_bundle(
        binding=binding, tokenizer=tokenizer, bindings=bindings, capability_profile=profile
    )
    return bound, tokenizer


def load_gold_pages(
    store_dir: Path,
    samples: Sequence[Any],
    ast: dict[str, Any],
    *,
    bound: Any,
    tokenizer: Any,
) -> GoldPages:
    """Design §8 steps 1–2 over `samples` (one document's `PageSample`s,
    sorted by page index, contiguous from 0)."""
    import torch

    from unflatten.dataset.items import build_item
    from unflatten.model import build_exact_text_cache as bxc
    from unflatten.model.text_authority import (
        TextAuthorityAlignmentError,
        build_text_span_authorities,
    )

    stub_features = torch.zeros(1, 8, dtype=torch.float16)
    stub_grid = torch.tensor([1, 1, 1], dtype=torch.int64)
    pages = GoldPages(
        targets={}, ast=ast, fragments=[], dropped=[], text_spans=[], typed_targets=[],
        hole_reasons=[], hypotheses=[], gold_ids=[],
    )
    if [s.page_index for s in samples] != list(range(len(samples))):
        raise DocumentError(
            f"pages are not contiguous from 0: {[s.page_index for s in samples]}"
        )
    for sample in samples:
        item = build_item(store_dir, sample, with_geometry=True)
        if not pages.targets:
            pages.targets.update(item.targets)
        fragment = item.fragment
        mask_projection = item.mask_projection or {}
        mask_hash = (sample.mask_projection or {}).get("sha256")
        schema_versions = bxc.page_schema_versions(item.targets, fragment, mask_projection)
        row = skip = None
        ledger = None
        if fragment is None or not fragment.get("slices"):
            skip = "no_slices"
        elif mask_hash is None or schema_versions is None:
            skip = "skipped_unencodable"
        else:
            row, skip, _coverage, _typed, ledger, _region = bxc.row_for_page(
                document_bundle_id=sample.document_bundle_id,
                page_index=sample.page_index,
                targets=item.targets,
                fragment=fragment,
                features=stub_features,
                grid_thw=stub_grid,
                mask_projection=item.mask_projection,
                mask_projection_hash=mask_hash,
                mask_projection_schema_version=mask_projection.get(
                    "mask_projection_schema_version"
                ),
                page_font_pack_hash=bxc.render_font_pack_hash(store_dir, sample),
                bound=bound,
                schema_versions=schema_versions,
                observation_sidecar=item.observation_sidecar_data,
                min_text_coverage=0.0,
            )
        if row is None or ledger is None:
            _append_hole(pages, skip or "skipped_unknown")
            continue
        try:
            text_spans = build_text_span_authorities(
                document_bundle_id=sample.document_bundle_id,
                target_bundle_hash=item.targets["target_bundle_hash"],
                text=item.targets.get("text") or {"spans": []},
                fragment=fragment,
                ownership_projection_hash=item.targets["target_bundle_hash"],
            )
        except TextAuthorityAlignmentError as exc:  # row_for_page passed, so unexpected
            raise DocumentError(f"page {sample.page_index}: text authority: {exc}") from exc
        token_ids = row.packed_token_ids.tolist()
        size = fragment["pages"][0]
        events = item.targets["structural"]["events"]
        typed = typed_by_anchor(row.typed_targets, row.typed_cardinalities)
        try:
            hypothesis = from_token_ids(
                token_ids,
                vocabulary=bound.vocabulary,
                tokenizer=tokenizer,
                typed_by_anchor=typed,
                page_input=PageInput(
                    page_ordinal=sample.page_index,
                    width_px=int(size["width_px"]),
                    height_px=int(size["height_px"]),
                ),
            )
        except PageStreamInvalid as exc:
            raise DocumentError(f"page {sample.page_index}: {exc}") from exc
        pages.fragments.append(fragment)
        pages.dropped.append(frozenset(ledger["dropped_node_ids"]))
        pages.text_spans.append(text_spans)
        pages.typed_targets.append(row.typed_targets)
        pages.hole_reasons.append(None)
        pages.hypotheses.append(hypothesis)
        pages.gold_ids.append(
            gold_ids_by_local(
                hypothesis, token_ids, row.source_event_index.tolist(), events,
                vocabulary=bound.vocabulary,
            )
        )
    return pages


def _append_hole(pages: GoldPages, reason: str) -> None:
    pages.fragments.append(None)
    pages.dropped.append(None)
    pages.text_spans.append(None)
    pages.typed_targets.append(None)
    pages.hole_reasons.append(reason)
    pages.hypotheses.append(None)
    pages.gold_ids.append(None)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _rule_class(label: SeamLabel) -> str:
    if label.level == "block":
        if label.a_node_type == label.b_node_type and label.a_node_type in _BLOCK_CLASSES:
            return label.a_node_type
        return "mixed"
    return label.level


def score(
    seams: Sequence[Seam],
    decisions: Sequence[SeamDecision],
    gold_ids: Sequence[Mapping[str, str] | None],
    dropped: Sequence[frozenset[str] | None] | None = None,
) -> tuple[list[ScoredDecision], int, int]:
    """`(scored, unlabelled, labels_at_dropped_nodes)`: gold labels x
    assembler decisions, matched by (seam, gold id left, gold id right).

    A label the assembler never decided is predicted "no join". A decision
    at a pair gold did not label is still scored -- its truth is the gold
    definition itself, "the same node on both pages" -- and counted as
    `unlabelled`. A label naming a node the page projection dropped is
    skipped and counted: that node is in neither `N` (design §7.2) nor any
    hypothesis. Cell pairings (`level == "cell"`) are bookkeeping of a row
    join, never a scored decision."""
    by_key: dict[tuple[tuple[int, int], str, str], SeamDecision] = {}
    for decision in decisions:
        if decision.level == "cell":
            continue
        left = (gold_ids[decision.seam[0]] or {}).get(decision.left)
        right = (gold_ids[decision.seam[1]] or {}).get(decision.right)
        if left is None or right is None:
            raise DocumentError(
                f"seam {decision.seam} {decision.rule}: page-local ids "
                f"{decision.left!r}/{decision.right!r} have no gold id"
            )
        by_key[(decision.seam, left, right)] = decision
    scored: list[ScoredDecision] = []
    labelled: set[tuple[tuple[int, int], str, str]] = set()
    at_dropped = 0
    for seam in seams:
        if seam.status != "scored":
            continue
        dropped_a = (dropped[seam.a] if dropped else None) or frozenset()
        dropped_b = (dropped[seam.b] if dropped else None) or frozenset()
        for label in seam.labels:
            if label.a_node_id in dropped_a or label.b_node_id in dropped_b:
                at_dropped += 1
                continue
            key = ((seam.a, seam.b), label.a_node_id, label.b_node_id)
            labelled.add(key)
            decision = by_key.get(key)
            scored.append(
                ScoredDecision(
                    seam=(seam.a, seam.b),
                    rule_class=_rule_class(label),
                    gold_a=label.a_node_id,
                    gold_b=label.b_node_id,
                    gold_join=label.gold_join,
                    predicted=decision is not None and decision.joined,
                    decided=decision is not None,
                )
            )
    unlabelled = 0
    for key, decision in by_key.items():
        if key in labelled:
            continue
        unlabelled += 1
        scored.append(
            ScoredDecision(
                seam=key[0],
                rule_class=_RULE_CLASS_BY_RULE[decision.rule],
                gold_a=key[1],
                gold_b=key[2],
                gold_join=key[1] == key[2],
                predicted=decision.joined,
                decided=True,
            )
        )
    return scored, unlabelled, at_dropped


def confusion(scored: Iterable[Any]) -> dict[str, dict[str, int]]:
    """Per rule class `{tp, fp, fn, tn}`; works on `ScoredDecision`s and on
    their JSON rows."""
    out: dict[str, dict[str, int]] = {}
    for item in scored:
        gold = item.gold_join if isinstance(item, ScoredDecision) else item["gold_join"]
        pred = item.predicted if isinstance(item, ScoredDecision) else item["predicted"]
        cls = item.rule_class if isinstance(item, ScoredDecision) else item["rule_class"]
        cell = out.setdefault(cls, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        cell[("tp" if pred else "fn") if gold else ("fp" if pred else "tn")] += 1
    return out


def _split_run_left_ends(
    seams: Sequence[Seam],
    hypotheses: Sequence[PageHypothesis | None],
    gold_ids: Sequence[Mapping[str, str] | None],
) -> Counter[str]:
    """Design §10 M1 step 3 "whitespace/hyphen seam re-count": at every gold
    split run, what the page-k piece ends with."""
    counts: Counter[str] = Counter()
    for seam in seams:
        if seam.status != "scored":
            continue
        runs = [
            label for label in seam.labels if label.level == "run" and label.gold_join
        ]
        if not runs:
            continue
        hypothesis, ids = hypotheses[seam.a], gold_ids[seam.a] or {}
        text_by_gold = {
            ids.get(node.local_id): node.text
            for node in walk([*hypothesis.sections, *hypothesis.notes])
            if node.node_type == "run"
        }
        for label in runs:
            text = text_by_gold.get(label.a_node_id) or ""
            if not text:
                counts["empty"] += 1
            elif text[-1].isspace():
                counts["whitespace"] += 1
            elif text[-1] in _HYPHENS:
                counts["hyphen"] += 1
            else:
                counts["other"] += 1
    return counts


def _stories(document: AssembledDocumentHypothesis) -> dict[str, int]:
    sets = document.story_sets
    return {
        "sets": len(sets),
        "unresolved": sum(1 for s in sets if s.issue == STORY_UNRESOLVED),
        "misrepresented_pages": sum(len(s.misrepresented_pages) for s in sets),
        "digit_runs_normalized": sum(s.story_digit_runs_normalized for s in sets),
    }


def run_policy(
    name: str,
    policy: JoinPolicy,
    pages: GoldPages,
    seams: Sequence[Seam],
    *,
    profile: str,
) -> PolicyOutcome:
    hypotheses = [h for h in pages.hypotheses if h is not None]
    try:
        document = assemble(hypotheses, policy, profile=profile)
    except AssemblyInvariantViolation as exc:
        return PolicyOutcome(name, "invariant_violation", f"{exc.check}: {exc}", 0, 0, 0, 0, {}, ())
    except ValueError as exc:
        return PolicyOutcome(name, "error", str(exc), 0, 0, 0, 0, {}, ())
    scored, unlabelled, at_dropped = score(
        seams, document.seam_decisions, pages.gold_ids, pages.dropped
    )
    return PolicyOutcome(
        policy=name,
        status="assembled",
        detail=None,
        conflicts=len(document.conflicts),
        joins=sum(1 for d in document.seam_decisions if d.joined and d.level != "cell"),
        unlabelled_decisions=unlabelled,
        labels_at_dropped_nodes=at_dropped,
        stories=_stories(document),
        decisions=tuple(scored),
    )


def evaluate_document(
    *,
    store_dir: Path,
    samples: Sequence[Any],
    ast: dict[str, Any],
    bound: Any,
    tokenizer: Any,
    policies: Mapping[str, JoinPolicy] = DEFAULT_POLICIES,
    profile: str,
) -> DocumentResult:
    """Design §8 for one document; every refusal lands as `status="failed"`
    with its reason, never as an exception out of a worker."""
    bundle_id = samples[0].document_bundle_id
    lineage = getattr(samples[0], "lineage_root_id", None)
    try:
        pages = load_gold_pages(store_dir, samples, ast, bound=bound, tokenizer=tokenizer)
        reference = build(
            targets=pages.targets,
            fragments=pages.fragments,
            dropped=pages.dropped,
            text_spans=pages.text_spans,
            typed_targets=pages.typed_targets,
            ast=pages.ast,
            hole_reasons=pages.hole_reasons,
        )
        outcomes = tuple(
            run_policy(name, policy, pages, reference.seams, profile=profile)
            for name, policy in policies.items()
        )
        return DocumentResult(
            document_bundle_id=bundle_id,
            lineage_root_id=lineage,
            n_pages=len(samples),
            holes={k: r for k, r in enumerate(pages.hole_reasons) if r is not None},
            status="done",
            error=None,
            seams_scored=sum(1 for s in reference.seams if s.status == "scored"),
            seams_hole=sum(1 for s in reference.seams if s.status != "scored"),
            reference_excluded=dict(reference.excluded),
            reference_attribution=dict(reference.attribution),
            split_run_left_ends=dict(
                _split_run_left_ends(reference.seams, pages.hypotheses, pages.gold_ids)
            ),
            policies=outcomes,
        )
    except (DocumentError, ReferenceError, ValueError, KeyError, OSError) as exc:
        return DocumentResult(
            document_bundle_id=bundle_id,
            lineage_root_id=lineage,
            n_pages=len(samples),
            holes={},
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            seams_scored=0,
            seams_hole=0,
            reference_excluded=None,
            reference_attribution=None,
            split_run_left_ends={},
            policies=(),
        )


# ---------------------------------------------------------------------------
# JSONL ledger (append-only, resume)
# ---------------------------------------------------------------------------


def result_to_json(result: DocumentResult) -> dict[str, Any]:
    row = asdict(result)
    row["holes"] = {str(k): v for k, v in result.holes.items()}
    row["policies"] = {
        outcome.policy: {
            key: value for key, value in asdict(outcome).items() if key != "policy"
        }
        for outcome in result.policies
    }
    for outcome in row["policies"].values():
        outcome["decisions"] = [
            {**d, "seam": list(d["seam"])} for d in outcome["decisions"]
        ]
    return row


def append_result(path: Path, result: DocumentResult) -> None:
    with open(path, "a") as handle:
        handle.write(json.dumps(result_to_json(result), sort_keys=True) + "\n")


def read_ledger(path: Path) -> dict[str, dict[str, Any]]:
    """Read a JSONL eval ledger, returning the LAST line per bundle id."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["document_bundle_id"]] = row
    return rows


def pending_bundle_ids(
    manifest_path: Path,
    ledger_path: Path | None,
    *,
    limit: int | None = None,
) -> list[str]:
    """Bundle ids from the manifest not yet final in the ledger."""
    manifest = json.loads(manifest_path.read_text())
    bundle_ids: list[str] = manifest.get("bundle_ids", manifest.get("document_bundle_ids", []))
    if ledger_path is not None and ledger_path.exists():
        ledger = read_ledger(ledger_path)
        bundle_ids = [
            bid
            for bid in bundle_ids
            if bid not in ledger or ledger[bid].get("status") not in _RESUME_FINAL_STATUSES
        ]
    if limit is not None:
        bundle_ids = bundle_ids[:limit]
    return bundle_ids


def backup_ledger(path: Path) -> None:
    """Back up an existing ledger file to ``<path>.bak``."""
    if path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))


def ledger_hash(path: Path) -> str:
    """sha256 over the ledger's rows, sorted by bundle id and re-serialised
    canonically -- the G-A2 identity of a run's output."""
    rows = read_ledger(path)
    payload = "".join(
        json.dumps(rows[bid], sort_keys=True, separators=(",", ":")) + "\n"
        for bid in sorted(rows)
    )
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _lower_bound(correct: list[int], total: list[int]) -> float | None:
    """One-sided lower bootstrap bound (alpha 0.05) of the accuracy under
    lineage-cluster resampling; `None` below two lineages."""
    if len(total) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    c = np.asarray(correct, dtype=np.int64)
    t = np.asarray(total, dtype=np.int64)
    idx = rng.integers(0, len(t), size=(BOOTSTRAP_RESAMPLES, len(t)))
    acc = c[idx].sum(axis=1) / t[idx].sum(axis=1)
    return float(np.quantile(acc, _ALPHA))


def compute_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    done = [r for r in rows if r.get("status") == "done"]
    holes_by_reason: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    attribution: Counter[str] = Counter()
    left_ends: Counter[str] = Counter()
    labels: dict[str, Counter[str]] = {}
    label_lineages: dict[str, set[str]] = {}
    per_policy: dict[str, dict[str, Any]] = {}
    docs_with_hole = 0
    for row in done:
        holes = row.get("holes") or {}
        docs_with_hole += bool(holes)
        holes_by_reason.update(holes.values())
        excluded.update(row.get("reference_excluded") or {})
        attribution.update(row.get("reference_attribution") or {})
        left_ends.update(row.get("split_run_left_ends") or {})
        lineage = row.get("lineage_root_id") or row["document_bundle_id"]
        for name, outcome in (row.get("policies") or {}).items():
            agg = per_policy.setdefault(
                name,
                {
                    "invariant_violations": 0, "errors": 0, "conflicts": 0, "joins": 0,
                    "unlabelled_decisions": 0, "labels_at_dropped_nodes": 0,
                    "seam_exact_documents_hole_free": 0,
                    "stories": Counter(), "confusion": {},
                    "lineage_correct": {}, "lineage_total": {},
                },
            )
            status = outcome.get("status")
            agg["invariant_violations"] += status == "invariant_violation"
            agg["errors"] += status == "error"
            agg["conflicts"] += outcome.get("conflicts", 0)
            agg["joins"] += outcome.get("joins", 0)
            agg["unlabelled_decisions"] += outcome.get("unlabelled_decisions", 0)
            agg["labels_at_dropped_nodes"] += outcome.get("labels_at_dropped_nodes", 0)
            agg["stories"].update(outcome.get("stories") or {})
            decisions = outcome.get("decisions") or []
            if (
                status == "assembled"
                and not holes
                and outcome.get("conflicts", 0) == 0
                and all(d["gold_join"] == d["predicted"] for d in decisions)
            ):
                agg["seam_exact_documents_hole_free"] += 1
            for cls, cell in confusion(decisions).items():
                total = agg["confusion"].setdefault(cls, Counter())
                total.update(cell)
            for d in decisions:
                cls = d["rule_class"]
                lc = agg["lineage_correct"].setdefault(cls, Counter())
                lt = agg["lineage_total"].setdefault(cls, Counter())
                lc[lineage] += d["gold_join"] == d["predicted"]
                lt[lineage] += 1
        first = next(iter((row.get("policies") or {}).values()), None)
        if first is not None:
            for d in first.get("decisions") or []:
                counter = labels.setdefault(d["rule_class"], Counter())
                counter["total"] += 1
                counter["gold_joins"] += bool(d["gold_join"])
                label_lineages.setdefault(d["rule_class"], set()).add(lineage)

    policies: dict[str, Any] = {}
    for name in sorted(per_policy):
        agg = per_policy[name]
        classes: dict[str, Any] = {}
        for cls in sorted(agg["confusion"]):
            cell = agg["confusion"][cls]
            n = sum(cell.values())
            lineages = sorted(agg["lineage_total"][cls])
            classes[cls] = {
                **{k: int(cell[k]) for k in ("tp", "fp", "fn", "tn")},
                "n": n,
                "gold_joins": int(cell["tp"] + cell["fn"]),
                "accuracy": (cell["tp"] + cell["tn"]) / n if n else None,
                "lineages": len(lineages),
                "lower_bound_95": _lower_bound(
                    [agg["lineage_correct"][cls][lin] for lin in lineages],
                    [agg["lineage_total"][cls][lin] for lin in lineages],
                ),
            }
        policies[name] = {
            "invariant_violations": agg["invariant_violations"],
            "errors": agg["errors"],
            "conflicts": agg["conflicts"],
            "joins": agg["joins"],
            "unlabelled_decisions": agg["unlabelled_decisions"],
            "labels_at_dropped_nodes": agg["labels_at_dropped_nodes"],
            "seam_exact_documents_hole_free": agg["seam_exact_documents_hole_free"],
            "stories": dict(sorted(agg["stories"].items())),
            "classes": classes,
        }
    return {
        "schema_version": "1.0.0",
        "documents": {
            "total": len(rows),
            "done": len(done),
            "failed": len(rows) - len(done),
            "hole_free": len(done) - docs_with_hole,
            "with_holes": docs_with_hole,
        },
        "pages": {
            "total": sum(r.get("n_pages", 0) for r in done),
            "holes": sum(holes_by_reason.values()),
        },
        "holes": {"by_reason": dict(sorted(holes_by_reason.items()))},
        "seams": {
            "scored": sum(r.get("seams_scored", 0) for r in done),
            "excluded_hole_seam": sum(r.get("seams_hole", 0) for r in done),
        },
        "labels_by_class": {
            cls: {"total": int(c["total"]), "gold_joins": int(c["gold_joins"])}
            for cls, c in sorted(labels.items())
        },
        "lineages_by_class": {cls: len(v) for cls, v in sorted(label_lineages.items())},
        "excluded": {key: int(excluded.get(key, 0)) for key in EXCLUDED_KEYS},
        "attribution": {key: int(attribution.get(key, 0)) for key in ATTRIBUTION_KEYS},
        "split_run_left_ends": dict(sorted(left_ends.items())),
        "failed": sorted(
            {r["document_bundle_id"]: r.get("error") for r in rows if r.get("status") != "done"}
            .items()
        ),
        "policies": policies,
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "lineage_root_id",
            "alpha": _ALPHA,
        },
    }


def summary_hash(summary: Mapping[str, Any]) -> str:
    """sha256 over the summary as sorted-key JSON (floats via `repr`, which
    is deterministic for equal inputs; RFC001 canonical JSON refuses floats)."""
    payload = json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def format_summary(s: Mapping[str, Any]) -> str:
    d = s["documents"]
    lines = [
        f"Documents: {d['done']} done / {d['failed']} failed / {d['total']} total "
        f"({d['hole_free']} hole-free, {d['with_holes']} with holes)",
        f"Pages: {s['pages']['total']} total, {s['pages']['holes']} holes "
        f"{s['holes']['by_reason']}",
        f"Seams: {s['seams']['scored']} scored, "
        f"{s['seams']['excluded_hole_seam']} excluded_hole_seam",
        f"Split-run left ends: {s['split_run_left_ends']}",
        f"Excluded: {s['excluded']}",
        f"Attribution: {s['attribution']}",
    ]
    for name, policy in s["policies"].items():
        lines.append(
            f"Policy {name}: invariant_violations={policy['invariant_violations']} "
            f"errors={policy['errors']} conflicts={policy['conflicts']} joins={policy['joins']} "
            f"unlabelled={policy['unlabelled_decisions']} "
            f"labels_at_dropped={policy['labels_at_dropped_nodes']} "
            f"seam_exact_hole_free={policy['seam_exact_documents_hole_free']} "
            f"stories={policy['stories']}"
        )
        lines.append(
            f"  {'class':<10} {'n':>6} {'gold':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} "
            f"{'acc':>8} {'LB95':>8} {'lin':>5}"
        )
        for cls, c in policy["classes"].items():
            acc = "-" if c["accuracy"] is None else f"{c['accuracy']:.4f}"
            lb = "-" if c["lower_bound_95"] is None else f"{c['lower_bound_95']:.4f}"
            lines.append(
                f"  {cls:<10} {c['n']:>6} {c['gold_joins']:>6} {c['tp']:>6} {c['fp']:>6} "
                f"{c['fn']:>6} {c['tn']:>6} {acc:>8} {lb:>8} {c['lineages']:>5}"
            )
    if s["failed"]:
        lines.append(f"Failed documents: {len(s['failed'])}")
        for bid, error in s["failed"][:10]:
            lines.append(f"  {bid}: {error}")
    return "\n".join(lines)
