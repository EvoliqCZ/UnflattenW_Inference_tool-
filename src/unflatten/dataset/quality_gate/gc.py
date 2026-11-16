"""G-C of the DEC-057 data-quality gate: the visual spot check.

* `sample_pages` draws the policy's page count (DEC-057 addendum A: 300),
  stratified by content kind, from the TRAIN split only -- every
  evaluation, selection and control cohort
  (val_iid, test, template OOD, the S5 control scans) lives in the other
  splits by construction (lineage-pinned split allocation), and any page
  whose bundle or lineage appears in an extra exclusion file is refused
  as well (DEC-053 point 4, DEC-057 point 4).
* `gold_page_decode` turns a cache row (gold tokens + supervised typed
  targets) into the same `PageDecodeResult` shape a model decode has, so
  the builder chain renders the TARGET through our own AST builder and
  renderer (`scripts/data_quality_gate.py gallery`).
* the findings register is a TSV `class<TAB>page<TAB>note`; `page` is
  `<lineage_root_id>#<page_index>` (survives a store revision) or
  `<document_bundle_id>#<page_index>`. The gate fails while a finding of a
  blocking class names a page still in the cache, and while a sampled
  page has no register row at all (the spot check was not done).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.dataset.quality_gate.token_tree import ID

GC_VERSION = "1"

_BUNDLE = re.compile(r"bundle_[0-9a-f]{64}")
_LINEAGE = re.compile(r"lin_[0-9a-f]{64}")

#: content kinds, first match wins (rarer kinds first so they get sampled)
KINDS: tuple[tuple[str, frozenset[int]], ...] = (
    ("text_frame", frozenset({ID["TEXT_FRAME_OPEN"]})),
    ("note", frozenset({ID["FOOTNOTE_OPEN"], ID["ENDNOTE_OPEN"]})),
    ("header_footer", frozenset({ID["HEADER_OPEN"], ID["FOOTER_OPEN"]})),
    ("image", frozenset({ID["INLINE_IMAGE"], ID["FLOATING_IMAGE"]})),
    ("table", frozenset({ID["TABLE_OPEN"]})),
    ("list", frozenset({ID["LIST_OPEN"]})),
)


def content_kind(tokens: Iterable[int]) -> str:
    present = set(tokens)
    for kind, ids in KINDS:
        if present & ids:
            return kind
    return "text"


def exclusion_ids(paths: Sequence[Path]) -> tuple[frozenset[str], frozenset[str]]:
    """Every bundle id and lineage id named anywhere in the given files
    (cohort manifests, control-sample JSON, page lists -- format-agnostic
    on purpose: a cohort file in any shape still excludes its pages)."""
    bundles: set[str] = set()
    lineages: set[str] = set()
    for path in paths:
        text = Path(path).read_text(errors="replace")
        bundles.update(_BUNDLE.findall(text))
        lineages.update(_LINEAGE.findall(text))
    return frozenset(bundles), frozenset(lineages)


@dataclass(frozen=True)
class Candidate:
    bundle: str
    page: int
    lineage: str
    kind: str

    @property
    def page_id(self) -> str:
        return f"{self.bundle}#{self.page}"

    @property
    def page_key(self) -> str:
        return f"{self.lineage}#{self.page}"


def sample_pages(
    candidates: Sequence[Candidate],
    *,
    n: int,
    seed: int,
    excluded_bundles: frozenset[str],
    excluded_lineages: frozenset[str],
    cache_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """`n` pages, round-robin over content kinds (a kind with too few
    pages gives its share to the others), never an excluded bundle or
    lineage. Deterministic in `seed` and the candidate set. `cache_binding`
    (the train manifest's `gate.cache_binding`) is sealed into the body: the
    gate refuses a sample drawn from another cache than the one it reads
    (DEC-057 addendum A: a new sample after every rebuild)."""
    eligible = [
        c
        for c in candidates
        if c.bundle not in excluded_bundles and c.lineage not in excluded_lineages
    ]
    by_kind: dict[str, list[Candidate]] = {}
    for c in sorted(eligible, key=lambda c: c.page_id):
        by_kind.setdefault(c.kind, []).append(c)
    rng = random.Random(seed)
    for pool in by_kind.values():
        rng.shuffle(pool)
    picked: list[Candidate] = []
    kinds = sorted(by_kind)
    while len(picked) < n and any(by_kind[k] for k in kinds):
        for k in kinds:
            if by_kind[k] and len(picked) < n:
                picked.append(by_kind[k].pop())
    pages = [
        {"page": c.page_id, "page_key": c.page_key, "kind": c.kind}
        for c in sorted(picked, key=lambda c: (c.kind, c.page_id))
    ]
    body: dict[str, Any] = {
        "gc_version": GC_VERSION,
        "seed": seed,
        "requested_sample_pages": n,
        "candidates": len(candidates),
        "excluded_candidates": len(candidates) - len(eligible),
        "pages": pages,
    }
    if cache_binding is not None:
        body["train_cache_binding"] = dict(cache_binding)
    return {**body, "sample_sha256": sha256_json(body)}


def sha256_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(data).hexdigest()


# -- register --------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    cls: str
    page: str  # lineage#page or bundle#page
    note: str


class RegisterInvalid(ValueError):
    """A register row is malformed or names an unknown class."""


def load_register(path: Path, policy: Mapping[str, Any]) -> list[Finding]:
    known = set(policy["blocking_classes"]) | set(policy["non_blocking_classes"])
    out = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 2:
            raise RegisterInvalid(f"{path}:{number}: expected class<TAB>page<TAB>note")
        cls, page = cols[0].strip(), cols[1].strip()
        if cls not in known:
            raise RegisterInvalid(f"{path}:{number}: unknown finding class {cls!r}")
        if not re.fullmatch(r"(bundle|lin)_[0-9a-f]{64}#\d+", page):
            raise RegisterInvalid(f"{path}:{number}: page {page!r} is not <id>#<page_index>")
        out.append(Finding(cls, page, cols[2].strip() if len(cols) > 2 else ""))
    return out


def evaluate_register(
    findings: Sequence[Finding],
    sample: Mapping[str, Any],
    policy: Mapping[str, Any],
    present: Mapping[str, str],
) -> dict[str, Any]:
    """`present` maps every page still in the caches, by BOTH keys
    (`bundle#page` and `lineage#page`), to its `bundle#page`. A blocking
    finding whose page left the caches (excluded and rebuilt) is resolved;
    one whose page is still there is open. A sampled page with no row is
    unreviewed."""
    blocking = set(policy["blocking_classes"])
    reviewed = {f.page for f in findings}
    unreviewed = [
        p["page"]
        for p in sample["pages"]
        if p["page"] not in reviewed and p["page_key"] not in reviewed
    ]
    open_findings = [
        {"class": f.cls, "page": f.page, "note": f.note}
        for f in findings
        if f.cls in blocking and f.page in present
    ]
    waived = frozenset(
        present[f.page] for f in findings if f.cls == "genuine_keep" and f.page in present
    )
    too_small = len(sample["pages"]) < policy["min_sample_pages"]
    return {
        "gc_version": GC_VERSION,
        "sample_sha256": sample["sample_sha256"],
        "sample_pages": len(sample["pages"]),
        "findings": len(findings),
        "open_blocking": open_findings,
        "unreviewed_sample_pages": unreviewed,
        "violation": bool(open_findings or unreviewed or too_small),
        "waived_pages": sorted(waived),
    }


# -- the target as a page decode (for rendering) -----------------------------------


def gold_page_decode(
    bundle: str,
    tokens: Sequence[int],
    typed_targets: Iterable[Any],
    typed_cardinalities: Iterable[Any] = (),
    *,
    zero_missing_tab_stops: bool = False,
    region_targets: Iterable[Any] = (),
) -> Any:
    """A `sealed.PageDecodeResult` whose hypothesis IS the target: the gold
    tokens, node states replayed by the same lifecycle tracker the parser
    uses, every KNOWN typed target as a typed prediction on the node its
    anchor token opens, and every known repeated-container count at its
    container pointer (the cardinality channel a decode also emits).
    Inputs are cache-row payload dicts.

    `zero_missing_tab_stops` (rendering only): a paragraph whose tab-stop
    count the target does not carry gets 0 tab stops, so the strict builder
    can render the page (a decode always emits the count; the cache row
    only has it where the target knows it)."""
    from unflatten.eval.s2_gate import sealed
    from unflatten.model import events_vocab
    from unflatten.model.node_lifecycle import NodeLifecycleTracker

    names = {v: k for k, v in events_vocab.TOKEN_IDS.items()}
    structural = (
        frozenset(events_vocab.OPEN_TOKEN.values())
        | frozenset(events_vocab.LEAF_TOKEN.values())
        | frozenset(events_vocab.CLOSE_TOKEN.values())
    )
    tracker = NodeLifecycleTracker()
    by_anchor: dict[int, Any] = {}
    for position, token in enumerate(tokens):
        name = names.get(token)
        if name not in structural:
            continue
        state = tracker.apply_structural_token(name, event_index=position)
        if state is not None and (
            state.open_event_index == position or state.leaf_event_index == position
        ):
            by_anchor[position] = state
    states = tuple(
        sealed.NodeStateProjection(
            local_node_id=s.local_node_id,
            slot_kind=s.slot_kind,
            node_type=s.node_type,
            parent_local_node_id=s.parent_local_node_id,
            open_event_index=s.open_event_index,
            leaf_event_index=s.leaf_event_index,
            resume_event_index=s.resume_event_index,
            close_event_index=s.close_event_index,
        )
        for s in tracker.states
    )
    typed = []
    seen: set[tuple[str, str]] = set()
    entries = [(t, t["node_property_pointer"], t["value"]) for t in typed_targets] + [
        (c, c["container_pointer"], c["count"]) for c in typed_cardinalities
    ]
    for t, pointer, value in entries:
        if not t["known_mask"]:
            continue
        state = by_anchor.get(int(t["anchor_token_index"]))
        if state is None or state.node_type != t.get("node_type", state.node_type):
            raise ValueError(
                f"gold target {pointer!r}: no matching anchor node at {t['anchor_token_index']}"
            )
        key = (state.local_node_id, pointer)
        if key in seen:
            continue
        seen.add(key)
        typed.append(
            {
                "local_node_id": state.local_node_id,
                "node_type": state.node_type,
                "pointer": pointer,
                "value": value,
            }
        )
    if zero_missing_tab_stops:
        for state in by_anchor.values():
            if state.node_type == "paragraph" and (state.local_node_id, "/layout/tabs") not in seen:
                typed.append(
                    {
                        "local_node_id": state.local_node_id,
                        "node_type": "paragraph",
                        "pointer": "/layout/tabs",
                        "value": 0,
                    }
                )
    return sealed.PageDecodeResult(
        document_bundle_id=bundle,
        outcome="terminated",
        accepted_tokens=tuple(int(t) for t in tokens),
        node_states=states,
        typed_predictions=tuple(typed),
        region_predictions=_gold_regions(region_targets, by_anchor),
    )


def _gold_regions(targets: Iterable[Any], by_anchor: Mapping[int, Any]) -> tuple[Any, ...]:
    """Apply the existing load-time policy; never fabricate or merge boxes."""
    from unflatten.eval.s2_gate import sealed
    from unflatten.model.node_visual_aligner import STATUS_VOCABULARY
    from unflatten.model.region_targets import region_target_from_payload
    from unflatten.model.s4_typed_loss import _derive_region_row, _RegionExcluded

    predictions = []
    for payload in targets:
        target = region_target_from_payload(payload)
        state = by_anchor.get(target.anchor_token_index)
        if state is None or state.node_type != target.node_type:
            raise ValueError(f"gold region {target.node_id!r}: no matching anchor node")
        try:
            status, bbox, supervise_box = _derive_region_row(target)
        except _RegionExcluded:
            continue
        predictions.append(
            {
                "local_node_id": state.local_node_id,
                "node_type": state.node_type,
                "alignment_status": STATUS_VOCABULARY[status],
                "confidence_ppm": 1_000_000,
                "regions": [
                    {
                        "page_ordinal": 0,
                        "bbox_ppm": list(bbox),
                        "continuation": "complete",
                        "confidence_ppm": 1_000_000,
                    }
                ]
                if supervise_box
                else [],
            }
        )
    return sealed._sealed_region_predictions(predictions)


def source_property_targets(
    targets: Sequence[Mapping[str, Any]],
    cardinalities: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Gallery-only source evidence, preserving every known cache target."""
    anchors = {}
    for target in (*targets, *cardinalities, *regions):
        node_id = target["node_id"]
        source_type = labels.get(node_id, {}).get("node_type")
        identity = (int(target["anchor_token_index"]), target.get("node_type", source_type))
        node_id = target["node_id"]
        if node_id in anchors and anchors[node_id] != identity:
            raise ValueError(f"gold source node {node_id!r}: conflicting anchor identity")
        anchors[node_id] = identity
    known = {
        (int(t["anchor_token_index"]), t["node_property_pointer"])
        for t in targets
        if t["known_mask"]
    }
    out = list(targets)
    hydrated = set()
    for node_id, (anchor, node_type) in anchors.items():
        label = labels.get(node_id)
        if label is None or label["node_type"] != node_type:
            raise ValueError(f"gold source node {node_id!r}: absent or mismatched canonical owner")
        for field in label["fields"]:
            pointer = field["node_property_pointer"]
            if pointer in {"/resource_ref", "/note_id"} or (anchor, pointer) in known:
                continue
            out.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "anchor_token_index": anchor,
                    "node_property_pointer": pointer,
                    "value": field["value"],
                    "known_mask": True,
                    "gallery_provenance": "source_only_unsupervised",
                }
            )
            hydrated.add(node_id)
    return out, {
        "source_only_properties": len(out) - len(targets),
        "source_only_nodes": len(hydrated),
    }


def source_cardinality_targets(
    targets: Sequence[Mapping[str, Any]],
    cardinalities: Sequence[Mapping[str, Any]],
    regions: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Gallery counts come from actual source arrays, including empty ones."""
    anchors = {
        t["node_id"]: (
            int(t["anchor_token_index"]),
            t.get("node_type", labels[t["node_id"]]["node_type"]),
        )
        for t in (*targets, *cardinalities, *regions)
    }
    known = {
        (int(c["anchor_token_index"]), c["container_pointer"])
        for c in cardinalities
        if c["known_mask"]
    }
    out = list(cardinalities)
    for node_id, (anchor, node_type) in anchors.items():
        label = labels[node_id]
        if label["node_type"] != node_type:
            raise ValueError(f"gold source node {node_id!r}: mismatched canonical owner")
        for pointer, count in label.get("containers", {}).items():
            if (anchor, pointer) in known:
                continue
            out.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "anchor_token_index": anchor,
                    "container_pointer": pointer,
                    "count": count,
                    "known_mask": True,
                    "gallery_provenance": "source_only_unsupervised",
                }
            )
    return out, {"source_only_cardinalities": len(out) - len(cardinalities)}
