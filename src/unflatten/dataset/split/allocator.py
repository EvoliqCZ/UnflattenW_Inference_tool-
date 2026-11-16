"""Lineage-first split allocator (RFC003 §40–§41), fully deterministic.

Input: gate-passing records ``{document_bundle_id, lineage_root_id,
recipe_id, fingerprint}`` (fingerprints from
:mod:`unflatten.dataset.split.fingerprints`). The split atom is the
connected component of the leakage graph over lineage_root_ids (§40.1):

- hard edges: equal ``exact_text_hash``, shared resource sha256 (§41.1);
- near-duplicate edges: MinHash jaccard estimate >= the policy's
  850000 ppm threshold (§41.2);
- ambiguous band [700000, 850000): per the policy's
  ``ambiguous_pair_action: quarantine`` BOTH components are quarantined
  (reason ``NEAR_DUPLICATE_COLLISION``) unless another edge already
  joined them (§41.2 — an ambiguous pair must never land in different
  splits).

Allocation is RNG-free: components are sorted by their smallest
lineage_root_id and greedily fill the 80/10/10 lineage targets (largest
remainder integer targets, deficit-first fill with the fixed tie order
train < val_iid < test), per the §40.4 [PROPOSED] proportions.

Template families (V0 simplification, tagged): [SIMPLIFIED-VS-RFC]
RFC003 §41.1 wants a template-holdout constraint solver over pinned
policies. V0 instead (a) reserves the smallest recipe family (ties by
recipe_id) entirely for ``val_template_ood`` — skipped when only one
family exists — and (b) routes every other recipe's components
consistently: the first component of a recipe fixes that recipe's split
and later components follow it, so no template family straddles
train/test. A component entangled with the reserved family via leakage
edges moves wholly to ``val_template_ood``; other components of its
non-reserved recipes are then routed by their own first component
(partial template overlap with the OOD branch is possible and is the
documented V0 gap vs the solver). The reserved family sits outside the
80/10/10 accounting (the RFC allows holdouts outside the percentages).

[SIMPLIFIED-VS-RFC] Branch names: V0 uses ``test`` (synthetic-only
corpus) instead of the RFC's recommended ``test_real_gold`` — no real
gold exists yet.

Conflicting routed recipes: a leakage component spanning recipes whose
first components were already routed to DIFFERENT splits is resolved by
the test-wins generalization — any test involvement sends the whole
component to test; a train-vs-val conflict routes to the val side
(never train, so a test/val-routed recipe's lineages cannot leak into
train). Every such resolution is recorded in
``SplitAssignment.routing_conflicts`` (component, recipes, routed
splits, chosen split, rule) — never silent.

Tolerance (§40.4): after allocation the per-split lineage proportion is
compared against ``targets_ppm``; a deviation beyond ``tolerance_ppm``
is recorded in ``SplitAssignment.tolerance_warnings`` as a warning, not
a failure — small populations legitimately deviate.

Late collision (§41.4, pinned): ``test wins`` — the non-test component
is quarantined (``SPLIT_COLLISION``); a collision between two non-test
branches conservatively quarantines both components. The released split
is never silently rewritten.

Pinning a prior release (§4.2, spec-zarovnanie-korpusu design): ``allocate``
accepts an optional ``prior_assignment`` (lineage_root_id -> split_name from
an earlier release) and its ``pinned_from_release_root_hash`` — both or
neither, never one alone. Pinning is an ALLOCATION INPUT, not a policy
change: proportions, tolerance and ``split_policy_hash`` are unaffected.
Two passes run before the ordinary greedy fill: **inherit** — a component
whose pinned lineages agree on one split keeps it (outside the usual
deficit fill, even for ``val_template_ood``); a component whose pinned
lineages disagree is a pin conflict, recorded in ``pin_conflicts`` and
resolved test-wins (a ``test``-pinned lineage keeps ``test``, every other
lineage in the component is quarantined ``SPLIT_COLLISION``; with no
``test`` involved the whole component is quarantined). **Newcomers** —
components with no pinned lineage — then fill the largest deficit exactly
as before, against targets computed over the WHOLE population (pinned +
newcomers), so the inherited population counts toward the 80/10/10 target
it already occupies. ``prior_lineages_absent`` names every pinned lineage
that is not part of the current population at all, or that lands in
``quarantined_lineages`` for any reason — informational, since its old
placement cannot be honoured either way.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from unflatten.dataset.identity import domain_hash
from unflatten.dataset.split.fingerprints import (
    DEFAULT_LEAKAGE_POLICY,
    LeakageFingerprintPolicy,
    minhash_similarity_ppm,
)

#: Proportioned branches, in deterministic fill/tie order.
SPLIT_NAMES: tuple[str, ...] = ("train", "val_iid", "test")
#: Reserved template-holdout branch (outside the proportion targets).
TEMPLATE_OOD_SPLIT = "val_template_ood"

SPLIT_POLICY_DOMAIN = "rfc003.split-policy/v1"


@dataclass(frozen=True)
class SplitPolicy:
    """Pinned split policy (§40.4 [PROPOSED] proportions, integer ppm)."""

    policy_schema_version: str = "0.1.0"
    policy_id: str = "split-word-core-v0"
    targets_ppm: dict[str, int] | None = None
    tolerance_ppm: int = 20_000
    late_collision_rule: str = "test_wins"
    reserved_template_ood_families: int = 1

    def __post_init__(self) -> None:
        if self.targets_ppm is None:
            object.__setattr__(
                self,
                "targets_ppm",
                {"train": 800_000, "val_iid": 100_000, "test": 100_000},
            )
        if set(self.targets_ppm) != set(SPLIT_NAMES):
            raise ValueError(
                f"targets_ppm keys must be exactly {SPLIT_NAMES}, "
                f"got {sorted(self.targets_ppm)}"
            )
        if sum(self.targets_ppm.values()) != 1_000_000:
            raise ValueError("targets_ppm must sum to 1000000")

    def to_dict(self) -> dict[str, Any]:
        """JCS-safe policy projection — never contains `policy_hash`."""
        return {
            "split_policy_schema_version": self.policy_schema_version,
            "policy_id": self.policy_id,
            "targets_ppm": dict(self.targets_ppm),
            "tolerance_ppm": self.tolerance_ppm,
            "late_collision_rule": self.late_collision_rule,
            "reserved_template_ood_families": self.reserved_template_ood_families,
            "template_ood_split": TEMPLATE_OOD_SPLIT,
        }

    def policy_hash(self) -> str:
        return domain_hash(SPLIT_POLICY_DOMAIN, self.to_dict())


DEFAULT_SPLIT_POLICY = SplitPolicy()


@dataclass(frozen=True)
class SplitAssignment:
    """Frozen allocation result: lineage -> split, plus the evidence."""

    assignments: dict[str, str]
    components: list[list[str]]
    quarantined_lineages: dict[str, str]  # lineage -> §45 reason code
    edges: list[dict[str, Any]]
    held_out_template_family_ids: list[str]
    split_policy_hash: str
    leakage_fingerprint_policy_hash: str
    #: Recorded recipe-routing conflicts (component spanning recipes
    #: already routed to different splits) — auditable, never silent.
    routing_conflicts: list[dict[str, Any]]
    #: Recorded per-split lineage-proportion deviations beyond the
    #: policy's tolerance_ppm — a warning, not a failure (small
    #: populations legitimately deviate).
    tolerance_warnings: list[dict[str, Any]]
    #: The prior release this allocation pins against, or ``None`` when
    #: unpinned (§4.2). Travels into ``component_lock`` unchanged.
    pinned_from_release_root_hash: str | None = None
    #: Recorded pin conflicts (component, disagreeing pinned splits,
    #: which lineages the test-wins rule quarantined) — never silent.
    pin_conflicts: list[dict[str, Any]] = field(default_factory=list)
    #: Lineages with no prior pin — routed by the ordinary deficit fill.
    newcomer_lineages: list[str] = field(default_factory=list)
    #: Pinned lineages absent from the current population (dropped from
    #: the store, gate-quarantined upstream, or quarantined here) —
    #: informational: their old placement cannot be honoured either way.
    prior_lineages_absent: list[str] = field(default_factory=list)


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root choice: smaller id wins
            if rb < ra:
                ra, rb = rb, ra
            self._parent[rb] = ra

    def components(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for key in self._parent:
            groups.setdefault(self.find(key), []).append(key)
        return sorted((sorted(members) for members in groups.values()), key=lambda c: c[0])


# Pool data is installed once per process, not serialized with every pair block.
_GRAPH_RECORDS: list[tuple[str, str, frozenset[str], list[int]]] = []
_GRAPH_ROW_STARTS: list[int] = []
_GRAPH_THRESHOLDS: tuple[int, int] = (0, 0)


def _init_graph_worker(prepared, row_starts, thresholds) -> None:
    global _GRAPH_RECORDS, _GRAPH_ROW_STARTS, _GRAPH_THRESHOLDS
    _GRAPH_RECORDS = prepared
    _GRAPH_ROW_STARTS = row_starts
    _GRAPH_THRESHOLDS = thresholds


def _graph_pair_block(bounds: tuple[int, int]):
    """One contiguous range in the sorted upper-triangular pair traversal."""
    start, stop = bounds
    i = bisect_right(_GRAPH_ROW_STARTS, start) - 1
    j = i + 1 + start - _GRAPH_ROW_STARTS[i]
    n = len(_GRAPH_RECORDS)
    near, low = _GRAPH_THRESHOLDS
    edges = {}
    ambiguous = set()
    for _ in range(start, stop):
        lin_a, exact_a, resources_a, minhash_a = _GRAPH_RECORDS[i]
        lin_b, exact_b, resources_b, minhash_b = _GRAPH_RECORDS[j]
        if lin_a != lin_b:
            if exact_a == exact_b:
                edges.setdefault(("exact_text", lin_a, lin_b), 1_000_000)
            if resources_a & resources_b:
                edges.setdefault(("resource_shared", lin_a, lin_b), None)
            similarity = minhash_similarity_ppm(minhash_a, minhash_b)
            if similarity >= near:
                # First record-pair wins, exactly as the serial seen_edges.
                edges.setdefault(("near_duplicate", lin_a, lin_b), similarity)
            elif similarity >= low:
                ambiguous.add((lin_a, lin_b))
        j += 1
        if j == n:
            i += 1
            j = i + 1
    return edges, ambiguous


def _parallel_leakage_graph(ordered, policy, workers: int, progress):
    n = len(ordered)
    total = n * (n - 1) // 2
    if not total:
        if progress is not None:
            progress(0, 0)
        return [], []
    prepared = [
        (
            r["lineage_root_id"],
            r["fingerprint"]["exact_text_hash"],
            frozenset(r["fingerprint"]["resource_sha256s"]),
            r["fingerprint"]["minhash"],
        )
        for r in ordered
    ]
    row_starts = [i * (2 * n - i - 1) // 2 for i in range(n - 1)]
    # Equal pair counts keep the triangular traversal balanced. Submission is
    # bounded independently of population size; only integer bounds travel.
    block_size = max(1, (total + workers * 16 - 1) // (workers * 16))
    blocks = [(start, min(start + block_size, total)) for start in range(0, total, block_size)]
    edges = {}
    ambiguous = set()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_graph_worker,
        initargs=(
            prepared,
            row_starts,
            (policy.near_duplicate_threshold_ppm, policy.ambiguous_low_threshold_ppm),
        ),
    ) as pool:
        # map preserves block order even if completion order changes: the
        # earliest duplicate edge's similarity is the serial result.
        for bounds, (block_edges, block_ambiguous) in zip(
            blocks, pool.map(_graph_pair_block, blocks), strict=True
        ):
            for key, similarity in block_edges.items():
                edges.setdefault(key, similarity)
            ambiguous.update(block_ambiguous)
            if progress is not None:
                progress(bounds[1], total)
    result = [
        {"edge_kind": kind, "a": a, "b": b, "similarity_ppm": similarity}
        for (kind, a, b), similarity in edges.items()
    ]
    result.sort(key=lambda edge: (edge["a"], edge["b"], edge["edge_kind"]))
    return result, sorted(ambiguous)


def build_leakage_graph(
    records: list[dict[str, Any]],
    fingerprint_policy: LeakageFingerprintPolicy = DEFAULT_LEAKAGE_POLICY,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Pairwise leakage edges + ambiguous-band pairs over lineages.

    Returns ``(edges, ambiguous_pairs)``; both use lineage_root_ids with
    ``a < b`` and are sorted, so the graph is order-independent.
    ``workers=1`` retains the serial comparison oracle; larger values compare
    every pair in balanced process blocks. ``progress(done_pairs, total_pairs)``
    runs only in the caller and never enters a policy or allocation identity.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("build_leakage_graph workers must be an integer >= 1")
    ordered = sorted(records, key=lambda r: (r["lineage_root_id"], r["document_bundle_id"]))
    if workers > 1:
        return _parallel_leakage_graph(ordered, fingerprint_policy, workers, progress)
    edges: list[dict[str, Any]] = []
    ambiguous: set[tuple[str, str]] = set()
    seen_edges: set[tuple[str, str, str]] = set()

    def add_edge(kind: str, lin_a: str, lin_b: str, similarity_ppm: int | None) -> None:
        a, b = sorted((lin_a, lin_b))
        if (kind, a, b) in seen_edges:
            return
        seen_edges.add((kind, a, b))
        edges.append(
            {"edge_kind": kind, "a": a, "b": b, "similarity_ppm": similarity_ppm}
        )

    for i, rec_a in enumerate(ordered):
        for rec_b in ordered[i + 1 :]:
            lin_a = rec_a["lineage_root_id"]
            lin_b = rec_b["lineage_root_id"]
            if lin_a == lin_b:
                continue
            fp_a, fp_b = rec_a["fingerprint"], rec_b["fingerprint"]
            if fp_a["exact_text_hash"] == fp_b["exact_text_hash"]:
                add_edge("exact_text", lin_a, lin_b, 1_000_000)
            if set(fp_a["resource_sha256s"]) & set(fp_b["resource_sha256s"]):
                add_edge("resource_shared", lin_a, lin_b, None)
            similarity = minhash_similarity_ppm(fp_a["minhash"], fp_b["minhash"])
            if similarity >= fingerprint_policy.near_duplicate_threshold_ppm:
                add_edge("near_duplicate", lin_a, lin_b, similarity)
            elif similarity >= fingerprint_policy.ambiguous_low_threshold_ppm:
                ambiguous.add(tuple(sorted((lin_a, lin_b))))

    edges.sort(key=lambda e: (e["a"], e["b"], e["edge_kind"]))
    if progress is not None:
        total = len(ordered) * (len(ordered) - 1) // 2
        progress(total, total)
    return edges, sorted(ambiguous)


def _largest_remainder_targets(total: int, targets_ppm: dict[str, int]) -> dict[str, int]:
    base = {name: total * targets_ppm[name] // 1_000_000 for name in SPLIT_NAMES}
    remainders = {
        name: total * targets_ppm[name] - base[name] * 1_000_000 for name in SPLIT_NAMES
    }
    leftover = total - sum(base.values())
    for name in sorted(
        SPLIT_NAMES, key=lambda n: (-remainders[n], SPLIT_NAMES.index(n))
    )[:leftover]:
        base[name] += 1
    return base


def allocate(
    records: list[dict[str, Any]],
    split_policy: SplitPolicy = DEFAULT_SPLIT_POLICY,
    fingerprint_policy: LeakageFingerprintPolicy = DEFAULT_LEAKAGE_POLICY,
    *,
    prior_assignment: dict[str, str] | None = None,
    pinned_from_release_root_hash: str | None = None,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> SplitAssignment:
    """Deterministic lineage-first allocation (see module docstring).

    ``workers`` and ``progress`` control only leakage-graph execution; neither
    changes split/fingerprint policies, pinned assignments or result identity.
    """
    if (prior_assignment is None) != (pinned_from_release_root_hash is None):
        raise ValueError(
            "prior_assignment and pinned_from_release_root_hash must be "
            "given together, or not at all"
        )
    prior_assignment = prior_assignment or {}
    edges, ambiguous_pairs = build_leakage_graph(
        records, fingerprint_policy, workers=workers, progress=progress
    )

    uf = _UnionFind()
    for record in records:
        uf.add(record["lineage_root_id"])
    for edge in edges:
        uf.union(edge["a"], edge["b"])

    # Ambiguous pairs still in different components => quarantine both
    # components wholly (§41.2: never split an undecided pair).
    quarantined: dict[str, str] = {}
    quarantined_roots: set[str] = set()
    for lin_a, lin_b in ambiguous_pairs:
        if uf.find(lin_a) != uf.find(lin_b):
            quarantined_roots.add(uf.find(lin_a))
            quarantined_roots.add(uf.find(lin_b))

    components = []
    for component in uf.components():
        if uf.find(component[0]) in quarantined_roots:
            for lineage in component:
                quarantined[lineage] = "NEAR_DUPLICATE_COLLISION"
        else:
            components.append(component)

    # recipe_id None marks a recipe-less record (imported real document):
    # it participates through fingerprint/resource edges only and never
    # forms or joins a template family.
    recipes_by_lineage: dict[str, set[str]] = {}
    for record in records:
        lineage_recipes = recipes_by_lineage.setdefault(
            record["lineage_root_id"], set()
        )
        if record["recipe_id"] is not None:
            lineage_recipes.add(record["recipe_id"])

    def component_recipes(component: list[str]) -> list[str]:
        found: set[str] = set()
        for lineage in component:
            found |= recipes_by_lineage[lineage]
        return sorted(found)

    # Reserved template family for val_template_ood (V0 simplification).
    family_sizes: dict[str, int] = {}
    for lineage, recipes in recipes_by_lineage.items():
        if lineage in quarantined:
            continue
        for recipe in recipes:
            family_sizes[recipe] = family_sizes.get(recipe, 0) + 1
    held_out: list[str] = []
    if len(family_sizes) > 1 and split_policy.reserved_template_ood_families > 0:
        held_out = sorted(
            family_sizes, key=lambda r: (family_sizes[r], r)
        )[: split_policy.reserved_template_ood_families]

    assignments: dict[str, str] = {}
    remaining: list[list[str]] = []
    for component in components:
        if set(component_recipes(component)) & set(held_out):
            for lineage in component:
                assignments[lineage] = TEMPLATE_OOD_SPLIT
        else:
            remaining.append(component)

    # --- pass 1: inherit prior pins (§4.2) --------------------------------
    # A component whose pinned lineages agree on one split keeps it,
    # bypassing the deficit fill entirely (this is how a pin to
    # TEMPLATE_OOD_SPLIT lands outside the 80/10/10 accounting, exactly
    # like the recipe-family reservation above). Disagreeing pins are a
    # conflict: test wins if any pinned lineage says test, otherwise the
    # whole component is quarantined — never silently resolved.
    pin_conflicts: list[dict[str, Any]] = []
    newcomer_components: list[list[str]] = []
    assigned_counts = {name: 0 for name in SPLIT_NAMES}
    routed_recipes: dict[str, str] = {}
    for component in remaining:
        pinned_splits = sorted(
            {prior_assignment[lineage] for lineage in component if lineage in prior_assignment}
        )
        if not pinned_splits:
            newcomer_components.append(component)
            continue
        if len(pinned_splits) == 1:
            split = pinned_splits[0]
            for lineage in component:
                assignments[lineage] = split
            if split in assigned_counts:
                assigned_counts[split] += len(component)
            for recipe in component_recipes(component):
                routed_recipes[recipe] = split
        else:
            if "test" in pinned_splits:
                losers = [
                    lineage for lineage in component
                    if prior_assignment.get(lineage) != "test"
                ]
                winners = [lineage for lineage in component if lineage not in losers]
                for lineage in winners:
                    assignments[lineage] = "test"
                assigned_counts["test"] += len(winners)
            else:
                losers = list(component)
            for lineage in losers:
                quarantined[lineage] = "SPLIT_COLLISION"
            pin_conflicts.append(
                {
                    "component": list(component),
                    "pinned_splits": pinned_splits,
                    "quarantined_lineages": sorted(losers),
                }
            )

    newcomer_lineages = sorted(
        lineage for component in newcomer_components for lineage in component
    )

    # --- pass 2: newcomers into the largest deficit -----------------------
    # `total` already counts the pinned population against the SAME
    # targets, so a newcomer fills whatever proportion the inherited
    # split assignments have not already occupied.
    total = sum(assigned_counts.values()) + sum(
        len(component) for component in newcomer_components
    )
    targets = _largest_remainder_targets(total, split_policy.targets_ppm)
    routing_conflicts: list[dict[str, Any]] = []
    for component in newcomer_components:  # already sorted by smallest lineage id
        recipes = component_recipes(component)
        routed = sorted({routed_recipes[r] for r in recipes if r in routed_recipes})
        if len(routed) == 1:
            split = routed[0]
        elif routed:
            # Conflicting routed targets: test-wins generalization —
            # any test involvement pulls the whole component to test;
            # a train-vs-val conflict routes to the val side. ALWAYS
            # recorded — a conflict is never resolved silently.
            if "test" in routed:
                split, rule = "test", "test_wins"
            else:
                split, rule = "val_iid", "val_wins"
            routing_conflicts.append(
                {
                    "component": list(component),
                    "recipes": recipes,
                    "routed_splits": routed,
                    "chosen_split": split,
                    "rule": rule,
                }
            )
        else:
            split = max(
                SPLIT_NAMES,
                key=lambda n: (
                    targets[n] - assigned_counts[n],
                    -SPLIT_NAMES.index(n),
                ),
            )
        for lineage in component:
            assignments[lineage] = split
        assigned_counts[split] += len(component)
        for recipe in recipes:
            # After a conflict resolution the component's split WINS for
            # every participating recipe — later components of a
            # conflicted recipe must follow the resolved split, or the
            # template family straddles train/test after all.
            routed_recipes[recipe] = split

    tolerance_warnings: list[dict[str, Any]] = []
    if total:
        for name in SPLIT_NAMES:
            actual_ppm = assigned_counts[name] * 1_000_000 // total
            deviation_ppm = abs(actual_ppm - split_policy.targets_ppm[name])
            if deviation_ppm > split_policy.tolerance_ppm:
                tolerance_warnings.append(
                    {
                        "split": name,
                        "target_ppm": split_policy.targets_ppm[name],
                        "actual_ppm": actual_ppm,
                        "deviation_ppm": deviation_ppm,
                        "tolerance_ppm": split_policy.tolerance_ppm,
                    }
                )

    # Pinned lineages that cannot be honoured either way: absent from the
    # current population entirely, or quarantined here (ambiguous-pair
    # NEAR_DUPLICATE_COLLISION or pin-conflict SPLIT_COLLISION) —
    # informational, per §4.2.
    current_lineages = {record["lineage_root_id"] for record in records}
    prior_lineages_absent = sorted(
        lineage
        for lineage in prior_assignment
        if lineage not in current_lineages or lineage in quarantined
    )

    return SplitAssignment(
        assignments=assignments,
        components=components,
        quarantined_lineages=quarantined,
        edges=edges,
        held_out_template_family_ids=held_out,
        split_policy_hash=split_policy.policy_hash(),
        leakage_fingerprint_policy_hash=fingerprint_policy.policy_hash(),
        routing_conflicts=routing_conflicts,
        tolerance_warnings=tolerance_warnings,
        pinned_from_release_root_hash=pinned_from_release_root_hash,
        pin_conflicts=pin_conflicts,
        newcomer_lineages=newcomer_lineages,
        prior_lineages_absent=prior_lineages_absent,
    )


def resolve_late_collision(
    assignment: SplitAssignment, lineage_a: str, lineage_b: str
) -> SplitAssignment:
    """§41.4 late-collision rule, pinned: test wins.

    A new near-duplicate edge joining two allocated lineages: the same
    split is a no-op; if one side is ``test`` its lineage keeps ``test``
    and the OTHER side's whole component is quarantined
    (``SPLIT_COLLISION``); two non-test branches conservatively
    quarantine both components. Returns a new assignment — the input is
    never mutated (§46.2: a published split is not edited in place).
    """
    split_a = assignment.assignments.get(lineage_a)
    split_b = assignment.assignments.get(lineage_b)
    if split_a is None or split_b is None or split_a == split_b:
        return assignment

    losers: set[str] = set()
    if "test" in (split_a, split_b):
        losers.add(lineage_b if split_a == "test" else lineage_a)
    else:
        losers.update((lineage_a, lineage_b))

    loser_components = [
        component
        for component in assignment.components
        if losers & set(component)
    ]
    new_assignments = dict(assignment.assignments)
    new_quarantined = dict(assignment.quarantined_lineages)
    for component in loser_components:
        for lineage in component:
            new_assignments.pop(lineage, None)
            new_quarantined[lineage] = "SPLIT_COLLISION"
    return replace(
        assignment,
        assignments=new_assignments,
        quarantined_lineages=new_quarantined,
        components=[
            component
            for component in assignment.components
            if component not in loser_components
        ],
    )
