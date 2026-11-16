"""Build-time region targets for ONE page — RFC004 §21, design
`docs/plans/spec-node-visual-aligner-design.md` T1.

Reads the Observation Sidecar's per-page ``observations`` (RFC003 §24.1)
and turns them into one flat record per `(node_id, page)` slice: **all**
observed boxes for that node on this page, plus the page-evidence and
mask facts the fragment slice already carries.

**Raw values only, no policy at build time** — the same rule
`typed_property_targets.py` states for the typed channel (DEC-020): a
node with `k >= 2` observations keeps every box, a node with `k == 0`
keeps its `page_evidence` and gets no box at all. Which `k == 0` state is
a trainable abstention and which is `no_supervision` is a *load-time*
policy decision (design §4.3), not a build-time one — baking that choice
into the cache would make a policy ablation a rebuild.

An unobserved node never gets a synthetic zero-area box: `regions` is
simply the empty tuple. `RegionTarget` carries no boolean "has a region"
flag because `len(regions)` already says that, and a second field could
drift from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# `_anchor_token_indices` is imported private on purpose, same reason
# `exact_text_cache.py` imports `_UNION_DISCRIMINANT`/`_split_container_index`
# from this module: the anchor-token mechanism is owned by
# `typed_property_targets.py`, and a second copy here would be a second
# rule, free to stop matching the first.
from unflatten.model.typed_property_targets import (
    _anchor_token_indices as _anchor_token_indices,
)

#: The RFC003 §24.1 page-evidence vocabulary a fragment slice carries
#: (`fragments/envelope.py::_page_evidence`).
PAGE_EVIDENCE_VALUES: frozenset[str] = frozenset({"observed", "structural", "unresolved"})


class RegionTargetError(ValueError):
    """A region target could not be derived fail-closed, or a stored
    payload does not match the `RegionTarget` shape this module writes."""


@dataclass(frozen=True, slots=True)
class RegionTarget:
    """One node's raw region observations on ONE page (design §4.1).

    `regions`/`continuation` are parallel tuples, one entry per sidecar
    observation of `node_id` on this page, ordered by the sidecar's own
    `fragment_index` — never deduplicated, never unioned. `mapping_method`
    and `chain_scope` are `None` when `regions` is empty: with no
    observation to read them from, a fabricated default would be
    inventing a label, not reading one.
    """

    node_id: str
    node_type: str
    anchor_token_index: int
    #: `observed | structural | unresolved` (`fragments/envelope.py`).
    page_evidence: str
    observable_mask: bool
    supervision_mask: bool
    regions: tuple[tuple[int, int, int, int], ...]
    #: one `(continuation_prev, continuation_next)` pair per region.
    continuation: tuple[tuple[bool, bool], ...]
    #: `text | image_object_match`, or `None` when `regions` is empty.
    mapping_method: str | None
    #: `document | page`, or `None` when `regions` is empty.
    chain_scope: str | None

    def __post_init__(self) -> None:
        if self.page_evidence not in PAGE_EVIDENCE_VALUES:
            raise RegionTargetError(
                f"node {self.node_id!r}: page_evidence must be one of "
                f"{sorted(PAGE_EVIDENCE_VALUES)}, got {self.page_evidence!r}"
            )
        if len(self.continuation) != len(self.regions):
            raise RegionTargetError(
                f"node {self.node_id!r}: {len(self.regions)} region(s) but "
                f"{len(self.continuation)} continuation fact(s) — one per region, "
                "always"
            )
        for bbox in self.regions:
            if len(bbox) != 4:
                raise RegionTargetError(
                    f"node {self.node_id!r}: a region must be (x1, y1, x2, y2), "
                    f"got {bbox!r}"
                )


def build_region_targets(
    *,
    observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    slices: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    anchors: dict[str, int],
) -> tuple[tuple[RegionTarget, ...], int]:
    """One `RegionTarget` per node with a slice on this page.

    `observations` is this page's `sidecar["pages"][page_index]
    ["observations"]` list — every sidecar observation for every node
    that has one here, not filtered to one node. `slices` is this page's
    fragment slices (`fragment["slices"]`); a node's `page_evidence`,
    `observable_mask` and `supervision_mask` come from there, verbatim
    (raw values, RFC003's own fields — never recomputed here).

    A node with no anchor on this page (`not_opened_here`, the same
    resumed-node case `build_typed_property_targets` names) gets no
    record at all and is counted in the second return value, exactly like
    the typed channel's fourth return value.
    """
    by_node: dict[str, list[dict[str, Any]]] = {}
    for observation in observations:
        by_node.setdefault(observation["node_id"], []).append(observation)

    targets: list[RegionTarget] = []
    not_opened_here = 0
    for entry in slices:
        node_id = entry["node_id"]
        anchor = anchors.get(node_id)
        if anchor is None:
            not_opened_here += 1
            continue
        node_observations = sorted(
            by_node.get(node_id, ()), key=lambda o: o["fragment_index"]
        )
        targets.append(
            RegionTarget(
                node_id=node_id,
                node_type=entry["node_type"],
                anchor_token_index=anchor,
                page_evidence=entry["page_evidence"],
                observable_mask=bool(entry["observable_mask"]),
                supervision_mask=bool(entry["supervision_mask"]),
                regions=tuple(
                    tuple(observation["bbox_ppm"]) for observation in node_observations
                ),
                continuation=tuple(
                    (
                        bool(observation["continuation_prev"]),
                        bool(observation["continuation_next"]),
                    )
                    for observation in node_observations
                ),
                mapping_method=(
                    node_observations[0].get("mapping_method", "text")
                    if node_observations
                    else None
                ),
                chain_scope=(
                    node_observations[0].get("chain_scope", "document")
                    if node_observations
                    else None
                ),
            )
        )
    return tuple(targets), not_opened_here


_PAYLOAD_KEYS: tuple[str, ...] = (
    "node_id",
    "node_type",
    "anchor_token_index",
    "page_evidence",
    "observable_mask",
    "supervision_mask",
    "regions",
    "continuation",
    "mapping_method",
    "chain_scope",
)


def region_target_payload(target: RegionTarget) -> dict[str, Any]:
    """The plain-dict payload for one record — the shape `ExactTextCacheRow`
    stores it under, on the `typed_target_payload` precedent."""
    return {
        "node_id": target.node_id,
        "node_type": target.node_type,
        "anchor_token_index": target.anchor_token_index,
        "page_evidence": target.page_evidence,
        "observable_mask": target.observable_mask,
        "supervision_mask": target.supervision_mask,
        "regions": [list(bbox) for bbox in target.regions],
        "continuation": [list(pair) for pair in target.continuation],
        "mapping_method": target.mapping_method,
        "chain_scope": target.chain_scope,
    }


def region_target_from_payload(payload: dict[str, Any]) -> RegionTarget:
    """Read one record back, refusing a payload with any other key set —
    the `typed_target_from_payload` precedent: a payload this module did
    not write is refused, never guessed at."""
    keys = frozenset(payload)
    if keys != frozenset(_PAYLOAD_KEYS):
        raise RegionTargetError(
            f"payload with keys {sorted(keys)} does not match the RegionTarget "
            f"shape {sorted(_PAYLOAD_KEYS)} — it was not written by this module"
        )
    return RegionTarget(
        node_id=payload["node_id"],
        node_type=payload["node_type"],
        anchor_token_index=payload["anchor_token_index"],
        page_evidence=payload["page_evidence"],
        observable_mask=payload["observable_mask"],
        supervision_mask=payload["supervision_mask"],
        regions=tuple(tuple(bbox) for bbox in payload["regions"]),
        continuation=tuple(tuple(pair) for pair in payload["continuation"]),
        mapping_method=payload["mapping_method"],
        chain_scope=payload["chain_scope"],
    )
