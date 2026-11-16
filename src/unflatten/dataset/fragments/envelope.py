"""Page Fragment Envelope builder + validator (RFC003 §26).

`build_fragments` derives page-window fragments from a canonical AST,
its Canonical Target Bundle and its Observation Sidecar. Fragments are
a derived view: they add no `Page` node and are re-derivable at will.

Window policies (§26.4), V0:

- ``single_page``    — one fragment per rendered page.
- ``two_page_overlap`` — sliding windows [0,1], [1,2], ...; a one-page
  document degenerates to a single [0] window.
- ``full_document``  — one fragment over all pages (the trivial case).
- ``bounded_multi_page`` — deferred (`NotImplementedError`); it is a
  parameterized generalization between the two extremes and needs a
  windowing policy input this signature does not carry yet.

Story placement (DEC-003 §2): a header has ink on every page of its
section, but exactly one node may own one placement. The Observation
Sidecar resolves this upstream — `observations.stories` observes a
story on the FIRST page that actually renders it and on no other — so a
header arrives here with a single page and is `observed` like any other
node. Nothing in this module special-cases stories; a header that
reaches it unobserved is `unresolved`, which is the honest answer for a
story the aligner could not place, and blocks quality training on it
per RFC005 §9.2.

Page assignment: a node is visible on the pages where the sidecar holds
observations for it or for any descendant. A node with no observed
surface anywhere (empty paragraph, alignment miss, explicit control
node such as ``page_break``) borrows its page deterministically from
its neighbors, resolved in preorder:

1. the nearest PREVIOUS sibling with resolved pages — its LAST page;
2. else the nearest NEXT sibling with raw (observation-derived) pages —
   its FIRST page;
3. else the parent's first resolved page (transitively the section's
   page; the synthesized document root falls back to page 0).

Because a borrowed placement and an observed one used to produce
byte-identical slices, every slice carries ``page_evidence``:

- ``observed``   — the node or a descendant has an observation;
- ``structural`` — nothing to observe AND nothing unknown: a control
  node, an empty paragraph, a whitespace-only run. No glyphs exist, the
  placement follows from the event partition, and nothing is fabricated;
- ``unresolved`` — the node carries real INK yet was never observed, so
  its borrowed page may be the wrong page. That is text the aligner
  missed, and also every ``inline_image``: the aligner matches pdftotext
  glyphs (`aligner.py`: "inline_image and breaks contribute no matchable
  glyphs"), so an image is never observed, while it does occupy a real
  position on a real page. Calling that ``structural`` would assert the
  structure fixes a placement nobody derived.

OWNERSHIP AND SUPERVISION ARE NOT THE SAME QUESTION, and conflating them
was a real defect. Ownership is partition bookkeeping: the invariants
below require exactly one structure owner per node whatever the
evidence, because the event stream must be covered exactly once. That
says nothing about whether the label is true.

Supervision is scoped per §29 and lives on the slice, because the
fragment IS the TaskViewProjection (§27: ``observable_mask`` and
``supervision_mask`` "patria do konkrétneho TaskViewProjection"):

- ``observable_mask`` — false exactly for ``unresolved``: the page was
  invented by ``_resolve_pages``, so nothing about this node is derivable
  from this view;
- ``supervision_mask`` — §29.3 forces ``supervision ⇒ observable``, so an
  unresolved node is supervised on NOTHING here, structural events
  included. Emitting its owned events as a target would be a fabricated
  positive on an invented page — worse than a numeric zero, because it
  looks like evidence. Consumers must read
  :func:`supervised_event_ranges`, never ``structural_event_ranges``,
  when building targets;
- ``placement_origin`` — §29.4 label origin of the PLACEMENT (not of the
  property values): ``imputed_for_render`` for ``unresolved``, which
  "nikdy nie je exact supervised ground truth"; ``mapping_deterministic``
  otherwise.

``structural`` keeps full supervision: a control node or an empty
paragraph has nothing to observe (§29 ``not_applicable``), its placement
follows deterministically from the event partition, and nothing is
fabricated. Typed property supervision for an unresolved node is
withdrawn by
:func:`unflatten.dataset.targets.masks.apply_placement_mask` — an
explicit no-supervision, never a numeric zero.

Deterministic ownership (§26.2) per node:

- ``structure_owner``    — the FIRST fragment (lowest sequence_index)
  whose pages intersect the node's resolved pages;
- ``typed_target_owner`` — identical to ``structure_owner`` in V0;
- ``text_range_owner``   — per char range: the observation char ranges
  on page *p* belong to the first fragment containing *p* (so with
  overlapping windows every character is still owned exactly once).

Every structural event is owned exactly once, via its node's structure
owner; a slice's ``structural_event_ranges`` therefore concatenate
across fragments to the full bundle event stream with no gap or
duplicate (§26.3 invariant 9). ``control_event_ownership`` re-lists the
owned leaf-control events as a convenience projection — the partition
authority is the slices (documented deviation from the RFC example,
where the two surfaces are disjoint).

``visible_text_event_ranges`` from the RFC example are omitted in this
V0 envelope (closed schema; adding them later is a schema MINOR bump) —
they are mechanically derivable from RUN_TEXT span ownership.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from unflatten.ast.hashing import ast_hash
from unflatten.ast.issues import Issue
from unflatten.dataset.enums import WINDOW_POLICY
from unflatten.dataset.fragments.identity import FRAGMENT_SCHEMA_VERSION, fragment_id
from unflatten.dataset.identity import domain_hash

FRAGMENT_INDEX_SCHEMA_VERSION = "0.2.0"

#: RFC003 §26 builder identity. `config_hash` covers the builder's only
#: configuration input, the window policy; the implementation itself is
#: pinned one level up by the M2b derivation hash.
BUILDER_NAME = "page-fragment-builder"
#: 0.3.0 (RFC003-ERRATA-002 E3): `_resolve_pages` gained rule 0 (a page-
#: scoped story's unobserved descendant inherits its story ancestor's
#: full resolved page set) and `_text_partition_issues` now checks the
#: partition per (node, fragment) for page-scoped story nodes instead of
#: per node. Both change which slices a fragment carries, so this feeds
#: `fragment_id` (via `builder`) — every fragment_id changes.
BUILDER_VERSION = "0.3.0"

_BUILDER_CONFIG_DOMAIN = "rfc003.fragment-builder-config/v1"
_FRAGMENT_INDEX_DOMAIN = "rfc003.fragment-index/v1"
_SIDECAR_HASH_DOMAIN = "rfc003.observation-sidecar/v1"

#: Node types that put ink on the page without emitting glyphs the
#: aligner can match. Their placement is real but unevidenced, so an
#: unobserved one is `unresolved`, never `structural`.
#: `floating_image` is an anchored image: exactly the same case as
#: `inline_image` — real pixels, no matchable glyph — so it belongs
#: here, and calling it `structural` would assert that the event
#: partition fixes a placement nobody derived.
INK_WITHOUT_GLYPHS: frozenset[str] = frozenset({"inline_image", "floating_image"})

#: §29.4 label origin of a slice's PLACEMENT, by page evidence.
PLACEMENT_ORIGIN: dict[str, str] = {
    "observed": "mapping_deterministic",
    "structural": "mapping_deterministic",
    "unresolved": "imputed_for_render",
}

#: Page evidence values whose placement is derivable from this view, and
#: which may therefore carry supervision (§29.2/§29.3).
OBSERVABLE_EVIDENCE: frozenset[str] = frozenset({"observed", "structural"})

#: leaf control node types (open + immediate close in the event stream).
CONTROL_NODE_TYPES: frozenset[str] = frozenset(
    {"tab", "line_break", "page_break", "column_break"}
)

#: page-scoped story roots (RFC003-ERRATA-002 E1/E2, DEC-033 §3). A pure
#: structural fact (`_Tree.node_types`), so `_resolve_pages` rule 0 needs
#: no observations-package import: the envelope stays a leaf of the
#: converter closure (`unflatten.model.converter_hash`).
STORY_NODE_TYPES: frozenset[str] = frozenset({"header", "footer"})

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "fragment_envelope.schema.json"


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


# ------------------------------------------------------------------ builder --


def fragment_index_hash(index: dict[str, Any]) -> str:
    """`H("rfc003.fragment-index/v1", projection)` for a fragment index.

    The projection is the index minus its own hash field, with each entry
    reduced to ``(sequence_index, fragment_id, page_indices)``. Storing
    the index appends a transport ``ArtifactRef`` to every entry; per
    §12.1 a transport path is not semantic identity, and the entry's
    ``fragment_id`` already binds the fragment's full content, so the
    hash is deliberately stable across that append — an index validates
    the same before and after it is written to the store.

    ``task_view_id`` and the ``fragment_policy_*`` triple the RFC lists
    are absent for the reason given in
    :mod:`unflatten.dataset.fragments.identity`: no such artifact exists
    to bind to, and a fabricated one would be worse than a gap.
    """
    return domain_hash(
        _FRAGMENT_INDEX_DOMAIN,
        {
            "fragment_index_schema_version": index["fragment_index_schema_version"],
            "ast_hash": index["ast_hash"],
            "window_policy": index["window_policy"],
            "fragment_count": index["fragment_count"],
            "entries": [
                {
                    "sequence_index": entry["sequence_index"],
                    "fragment_id": entry["fragment_id"],
                    "page_indices": list(entry["page_indices"]),
                }
                for entry in index["entries"]
            ],
        },
    )


def observation_sidecar_hash(sidecar: dict[str, Any]) -> str:
    """`H("rfc003.observation-sidecar/v1", sidecar)` — the digest of the
    exact sidecar a fragment is a view of."""
    return domain_hash(_SIDECAR_HASH_DOMAIN, sidecar)


def _builder(window_policy: str) -> dict[str, Any]:
    return {
        "name": BUILDER_NAME,
        "version": BUILDER_VERSION,
        "config_hash": domain_hash(
            _BUILDER_CONFIG_DOMAIN, {"window_policy": window_policy}
        ),
    }


def build_fragments(
    canonical_ast: dict[str, Any],
    target_bundle: dict[str, Any],
    sidecar: dict[str, Any],
    window_policy: str,
    *,
    document_bundle_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Derive fragments + their fragment index for one rendered document.

    All three inputs must describe the same AST (`ast_hash` equality is
    enforced); the sidecar must have at least one page. Returns
    ``(fragments, fragment_index)``.

    `document_bundle_id` binds the fragments to the DocumentBundle they
    belong to; `None` records "not bound", which is the truth for the
    synthetic factory. It is NOT a placeholder — see
    :mod:`unflatten.dataset.fragments.identity` on why an unbound closure
    field is an explicit null.
    """
    doc_hash = ast_hash(canonical_ast)
    for name, carried in (
        ("target_bundle", target_bundle["ast_hash"]),
        ("sidecar", sidecar["ast_hash"]),
    ):
        if carried != doc_hash:
            raise ValueError(
                f"{name} ast_hash {carried!r} does not match the canonical AST "
                f"({doc_hash!r}) — fragments must bind one document."
            )
    bundle_hash = target_bundle.get("target_bundle_hash")
    if bundle_hash is None:
        raise ValueError(
            "target_bundle carries no target_bundle_hash — a fragment must bind "
            "the target authority it is a view of (RFC003 §26/§27)."
        )
    sidecar_hash = observation_sidecar_hash(sidecar)
    builder = _builder(window_policy)
    pages = sidecar["pages"]
    if not pages:
        raise ValueError("sidecar has no pages — cannot derive page fragments.")
    observed_indices = [page["page_index"] for page in pages]
    if observed_indices != list(range(len(pages))):
        raise ValueError(
            f"sidecar page_index values must be 0-based contiguous, got "
            f"{observed_indices!r} — fragment windows index pages positionally."
        )
    windows = _windows(window_policy, len(pages))

    tree = _Tree(target_bundle["structural"])
    obs_pages, obs_ranges = _observations_by_node(pages)
    resolved, observed = _resolve_pages(tree, obs_pages)
    evidence = _page_evidence(tree, observed, target_bundle["text"]["spans"])

    window_sets = [frozenset(w) for w in windows]
    # first fragment whose pages intersect the node's resolved pages
    structure_owner = [
        min(k for k, w in enumerate(window_sets) if w & set(resolved[i]))
        for i in range(tree.n)
    ]
    # first fragment containing a given page (text-range ownership)
    page_owner = {
        p: min(k for k, w in enumerate(window_sets) if p in w) for p in range(len(pages))
    }

    fragments: list[dict[str, Any]] = []
    for k, window in enumerate(windows):
        window_set = window_sets[k]
        slices: list[dict[str, Any]] = []
        controls: list[dict[str, Any]] = []
        for i in range(tree.n):
            node_pages = resolved[i]
            visible = sorted(set(node_pages) & window_set)
            if not visible:
                continue
            node_id = tree.node_order[i]
            node_type = tree.node_types[i]
            owns_structure = structure_owner[i] == k
            char_ranges = _owned_char_ranges(obs_ranges.get(node_id, {}), page_owner, k)
            observable = evidence[i] in OBSERVABLE_EVIDENCE
            slices.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "continuation": _continuation(node_pages, visible),
                    "structure_owner": owns_structure,
                    "typed_target_owner": owns_structure,
                    "text_range_owner": bool(char_ranges),
                    "page_evidence": evidence[i],
                    "placement_origin": PLACEMENT_ORIGIN[evidence[i]],
                    "observable_mask": observable,
                    # §29.3: supervision requires observability. Ownership
                    # above is unaffected — it is partition bookkeeping.
                    "supervision_mask": observable,
                    "structural_event_ranges": (
                        _compress(tree.events_of[node_id]) if owns_structure else []
                    ),
                    "char_ranges": char_ranges,
                }
            )
            if owns_structure and node_type in CONTROL_NODE_TYPES:
                open_event = tree.open_event_of[i]
                controls.append(
                    {
                        "node_id": node_id,
                        "node_type": node_type,
                        "structural_event_range": {
                            "start": open_event,
                            "end": tree.close_event_index[i] + 1,
                        },
                    }
                )
        page_indices = list(window)
        page_records = [
            {
                "page_index": pages[p]["page_index"],
                "width_px": pages[p]["width_px"],
                "height_px": pages[p]["height_px"],
            }
            for p in window
        ]
        fragment = {
            "fragment_schema_version": FRAGMENT_SCHEMA_VERSION,
            "fragment_id": fragment_id(
                ast_hash=doc_hash,
                document_bundle_id=document_bundle_id,
                task_view_id=None,
                target_bundle_hash=bundle_hash,
                observation_sidecar_hash=sidecar_hash,
                builder=builder,
                window_policy=window_policy,
                sequence_index=k,
                page_indices=page_indices,
                pages=page_records,
                renderer=sidecar["renderer"],
                slices=slices,
                control_event_ownership=controls,
            ),
            "ast_hash": doc_hash,
            "document_bundle_id": document_bundle_id,
            # no TaskView artifact exists to bind to; an invented ID would
            # be a claim nobody can verify (see fragments.identity)
            "task_view_id": None,
            "target_bundle_hash": bundle_hash,
            "observation_sidecar_hash": sidecar_hash,
            "builder": builder,
            "window_policy": window_policy,
            "sequence_index": k,
            "page_indices": page_indices,
            "pages": page_records,
            # ancestors of visible nodes are themselves visible under the
            # descendant-union page rule, so the RFC ancestor_context is
            # always empty here (kept for schema/shape compatibility).
            "ancestor_context": [],
            "slices": slices,
            "control_event_ownership": controls,
        }
        fragments.append(fragment)

    index: dict[str, Any] = {
        "fragment_index_schema_version": FRAGMENT_INDEX_SCHEMA_VERSION,
        "ast_hash": doc_hash,
        "window_policy": window_policy,
        "fragment_count": len(fragments),
        "entries": [
            {
                "sequence_index": f["sequence_index"],
                "fragment_id": f["fragment_id"],
                "page_indices": f["page_indices"],
            }
            for f in fragments
        ],
    }
    index["fragment_index_hash"] = fragment_index_hash(index)
    return fragments, index


def supervised_event_ranges(fragment: dict[str, Any]) -> list[dict[str, int]]:
    """Structural event ranges this fragment may be SUPERVISED on.

    The supervised subset of the owned ranges: slices whose
    ``supervision_mask`` is false (an ``unresolved`` placement) own their
    events for partition completeness but contribute no target. Target
    builders must use this, not ``structural_event_ranges``.
    """
    return _compress(
        [
            index
            for entry in fragment.get("slices", ())
            if entry.get("supervision_mask")
            for rng in entry["structural_event_ranges"]
            for index in range(rng["start"], rng["end"])
        ]
    )


def _windows(window_policy: str, page_count: int) -> list[list[int]]:
    if window_policy not in WINDOW_POLICY:
        raise ValueError(f"unknown window_policy {window_policy!r}")
    if window_policy == "single_page":
        return [[p] for p in range(page_count)]
    if window_policy == "two_page_overlap":
        if page_count == 1:
            return [[0]]
        return [[p, p + 1] for p in range(page_count - 1)]
    if window_policy == "full_document":
        return [list(range(page_count))]
    raise NotImplementedError(
        "bounded_multi_page windows are deferred (V0 implements single_page, "
        "two_page_overlap and full_document; see module docstring)."
    )


class _Tree:
    """Node tree derived from the target bundle's structural arrays."""

    def __init__(self, structural: dict[str, Any]) -> None:
        self.node_order: list[str] = structural["node_order"]
        self.node_types: list[str] = structural["node_types"]
        self.close_event_index: list[int] = structural["close_event_index"]
        self.n = len(self.node_order)

        events = structural["events"]
        opens = [e["index"] for e in events if e["op"] == "open"]
        ordinal_by_open = {open_index: i for i, open_index in enumerate(opens)}
        self.open_event_of: list[int] = opens
        self.parent_ordinal: list[int] = [
            ordinal_by_open[p] if p >= 0 else -1 for p in structural["parent_index"]
        ]
        self.children: dict[int, list[int]] = {}
        for i, parent in enumerate(self.parent_ordinal):
            if parent >= 0:
                self.children.setdefault(parent, []).append(i)
        # every event index carried by a node (open/text/close), by node_id
        self.events_of: dict[str, list[int]] = {}
        for event in events:
            self.events_of.setdefault(event["node_id"], []).append(event["index"])


def _observations_by_node(
    pages: list[dict[str, Any]],
) -> tuple[dict[str, set[int]], dict[str, dict[int, list[tuple[int, int]]]]]:
    """Per node: the set of observed pages, and char ranges per page.

    An image observation (RFC003 §24.3/§25) carries no char range — it
    still makes its node ``observed`` on that page, but contributes no
    entry to ``obs_ranges``.
    """
    obs_pages: dict[str, set[int]] = {}
    obs_ranges: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for page in pages:
        page_index = page["page_index"]
        for obs in page["observations"]:
            node_id = obs["node_id"]
            obs_pages.setdefault(node_id, set()).add(page_index)
            if "char_start" not in obs:
                continue
            obs_ranges.setdefault(node_id, {}).setdefault(page_index, []).append(
                (obs["char_start"], obs["char_end"])
            )
    return obs_pages, obs_ranges


def _chain_scoped_node_ids(sidecar: dict[str, Any]) -> frozenset[str]:
    """Node ids with at least one ``chain_scope: "page"`` observation
    (RFC003-ERRATA-002 E3): each page's full text is its own closed
    chain, so :func:`_text_partition_issues` checks these per (node,
    fragment) instead of merging across the whole document."""
    return frozenset(
        obs["node_id"]
        for page in sidecar["pages"]
        for obs in page["observations"]
        if obs.get("chain_scope") == "page"
    )


def _resolve_pages(
    tree: _Tree, obs_pages: dict[str, set[int]]
) -> tuple[list[list[int]], list[bool]]:
    """Resolved page set per node ordinal (see module docstring rule).

    Returns the resolved pages and, per ordinal, whether the placement
    rests on an observation of the node or of one of its descendants —
    the input of :func:`_page_evidence`.

    Rule 0 (RFC003-ERRATA-002 E1, DEC-033 §3) runs BEFORE rules 1-3: an
    unobserved node whose PARENT lies inside a page-scoped story
    (``under_story``, True for a :data:`STORY_NODE_TYPES` root and every
    one of its descendants, computed from ``tree.node_types`` alone — a
    purely structural fact, so it needs no observations-package import)
    inherits the parent's full resolved page set — e.g. a bare
    `page_break` between two runs inside a repeated header. This must
    not fall through to the sibling rules below: an empty run would
    otherwise borrow the PREVIOUS SIBLING'S page, which is simply the
    wrong page for a node whose real ancestor repeats on every page of
    the section. The story ROOT itself is excluded (its own parent, the
    section, is never ``under_story``), so a glyphless header/footer
    still falls through to rules 1-3 and stays on one page (design R5)
    — there is no observed story ancestor to inherit from.
    """
    raw: list[set[int]] = [set(obs_pages.get(tree.node_order[i], ())) for i in range(tree.n)]
    for i in range(tree.n - 1, 0, -1):  # postorder union into parents
        raw[tree.parent_ordinal[i]] |= raw[i]

    under_story: list[bool] = [False] * tree.n
    for i in range(tree.n):  # preorder: a header/footer's ancestry is set first
        under_story[i] = tree.node_types[i] in STORY_NODE_TYPES or (
            i > 0 and under_story[tree.parent_ordinal[i]]
        )

    resolved: list[list[int]] = [[] for _ in range(tree.n)]
    for i in range(tree.n):  # preorder: parents and earlier siblings first
        if raw[i]:
            resolved[i] = sorted(raw[i])
            continue
        if i > 0 and under_story[tree.parent_ordinal[i]]:
            resolved[i] = list(resolved[tree.parent_ordinal[i]])
            continue
        siblings = tree.children.get(tree.parent_ordinal[i], []) if i > 0 else []
        position = siblings.index(i) if siblings else 0
        pages: list[int] | None = None
        for j in reversed(siblings[:position]):  # 1. previous sibling's last page
            if resolved[j]:
                pages = [resolved[j][-1]]
                break
        if pages is None:  # 2. next sibling's first page (raw only)
            for j in siblings[position + 1 :]:
                if raw[j]:
                    pages = [min(raw[j])]
                    break
        if pages is None:  # 3. parent's page (root: page 0)
            pages = [resolved[tree.parent_ordinal[i]][0]] if i > 0 else [0]
        resolved[i] = pages
    return resolved, [bool(pages) for pages in raw]


def _page_evidence(
    tree: _Tree, observed: list[bool], spans: list[dict[str, Any]]
) -> list[str]:
    """Provenance of each node's placement (see module docstring).

    A node that was not observed is ``unresolved`` when its subtree puts
    ink on the page — visible text the alignment failed to find, or an
    :data:`INK_WITHOUT_GLYPHS` node it could never have found — and
    ``structural`` when it puts none, the only case where a borrowed
    page is sound by construction.

    Ink propagates UP only. A paragraph whose own text was observed stays
    ``observed`` even when it holds an unobservable image: it is the
    image node whose placement is unevidenced, not its parent's. A
    paragraph holding nothing but that image is itself unresolved.
    """
    has_ink = [
        tree.node_types[i] in INK_WITHOUT_GLYPHS for i in range(tree.n)
    ]
    ordinal_of = {node_id: i for i, node_id in enumerate(tree.node_order)}
    for span in spans:
        if span["text"].strip():
            has_ink[ordinal_of[span["node_id"]]] = True
    for i in range(tree.n - 1, 0, -1):  # postorder union into parents
        if has_ink[i]:
            has_ink[tree.parent_ordinal[i]] = True
    return [
        "observed" if observed[i] else ("unresolved" if has_ink[i] else "structural")
        for i in range(tree.n)
    ]


def _owned_char_ranges(
    ranges_by_page: dict[int, list[tuple[int, int]]],
    page_owner: dict[int, int],
    fragment: int,
) -> list[dict[str, int]]:
    owned = sorted(
        rng
        for page, ranges in ranges_by_page.items()
        if page_owner[page] == fragment
        for rng in ranges
    )
    merged: list[list[int]] = []
    for start, end in owned:
        if merged and merged[-1][1] == start:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [{"start": start, "end": end} for start, end in merged]


def _continuation(node_pages: list[int], visible: list[int]) -> str:
    if visible == node_pages:
        return "complete"
    if node_pages[0] in visible:
        return "start"
    if node_pages[-1] in visible:
        return "end"
    return "middle"


def _compress(indexes: list[int]) -> list[dict[str, int]]:
    ranges: list[list[int]] = []
    for index in sorted(indexes):
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index, index + 1])
    return [{"start": start, "end": end} for start, end in ranges]


# ---------------------------------------------------------------- validator --


def validate_fragments(
    fragments: list[dict[str, Any]],
    fragment_index: dict[str, Any],
    target_bundle: dict[str, Any],
    sidecar: dict[str, Any],
) -> list[Issue]:
    """Check the §26.3 invariants; returns Issues, never raises on invalid
    input. Schema violations short-circuit the semantic checks."""
    issues: list[Issue] = []
    for k, fragment in enumerate(fragments):
        issues.extend(
            Issue(
                code="FRAGMENT_SCHEMA_VIOLATION",
                severity="error",
                node_id=None,
                path=f"/fragments/{k}/" + "/".join(str(p) for p in error.absolute_path),
                message=error.message,
            )
            for error in sorted(
                _validator().iter_errors(fragment), key=lambda e: str(e.absolute_path)
            )
        )
    if issues:
        return issues
    issues.extend(_hash_issues(fragments, fragment_index, target_bundle, sidecar))
    issues.extend(_index_issues(fragments, fragment_index))
    issues.extend(_page_issues(fragments, sidecar))
    issues.extend(_ownership_issues(fragments, target_bundle))
    issues.extend(_evidence_issues(fragments))
    issues.extend(_event_partition_issues(fragments, target_bundle))
    issues.extend(_slice_event_issues(fragments, target_bundle))
    issues.extend(_text_partition_issues(fragments, sidecar, target_bundle))
    issues.extend(_control_issues(fragments, target_bundle))
    return issues


def _issue(code: str, path: str, message: str, node_id: str | None = None) -> Issue:
    return Issue(code=code, severity="error", node_id=node_id, path=path, message=message)


def _hash_issues(
    fragments: list[dict[str, Any]],
    fragment_index: dict[str, Any],
    target_bundle: dict[str, Any],
    sidecar: dict[str, Any],
) -> list[Issue]:
    issues: list[Issue] = []
    expected = target_bundle["ast_hash"]
    if sidecar["ast_hash"] != expected:
        issues.append(
            _issue(
                "FRAGMENT_HASH_MISMATCH",
                "/",
                "target bundle and sidecar carry different ast_hash values.",
            )
        )
    # the fragment must bind the authorities it claims to be a view of
    expected_bundle_hash = target_bundle.get("target_bundle_hash")
    expected_sidecar_hash = observation_sidecar_hash(sidecar)
    for k, fragment in enumerate(fragments):
        for path, carried, expected_value in (
            ("target_bundle_hash", fragment["target_bundle_hash"], expected_bundle_hash),
            (
                "observation_sidecar_hash",
                fragment["observation_sidecar_hash"],
                expected_sidecar_hash,
            ),
        ):
            if carried != expected_value:
                issues.append(
                    _issue(
                        "FRAGMENT_HASH_MISMATCH",
                        f"/fragments/{k}/{path}",
                        f"{path} {carried!r} does not match the artifact it binds "
                        f"({expected_value!r}).",
                    )
                )
    carriers = [(f"/fragments/{k}/ast_hash", f["ast_hash"]) for k, f in enumerate(fragments)]
    carriers.append(("/fragment_index/ast_hash", fragment_index.get("ast_hash")))
    issues.extend(
        _issue(
            "FRAGMENT_HASH_MISMATCH",
            path,
            f"ast_hash {carried!r} does not match the target bundle ({expected!r}).",
        )
        for path, carried in carriers
        if carried != expected
    )
    for k, fragment in enumerate(fragments):
        recomputed = fragment_id(
            ast_hash=fragment["ast_hash"],
            document_bundle_id=fragment["document_bundle_id"],
            task_view_id=fragment["task_view_id"],
            target_bundle_hash=fragment["target_bundle_hash"],
            observation_sidecar_hash=fragment["observation_sidecar_hash"],
            builder=fragment["builder"],
            window_policy=fragment["window_policy"],
            sequence_index=fragment["sequence_index"],
            page_indices=fragment["page_indices"],
            pages=fragment["pages"],
            renderer=sidecar["renderer"],
            slices=fragment["slices"],
            control_event_ownership=fragment["control_event_ownership"],
        )
        if fragment["fragment_id"] != recomputed:
            issues.append(
                _issue(
                    "FRAGMENT_HASH_MISMATCH",
                    f"/fragments/{k}/fragment_id",
                    f"fragment_id does not recompute from the envelope "
                    f"(expected {recomputed}).",
                )
            )
    return issues


def _index_issues(
    fragments: list[dict[str, Any]], fragment_index: dict[str, Any]
) -> list[Issue]:
    issues: list[Issue] = []
    entries = fragment_index.get("entries", [])
    try:
        recomputed = fragment_index_hash(fragment_index)
    except (KeyError, TypeError):
        recomputed = None
    if recomputed is None or fragment_index.get("fragment_index_hash") != recomputed:
        issues.append(
            _issue(
                "FRAGMENT_HASH_MISMATCH",
                "/fragment_index/fragment_index_hash",
                "fragment_index_hash does not recompute from the index"
                + (f" (expected {recomputed})." if recomputed else " (malformed index)."),
            )
        )
    if fragment_index.get("fragment_count") != len(fragments) or len(entries) != len(
        fragments
    ):
        issues.append(
            _issue(
                "FRAGMENT_INDEX_INVALID",
                "/fragment_index/fragment_count",
                "fragment_count/entries do not match the fragment list.",
            )
        )
        return issues
    for k, (fragment, entry) in enumerate(zip(fragments, entries)):
        if fragment["sequence_index"] != k or entry.get("sequence_index") != k:
            issues.append(
                _issue(
                    "FRAGMENT_INDEX_INVALID",
                    f"/fragments/{k}/sequence_index",
                    f"sequence_index must be unique and contiguous from 0 "
                    f"(expected {k}).",
                )
            )
        elif entry.get("fragment_id") != fragment["fragment_id"] or (
            entry.get("page_indices") != fragment["page_indices"]
        ):
            issues.append(
                _issue(
                    "FRAGMENT_INDEX_INVALID",
                    f"/fragment_index/entries/{k}",
                    "index entry does not mirror the fragment envelope.",
                )
            )
    return issues


def _page_issues(fragments: list[dict[str, Any]], sidecar: dict[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    frozen = {
        page["page_index"]: (page["width_px"], page["height_px"])
        for page in sidecar["pages"]
    }
    for k, fragment in enumerate(fragments):
        if fragment["page_indices"] != [p["page_index"] for p in fragment["pages"]]:
            issues.append(
                _issue(
                    "FRAGMENT_PAGE_INVALID",
                    f"/fragments/{k}/page_indices",
                    "page_indices do not mirror the pages records.",
                )
            )
        for j, page in enumerate(fragment["pages"]):
            if frozen.get(page["page_index"]) != (page["width_px"], page["height_px"]):
                issues.append(
                    _issue(
                        "FRAGMENT_PAGE_INVALID",
                        f"/fragments/{k}/pages/{j}",
                        "fragment page is not a parent frozen page (index/dimension "
                        "mismatch with the Observation Sidecar).",
                    )
                )
    return issues


def _ownership_issues(
    fragments: list[dict[str, Any]], target_bundle: dict[str, Any]
) -> list[Issue]:
    issues: list[Issue] = []
    owners: dict[str, int] = {}
    for k, fragment in enumerate(fragments):
        for j, entry in enumerate(fragment["slices"]):
            if entry["typed_target_owner"] != entry["structure_owner"]:
                issues.append(
                    _issue(
                        "TYPED_OWNER_VIOLATION",
                        f"/fragments/{k}/slices/{j}",
                        "typed_target_owner must equal structure_owner in V0.",
                        node_id=entry["node_id"],
                    )
                )
            if entry["structure_owner"]:
                owners[entry["node_id"]] = owners.get(entry["node_id"], 0) + 1
    for node_id in target_bundle["structural"]["node_order"]:
        count = owners.pop(node_id, 0)
        if count != 1:
            issues.append(
                _issue(
                    "STRUCTURE_OWNER_VIOLATION",
                    "/fragments",
                    f"node {node_id!r} has {count} structure owner(s); exactly one "
                    "is required.",
                    node_id=node_id,
                )
            )
    issues.extend(
        _issue(
            "STRUCTURE_OWNER_VIOLATION",
            "/fragments",
            f"structure owner declared for unknown node {node_id!r}.",
            node_id=node_id,
        )
        for node_id in owners
    )
    return issues


def _evidence_issues(fragments: list[dict[str, Any]]) -> list[Issue]:
    """``page_evidence`` is a property of the NODE, not of the window:
    every slice of a node must carry the same value, and a slice that
    owns observed characters cannot claim its page was inferred."""
    issues: list[Issue] = []
    seen: dict[str, str] = {}
    for k, fragment in enumerate(fragments):
        for j, entry in enumerate(fragment["slices"]):
            evidence = entry["page_evidence"]
            node_id = entry["node_id"]
            if entry["char_ranges"] and evidence != "observed":
                issues.append(
                    _issue(
                        "PAGE_EVIDENCE_VIOLATION",
                        f"/fragments/{k}/slices/{j}/page_evidence",
                        f"slice of node {node_id!r} owns observed char ranges but "
                        f"declares page_evidence {evidence!r}.",
                        node_id=node_id,
                    )
                )
            if entry["placement_origin"] != PLACEMENT_ORIGIN.get(evidence):
                issues.append(
                    _issue(
                        "PAGE_EVIDENCE_VIOLATION",
                        f"/fragments/{k}/slices/{j}/placement_origin",
                        f"placement_origin {entry['placement_origin']!r} does not "
                        f"follow from page_evidence {evidence!r} (expected "
                        f"{PLACEMENT_ORIGIN.get(evidence)!r}).",
                        node_id=node_id,
                    )
                )
            if entry["observable_mask"] != (evidence in OBSERVABLE_EVIDENCE):
                issues.append(
                    _issue(
                        "SUPERVISION_SCOPE_VIOLATION",
                        f"/fragments/{k}/slices/{j}/observable_mask",
                        f"observable_mask {entry['observable_mask']!r} does not "
                        f"follow from page_evidence {evidence!r}.",
                        node_id=node_id,
                    )
                )
            if entry["supervision_mask"] and not entry["observable_mask"]:
                issues.append(
                    _issue(
                        "SUPERVISION_SCOPE_VIOLATION",
                        f"/fragments/{k}/slices/{j}/supervision_mask",
                        "supervision_mask requires observable_mask (RFC003 §29.3); "
                        f"node {node_id!r} is placed on an imputed page.",
                        node_id=node_id,
                    )
                )
            previous = seen.setdefault(node_id, evidence)
            if previous != evidence:
                issues.append(
                    _issue(
                        "PAGE_EVIDENCE_VIOLATION",
                        f"/fragments/{k}/slices/{j}/page_evidence",
                        f"node {node_id!r} carries page_evidence {evidence!r} here "
                        f"and {previous!r} in an earlier fragment.",
                        node_id=node_id,
                    )
                )
    return issues


def _event_partition_issues(
    fragments: list[dict[str, Any]], target_bundle: dict[str, Any]
) -> list[Issue]:
    owned: list[int] = []
    for fragment in fragments:
        for entry in fragment["slices"]:
            for rng in entry["structural_event_ranges"]:
                owned.extend(range(rng["start"], rng["end"]))
    total = len(target_bundle["structural"]["events"])
    if sorted(owned) != list(range(total)):
        return [
            _issue(
                "EVENT_RANGE_PARTITION_VIOLATION",
                "/fragments",
                f"owned structural event ranges do not partition the full stream "
                f"of {total} events (owned {len(owned)}, distinct {len(set(owned))}).",
            )
        ]
    return []


def _slice_event_issues(
    fragments: list[dict[str, Any]], target_bundle: dict[str, Any]
) -> list[Issue]:
    """Per-slice event attribution (D4 should_fix 2): the global partition
    check alone lets two owned slices SWAP their structural_event_ranges.
    Each owning slice must own exactly the events of its OWN node; a
    non-owning slice must own none."""
    issues: list[Issue] = []
    events_of: dict[str, set[int]] = {}
    for event in target_bundle["structural"]["events"]:
        events_of.setdefault(event["node_id"], set()).add(event["index"])
    for k, fragment in enumerate(fragments):
        for j, entry in enumerate(fragment["slices"]):
            owned: set[int] = set()
            for rng in entry["structural_event_ranges"]:
                owned.update(range(rng["start"], rng["end"]))
            expected = (
                events_of.get(entry["node_id"], set())
                if entry["structure_owner"]
                else set()
            )
            if owned != expected:
                issues.append(
                    _issue(
                        "SLICE_EVENT_MISMATCH",
                        f"/fragments/{k}/slices/{j}/structural_event_ranges",
                        f"slice of node {entry['node_id']!r} owns event indexes "
                        f"{sorted(owned)} but must own exactly {sorted(expected)} "
                        "(its own node's events; empty unless structure_owner).",
                        node_id=entry["node_id"],
                    )
                )
    return issues


def _text_partition_issues(
    fragments: list[dict[str, Any]],
    sidecar: dict[str, Any],
    target_bundle: dict[str, Any],
) -> list[Issue]:
    """RFC003-ERRATA-002 E3.

    Body nodes: owned char ranges must concatenate losslessly to the
    node's full text ACROSS THE WHOLE DOCUMENT, unchanged from §26.3.

    Page-scoped story nodes (header/footer, ``chain_scope: "page"``,
    §3): the full range is a self-contained chain PER PAGE, so it may
    legitimately be owned again on every page that renders it. The
    partition is checked per (node, fragment) instead.
    """
    issues: list[Issue] = []
    # full extent per observed node: sidecar chains are validated elsewhere
    # to partition [0, end) — the owned ranges must re-partition the same.
    full_extent: dict[str, int] = {}
    for page in sidecar["pages"]:
        for obs in page["observations"]:
            if "char_end" not in obs:  # image observation: no char range to partition
                continue
            node_id = obs["node_id"]
            full_extent[node_id] = max(full_extent.get(node_id, 0), obs["char_end"])
    span_lengths: dict[str, int] = {}
    for span in target_bundle["text"]["spans"]:
        span_lengths[span["node_id"]] = span_lengths.get(span["node_id"], 0) + len(
            span["text"]
        )
    page_scoped_ids = _chain_scoped_node_ids(sidecar)
    # A repeated container's synthetic text can differ on each page.
    # Match the builder's first-fragment ownership of each page rather
    # than requiring the longest occurrence everywhere.
    page_owner: dict[int, int] = {}
    for k, fragment in enumerate(fragments):
        for page_index in fragment.get("page_indices", []):
            page_owner.setdefault(page_index, k)
    scoped_extent: dict[tuple[int, str], int] = {}
    for page in sidecar["pages"]:
        owner = page_owner.get(page.get("page_index"))
        if owner is None:
            continue
        for obs in page["observations"]:
            if obs["node_id"] in page_scoped_ids and "char_end" in obs:
                key = (owner, obs["node_id"])
                scoped_extent[key] = max(scoped_extent.get(key, 0), obs["char_end"])

    def _partition_issue(node_id: str, expected_end: int, ranges: list[tuple[int, int]]) -> bool:
        """Checks one range set against ``[0, expected_end)``, appending an
        issue and returning False on a gap/overlap/mismatch."""
        cursor = 0
        for start, end in ranges:
            if start != cursor or end <= start:
                cursor = -1
                break
            cursor = end
        if cursor != expected_end:
            issues.append(
                _issue(
                    "TEXT_RANGE_PARTITION_VIOLATION",
                    "/fragments",
                    f"owned char ranges of node {node_id!r} do not concatenate "
                    f"losslessly to its full text [0, {expected_end}).",
                    node_id=node_id,
                )
            )
            return False
        # for run nodes the observed extent must also cover the bundle text
        if node_id in span_lengths and span_lengths[node_id] != expected_end:
            issues.append(
                _issue(
                    "TEXT_RANGE_PARTITION_VIOLATION",
                    "/fragments",
                    f"node {node_id!r} owns [0, {expected_end}) but its target-bundle "
                    f"text has {span_lengths[node_id]} scalar(s).",
                    node_id=node_id,
                )
            )
        return True

    owned_by_node: dict[str, list[tuple[int, int]]] = {}
    owned_by_fragment_node: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for k, fragment in enumerate(fragments):
        for entry in fragment["slices"]:
            node_id = entry["node_id"]
            for rng in entry["char_ranges"]:
                if node_id in page_scoped_ids:
                    owned_by_fragment_node.setdefault((k, node_id), []).append(
                        (rng["start"], rng["end"])
                    )
                else:
                    owned_by_node.setdefault(node_id, []).append((rng["start"], rng["end"]))

    for node_id, expected_end in sorted(full_extent.items()):
        if node_id in page_scoped_ids:
            continue
        # Fragment sequence order is page order, not text order: a table
        # row spanning a page break wraps its cells independently per
        # column, so its owned ranges can arrive page0/page1/page0/page1.
        # Sort by start before checking contiguity — a real gap or overlap
        # in the range set stays detected regardless of arrival order.
        _partition_issue(node_id, expected_end, sorted(owned_by_node.pop(node_id, [])))
    issues.extend(
        _issue(
            "TEXT_RANGE_PARTITION_VIOLATION",
            "/fragments",
            f"char ranges owned for node {node_id!r} which has no observations.",
            node_id=node_id,
        )
        for node_id in owned_by_node
    )

    for (k, node_id), ranges in sorted(owned_by_fragment_node.items()):
        expected_end = scoped_extent.get((k, node_id))
        if expected_end is None:
            issues.append(
                _issue(
                    "TEXT_RANGE_PARTITION_VIOLATION",
                    f"/fragments/{k}",
                    f"char ranges owned for node {node_id!r} which has no observations.",
                    node_id=node_id,
                )
            )
            continue
        _partition_issue(node_id, expected_end, sorted(ranges))
    return issues


def _control_issues(
    fragments: list[dict[str, Any]], target_bundle: dict[str, Any]
) -> list[Issue]:
    issues: list[Issue] = []
    structural = target_bundle["structural"]
    control_nodes = {
        node_id
        for node_id, node_type in zip(structural["node_order"], structural["node_types"])
        if node_type in CONTROL_NODE_TYPES
    }
    seen: set[str] = set()
    for k, fragment in enumerate(fragments):
        owned_events: dict[str, set[int]] = {}
        for entry in fragment["slices"]:
            for rng in entry["structural_event_ranges"]:
                owned_events.setdefault(entry["node_id"], set()).update(
                    range(rng["start"], rng["end"])
                )
        for j, control in enumerate(fragment["control_event_ownership"]):
            node_id = control["node_id"]
            rng = control["structural_event_range"]
            events = set(range(rng["start"], rng["end"]))
            if (
                node_id not in control_nodes
                or node_id in seen
                or not events <= owned_events.get(node_id, set())
            ):
                issues.append(
                    _issue(
                        "CONTROL_EVENT_OWNERSHIP_VIOLATION",
                        f"/fragments/{k}/control_event_ownership/{j}",
                        f"control entry for {node_id!r} is not a control node owned "
                        "exactly once by this fragment's slices.",
                        node_id=node_id,
                    )
                )
            seen.add(node_id)
    issues.extend(
        _issue(
            "CONTROL_EVENT_OWNERSHIP_VIOLATION",
            "/fragments",
            f"control node {node_id!r} missing from control_event_ownership.",
            node_id=node_id,
        )
        for node_id in sorted(control_nodes - seen)
    )
    return issues
