"""VC-1 rehearsal under DEC-054, discovery, not the formal VC-1 -- paired
page metrics beyond text similarity (DEC-054 point 4: typed properties,
table structure, regions).

Pure functions over what the eval already has: the packed token sequences
(decoded and gold) and the sealed page dict (`s2_gate.sealed.page_to_dict`,
the same JSON shape `builder_chain.visual_metrics.sealed_page_facts` and the
failure-taxonomy report read). No torch, no model.

* **Table structure** is read off the token sequence itself
  (`TABLE_OPEN` / `ROW_OPEN` / `CELL_OPEN`), so gold and every arm are
  measured by the same function: a table's shape is its tuple of per-row
  cell counts.
* **Typed properties and regions** are compared against the BYPASS decode of
  the same page (the frozen primary with full visual memory), since that is
  the reference a compressor must not move away from. Nodes of the two
  decodes are aligned by their `(node_type, text content)` sequence in
  document order (`difflib`, no junk heuristic); a bypass typed value on an
  unmatched node counts as a disagreement, never as missing. Region IoU is
  read on the same aligned pairs.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping, Sequence
from typing import Any

from unflatten.eval.failure_taxonomy.report import iou_ppm
from unflatten.model import events_vocab
from unflatten.model.composite_vocab import STRUCTURAL_COUNT, PackedRanges, StructuralRef, unpack

_TABLE_OPEN = events_vocab.TOKEN_IDS["TABLE_OPEN"]
_TABLE_CLOSE = events_vocab.TOKEN_IDS["TABLE_CLOSE"]
_ROW_OPEN = events_vocab.TOKEN_IDS["ROW_OPEN"]
_CELL_OPEN = events_vocab.TOKEN_IDS["CELL_OPEN"]


def table_shapes(tokens: Sequence[int], ranges: PackedRanges) -> list[tuple[int, ...]]:
    """Shape (cells per row) of every table in `tokens`, in `TABLE_OPEN`
    order; nested tables are their own entries. An unclosed table (a
    truncated decode) still reports what it opened."""
    shapes: list[list[int]] = []
    stack: list[int] = []
    for token in tokens:
        ref = unpack(int(token), ranges)
        if not isinstance(ref, StructuralRef):
            continue
        if ref.local_id == _TABLE_OPEN:
            shapes.append([])
            stack.append(len(shapes) - 1)
        elif ref.local_id == _TABLE_CLOSE and stack:
            stack.pop()
        elif ref.local_id == _ROW_OPEN and stack:
            shapes[stack[-1]].append(0)
        elif ref.local_id == _CELL_OPEN and stack and shapes[stack[-1]]:
            shapes[stack[-1]][-1] += 1
    return [tuple(shape) for shape in shapes]


def _states(page: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(page.get("node_states") or ())


_TEXT_KEYED = frozenset({"paragraph", "run"})


def _node_keys(page: Mapping[str, Any]) -> list[tuple[str, tuple[int, ...]]]:
    """`(node_type, text tokens inside the node)` per node, document order.
    A node's span is `accepted_tokens[open_event_index:close_event_index]`
    (event indices are token indices of the sealed hypothesis); text tokens
    are the packed ids `>= STRUCTURAL_COUNT`. Only paragraphs and runs are
    keyed by text: a container (section, header, table, cell, ...) spans its
    whole subtree, so one changed word anywhere would unmatch it; it aligns
    by type between its content-matched children instead. Leaves and pages
    without tokens get an empty text."""
    tokens = page.get("accepted_tokens") or ()
    keys = []
    for state in _states(page):
        start, end = state.get("open_event_index"), state.get("close_event_index")
        text: tuple[int, ...] = ()
        if (
            state["node_type"] in _TEXT_KEYED
            and tokens
            and isinstance(start, int)
            and isinstance(end, int)
        ):
            text = tuple(t for t in tokens[start:end] if t >= STRUCTURAL_COUNT)
        keys.append((state["node_type"], text))
    return keys


def align_nodes(
    page_a: Mapping[str, Any], page_b: Mapping[str, Any]
) -> list[tuple[str, str]]:
    """`(node id in a, node id in b)` for nodes matched by their
    `(node_type, text content)` sequence in document order -- by content, so
    an inserted empty paragraph or a dropped table does not shift every later
    pairing onto the wrong node."""
    states_a, states_b = _states(page_a), _states(page_b)
    matcher = difflib.SequenceMatcher(None, _node_keys(page_a), _node_keys(page_b), autojunk=False)
    pairs = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            pairs.append(
                (states_a[block.a + k]["local_node_id"], states_b[block.b + k]["local_node_id"])
            )
    return pairs


def typed_agreement(reference: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    """Share of the reference decode's typed values (node, pointer) that the
    arm reproduces exactly on the aligned node. `by_node_type` maps the
    reference node type to `[agree, n]`. Exact match suits an enum or a flag
    but not a length, so the values are also split: `categorical`
    (str / bool / None reference values) and `length_exact` (`*_mpt`
    pointers) as `[agree, n]`, plus `length_abs_error_mpt` where both sides
    carry an integer length on an aligned node."""
    node_type = {s["local_node_id"]: s["node_type"] for s in _states(reference)}
    mapping = dict(align_nodes(reference, arm))
    arm_values = {
        (t["local_node_id"], t["pointer"]): t["value"] for t in arm.get("typed_predictions") or ()
    }
    agree = 0
    total = 0
    by_type: dict[str, list[int]] = {}
    categorical = [0, 0]
    length_exact = [0, 0]
    length_errors: list[int] = []
    for item in reference.get("typed_predictions") or ():
        ref_node = item["local_node_id"]
        arm_node = mapping.get(ref_node)
        value = item["value"]
        arm_value = (
            _MISSING if arm_node is None else arm_values.get((arm_node, item["pointer"]), _MISSING)
        )
        hit = arm_value == value
        bucket = by_type.setdefault(node_type.get(ref_node, "?"), [0, 0])
        bucket[0] += int(hit)
        bucket[1] += 1
        agree += int(hit)
        total += 1
        if value is None or isinstance(value, bool | str):
            categorical[0] += int(hit)
            categorical[1] += 1
        elif item["pointer"].endswith("_mpt") and isinstance(value, int):
            length_exact[0] += int(hit)
            length_exact[1] += 1
            if isinstance(arm_value, int) and not isinstance(arm_value, bool):
                length_errors.append(abs(arm_value - value))
    return {
        "agree": agree,
        "n": total,
        "by_node_type": by_type,
        "categorical": categorical,
        "length_exact": length_exact,
        "length_abs_error_mpt": length_errors,
        "nodes_matched": len(mapping),
        "nodes": len(node_type),
    }


_MISSING = object()


def _first_box(region: Mapping[str, Any] | None) -> Sequence[int] | None:
    if not region or region.get("alignment_status") != "region" or not region.get("regions"):
        return None
    return region["regions"][0]["bbox_ppm"]


def region_agreement(reference: Mapping[str, Any], arm: Mapping[str, Any]) -> dict[str, Any]:
    """Node Visual Aligner output of the arm against the reference, on
    aligned nodes: alignment-status agreement, and the IoU (ppm) of the first
    region box where both aligned the node to a region."""
    ref_regions = {r["local_node_id"]: r for r in reference.get("region_predictions") or ()}
    arm_regions = {r["local_node_id"]: r for r in arm.get("region_predictions") or ()}
    status_agree = 0
    status_n = 0
    ious: list[int] = []
    for ref_node, arm_node in align_nodes(reference, arm):
        ref_region = ref_regions.get(ref_node)
        if ref_region is None:
            continue
        arm_region = arm_regions.get(arm_node)
        status_n += 1
        status_agree += int(
            arm_region is not None
            and arm_region.get("alignment_status") == ref_region.get("alignment_status")
        )
        ref_box, arm_box = _first_box(ref_region), _first_box(arm_region)
        if ref_box is not None and arm_box is not None:
            ious.append(iou_ppm(ref_box, arm_box))
    return {"status_agree": status_agree, "status_n": status_n, "iou_ppm": ious}
