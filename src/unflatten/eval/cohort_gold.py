"""Gold facts of one cohort page, read from an exact-text cache split.

`GoldRow` is the gold side of the failure-taxonomy report
(RFC005-ERRATA-007 E1): node-type counts, the first child of the first
section, image regions with their `page_evidence`, table geometry and the
text token sequence -- everything the report compares a sealed hypothesis
against, and nothing it does not read.

Rows are located through `manifest.json`'s `shard_index[*].keys`
(`[document_bundle_id, page_index]` per row, in shard order -- the format
`exact_text_cache.ExactTextCacheWriter` writes and its reader re-checks),
so only the shards that hold a requested page are opened. Rows load with
`torch.load(weights_only=True, mmap=True)`; a row found at a declared
position must carry the declared bundle, otherwise the split is refused
rather than read as something it is not.

Text tokens are the packed ids at or above `composite_vocab.STRUCTURAL_COUNT`:
the composite vocabulary places structural tokens at `[0, 71)` and every
text-bearing range above (`composite_vocab.packed_ranges`). On a real v15b
row this equals the `text_scalar_start != -1` positions exactly (checked
2026-09-26), and it is the one rule that also classifies a sealed page's
`accepted_tokens` without the tokenizer.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from unflatten.model.composite_vocab import STRUCTURAL_COUNT

IMAGE_TYPES: frozenset[str] = frozenset({"inline_image", "floating_image"})
CROP_SIDES: tuple[str, ...] = ("left", "top", "right", "bottom")


class CohortGoldError(ValueError):
    """The split or a row is not shaped like an exact-text cache."""


@dataclass(frozen=True, slots=True)
class GoldImage:
    node_id: str
    node_type: str
    page_evidence: str
    #: First region of the node, or None when the target carries no region.
    bbox_ppm: tuple[int, int, int, int] | None
    #: `/crop/<side>_ppm` values present on the node, in `CROP_SIDES` order.
    crop_ppm: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GoldTable:
    node_id: str
    width_mode: str | None
    width_value_mpt: int | None
    width_value_ppm: int | None
    autofit: bool | None
    column_widths_mpt: tuple[int, ...]
    #: `typed_cardinalities` count of `/columns`, else the number of width leaves.
    column_count: int
    cell_widths_mpt: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GoldRow:
    document_bundle_id: str
    page_index: int
    node_type_counts: Mapping[str, int]
    #: Node type of the pre-order successor of the first section (None if
    #: the page has no section or the section is empty).
    first_section_child: str | None
    images: tuple[GoldImage, ...]
    tables: tuple[GoldTable, ...]
    text_token_ids: tuple[int, ...] = field(repr=False)

    def count(self, node_type: str) -> int:
        return int(self.node_type_counts.get(node_type, 0))

    @property
    def text_length(self) -> int:
        return len(self.text_token_ids)


def _nid(node_id: str) -> int:
    try:
        return int(node_id[1:])
    except (TypeError, ValueError):
        return -1


def _as_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def gold_row_from_payload(row: Mapping[str, Any]) -> GoldRow:
    """Project one stored cache row (tensors or plain lists) to a `GoldRow`."""
    for key in ("document_bundle_id", "page_index", "region_targets", "typed_targets",
                "packed_token_ids"):
        if key not in row:
            raise CohortGoldError(f"cache row lacks {key!r}; not an exact-text cache row")

    # node inventory: region_targets carry every node with its anchor;
    # typed_targets may add nodes without a region record.
    nodes: dict[str, str] = {}
    anchors: dict[str, int] = {}
    for target in row["region_targets"]:
        nodes[target["node_id"]] = target["node_type"]
        anchors[target["node_id"]] = int(target["anchor_token_index"])
    for target in row["typed_targets"]:
        nodes.setdefault(target["node_id"], target["node_type"])
        anchors.setdefault(target["node_id"], int(target["anchor_token_index"]))

    by_node: dict[str, dict[str, Any]] = {}
    for target in row["typed_targets"]:
        by_node.setdefault(target["node_id"], {})[target["node_property_pointer"]] = target["value"]
    cardinality: dict[tuple[str, str], int] = {
        (c["node_id"], c["container_pointer"]): int(c["count"])
        for c in row.get("typed_cardinalities", ())
    }
    region_by_node: dict[str, Mapping[str, Any]] = {t["node_id"]: t for t in row["region_targets"]}

    sections = sorted((n for n, t in nodes.items() if t == "section"), key=lambda n: anchors[n])
    first_child: str | None = None
    if sections:
        after = anchors[sections[0]]
        successors = [n for n in nodes if anchors[n] > after]
        if successors:
            first_child = nodes[min(successors, key=lambda n: anchors[n])]

    images: list[GoldImage] = []
    for node_id in sorted((n for n, t in nodes.items() if t in IMAGE_TYPES), key=_nid):
        props = by_node.get(node_id, {})
        region = region_by_node.get(node_id)
        regions = list(region["regions"]) if region is not None else []
        bbox = tuple(int(v) for v in regions[0]) if regions else None
        images.append(
            GoldImage(
                node_id=node_id,
                node_type=nodes[node_id],
                page_evidence=str(region["page_evidence"]) if region is not None else "absent",
                bbox_ppm=bbox,  # type: ignore[arg-type]
                crop_ppm=tuple(
                    int(props[f"/crop/{side}_ppm"])
                    for side in CROP_SIDES
                    if props.get(f"/crop/{side}_ppm") is not None
                ),
            )
        )

    table_ids = sorted((n for n, t in nodes.items() if t == "table"), key=_nid)
    cells_by_table: dict[str, list[int]] = {t: [] for t in table_ids}
    for cell_id in sorted((n for n, t in nodes.items() if t == "cell"), key=_nid):
        owner = None
        for table_id in table_ids:
            if _nid(table_id) < _nid(cell_id):
                owner = table_id
            else:
                break
        width = by_node.get(cell_id, {}).get("/layout/width_mpt")
        if owner is not None and width is not None:
            cells_by_table[owner].append(int(width))
    tables: list[GoldTable] = []
    for table_id in table_ids:
        props = by_node.get(table_id, {})
        columns: dict[int, int] = {}
        for pointer, value in props.items():
            if pointer.startswith("/columns/") and pointer.endswith("/width_mpt"):
                columns[int(pointer.split("/")[2])] = int(value)
        widths = tuple(columns[i] for i in sorted(columns))
        mode = props.get("/layout/width/mode")
        value_mpt = props.get("/layout/width/value_mpt")
        value_ppm = props.get("/layout/width/value_ppm")
        autofit = props.get("/layout/autofit")
        tables.append(
            GoldTable(
                node_id=table_id,
                width_mode=str(mode) if mode is not None else None,
                width_value_mpt=int(value_mpt) if value_mpt is not None else None,
                width_value_ppm=int(value_ppm) if value_ppm is not None else None,
                autofit=bool(autofit) if autofit is not None else None,
                column_widths_mpt=widths,
                column_count=cardinality.get((table_id, "/columns"), len(widths)),
                cell_widths_mpt=tuple(cells_by_table[table_id]),
            )
        )

    tokens = _as_list(row["packed_token_ids"])
    return GoldRow(
        document_bundle_id=str(row["document_bundle_id"]),
        page_index=int(row["page_index"]),
        node_type_counts=dict(Counter(nodes.values())),
        first_section_child=first_child,
        images=tuple(images),
        tables=tuple(tables),
        text_token_ids=tuple(t for t in tokens if t >= STRUCTURAL_COUNT),
    )


def _manifest(split_dir: Path) -> Mapping[str, Any]:
    path = split_dir / "manifest.json"
    if not path.is_file():
        raise CohortGoldError(f"{split_dir}: no manifest.json")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest.get("shard_index"), list):
        raise CohortGoldError(f"{split_dir}: manifest carries no 'shard_index' list")
    return manifest


def split_entries(split_dir: Path | str) -> int:
    """The split's row count (`manifest.entries`)."""
    return int(_manifest(Path(split_dir))["entries"])


def load_gold_rows(
    split_dir: Path | str, bundle_ids: Iterable[str], *, page_index: int = 0
) -> dict[str, GoldRow]:
    """`GoldRow` per requested bundle at `page_index`. A bundle the split
    does not hold is simply absent from the result -- the caller accounts
    for it; nothing is invented."""
    import torch

    split = Path(split_dir)
    wanted = set(bundle_ids)
    manifest = _manifest(split)
    # shard name -> [(position, bundle)]
    plan: dict[str, list[tuple[int, str]]] = {}
    for shard in manifest["shard_index"]:
        keys = shard.get("keys")
        if not isinstance(keys, list):
            raise CohortGoldError(
                f"{split}: shard_index entry {shard.get('name')!r} carries no 'keys' list"
            )
        for position, key in enumerate(keys):
            bundle, page = key[0], int(key[1])
            if bundle in wanted and page == page_index:
                plan.setdefault(str(shard["name"]), []).append((position, bundle))

    result: dict[str, GoldRow] = {}
    for name in sorted(plan):
        rows = torch.load(split / name, weights_only=True, mmap=True)
        for position, bundle in plan[name]:
            if position >= len(rows):
                raise CohortGoldError(
                    f"{split}/{name}: declares position {position} beyond its rows"
                )
            row = rows[position]
            if row["document_bundle_id"] != bundle or int(row["page_index"]) != page_index:
                raise CohortGoldError(
                    f"{split}/{name}: manifest declares {bundle}@{page_index} at position "
                    f"{position}, the row carries {row['document_bundle_id']}@{row['page_index']}"
                )
            result[bundle] = gold_row_from_payload(row)
    return result
