"""S4 v17 boost page sets (design `spec-s4-v17-prep` §4.1).

Frozen, deterministic page sets over the TRAIN cache that a page-uniform plan
may draw extra times (`s4_plan --boost-sets`, task 15 consumes this module):

* `text_frame`          -- the token stream contains `TEXT_FRAME_OPEN`;
* `header_footer_image` -- an `INLINE_IMAGE` / `FLOATING_IMAGE` token INSIDE a
  `HEADER_OPEN..HEADER_CLOSE` or `FOOTER_OPEN..FOOTER_CLOSE` span (a depth
  counter, so a body image never counts);
* `bordered_table`      -- a `table` node with a SUPERVISED typed target on
  `/style/borders/{top,bottom,left,right}/style` whose value is neither
  `null` nor `"none"`.

The cache is read the way `s4_typed_census` reads it (`torch.load(mmap=True)`
per shard, manifest key list checked positionally, no GPU, no per-shard sha256)
and the number of rows counted in `E \\ X` must equal the expectation.
Pages named by `--exclude-pages` (the run's `s4_eval_pages.json`, a selection
policy's `cohort.pinned_pages`) are removed from every set: the lineage split
already keeps them out of a train cache, this is the DEC-053 b. 4 defence.

Output is sorted and serialized with `sort_keys`, so two builds over the same
inputs are byte-identical. A missing label is never a border: a target without
`supervision_mask` does not qualify a page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from unflatten.dataset.targets.properties import normalize_pointer
from unflatten.model.events_vocab import TOKEN_IDS
from unflatten.model.exact_text_cache import CacheExpectation, ExactTextCacheReader, PageKey
from unflatten.model.run_binding import (
    compute_exact_text_cache_expectation_hash,
    compute_exact_text_cache_identity_hash,
)
from unflatten.model.s4_selection_policy import pinned_pages_sha256

BOOST_SETS_SCHEMA = "unflatten.s4-boost-sets/1"
BOOST_SET_NAMES = ("text_frame", "bordered_table", "header_footer_image")

_TEXT_FRAME_OPEN = TOKEN_IDS["TEXT_FRAME_OPEN"]
_IMAGE_IDS = frozenset((TOKEN_IDS["INLINE_IMAGE"], TOKEN_IDS["FLOATING_IMAGE"]))
_STORY_OPEN = frozenset((TOKEN_IDS["HEADER_OPEN"], TOKEN_IDS["FOOTER_OPEN"]))
_STORY_CLOSE = frozenset((TOKEN_IDS["HEADER_CLOSE"], TOKEN_IDS["FOOTER_CLOSE"]))
_BORDER_STYLE_POINTERS = frozenset(
    f"/style/borders/{side}/style" for side in ("top", "bottom", "left", "right")
)


class BoostSetsError(ValueError):
    """The boost sets cannot be built or read truthfully: a shard disagrees
    with its manifest, the counted rows differ from the pinned expectation,
    the output exists, or an input file is malformed."""


def _field(target: Any, name: str) -> Any:
    return target[name] if isinstance(target, Mapping) else getattr(target, name)


def _has_bordered_table(typed_targets: Iterable[Any]) -> bool:
    for target in typed_targets:
        if (
            _field(target, "node_type") == "table"
            and _field(target, "supervision_mask")
            and normalize_pointer(_field(target, "node_property_pointer")) in _BORDER_STYLE_POINTERS
            and _field(target, "value") not in (None, "none")
        ):
            return True
    return False


def _has_story_image(token_ids: Iterable[int]) -> bool:
    depth = 0
    for token in token_ids:
        if token in _STORY_OPEN:
            depth += 1
        elif token in _STORY_CLOSE:
            depth = max(depth - 1, 0)
        elif depth and token in _IMAGE_IDS:
            return True
    return False


def classify_row(
    packed_token_ids: Sequence[int], typed_targets: Iterable[Any]
) -> frozenset[str]:
    """The boost sets one cache row belongs to. `typed_targets` are the
    stored payload dicts or `TypedPropertyTarget` objects."""
    tokens = list(packed_token_ids)
    names: set[str] = set()
    if _TEXT_FRAME_OPEN in tokens:
        names.add("text_frame")
    if _has_story_image(tokens):
        names.add("header_footer_image")
    if _has_bordered_table(typed_targets):
        names.add("bordered_table")
    return frozenset(names)


_worker_state: dict[str, Any] = {}


def _init_worker(cache_dir: str, target_keys: frozenset[PageKey]) -> None:
    _worker_state["cache_dir"] = Path(cache_dir)
    _worker_state["target_keys"] = target_keys


def _classify_one_shard(shard_entry: Mapping[str, Any]) -> list[tuple[PageKey, frozenset[str]]]:
    cache_dir: Path = _worker_state["cache_dir"]
    target_keys: frozenset[PageKey] = _worker_state["target_keys"]
    path = cache_dir / shard_entry["name"]
    declared: Sequence[Sequence[Any]] = shard_entry["keys"]
    rows = torch.load(path, weights_only=True, mmap=True)
    if len(rows) != len(declared):
        raise BoostSetsError(f"{path}: {len(rows)} row(s), manifest declares {len(declared)}")
    found: list[tuple[PageKey, frozenset[str]]] = []
    for position, row in enumerate(rows):
        key = (row["document_bundle_id"], int(row["page_index"]))
        if key != tuple(declared[position]):
            raise BoostSetsError(
                f"{path} row {position} is page {key!r}, but the manifest declares "
                f"{tuple(declared[position])!r}"
            )
        if key in target_keys:
            names = classify_row(row["packed_token_ids"].tolist(), row["typed_targets"])
            found.append((key, names))
    return found


def build_boost_sets(
    cache_dir: Path | str,
    *,
    expectation: CacheExpectation,
    exclude_pages: Collection[PageKey],
    workers: int = 1,
) -> dict[str, Any]:
    """The frozen boost-sets payload over `expectation.target_keys()` minus
    `exclude_pages`. Refuses when the rows counted differ from the expectation."""
    cache_dir = Path(cache_dir)
    reader = ExactTextCacheReader(cache_dir, gradient_bearing=False, expectation=expectation)
    target_keys = frozenset(expectation.target_keys())
    excluded = {(str(b), int(p)) for b, p in exclude_pages}
    shard_entries = list(reader.manifest["shard_index"])

    if workers <= 1:
        _init_worker(str(cache_dir), target_keys)
        per_shard = [_classify_one_shard(entry) for entry in shard_entries]
    else:
        with mp.Pool(
            processes=workers, initializer=_init_worker, initargs=(str(cache_dir), target_keys)
        ) as pool:
            per_shard = pool.map(_classify_one_shard, shard_entries)

    classified = [item for shard in per_shard for item in shard]
    if len(classified) != len(target_keys):
        raise BoostSetsError(
            f"counted {len(classified)} row(s) but the expectation names "
            f"{len(target_keys)} page(s) in E \\ X -- boost sets over a different "
            "cohort than the one they are pinned to are refused"
        )
    sets: dict[str, list[list[Any]]] = {name: [] for name in BOOST_SET_NAMES}
    for key, names in sorted(classified):
        if key in excluded:
            continue
        for name in BOOST_SET_NAMES:
            if name in names:
                sets[name].append([key[0], key[1]])
    union = {tuple(page) for pages in sets.values() for page in pages}
    counts = {name: len(pages) for name, pages in sets.items()}
    counts.update(union=len(union), rows=len(classified), excluded_in_cache=len(
        [key for key, _ in classified if key in excluded]
    ))
    return {
        "schema": BOOST_SETS_SCHEMA,
        "cache_identity_hash": compute_exact_text_cache_identity_hash(asdict(reader.identity)),
        "expectation_hash": compute_exact_text_cache_expectation_hash(expectation.payload()),
        "excluded_pages_sha256": pinned_pages_sha256([list(k) for k in sorted(excluded)]),
        "sets": sets,
        "counts": counts,
    }


def boost_sets_sha256(path: Path | str) -> str:
    """`sha256:<hex>` of the boost-sets file's bytes."""
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boost_draw_pages(
    sets: Mapping[str, Any], pages: Iterable[PageKey], *, factor: int
) -> list[PageKey]:
    """The extra draws a boost of `factor` adds: every page in the union of the
    sets that is also in `pages`, `factor - 1` times, sorted. `sets` is the
    payload (`{"sets": {...}}`)."""
    if factor < 1:
        raise ValueError(f"boost factor must be >= 1, got {factor}")
    union = {(str(b), int(p)) for members in sets["sets"].values() for b, p in members}
    selected = sorted(union & {(str(b), int(p)) for b, p in pages})
    return [page for page in selected for _ in range(factor - 1)]


def _load_exclude_pages(path: Path) -> list[PageKey]:
    doc = json.loads(path.read_text())
    if isinstance(doc, dict) and "pages" in doc:
        raw = doc["pages"]
    elif isinstance(doc, dict) and "pinned_pages" in doc.get("cohort", {}):
        raw = doc["cohort"]["pinned_pages"]
    else:
        raise BoostSetsError(f"{path}: neither 'pages' nor 'cohort.pinned_pages' present")
    try:
        return [(str(bundle), int(page)) for bundle, page in raw]
    except (TypeError, ValueError) as exc:
        raise BoostSetsError(f"{path}: pages must be [bundle, page_index] pairs") from exc


def _load_expectation(path: Path) -> CacheExpectation:
    return CacheExpectation.from_payload(json.loads(path.read_text()))


def _cmd_build(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise BoostSetsError(
            f"{args.out} already exists -- refusing to overwrite frozen boost sets"
        )
    exclude: list[PageKey] = []
    for path in args.exclude_pages:
        exclude.extend(_load_exclude_pages(path))
    payload = build_boost_sets(
        args.cache,
        expectation=_load_expectation(args.expectation),
        exclude_pages=exclude,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "counts": payload["counts"]}, sort_keys=True))
    return 0


def _cmd_count(args: argparse.Namespace) -> int:
    payload = json.loads(args.boost_sets.read_text())
    if payload.get("schema") != BOOST_SETS_SCHEMA:
        raise BoostSetsError(f"{args.boost_sets}: schema is not {BOOST_SETS_SCHEMA!r}")
    expectation = _load_expectation(args.expectation)
    expected_hash = compute_exact_text_cache_expectation_hash(expectation.payload())
    if payload.get("expectation_hash") != expected_hash:
        raise BoostSetsError(
            f"{args.boost_sets} was built for expectation {payload.get('expectation_hash')!r}, "
            f"not {expected_hash!r}"
        )
    pages = expectation.target_keys()
    if args.exclude_pages is not None:
        # the same reduced page set the plan draws from (`s4_plan --exclude-pages`)
        from unflatten.model.s4_plan import load_exclude_pages

        excluded = set(load_exclude_pages(args.exclude_pages))
        pages = [page for page in pages if page not in excluded]
    draws = boost_draw_pages(payload, pages, factor=args.factor)
    print(json.dumps({
        "draws": len(draws),
        "factor": args.factor,
        "sets_sha256": boost_sets_sha256(args.boost_sets),
        "set_sizes": payload["counts"],
    }, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m unflatten.model.s4_boost_sets",
        description="Frozen S4 v17 boost page sets (text frame / bordered table / logo).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="derive the sets from a train cache")
    build.add_argument("--cache", type=Path, required=True)
    build.add_argument("--expectation", type=Path, required=True)
    build.add_argument("--exclude-pages", type=Path, action="append", default=[])
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--workers", type=int, default=1)
    build.set_defaults(func=_cmd_build)
    count = sub.add_parser("count", help="print the extra draws a factor adds")
    count.add_argument("--boost-sets", type=Path, required=True)
    count.add_argument("--expectation", type=Path, required=True)
    count.add_argument("--factor", type=int, required=True)
    count.add_argument("--exclude-pages", type=Path, default=None,
                       help="JSONL of pages the plan never draws (s4_plan --exclude-pages)")
    count.set_defaults(func=_cmd_count)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
