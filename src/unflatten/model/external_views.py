"""External (scan / photo / augraphy) page views for S4 training draws.

Pure Python: a content-pinned table of external renderings of train pages and a
deterministic, table-bound draw selection. No torch. The table lives outside git;
every file it names is sha256-verified at load and again at every read.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema

KINDS = ("scan", "photo", "augraphy")
SELECTION_DOMAIN = "unflatten.s4-external-view-draw/v1"
MAX_SHARE_MILLI = 100
SCHEMA_ID = "unflatten.s4-external-views/v1"

_SCHEMAS_DIR = Path(__file__).parent / "schemas"
_GATE_POLICY = Path(__file__).parent / "policies" / "s5_capture_gate.v17.json"
_CONTROL_NAME = "control-sample.json"


class ExternalViewsRefused(ValueError):
    """The external-views table or one of its files was refused."""


def _refuse(message: str) -> ExternalViewsRefused:
    return ExternalViewsRefused(f"external views refused: {message}")


@dataclass(frozen=True)
class ExternalEntry:
    kind: str
    document_bundle_id: str
    page_index: int
    path: str
    sha256: str
    width_px: int
    height_px: int
    matrix_ref_to_view: tuple[tuple[float, ...], ...] | None
    alignment_usable: bool
    color_lost: bool

    @property
    def view_id(self) -> str:
        return (
            f"{self.document_bundle_id}:{self.page_index}:external:{self.kind}:{self.sha256[7:19]}"
        )


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMAS_DIR / "s4_external_views.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _bundle_ids(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "document_bundle_id" and isinstance(v, str):
                found.add(v)
            else:
                found |= _bundle_ids(v)
    elif isinstance(node, list):
        for v in node:
            found |= _bundle_ids(v)
    return found


def _pinned_control_sha() -> str:
    with open(_GATE_POLICY, encoding="utf-8") as f:
        return json.load(f)["sample"]["control_sample_sha256"]


def _check_control(table_dir: Path, header_sha: str, entries: Sequence[ExternalEntry]) -> None:
    control = table_dir / _CONTROL_NAME
    if not control.is_file():
        raise _refuse(f"{_CONTROL_NAME} missing beside the table")
    actual = _file_sha(control)
    if actual != header_sha:
        raise _refuse(f"{_CONTROL_NAME} sha256 {actual} != header control_sample_sha256")
    if actual != _pinned_control_sha():
        raise _refuse(f"{_CONTROL_NAME} sha256 {actual} != pinned s5_capture_gate value")
    with open(control, encoding="utf-8") as f:
        control_bundles = _bundle_ids(json.load(f))
    overlap = sorted({e.document_bundle_id for e in entries} & control_bundles)
    if overlap:
        raise _refuse(f"entries overlap the S5 control sample: {overlap[:3]}")


def _parse_entry(raw: dict[str, Any]) -> ExternalEntry:
    path = PurePosixPath(raw["path"])
    if path.is_absolute() or ".." in path.parts or "\\" in raw["path"]:
        raise _refuse(f"entry path {raw['path']!r} must be relative without '..'")
    m = raw["matrix_ref_to_view"]
    return ExternalEntry(
        kind=raw["kind"],
        document_bundle_id=raw["document_bundle_id"],
        page_index=raw["page_index"],
        path=raw["path"],
        sha256=raw["sha256"],
        width_px=raw["width_px"],
        height_px=raw["height_px"],
        matrix_ref_to_view=None if m is None else tuple(tuple(float(x) for x in r) for r in m),
        alignment_usable=raw["alignment_usable"],
        color_lost=raw["color_lost"],
    )


def page_fraction_overflow(
    fraction_milli: dict[str, int], entries: Sequence[ExternalEntry]
) -> tuple[tuple[str, int], int] | None:
    """First page whose present kinds' `fraction_milli` sum exceeds 1000, as
    `((bundle, page), sum)`; `select()` only walks the kinds a page has."""
    kinds: dict[tuple[str, int], set[str]] = {}
    for e in entries:
        kinds.setdefault((e.document_bundle_id, e.page_index), set()).add(e.kind)
    for page in sorted(kinds):
        total = sum(fraction_milli[k] for k in kinds[page])
        if total > 1000:
            return page, total
    return None


def _sort_key(e: ExternalEntry) -> tuple[str, int, str, str]:
    return (e.document_bundle_id, e.page_index, e.kind, e.sha256)


def _digest_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class ExternalViews:
    def __init__(
        self,
        *,
        table_sha256: str,
        seed: int,
        fraction_milli: dict[str, int],
        entries: Sequence[ExternalEntry],
        root: Path,
    ) -> None:
        self.table_sha256 = table_sha256
        self.seed = seed
        self.fraction_milli = dict(fraction_milli)
        self.entries = tuple(entries)
        self._root = root
        by_page: dict[tuple[str, int], dict[str, list[ExternalEntry]]] = {}
        for e in self.entries:
            by_page.setdefault((e.document_bundle_id, e.page_index), {}).setdefault(
                e.kind, []
            ).append(e)
        self._by_page = by_page

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        train_pages: Collection[tuple[str, int]],
        plan_pages: Collection[tuple[str, int]],
        eval_pages: Collection[tuple[str, int]],
        collapsed_region_rule: str,
    ) -> ExternalViews:
        path = Path(path)
        table_sha = _file_sha(path)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except json.JSONDecodeError as exc:
            raise _refuse(f"table is not valid JSON: {exc}") from exc
        errors = sorted(_validator().iter_errors(doc), key=lambda e: list(e.path))
        if errors:
            raise _refuse(f"schema: {errors[0].message} at {list(errors[0].path)}")
        fractions = doc["fraction_milli"]
        entries = [_parse_entry(r) for r in doc["entries"]]
        over = page_fraction_overflow(fractions, entries)
        if over is not None:
            raise _refuse(f"page {over[0]}: fraction_milli of its kinds sums to {over[1]} > 1000")
        keys = [_sort_key(e) for e in entries]
        if keys != sorted(keys):
            raise _refuse("entries are not sorted by (bundle, page, kind, sha256)")
        if len(set(keys)) != len(keys):
            raise _refuse("duplicate (bundle, page, kind, sha256) entries")
        train, plan, evalp = set(train_pages), set(plan_pages), set(eval_pages)
        for e in entries:
            page = (e.document_bundle_id, e.page_index)
            if page not in train:
                raise _refuse(f"page {page} is not a train page")
            if page not in plan:
                raise _refuse(f"page {page} is not in the plan")
            if page in evalp:
                raise _refuse(f"page {page} is an eval page")
            if not e.alignment_usable and collapsed_region_rule != "exclude":
                raise _refuse(
                    f"entry {e.view_id} is unaligned but collapsed_region_rule is "
                    f"{collapsed_region_rule!r}, not 'exclude'"
                )
        table_dir = path.resolve().parent
        _check_control(table_dir, doc["control_sample_sha256"], entries)
        for e in entries:
            file = table_dir / e.path
            if not file.is_file():
                raise _refuse(f"entry file missing: {e.path}")
            actual = _file_sha(file)
            if actual != e.sha256:
                raise _refuse(f"entry file {e.path} sha256 {actual} != table {e.sha256}")
        return cls(
            table_sha256=table_sha,
            seed=doc["seed"],
            fraction_milli=fractions,
            entries=entries,
            root=table_dir,
        )

    def select(self, draw_index: int, bundle: str, page: int) -> ExternalEntry | None:
        kinds = self._by_page.get((bundle, page))
        if not kinds:
            return None
        base = f"{SELECTION_DOMAIN}|{self.seed}|{self.table_sha256}|{bundle}|{page}|{draw_index}"
        u = (_digest_int(base) * 1000) >> 64
        cum = 0
        for kind in KINDS:
            if kind not in kinds:
                continue
            cum += self.fraction_milli[kind]
            if u < cum:
                pool = kinds[kind]
                return pool[_digest_int(base + "|pick") % len(pool)]
        return None

    def read(self, entry: ExternalEntry) -> bytes:
        data = (self._root / entry.path).read_bytes()
        actual = "sha256:" + hashlib.sha256(data).hexdigest()
        if actual != entry.sha256:
            raise _refuse(f"read {entry.path}: sha256 {actual} != {entry.sha256}")
        return data

    def count(self, page_by_draw: Sequence[tuple[str, int]], start: int, stop: int) -> dict:
        draws: Counter[str] = Counter()
        uses: dict[str, Counter[str]] = {k: Counter() for k in KINDS}
        pages: dict[str, set[tuple[str, int]]] = {k: set() for k in KINDS}
        for i in range(start, stop):
            bundle, page = page_by_draw[i]
            e = self.select(i, bundle, page)
            if e is None:
                continue
            draws[e.kind] += 1
            uses[e.kind][e.sha256] += 1
            pages[e.kind].add((bundle, page))
        by_kind = {
            k: {
                "draws": draws[k],
                "entries_used": len(uses[k]),
                "pages": len(pages[k]),
                "max_uses_per_entry": max(uses[k].values(), default=0),
            }
            for k in KINDS
        }
        external = sum(draws.values())
        total = stop - start
        return {
            "by_kind": by_kind,
            "external_draws": external,
            "draws": total,
            "share_milli": (1000 * external) // total if total > 0 else 0,
        }

    def manifest_fields(self) -> dict:
        counts = Counter(e.kind for e in self.entries)
        return {
            "table_sha256": self.table_sha256,
            "seed": self.seed,
            "selection_domain": SELECTION_DOMAIN,
            "fraction_milli": {k: self.fraction_milli[k] for k in KINDS},
            "entries": {k: counts[k] for k in KINDS if counts[k] >= 1},
        }
