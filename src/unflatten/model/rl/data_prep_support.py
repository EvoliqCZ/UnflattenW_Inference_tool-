"""Filesystem and JSON helpers of the ``data_prep`` CLI (torch-free).

Every write of the preparation tools goes through :func:`publish`: tmp file,
``fsync``, hard link -- atomic and never replacing an existing artefact.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json


class DataPrepRefused(ValueError):
    """A named refusal of the preparation CLI (exit code 2)."""


def read_json(path: Path | str, what: str) -> Any:
    p = Path(path)
    if not p.is_file():
        raise DataPrepRefused(f"{what}: {p} is not a file")
    try:
        return json.loads(p.read_bytes())
    except ValueError as exc:
        raise DataPrepRefused(f"{what}: {p} is not JSON ({exc})") from exc


def out_dir(out: str) -> Path:
    path = Path(out)
    if path.exists() and not path.is_dir():
        raise DataPrepRefused(f"--out {path} exists and is not a directory")
    return path


def require_free(out: Path, names: Iterable[str]) -> None:
    taken = sorted(n for n in names if (out / n).exists())
    if taken:
        raise DataPrepRefused(f"refusing to overwrite existing artefact(s) in {out}: {taken}")


def publish(out: Path, name: str, payload: bytes) -> Path:
    """tmp + fsync + hard link: atomic, and never replaces an existing artefact."""
    out.mkdir(parents=True, exist_ok=True)
    target = out / name
    tmp = out / f".{name}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, target)
        except FileExistsError as exc:
            raise DataPrepRefused(f"refusing to overwrite existing artefact {target}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return target


def write_docs(out: Path, docs: Mapping[str, Any]) -> list[str]:
    require_free(out, docs)
    for name, doc in docs.items():
        publish(out, name, to_canonical_json(doc))
    return sorted(docs)


def train_lineages(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {r["lineage_root_id"] for r in records if r.get("split_name") == "train"}


def cache_identity(cache_dir: str) -> Mapping[str, Any]:
    manifest = read_json(Path(cache_dir) / "manifest.json", "cache manifest")
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if not isinstance(identity, dict):
        raise DataPrepRefused(f"cache {cache_dir}: manifest.json carries no 'identity' object")
    if identity.get("split") != "train":
        raise DataPrepRefused(
            f"cache identity split is {identity.get('split')!r}; S4.5 trains on 'train' only"
        )
    return identity


def gold_screen_document(screen: Any) -> dict[str, Any]:
    """JSON form of an ``adapters.gold.GoldScreen`` (``passed`` / ``refused`` attrs)."""
    return {
        "passed": sorted([b, int(k)] for b, k in screen.passed),
        "refused": sorted(
            [b, int(k), counter, message] for (b, k), (counter, message) in screen.refused.items()
        ),
    }


def gold_screen(path: str | None, bundle_lineage: Mapping[str, str], selected: set[str]) -> Any:
    if path is None:
        return "not_run"
    doc = read_json(path, "gold screen report")
    try:
        passed = [(str(b), int(k)) for b, k in doc["passed"]]
        refused = [(str(b), int(k), str(c)) for b, k, c, _m in doc["refused"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise DataPrepRefused(f"gold screen report {path} is malformed: {exc!r}") from exc
    uncovered = sorted(selected - {bundle_lineage[b] for b, _k in passed if b in bundle_lineage})
    if uncovered:
        raise DataPrepRefused(
            f"{len(uncovered)} selected lineage(s) have no page that passed the gold screen: "
            f"{uncovered[:5]}"
        )
    return {
        "report_sha256": f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}",
        "passed": len(passed),
        "refused": len(refused),
        "refused_by_counter": dict(sorted(Counter(c for _b, _k, c in refused).items())),
    }
