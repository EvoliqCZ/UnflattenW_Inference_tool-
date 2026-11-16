"""Verified exclusion-only replay of immutable, complete page-image caches."""

from __future__ import annotations

import ast
import hashlib
import json
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from unflatten.model.exact_text_cache import ExactTextCacheIncompatible, ExactTextCacheReader
from unflatten.model.piece_withdrawal import PIECE_WITHDRAWAL_VERSION


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_build_config(path: Path | None, parent: Path, budget: int) -> dict[str, Any]:
    if path is None:
        raise ExactTextCacheIncompatible("reuse requires explicit parent build configuration")
    record = json.loads(Path(path).read_text())
    if budget != 10000 or record.get("text_encoding_budget_ops") != budget:
        raise ExactTextCacheIncompatible("reuse text_encoding_budget_ops mismatch")
    if record.get("parent_manifest_sha256") != digest(parent / "manifest.json"):
        raise ExactTextCacheIncompatible("reuse parent manifest sha256 mismatch")
    for name in ("launcher", "builder_source"):
        evidence = record.get(name) or {}
        evidence_path = Path(evidence.get("path", ""))
        if not evidence_path.is_file() or digest(evidence_path) != evidence.get("sha256"):
            raise ExactTextCacheIncompatible(f"reuse {name} evidence sha256 mismatch")
    if "--text-encoding-budget-ops" in Path(record["launcher"]["path"]).read_text():
        raise ExactTextCacheIncompatible("reuse launcher must attest the default text budget")
    tree = ast.parse(Path(record["builder_source"]["path"]).read_text())
    defaults = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_TEXT_ENCODING_BUDGET_OPS"
            for target in node.targets
        )
    ]
    if defaults != [budget]:
        raise ExactTextCacheIncompatible("reuse frozen builder default budget mismatch")
    return {**record, "record_sha256": digest(Path(path))}


@dataclass
class VerifiedParent:
    reader: ExactTextCacheReader
    locations: dict[tuple[str, int], tuple[int, int] | dict[str, Any]]
    changed: set[tuple[str, int]]
    provenance: dict[str, Any]

    def retained(
        self, key: tuple[str, int]
    ) -> tuple[dict[str, Any] | None, dict[str, Any], str | None, bool] | None:
        if key in self.changed:
            return None
        location = self.locations[key]
        if isinstance(location, dict):
            return None, location["metadata"], location["reason"], location["continuation"]
        shard, index = location
        stored = self.reader._shard_rows(shard)[index]
        return stored, stored["_build_metadata"], None, False


_VERIFY_READER: ExactTextCacheReader | None = None


def _verify_init(reader: ExactTextCacheReader) -> None:
    global _VERIFY_READER
    _VERIFY_READER = reader
    import torch

    torch.set_num_threads(1)


def _checked_position(metadata: Any, key: tuple[str, int]) -> int:
    required = (
        "position",
        "document_bundle_id",
        "page_index",
        "coverage",
        "typed",
        "ledger",
        "node_shares",
        "region",
    )
    if not isinstance(metadata, dict) or any(field not in metadata for field in required):
        raise ExactTextCacheIncompatible("reuse parent missing build metadata")
    if (
        type(metadata["position"]) is not int
        or (metadata["document_bundle_id"], metadata["page_index"]) != key
    ):
        raise ExactTextCacheIncompatible("reuse parent metadata key/position mismatch")
    for field, fields in (
        ("coverage", {"granted_chars", "store_chars"}),
        ("typed", {"flipped", "node_not_opened_here"}),
        ("region", {"node_not_opened_here"}),
    ):
        value = metadata[field]
        if value is not None and (
            not isinstance(value, dict)
            or set(value) != fields
            or any(type(count) is not int or count < 0 for count in value.values())
        ):
            raise ExactTextCacheIncompatible(f"reuse parent malformed {field} metadata")
    if metadata["ledger"] is not None and not isinstance(metadata["ledger"], dict):
        raise ExactTextCacheIncompatible("reuse parent malformed ledger metadata")
    return metadata["position"]


def _verify_shard(ordinal: int) -> list[tuple[int, tuple[str, int], int, dict[str, Any]]]:
    assert _VERIFY_READER is not None
    return [
        (
            _checked_position(
                stored.get("_build_metadata"), (stored["document_bundle_id"], stored["page_index"])
            ),
            (stored["document_bundle_id"], stored["page_index"]),
            index,
            stored["page_image"],
        )
        for index, stored in enumerate(_VERIFY_READER._shard_rows(ordinal))
    ]


def _verified_shards(reader: ExactTextCacheReader, workers: int):
    ordinals = iter(range(len(reader.manifest["shard_index"])))
    if workers == 1:
        _verify_init(reader)
        for ordinal in ordinals:
            yield ordinal, _verify_shard(ordinal)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("fork"),
        initializer=_verify_init,
        initargs=(reader,),
    ) as pool:
        pending = deque()
        for _ in range(workers * 2):
            ordinal = next(ordinals, None)
            if ordinal is None:
                break
            pending.append((ordinal, pool.submit(_verify_shard, ordinal)))
        while pending:
            ordinal, future = pending.popleft()
            yield ordinal, future.result()
            ordinal = next(ordinals, None)
            if ordinal is not None:
                pending.append((ordinal, pool.submit(_verify_shard, ordinal)))


def verify_parent(
    *,
    parent: Path,
    cohort: list[Any],
    expected_identity: dict[str, Any],
    scope: dict[str, Any],
    min_coverage: float,
    old_entries: dict[Any, str],
    new_entries: dict[Any, str],
    config: dict[str, Any],
    skip_reasons: tuple[str, ...],
    feature_sources: list[dict[str, Any]],
    workers: int,
    parent_policy: dict[str, Any],
    shard_size: int,
) -> VerifiedParent:
    reader = ExactTextCacheReader(parent, gradient_bearing=False, cache_shards=2)
    manifest = reader.manifest

    def refuse(message: str) -> None:
        raise ExactTextCacheIncompatible(f"reuse parent {message}")

    if manifest.get("piece_withdrawal_version") != PIECE_WITHDRAWAL_VERSION:
        refuse("piece withdrawal version mismatch or missing")
    if manifest.get("cache_version") != "7.0.0":
        refuse("must be a complete 7.0.0 page-image cache")
    if (
        manifest["identity"] != expected_identity
        or manifest.get("min_text_coverage") != min_coverage
    ):
        refuse("identity/configuration mismatch")
    if manifest.get("shard_size") != shard_size:
        refuse("shard size configuration mismatch")
    if any(manifest["scope"].get(key) != value for key, value in scope.items()):
        refuse("cohort scope mismatch")
    if (manifest.get("target_policy") or {}).get("descriptor") != parent_policy:
        refuse("target policy descriptor mismatch")
    if manifest.get("feature_sources") != feature_sources:
        refuse("visual input policy/configuration mismatch")
    if not old_entries.keys() <= new_entries.keys():
        refuse("exclusions must be monotonic")
    keys = [(sample.document_bundle_id, sample.page_index) for sample in cohort]
    if len(set(keys)) != len(keys):
        refuse("cohort keys are duplicated")
    changed = {
        keys[i]
        for i, sample in enumerate(cohort)
        if old_entries.get((sample.lineage_root_id, sample.page_index))
        != new_entries.get((sample.lineage_root_id, sample.page_index))
    }
    locations = {}
    positions = set()

    def check(position: int, key: tuple[str, int]) -> None:
        if not 0 <= position < len(keys) or position in positions:
            refuse("invalid/duplicated cohort position")
        if keys[position] != key or key in locations:
            refuse("metadata key closure mismatch")
        positions.add(position)

    for ordinal, records in _verified_shards(reader, workers):
        if (ordinal + 1) % 100 == 0:
            print(
                f"[cache-reuse] {ordinal + 1}/{len(manifest['shard_index'])} "
                "parent shards verified",
                flush=True,
            )
        for position, key, index, image in records:
            check(position, key)
            if image != dict(cohort[position].image):
                refuse("page-image reference/cohort mismatch")
            locations[key] = (ordinal, index)
    progress = manifest.get("build_progress") or {}
    if progress.get("next_position") != len(keys):
        refuse("incomplete cohort progress")
    for record in progress.get("skip_records", []):
        key = (record["document_bundle_id"], record["page_index"])
        check(_checked_position(record.get("metadata"), key), key)
        if (
            record.get("position") != record["metadata"]["position"]
            or record.get("reason") not in skip_reasons
            or type(record.get("continuation")) is not bool
        ):
            refuse("invalid skip record")
        locations[key] = record
    if positions != set(range(len(keys))) or set(locations) != set(keys):
        refuse("incomplete cohort position/key closure")
    # Drop verification residency; workers load bounded shards, never inherit all rows.
    print(
        f"[cache-reuse] verified {len(manifest['shard_index'])} parent shards, "
        f"{len(positions)} complete cohort pages",
        flush=True,
    )
    reader._loaded.clear()
    reader._verified.clear()
    if digest(parent / "manifest.json") != config["parent_manifest_sha256"]:
        refuse("manifest changed during verification")
    return VerifiedParent(
        reader,
        locations,
        changed,
        {
            "kind": "verified-page-image-exclusion-only.v1",
            "parent_manifest_sha256": config["parent_manifest_sha256"],
            "parent_build_configuration": config,
            "parent_shards": [
                {"name": s["name"], "sha256": s["sha256"], "count": s["count"]}
                for s in manifest["shard_index"]
            ],
            "parent_policy": manifest.get("target_policy"),
            "reused_pages": len(keys) - len(changed),
            "rematerialized_pages": len(changed),
            "rematerialized_keys": [list(key) for key in keys if key in changed],
            "cohort_keys_sha256": hashlib.sha256(
                json.dumps(keys, separators=(",", ":")).encode()
            ).hexdigest(),
        },
    )
