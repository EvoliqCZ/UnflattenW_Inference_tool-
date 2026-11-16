"""CPU-only composition of immutable M2b stores, with checked provenance.

Root ledgers are canonical deterministic unions; objects are hardlinked, never
rewritten. Default object checking binds each reference to its source owner and
checks existence/size. Full byte hashing is explicit and recorded separately.
Composition manifests are not render summaries and cannot be nested.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.dataset.split.release import ReleaseError

SCHEMA = "unflatten.store-composition/1"
LEDGERS = ("index.jsonl", "renders.jsonl", "target-bundles.jsonl", "mask-projections.jsonl")
EXCLUSION_FILES = ("bundle-exclusions.jsonl", "exclusions.jsonl", "release-exclusions.jsonl")
PROFILE_KEYS = ("derivation_hash", "render_dpi", "font_pack_hash", "aligner_policy")


def _pin(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return {
        "path": str(path.resolve()),
        "sha256": f"sha256:{digest.hexdigest()}",
        "size_bytes": size,
    }


def _verify_pin(pin: dict[str, Any], path: Path | None = None) -> None:
    actual_path = Path(pin["path"])
    if path is not None and actual_path.resolve() != path.resolve():
        raise ReleaseError(f"composition pin path mismatch: {actual_path}, expected {path}")
    if _pin(actual_path) != pin:
        raise ReleaseError(f"composition file hash/size mismatch: {actual_path}")


def _rows(path: Path) -> list[dict[str, Any]]:
    # Immutable inputs must be complete. Unlike resumable M2b readers, no torn
    # tail may silently vanish from a purported exact union.
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def _bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        for row in rows
    ).encode()


def _bundle(row: dict[str, Any], index: bool = False) -> str:
    value = row["envelope"]["document_bundle_id"] if index else row["document_bundle_id"]
    if not isinstance(value, str) or not value:
        raise ReleaseError("composition invalid bundle id")
    return value


def _component(store: Path, summary: Path) -> dict[str, Any]:
    from unflatten.dataset.migration.m2c import build_migration_record

    raw = json.loads(summary.read_bytes())
    if raw.get("kind") == "combined_store" or raw.get("schema") == SCHEMA:
        raise ReleaseError(f"nested composition unsupported: {summary}")
    files = {}
    for name in (*LEDGERS, "run-summary.json", *EXCLUSION_FILES):
        path = store / name
        if path.exists():
            files[name] = _pin(path)
        elif name in ("index.jsonl", "renders.jsonl", "run-summary.json"):
            raise ReleaseError(f"composition required source file missing: {path}")
    return {
        "store": str(store.resolve()),
        "summary": _pin(summary),
        "files": files,
        "migration": build_migration_record(summary, store),
    }


def _union(components: list[dict[str, Any]], excluded: dict[str, str]):
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in LEDGERS}
    owners: dict[str, Path] = {}
    expected_profile = None
    profile_exceptions = []
    from unflatten.dataset.migration.m2c import load_bundle_exclusions

    for component in components:
        store = Path(component["store"])
        index = _rows(store / "index.jsonl")
        local_ids = set()
        ast_hashes = {}
        for row in index:
            bundle = _bundle(row, True)
            if bundle in owners:
                raise ReleaseError(f"composition duplicate bundle id: {bundle}")
            owners[bundle] = store
            local_ids.add(bundle)
            ast_hashes[bundle] = row["envelope"].get("ast_hash")
        result["index.jsonl"].extend(index)
        for name in EXCLUSION_FILES:
            if name in component["files"]:
                for bundle, reason in load_bundle_exclusions(store / name).items():
                    if bundle not in local_ids or excluded.get(bundle) != reason:
                        raise ReleaseError(f"composition source exclusion lost/changed: {bundle}")
        effective = {}
        for name in LEDGERS[1:]:
            rows = {}
            if name in component["files"]:
                for row in _rows(store / name):
                    bundle = _bundle(row)
                    if bundle not in local_ids:
                        raise ReleaseError(f"composition foreign {name} bundle: {bundle}")
                    rows[bundle] = row  # append-only source resume: last wins
            effective[name] = rows
            result[name].extend(rows.values())
        missing_renders = local_ids - effective["renders.jsonl"].keys()
        if missing_renders:
            raise ReleaseError(
                f"composition source renders omit bundle ids: {sorted(missing_renders)[:5]}"
            )
        for bundle, mask in effective["mask-projections.jsonl"].items():
            if bundle not in excluded and mask.get("ast_hash") != ast_hashes[bundle]:
                raise ReleaseError(f"composition stale mask overlay: {bundle}")
        for bundle, row in effective["renders.jsonl"].items():
            status = row.get("render_status")
            if status not in ("rendered", "failed", "skipped"):
                raise ReleaseError(f"composition invalid render status: {bundle}: {status}")
            if bundle in excluded or status != "rendered":
                profile_exceptions.append(
                    {
                        "document_bundle_id": bundle,
                        "render_status": status,
                        "render_profile": row.get("render_profile"),
                        "reason": excluded.get(bundle, "non_rendered_source_row"),
                    }
                )
                continue
            profile = dict(row.get("render_profile") or {})
            target = effective["target-bundles.jsonl"].get(bundle)
            if target is not None:
                ast_hash = ast_hashes[bundle]
                if target.get("ast_hash") != ast_hash:
                    raise ReleaseError(f"composition stale target overlay: {bundle}")
                profile["derivation_hash"] = target.get("derivation_hash")
            if any(profile.get(k) is None for k in PROFILE_KEYS[:3]):
                raise ReleaseError(f"composition missing effective profile: {bundle}")
            current = {k: profile.get(k) for k in PROFILE_KEYS}
            if current["derivation_hash"] != component["migration"]["aligner_derivation_hash"]:
                raise ReleaseError(f"composition summary derivation mismatch: {bundle}")
            summary_profile = component["migration"].get("render_profile")
            if summary_profile and any(current[k] != summary_profile.get(k) for k in PROFILE_KEYS):
                raise ReleaseError(f"composition summary profile mismatch: {bundle}")
            if expected_profile is not None and current != expected_profile:
                raise ReleaseError(f"composition mixed effective profiles: {bundle}")
            expected_profile = current
    missing = set(excluded) - owners.keys()
    if missing:
        raise ReleaseError(f"composition foreign exclusions: {sorted(missing)}")
    for name, rows in result.items():
        rows.sort(key=lambda r: _bundle(r, name == "index.jsonl"))
    profile_exceptions.sort(key=lambda row: row["document_bundle_id"])
    return result, owners, expected_profile, profile_exceptions


def _refs(value):
    if isinstance(value, dict):
        if {"ref", "sha256", "size_bytes", "mime_type"} <= value.keys():
            yield ArtifactRef.from_dict(value)
        else:
            for child in value.values():
                yield from _refs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _refs(child)


def _objects(rows, owners, out: Path) -> None:
    """Check direct refs against the source owner's preserved object namespace."""
    checked = set()
    for name, ledger in rows.items():
        for row in ledger:
            store = owners[_bundle(row, name == "index.jsonl")]
            for ref in _refs(row):
                if not ref.ref.startswith("objects/") or Path(ref.ref).name != ref.sha256[7:]:
                    raise ReleaseError(f"composition non-content-addressed ref: {ref.ref}")
                key = (store, ref.ref, ref.sha256, ref.size_bytes)
                if key in checked:
                    continue
                source = store / ref.ref
                if not source.resolve().is_relative_to(store) or source.is_symlink():
                    raise ReleaseError(f"composition unsafe source object: {source}")
                if not source.is_file() or source.stat().st_size != ref.size_bytes:
                    raise ReleaseError(f"composition source object missing/size mismatch: {source}")
                destination = out / ref.ref
                if not destination.resolve().is_relative_to(out):
                    raise ReleaseError(f"composition unsafe output object: {destination}")
                if not destination.is_file() or destination.stat().st_size != ref.size_bytes:
                    raise ReleaseError(
                        f"composition output object missing/size mismatch: {destination}"
                    )
                checked.add(key)


def _object_inventory(components, out: Path, *, link: bool, full_hash: bool):
    """Preserve complete object namespaces, including indirect JSON refs.

    SHA fields come from content-addressed names; only full_hash claims their
    byte verification. Every owner/path/size binding is pinned in the manifest.
    """
    inventory = []
    seen = {}
    for component in components:
        store = Path(component["store"])
        root = store / "objects"
        if not root.is_dir() or root.is_symlink():
            raise ReleaseError(f"composition missing/unsafe object root: {root}")
        for source in sorted(root.rglob("*")):
            if source.is_symlink():
                raise ReleaseError(f"composition symlink object unsupported: {source}")
            if source.is_dir():
                continue
            if (
                not source.is_file()
                or len(source.name) != 64
                or any(c not in "0123456789abcdef" for c in source.name)
            ):
                raise ReleaseError(f"composition invalid content-addressed object: {source}")
            relative = source.relative_to(store).as_posix()
            digest = f"sha256:{source.name}"
            size = source.stat().st_size
            row = {"store": str(store), "ref": relative, "sha256": digest, "size_bytes": size}
            inventory.append(row)
            old = seen.get(relative)
            if old is not None and old[1] != size:
                raise ReleaseError(f"composition duplicate object size mismatch: {relative}")
            if full_hash or old is not None:
                for candidate in (source, old[0]) if old else (source,):
                    if _pin(candidate)["sha256"] != digest:
                        raise ReleaseError(f"composition object hash mismatch: {candidate}")
            destination = out / relative
            if link and old is None:
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(source, destination)
            if not destination.is_file() or destination.stat().st_size != size:
                raise ReleaseError(
                    f"composition output object missing/size mismatch: {destination}"
                )
            if not os.path.samefile(source, destination):
                if _pin(destination)["sha256"] != digest:
                    raise ReleaseError(f"composition output object hash mismatch: {destination}")
            seen[relative] = (source, size)
    inventory.sort(key=lambda row: (row["store"], row["ref"]))
    return inventory


def compose_stores(
    components: list[tuple[Path, Path]],
    out: Path,
    *,
    exclusions: Path | None = None,
    verify_object_hashes: bool = False,
) -> Path:
    """Compose disjoint stores into a NEW directory; return its manifest path.

    On failure a partial new directory may remain; it has no accepted manifest.
    Inputs must stay immutable, including hardlinked object bytes, thereafter.
    """
    from unflatten.dataset.migration.m2c import load_bundle_exclusions

    try:
        if len(components) < 2:
            raise ReleaseError("composition requires at least two components")
        out = Path(out).resolve()
        for store, _summary in components:
            source = Path(store).resolve()
            if out.is_relative_to(source) or source.is_relative_to(out):
                raise ReleaseError(f"composition input/output overlap: {source}, {out}")
        records = [
            _component(Path(store).resolve(), Path(summary).resolve())
            for store, summary in components
        ]
        excluded = load_bundle_exclusions(exclusions) if exclusions is not None else {}
        rows, owners, profile, profile_exceptions = _union(records, excluded)
        out.mkdir(parents=True, exist_ok=False)
        inventory = _object_inventory(records, out, link=True, full_hash=verify_object_hashes)
        _objects(rows, owners, out)
        inventory_path = out / "object-owners.jsonl"
        inventory_path.write_bytes(_bytes(inventory))
        outputs = {"object-owners.jsonl": _pin(inventory_path)}
        for name, ledger in rows.items():
            if ledger or name in LEDGERS[:2]:
                path = out / name
                path.write_bytes(_bytes(ledger))
                outputs[name] = _pin(path)
        if excluded:
            path = out / "bundle-exclusions.jsonl"
            path.write_bytes(
                _bytes(
                    [{"document_bundle_id": k, "reason": v} for k, v in sorted(excluded.items())]
                )
            )
            outputs[path.name] = _pin(path)
        manifest = {
            "schema": SCHEMA,
            "kind": "combined_store",
            "store": str(out),
            "components": records,
            "outputs": outputs,
            "effective_profile": profile,
            "profile_exceptions": profile_exceptions,
            "object_verification": "sha256" if verify_object_hashes else "existence_size_owner",
            "source_exclusions": _pin(exclusions) if exclusions is not None else None,
        }
        path = out / "composition-manifest.json"
        # Verify source/output pins once more before publishing the manifest.
        _validate(manifest, out)
        path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        return path
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReleaseError(f"compose stores {out}: {exc}") from exc


def _validate(manifest: dict[str, Any], store: Path) -> None:
    from unflatten.dataset.migration.m2c import load_bundle_exclusions

    if manifest.get("schema") != SCHEMA or manifest.get("kind") != "combined_store":
        raise ReleaseError("invalid composition manifest schema/kind")
    if Path(manifest["store"]).resolve() != store.resolve():
        raise ReleaseError(f"composition store mismatch: {store}")
    components = manifest["components"]
    if not isinstance(components, list) or len(components) < 2:
        raise ReleaseError("composition requires at least two components")
    for component in components:
        source = Path(component["store"])
        if component["migration"].get("kind") not in ("align_only", "full_render"):
            raise ReleaseError("nested/unknown component migration unsupported")
        _verify_pin(component["summary"])
        for name, pin in component["files"].items():
            _verify_pin(pin, source / name)
        rebuilt = _component(source, Path(component["summary"]["path"]))
        if rebuilt != component:
            raise ReleaseError(f"composition component migration/file mismatch: {source}")
    outputs = manifest["outputs"]
    for name, pin in outputs.items():
        if name not in (*LEDGERS, "bundle-exclusions.jsonl", "object-owners.jsonl"):
            raise ReleaseError(f"composition unknown output: {name}")
        _verify_pin(pin, store / name)
    source_exclusions = manifest.get("source_exclusions")
    excluded = {}
    if source_exclusions is not None:
        _verify_pin(source_exclusions)
        excluded = load_bundle_exclusions(Path(source_exclusions["path"]))
        pin = outputs.get("bundle-exclusions.jsonl")
        if pin is None or load_bundle_exclusions(Path(pin["path"])) != excluded:
            raise ReleaseError("composition output exclusion mismatch")
    elif "bundle-exclusions.jsonl" in outputs:
        raise ReleaseError("composition output exclusions lack source provenance")
    rows, owners, profile, profile_exceptions = _union(components, excluded)
    expected_names = {n for n, r in rows.items() if r or n in LEDGERS[:2]}
    expected_names.add("object-owners.jsonl")
    if excluded:
        expected_names.add("bundle-exclusions.jsonl")
    if set(outputs) != expected_names:
        raise ReleaseError("composition missing/foreign output ledgers")
    for name, ledger in rows.items():
        path = store / name
        if name in expected_names:
            if path.read_bytes() != _bytes(ledger):
                raise ReleaseError(f"composition output is not exact deterministic union: {name}")
        elif path.exists():
            raise ReleaseError(f"composition unpinned overlay: {name}")
    if profile_exceptions != manifest.get("profile_exceptions"):
        raise ReleaseError("composition named profile exceptions mismatch")
    if profile != manifest.get("effective_profile"):
        raise ReleaseError("composition effective profile mismatch")
    mode = manifest.get("object_verification")
    if mode not in ("sha256", "existence_size_owner"):
        raise ReleaseError("composition invalid object verification mode")
    inventory = _object_inventory(components, store, link=False, full_hash=mode == "sha256")
    if (store / "object-owners.jsonl").read_bytes() != _bytes(inventory):
        raise ReleaseError("composition source object owner inventory mismatch")
    _objects(rows, owners, store)


def build_composition_record(path: Path, store: Path) -> dict[str, Any]:
    """Verify every pin and exact union before returning release provenance."""
    try:
        manifest = json.loads(path.read_bytes())
        _validate(manifest, store)
        return {**manifest, "store": store.name, "composition_manifest": _pin(path)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ReleaseError(f"composition manifest {path}: {exc}") from exc
