"""One-shot, audited correction of a feature cache's missing backbone revision.

Why this exists
---------------
The v12 candidate feature cache was built from a hub cache that was
downloaded by explicit SHA, so `refs/main` was absent and
`identity.backbone_revision` was recorded as the honest `"unavailable"`
gap. The weights were pinned anyway — the snapshot directory is named
by the commit sha, and the P5 preflight verified the weight digests
against that revision. Rebuilding 3.2 TB of features to fill one
metadata field would waste a full GPU day to state a fact that is
already on disk; this tool records that fact instead.

Fail-closed rules
-----------------
- Only a `complete` cache whose gap is exactly
  (`backbone_revision` null, `revision_source` `"unavailable"`) is
  correctable. Anything else — a set revision, a source that claims
  otherwise, an in-progress build — is refused.
- The revision is derived by `build_cache.resolve_backbone_revision`
  from the live hub cache, never typed by a user. Zero or ambiguous
  snapshot directories stay `"unavailable"` and refuse the correction.
- An optional `--pinned-hashes` file (the P5 pin) must agree on the
  revision; its sha256 is recorded as independent evidence.
- The manifest is rewritten atomically in the writer's exact format
  and re-validated before it is published. The event is recorded under
  the top-level key `extensions.identity_revision_correction` —
  `extensions.*` is the manifest's explicit 0.4.0/0.5.0 extension
  point, so the key set and version are unchanged.

The correction changes identity VALUES only; no shard, count or
converter field moves, so every check a reader applies to the data is
unaffected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from unflatten.model.build_cache import resolve_backbone_revision
from unflatten.model.feature_cache import (
    CacheIntegrityError,
    cache_manifest,
    validate_manifest,
)

EXTENSION_KEY = "extensions.identity_revision_correction"
_SNAPSHOT_SOURCE = "local_hf_snapshot_dir"


class FeatureCacheCorrectionError(RuntimeError):
    """The requested correction would not be an honest, verifiable one."""


def _write_manifest_atomic(out_dir: Path, manifest: dict[str, Any]) -> None:
    """The writer's exact publication: same format, tmp + fsync + replace."""
    manifest_path = out_dir / "manifest.json"
    temporary = out_dir / "manifest.json.tmp"
    with temporary.open("w") as handle:
        handle.write(json.dumps(manifest, indent=1, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
    try:
        descriptor = os.open(out_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _check_pinned_hashes(
    pinned_hashes: Path, revision: str
) -> tuple[str, str]:
    """Return (file sha256, recorded revision) or refuse on mismatch."""
    raw = pinned_hashes.read_bytes()
    try:
        pin = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FeatureCacheCorrectionError(
            f"pinned hashes file {pinned_hashes} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(pin, dict):
        raise FeatureCacheCorrectionError(
            f"pinned hashes file {pinned_hashes} must hold a JSON object"
        )
    recorded = pin.get("revision")
    if not isinstance(recorded, str) or len(recorded) < 8:
        raise FeatureCacheCorrectionError(
            f"pinned hashes file {pinned_hashes} has no usable 'revision' "
            f"field, got {recorded!r}"
        )
    if not revision.startswith(recorded):
        raise FeatureCacheCorrectionError(
            f"pinned hashes file {pinned_hashes} records revision "
            f"{recorded!r}, which does not match the derived {revision!r}"
        )
    return hashlib.sha256(raw).hexdigest(), recorded


def correct_cache_revision(
    out_dir: Path | str,
    *,
    hub_root: Path | None = None,
    pinned_hashes: Path | str | None = None,
) -> dict[str, Any]:
    """Fill the `backbone_revision` gap of a complete feature cache.

    Returns the audit record written under `EXTENSION_KEY`. Refuses with
    `FeatureCacheCorrectionError` whenever the correction would not be
    a verifiable fact (see the module docstring for the full rules).
    """
    out_dir = Path(out_dir)
    manifest = cache_manifest(out_dir)
    validate_manifest(manifest)

    if manifest["status"] != "complete":
        raise FeatureCacheCorrectionError(
            f"{out_dir}: status is {manifest['status']!r}, not 'complete'; "
            "a build may still write shards, and this correction must not "
            "race the writer's manifest checkpoints"
        )

    identity = manifest["identity"]
    if identity["backbone_revision"] is not None:
        raise FeatureCacheCorrectionError(
            f"{out_dir}: backbone_revision is already "
            f"{identity['backbone_revision']!r}; nothing to correct, and a "
            "second pass would be a patch on top of a patch"
        )
    if identity["revision_source"] != "unavailable":
        raise FeatureCacheCorrectionError(
            f"{out_dir}: backbone_revision is null but revision_source is "
            f"{identity['revision_source']!r} — that is not a gap, it is a "
            "corrupted manifest"
        )
    backbone = identity["backbone"]
    if not isinstance(backbone, str) or not backbone:
        raise FeatureCacheCorrectionError(
            f"{out_dir}: identity has no backbone name; there is nothing "
            "to resolve a revision for"
        )

    revision, source = resolve_backbone_revision(backbone, hub_root=hub_root)
    if source != _SNAPSHOT_SOURCE or not revision:
        raise FeatureCacheCorrectionError(
            f"{out_dir}: no verifiable revision for {backbone!r} "
            f"(source: {source}); the gap stays honest rather than filled"
        )

    pinned_sha: str | None = None
    pinned_revision: str | None = None
    if pinned_hashes is not None:
        pinned_sha, pinned_revision = _check_pinned_hashes(
            Path(pinned_hashes), revision
        )

    record: dict[str, Any] = {
        "corrected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corrected_by": "unflatten.model.feature_cache_correction",
        "tool_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "from": {"backbone_revision": None, "revision_source": "unavailable"},
        "to": {
            "backbone_revision": revision,
            "revision_source": _SNAPSHOT_SOURCE,
        },
        "evidence": {
            "backbone": backbone,
            "snapshot_dir": f"snapshots/{revision}",
            "pinned_hashes_file": (
                str(Path(pinned_hashes).resolve())
                if pinned_hashes is not None
                else None
            ),
            "pinned_hashes_sha256": pinned_sha,
            "pinned_revision": pinned_revision,
        },
    }

    identity["backbone_revision"] = revision
    identity["revision_source"] = _SNAPSHOT_SOURCE
    manifest[EXTENSION_KEY] = record
    validate_manifest(manifest)
    _write_manifest_atomic(out_dir, manifest)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="unflatten.model.feature_cache_correction",
        description="Fill a feature cache's honest backbone_revision gap "
        "from the hub cache's snapshot directory, with an audited record.",
    )
    parser.add_argument("out_dir", type=Path, help="the feature cache directory")
    parser.add_argument(
        "--pinned-hashes",
        type=Path,
        default=None,
        help="optional P5 pinned-hashes JSON; must agree on the revision",
    )
    parser.add_argument(
        "--hub-root",
        type=Path,
        default=None,
        help="hub cache root (default: $HF_HOME/hub or ~/.cache/huggingface/hub)",
    )
    args = parser.parse_args(argv)
    try:
        record = correct_cache_revision(
            args.out_dir,
            hub_root=args.hub_root,
            pinned_hashes=args.pinned_hashes,
        )
    except (FeatureCacheCorrectionError, CacheIntegrityError, FileNotFoundError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
