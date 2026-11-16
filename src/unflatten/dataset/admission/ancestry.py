"""§13.2 "audit the legacy Stage1 checkpoint" — as evidence, not assertion.

`legacy_ancestry: false` may not be asserted; it has to be shown. This
walks the data tree for every serialized-tensor file and classifies each
one:

* **feature cache shard** — named in a sibling `manifest.json`'s
  `shard_index` (the format `model/feature_cache.py` writes);
* **model checkpoint** — its pickle references `state_dict`, the key the
  only checkpoint writer in this repo (`model/train.py`) saves under;
* **approved foundation weights** — only the two byte-hash-verified upstream
  Qwen shards pinned by DEC-010, in their exact Hugging Face repo/revision;
* **cache negative marker** — the empty non-symlink HF marker for an absent
  monolithic weight file, only alongside that fully verified positive snapshot;
* **unclassified** — anything else.

`value` is `False` only when there are no checkpoints AND nothing is
unclassified. One unclassified file makes it `None`: an unaudited tensor
file is not evidence of absence, and this report does not get to round it
down to "clean".

The classification reads the zip central directory and the (small)
`data.pkl` entry — no torch import, no tensor load.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

#: File extensions a model checkpoint could plausibly use.
TENSOR_SUFFIXES: frozenset[str] = frozenset({".pt", ".pth", ".ckpt", ".safetensors"})

#: The key `unflatten.model.train` saves a checkpoint under.
_CHECKPOINT_MARKER = b"state_dict"


# DEC-010 (docs/decisions/DEC-010-qwen-license.md) approves this exact
# upstream foundation only. These content pins are recorded independently in
# ~/unflatten-data/qwen-pinned-hashes.json; filenames alone prove nothing.
_APPROVED_REPO = "Qwen/Qwen3-VL-4B-Instruct"
_APPROVED_CACHE_REPO = "models--Qwen--Qwen3-VL-4B-Instruct"
_APPROVED_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
_APPROVED_WEIGHTS_SET_SHA256 = (
    "4b782234871728edc68425be6e2de879b72d5b3449a0b2f3f4ad6265d5726d9b"
)
_APPROVED_WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors":
        "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    "model-00002-of-00002.safetensors":
        "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
}


def _file_sha256(path: Path, cached: dict[Path, str | None]) -> str | None:
    """Stream bytes, including resolved symlink targets; never load tensors."""
    try:
        resolved = path.resolve(strict=True)
        if resolved not in cached:
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            cached[resolved] = digest.hexdigest()
        return cached[resolved]
    except (OSError, RuntimeError):
        return None


def _approved_foundation(
    path: Path, verified: dict[Path, bool], hashes: dict[Path, str | None]
) -> tuple[str, dict[str, Any]] | None:
    """Classify a closed HF path set only after BOTH positive shards verify."""
    revision = path.parent
    lane = revision.parent
    repo = lane.parent
    if (repo.name != _APPROVED_CACHE_REPO or revision.name != _APPROVED_REVISION
            or lane.name not in {"snapshots", ".no_exist"}):
        return None
    is_weight = lane.name == "snapshots" and path.name in _APPROVED_WEIGHT_SHA256
    is_marker = lane.name == ".no_exist" and path.name == "model.safetensors"
    if not (is_weight or is_marker):
        return None
    if repo not in verified:
        snapshot = repo / "snapshots" / _APPROVED_REVISION
        verified[repo] = all(
            _file_sha256(snapshot / name, hashes) == expected
            for name, expected in _APPROVED_WEIGHT_SHA256.items()
        )
    if not verified[repo]:
        return None
    evidence = {
        "path": str(path), "repo": _APPROVED_REPO,
        "revision": _APPROVED_REVISION, "decision": "DEC-010",
        "weights_set_sha256": "sha256:" + _APPROVED_WEIGHTS_SET_SHA256,
    }
    if is_weight:
        return "weight", {
            **evidence, "filename": path.name,
            "sha256": "sha256:" + _APPROVED_WEIGHT_SHA256[path.name],
            "resolved_path": str(path.resolve(strict=True)),
        }
    try:
        if not path.is_symlink() and path.is_file() and path.stat().st_size == 0:
            return "marker", {**evidence, "kind": "cache-negative-marker"}
    except OSError:
        return None
    return None


def _cache_shards(directory: Path) -> set[str]:
    """Shard file names a feature-cache manifest in `directory` accounts for."""
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        return set()
    try:
        data = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if "shard_index" not in data:
        return set()
    return {
        entry["name"]
        for entry in data.get("shard_index", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }


def _is_checkpoint(path: Path) -> bool | None:
    """True/False if the pickle can be inspected, None if it cannot be."""
    if not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            entries = [n for n in archive.namelist() if n.endswith("data.pkl")]
            if not entries:
                return None
            return _CHECKPOINT_MARKER in archive.read(entries[0])
    except (zipfile.BadZipFile, OSError, KeyError):
        return None


def legacy_ancestry(data_roots: list[Path] | tuple[Path, ...]) -> dict[str, Any]:
    """Audit `data_roots` for model checkpoints (§13.2, §9.4 line 11)."""
    checkpoints: list[str] = []
    unclassified: list[str] = []
    shards = 0
    non_checkpoints = 0
    scanned = 0
    foundation_weights: list[dict[str, Any]] = []
    negative_markers: list[dict[str, Any]] = []
    verified: dict[Path, bool] = {}
    hashes: dict[Path, str | None] = {}
    cached: dict[Path, set[str]] = {}
    roots = [Path(root) for root in data_roots]
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.suffix not in TENSOR_SUFFIXES or not (
                path.is_file() or path.is_symlink()
            ):
                continue
            scanned += 1
            approved = _approved_foundation(path, verified, hashes)
            if approved is not None:
                kind, evidence = approved
                (foundation_weights if kind == "weight" else negative_markers).append(evidence)
                continue
            if path.parent not in cached:
                cached[path.parent] = _cache_shards(path.parent)
            accounted = cached[path.parent]
            if path.name in accounted:
                shards += 1
                continue
            verdict = _is_checkpoint(path)
            label = str(path.relative_to(root))
            if verdict is True:
                checkpoints.append(label)
            elif verdict is None:
                unclassified.append(label)
            else:
                non_checkpoints += 1
    if checkpoints:
        value: bool | None = True
    elif unclassified:
        value = None
    else:
        value = False
    return {
        "value": value,
        "evidence": {
            "data_roots": [str(root) for root in roots],
            "tensor_files_scanned": scanned,
            "model_checkpoints": sorted(checkpoints),
            "feature_cache_shards": shards,
            "approved_foundation_weights": sorted(foundation_weights, key=lambda x: x["path"]),
            "cache_negative_markers": sorted(negative_markers, key=lambda x: x["path"]),
            "non_checkpoint_tensor_files": non_checkpoints,
            "unclassified": sorted(unclassified),
            "rule": (
                "a file is a model checkpoint iff its pickle references "
                "'state_dict' (unflatten/model/train.py); a file named in a "
                "sibling manifest.json shard_index is a feature cache "
                "(unflatten/model/feature_cache.py); DEC-010 foundation weights "
                "require the exact HF repo/revision/filenames and both pinned "
                "streamed SHA256 hashes; its empty non-symlink HF negative marker "
                "requires the same fully verified positive snapshot"
            ),
        },
    }
