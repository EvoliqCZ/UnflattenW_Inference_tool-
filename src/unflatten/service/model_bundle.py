"""D7 model bundle: manifest read + full torch-free verify (plan T8).

A model bundle is one self-contained directory carrying everything a
diagnostic worker needs to reload the exact checkpoint it was built from:
the weights-only checkpoint, its ancestry sidecar (re-finalized for the
NEW bytes), the component lock, the qwen text allowlist, a pinned copy of
the loop-guard v2 caps, the frozen decode policy block, and -- when
available at build time -- the base Qwen model's pinned weight hashes.
Nothing inside it references `runs/` or any prod path (design D7).

**This module never imports torch**, directly or indirectly, so
`api_cli.py` (design D9 §6.1: reads only `identities.component_lock_hash`
from a bundle, never loads a model) stays torch-free by construction --
`scripts/build_model_bundle.py` is the only place a checkpoint's TENSOR
payload is ever inspected, and it imports torch itself for that.

`verify_bundle` is therefore necessarily lighter than a full checkpoint
load: it cannot inspect the `.pt` file's own embedded `ancestry_hash`
tensor payload (only `scripts/build_model_bundle.py verify` and the
worker's own `load_candidate_runtime` -- both torch-consuming -- do that).
It instead treats the manifest's own `identities.ancestry_hash` as the
sidecar's expected embedded value, and separately pins the WHOLE manifest
file's own SHA-256 (`UNFLATTEN_MODEL_BUNDLE_SHA256`, the "self-reference"
note in D7): a manifest that lied about `ancestry_hash` while shipping a
sidecar carrying a different one is still caught (`verify_sidecar`'s own
self-hash and cross-check), and every byte of the checkpoint file itself
is bound to the manifest by a direct SHA-256 comparison. The worker's own
startup (`candidate_runtime.load_candidate_runtime` /
`verify_decode_bindings`) independently re-verifies `optimizer_state_dict`
absence, `cache_identity` binding and `typed_head_config_hash` against the
REAL loaded checkpoint state -- this module's checks are a torch-free
pre-flight, not a replacement for those.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from unflatten.model import checkpoint_ancestry as ca
from unflatten.model.cbor import encode as cbor_encode
from unflatten.model.loop_guard_v2_caps import (
    LoopGuardCapsStale,
    parse_loop_guard_v2_caps,
    refuse_unless_caps_bind,
)
from unflatten.model.quantization.spec import bf16_spec
from unflatten.model.s4_selection_policy import (
    S4SelectionPolicyInvalid,
    load_s4_selection_policy,
    s4_selection_policy_hash,
)

BUNDLE_SCHEMA_VERSION = "1.0.0"

MANIFEST_FILENAME = "manifest.json"
#: DEC-050 §3/§1 -- when a bundle carries its own copy of the backbone, its
#: files live under this subdirectory. Absent entirely -> today's behaviour
#: byte-identical (base weights load offline from the HF cache instead).
BACKBONE_DIRNAME = "backbone"
CHECKPOINT_FILENAME = "checkpoint.weights.pt"
ANCESTRY_FILENAME = CHECKPOINT_FILENAME + ".ancestry.json"
COMPONENT_LOCK_FILENAME = "component-lock.json"
ALLOWLIST_FILENAME = "qwen_text_allowlist.json"
CAPS_FILENAME = "loop_guard_v2_caps.json"
DECODE_POLICY_FILENAME = "decode_policy.json"
PINNED_HASHES_FILENAME = "qwen-pinned-hashes.json"

_APPROVED_SELECTION_POLICY_PATH = (
    Path(__file__).parents[1] / "model" / "policies" / "s4_selection_policy.v15b.json"
)

#: Every bundle must carry these (manifest.json itself is never one of its
#: own `files` entries -- it names them, not itself).
REQUIRED_FILES: tuple[str, ...] = (
    CHECKPOINT_FILENAME,
    ANCESTRY_FILENAME,
    COMPONENT_LOCK_FILENAME,
    ALLOWLIST_FILENAME,
    CAPS_FILENAME,
    DECODE_POLICY_FILENAME,
    PINNED_HASHES_FILENAME,
)
OPTIONAL_FILES: tuple[str, ...] = ()

_CHUNK_BYTES = 1 << 20
_TOKENIZER_HASH_DOMAIN = "rfc004.qwen-tokenizer-artifact-set/v1"
_REQUIRED_TOKENIZER_FILES = frozenset({"tokenizer.json", "tokenizer_config.json"})
_OPTIONAL_TOKENIZER_FILES = frozenset(
    {
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
)


class ModelBundleInvalid(ValueError):
    """A model bundle directory failed manifest, file-hash, sidecar, caps
    binding, or base-model pin verification."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha_eq(actual: str, expected: str) -> bool:
    """Digest equality independent of the `sha256:` prefix: the real
    `qwen-pinned-hashes.json` stores bare hex, `_sha256_file` and the
    manifests use the prefixed form (26.9., the first real verify on prod
    refused a correct weight file for the prefix alone)."""
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    return _bare_hex(actual) == _bare_hex(expected) and bool(_bare_hex(actual))


def _bare_hex(digest: str) -> str:
    return digest.removeprefix("sha256:").strip().lower()


def _tokenizer_artifact_set_hash(snapshot_dir: Path, *, model_id: str, revision: str) -> str:
    """Torch-free equivalent of the tokenizer identity calculation."""
    present = {path.name for path in snapshot_dir.iterdir() if path.is_file()}
    missing = sorted(_REQUIRED_TOKENIZER_FILES - present)
    if missing:
        raise ModelBundleInvalid(
            f"offline HF snapshot is missing required tokenizer file(s): {', '.join(missing)}"
        )
    names_to_hash = _REQUIRED_TOKENIZER_FILES | _OPTIONAL_TOKENIZER_FILES
    files = [
        {
            "path": name,
            "sha256": _sha256_file(snapshot_dir / name),
            "size": (snapshot_dir / name).stat().st_size,
        }
        for name in sorted(present & names_to_hash)
    ]
    payload = {
        "model_id": model_id,
        "revision": revision,
        "files": files,
        "ignored_file_names": sorted(present - names_to_hash),
    }
    digest = hashlib.sha256(_TOKENIZER_HASH_DOMAIN.encode() + b"\x00" + cbor_encode(payload))
    return f"sha256:{digest.hexdigest()}"


def visual_feature_tap_from_identities(identities: Mapping[str, Any]) -> str:
    """DEC-056: the visual tap the bundle's checkpoint was trained on.
    Absent -- every v15b bundle -- means `"pooler_output"`; a DeepStack
    (v16) bundle carries e.g. `"pooler_output+deepstack:5,11,17"`, copied
    from the checkpoint by `scripts/build_model_bundle.py`."""
    return str(identities.get("visual_feature_tap") or "pooler_output")


def load_manifest(bundle_dir: Path | str) -> dict[str, Any]:
    """Parse and shape-check `manifest.json` -- the ONLY file this reads.

    This is the sole entry point `api_cli.py` may call (design D9 §6.1):
    it never opens the checkpoint, the caps file, or anything else in the
    bundle, so no bundle content can pull torch into `api_cli`'s process
    through this function.
    """
    path = Path(bundle_dir) / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelBundleInvalid(f"model bundle manifest {path} cannot be read: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ModelBundleInvalid(f"model bundle manifest {path} must be a JSON object")
    if manifest.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise ModelBundleInvalid(
            "model bundle manifest bundle_schema_version "
            f"{manifest.get('bundle_schema_version')!r} != {BUNDLE_SCHEMA_VERSION!r}"
        )
    for key in ("files", "identities", "base_model"):
        if not isinstance(manifest.get(key), dict):
            raise ModelBundleInvalid(f"model bundle manifest is missing object field {key!r}")
    return manifest


def precision_spec_hash_from_bundle(bundle_dir: Path | str) -> str:
    """`identities.precision.spec_hash`, defaulting to the bf16 sentinel
    spec's hash when absent -- every bundle built before quantization
    existed carries no `precision` block at all, and must keep verifying
    exactly as a bf16 bundle (owner requirement item 3: "existing bf16
    bundles unchanged and still verify")."""
    manifest = load_manifest(bundle_dir)
    precision = manifest["identities"].get("precision")
    if not isinstance(precision, Mapping):
        return bf16_spec().spec_hash
    value = precision.get("spec_hash")
    if not isinstance(value, str) or not value:
        raise ModelBundleInvalid("model bundle manifest identities.precision.spec_hash missing")
    return value


def component_lock_hash_from_bundle(bundle_dir: Path | str) -> str:
    """D9 §6.1: the one value `api_cli.py` reads from a model bundle --
    never a model load, never torch."""
    manifest = load_manifest(bundle_dir)
    value = manifest["identities"].get("component_lock_hash")
    if not isinstance(value, str) or not value:
        raise ModelBundleInvalid("model bundle manifest identities.component_lock_hash missing")
    return value


def manifest_sha256(bundle_dir: Path | str) -> str:
    """SHA-256 of the raw `manifest.json` bytes -- what
    `UNFLATTEN_MODEL_BUNDLE_SHA256` pins (design D7 self-reference note)."""
    return _sha256_file(Path(bundle_dir) / MANIFEST_FILENAME)


def _verify_file_set(bundle_dir: Path, files: Mapping[str, Any]) -> None:
    if not files:
        raise ModelBundleInvalid("model bundle manifest names no files")
    on_disk = {
        p.name for p in bundle_dir.iterdir() if p.is_file() and p.name != MANIFEST_FILENAME
    }
    named = set(files)
    missing = sorted(named - on_disk)
    if missing:
        raise ModelBundleInvalid(f"model bundle is missing file(s): {', '.join(missing)}")
    extra = sorted(on_disk - named)
    if extra:
        raise ModelBundleInvalid(
            f"model bundle has unaccounted-for file(s) not named in the manifest: "
            f"{', '.join(extra)}"
        )
    for name, meta in files.items():
        if (
            not isinstance(meta, Mapping)
            or not isinstance(meta.get("sha256"), str)
            or not isinstance(meta.get("size_bytes"), int)
        ):
            raise ModelBundleInvalid(f"model bundle manifest files[{name!r}] is malformed")
        path = bundle_dir / name
        size = path.stat().st_size
        if size != meta["size_bytes"]:
            raise ModelBundleInvalid(
                f"model bundle file {name} size {size} != manifest {meta['size_bytes']}"
            )
        if not _sha_eq(_sha256_file(path), meta["sha256"]):
            raise ModelBundleInvalid(f"model bundle file {name} sha256 does not match manifest")


def _verify_ancestry(bundle_dir: Path, identities: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_path = bundle_dir / CHECKPOINT_FILENAME
    try:
        doc = ca.verify_sidecar(
            checkpoint_path,
            expect_base_ancestry_hash=identities.get("base_ancestry_hash"),
            expect_s1_run_manifest_hash=identities.get("source_s1_run_manifest_hash"),
            embedded_ancestry_hash=identities.get("ancestry_hash"),
        )
    except ca.CheckpointAncestryInvalid as exc:
        raise ModelBundleInvalid(f"model bundle ancestry sidecar refused: {exc}") from exc
    if doc["ancestry_hash"] != identities.get("ancestry_hash"):
        raise ModelBundleInvalid("sidecar ancestry_hash does not match manifest identities")
    if doc["checkpoint"]["checkpoint_sha256"] != identities.get("checkpoint_sha256"):
        raise ModelBundleInvalid("sidecar checkpoint_sha256 does not match manifest identities")
    if doc["checkpoint"]["checkpoint_id"] != identities.get("checkpoint_id"):
        raise ModelBundleInvalid("sidecar checkpoint_id does not match manifest identities")
    return doc


def _verify_caps(bundle_dir: Path, identities: Mapping[str, Any]) -> None:
    caps_path = bundle_dir / CAPS_FILENAME
    try:
        raw = json.loads(caps_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleInvalid(f"model bundle caps file cannot be read: {exc}") from exc
    if not isinstance(raw, dict):
        raise ModelBundleInvalid("model bundle caps file must hold a JSON object")
    try:
        caps = parse_loop_guard_v2_caps(raw, origin=str(caps_path))
    except ValueError as exc:
        raise ModelBundleInvalid(f"model bundle caps file invalid: {exc}") from exc
    if not _sha_eq(caps.sha256, identities.get("loop_guard_v2_caps_sha256", "")):
        raise ModelBundleInvalid(
            "model bundle loop_guard_v2_caps.json sha256 does not match manifest identities"
        )
    eval_cache_identity = identities.get("eval_cache_identity")
    if not isinstance(eval_cache_identity, Mapping):
        raise ModelBundleInvalid("model bundle manifest identities.eval_cache_identity missing")
    try:
        refuse_unless_caps_bind(caps, eval_cache_identity)
    except LoopGuardCapsStale as exc:
        raise ModelBundleInvalid(f"model bundle loop guard v2 caps do not bind: {exc}") from exc


def approved_s4_decode_policy() -> tuple[dict[str, Any], str]:
    """Return the ratified S4-v15b decode block and its policy identity.

    This is the deployment trust anchor for D5/D7: a bundle may not select a
    different policy merely by refreshing its file hashes and manifest.
    """
    try:
        selection_policy = load_s4_selection_policy(_APPROVED_SELECTION_POLICY_PATH)
        return dict(selection_policy["decode"]), s4_selection_policy_hash(selection_policy)
    except S4SelectionPolicyInvalid as exc:
        raise ModelBundleInvalid(f"approved S4 selection policy is invalid: {exc}") from exc


def _verify_decode_policy(bundle_dir: Path, identities: Mapping[str, Any]) -> None:
    """Refuse a bundle whose D5 decode policy is not the approved S4-v15b one."""
    path = bundle_dir / DECODE_POLICY_FILENAME
    try:
        decode_policy = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelBundleInvalid(f"model bundle decode policy cannot be read: {exc}") from exc
    if not isinstance(decode_policy, Mapping):
        raise ModelBundleInvalid("model bundle decode policy must hold a JSON object")

    approved_decode, approved_hash = approved_s4_decode_policy()
    if decode_policy.get("decode") != approved_decode:
        raise ModelBundleInvalid(
            "model bundle decode policy is not the approved frozen S4-v15b block"
        )
    if decode_policy.get("selection_policy_hash") != approved_hash:
        raise ModelBundleInvalid(
            "model bundle decode policy selection_policy_hash is not the approved S4-v15b hash"
        )
    if identities.get("selection_policy_hash") != approved_hash:
        raise ModelBundleInvalid(
            "model bundle manifest selection_policy_hash is not the approved S4-v15b hash"
        )


def _verify_base_model_pin(
    bundle_dir: Path, base_model: Mapping[str, Any], *, hf_cache_dir: Path | str | None
) -> None:
    pinned_path = bundle_dir / PINNED_HASHES_FILENAME
    try:
        pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleInvalid(f"model bundle pinned-hashes file cannot be read: {exc}") from exc
    if not isinstance(pinned, dict):
        raise ModelBundleInvalid("model bundle pinned-hashes file must hold a JSON object")
    if pinned.get("revision") != base_model.get("revision"):
        raise ModelBundleInvalid(
            f"model bundle {PINNED_HASHES_FILENAME} revision does not match manifest base_model"
        )
    for key in ("weights_set_sha256", "processor_set_sha256"):
        if pinned.get(key) != base_model.get(key):
            raise ModelBundleInvalid(
                f"model bundle {PINNED_HASHES_FILENAME} {key} does not match manifest base_model"
            )
    model_id = base_model.get("model_id")
    revision = base_model.get("revision")
    if (
        not isinstance(model_id, str)
        or not isinstance(revision, str)
        or not model_id
        or not revision
    ):
        raise ModelBundleInvalid("model bundle manifest base_model.model_id/revision missing")

    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(
        model_id, "tokenizer.json", revision=revision, cache_dir=hf_cache_dir
    )
    if not isinstance(cached, str):
        raise ModelBundleInvalid(
            f"no local HF cache snapshot for {model_id}@{revision} "
            f"(try_to_load_from_cache returned {cached!r}); base weights are not "
            "distributed inside the bundle (design D7) and must be present offline"
        )
    snapshot_dir = Path(cached).parent
    _verify_hf_assets(
        snapshot_dir, pinned, fallback_dirs=_fallback_snapshot_dirs(model_id, revision)
    )
    tokenizer_hash = _tokenizer_artifact_set_hash(
        snapshot_dir, model_id=model_id, revision=revision
    )
    if not _sha_eq(tokenizer_hash, base_model.get("tokenizer_artifact_set_hash", "")):
        # `base_model` intentionally stays small on disk; the identity lives
        # in the manifest identities block and is injected by the caller.
        raise ModelBundleInvalid("offline HF tokenizer hash does not match bundle identity")


#: Optional HF cache dir(s) (``os.pathsep``-separated) that may supply a pinned
#: asset ABSENT from the tokenizer's snapshot. S4 v16 hashed its tokenizer over a
#: snapshot without ``video_preprocessor_config.json`` (file names enter that
#: hash) while its pinned processor set requires the file, so no single snapshot
#: satisfies both (17. 10.). The fallback asset is sha256-checked like any other;
#: an asset present in the primary snapshot is never looked up elsewhere.
HF_ASSET_FALLBACK_ENV = "UNFLATTEN_HF_ASSET_FALLBACK_CACHE"


def _fallback_snapshot_dirs(model_id: str, revision: str) -> tuple[Path, ...]:
    raw = os.environ.get(HF_ASSET_FALLBACK_ENV, "")
    repo = "models--" + model_id.replace("/", "--")
    return tuple(
        Path(entry) / repo / "snapshots" / revision for entry in raw.split(os.pathsep) if entry
    )


def _verify_hf_assets(
    snapshot_dir: Path, pinned: Mapping[str, Any], *, fallback_dirs: Sequence[Path] = ()
) -> None:
    """Verify every declared offline Qwen asset, not merely a sentinel tokenizer.

    The pinned-hashes record is the authority for the two asset families.  A
    missing declaration, unsafe path, missing cached asset, or changed byte is
    a refusal; accepting a partial snapshot would turn D7's pin into a probe.
    """
    for family in ("weights", "processor"):
        declared = pinned.get(family)
        if not isinstance(declared, Mapping) or not declared:
            raise ModelBundleInvalid(
                f"model bundle {PINNED_HASHES_FILENAME} {family} must be a non-empty object"
            )
        for relative_name, expected_sha256 in declared.items():
            if not isinstance(relative_name, str) or not relative_name:
                raise ModelBundleInvalid(f"model bundle pinned {family} asset name is invalid")
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ModelBundleInvalid(
                    f"model bundle pinned {family} asset path {relative_name!r} is unsafe"
                )
            if not isinstance(expected_sha256, str):
                raise ModelBundleInvalid(
                    f"model bundle pinned {family} asset {relative_name!r} has no sha256"
                )
            path = snapshot_dir / relative_path
            if not path.is_file():
                path = next(
                    (d / relative_path for d in fallback_dirs if (d / relative_path).is_file()),
                    path,
                )
            if not path.is_file():
                raise ModelBundleInvalid(
                    f"offline HF snapshot is missing pinned {family} asset {relative_name!r}"
                )
            if not _sha_eq(_sha256_file(path), expected_sha256):
                raise ModelBundleInvalid(
                    f"offline HF pinned {family} asset {relative_name!r} sha256 does not match"
                )


def _verify_backbone_dir(
    bundle_dir: Path, identities: Mapping[str, Any], base_model: Mapping[str, Any]
) -> None:
    """DEC-050 §3: when a bundle carries the backbone in `backbone/`, verify
    every declared file's SHA-256 and refuse identity drift.

    Absent `identities.backbone` -> no-op (today's behaviour byte-identical,
    per DEC-050 §1: "Absent -> today's behaviour byte-identical"). When
    present, the model_id/revision it names must agree with the manifest's
    own `base_model` block -- the mechanism that keeps a carried backbone's
    identity honest instead of letting it silently diverge from the rest of
    the bundle's pins (owner requirement item 1).
    """
    backbone = identities.get("backbone")
    if backbone is None:
        return
    if not isinstance(backbone, Mapping):
        raise ModelBundleInvalid("model bundle manifest identities.backbone must be an object")
    for key in ("source_model_id", "source_revision", "files"):
        if key not in backbone:
            raise ModelBundleInvalid(f"model bundle manifest identities.backbone missing {key!r}")
    if backbone["source_model_id"] != base_model.get("model_id"):
        raise ModelBundleInvalid(
            "model bundle identities.backbone.source_model_id does not match base_model.model_id"
        )
    if backbone["source_revision"] != base_model.get("revision"):
        raise ModelBundleInvalid(
            "model bundle identities.backbone.source_revision does not match base_model.revision"
        )
    files = backbone["files"]
    if not isinstance(files, Mapping) or not files:
        raise ModelBundleInvalid(
            "model bundle manifest identities.backbone.files must be a non-empty object"
        )
    backbone_dir = bundle_dir / BACKBONE_DIRNAME
    if not backbone_dir.is_dir():
        raise ModelBundleInvalid(
            f"model bundle names identities.backbone but has no {BACKBONE_DIRNAME}/ directory"
        )
    for relative_name, expected_sha256 in files.items():
        if not isinstance(relative_name, str) or not relative_name:
            raise ModelBundleInvalid("model bundle backbone file name is invalid")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ModelBundleInvalid(
                f"model bundle backbone file path {relative_name!r} is unsafe"
            )
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise ModelBundleInvalid(f"model bundle backbone file {relative_name!r} has no sha256")
        path = backbone_dir / relative_path
        if not path.is_file():
            raise ModelBundleInvalid(f"model bundle backbone is missing file {relative_name!r}")
        if not _sha_eq(_sha256_file(path), expected_sha256):
            raise ModelBundleInvalid(
                f"model bundle backbone file {relative_name!r} sha256 does not match manifest"
            )
    on_disk = {str(p.relative_to(backbone_dir)) for p in backbone_dir.rglob("*") if p.is_file()}
    extra = sorted(on_disk - set(files))
    if extra:
        raise ModelBundleInvalid(
            f"model bundle backbone has unaccounted-for file(s): {', '.join(extra)}"
        )


def add_backbone_to_bundle(
    bundle_dir: Path | str, snapshot_dir: Path | str, *, model_id: str, revision: str
) -> None:
    """DEC-050 §3: copy an already-verified offline HF snapshot into
    `bundle_dir/backbone/` and record it in `manifest.json`.

    Applied AFTER `scripts/build_model_bundle.py build` has already written
    the manifest -- a separate, additive post-processing step (never part of
    `build()` itself), so a caller that never wants a carried backbone pays
    nothing. Refuses if `model_id`/`revision` disagree with the bundle's own
    `base_model` block (identity honesty, owner requirement item 1) or if
    `backbone/` already exists.
    """
    import shutil

    bundle_dir = Path(bundle_dir)
    snapshot_dir = Path(snapshot_dir)
    manifest = load_manifest(bundle_dir)
    base_model = manifest["base_model"]
    if model_id != base_model.get("model_id") or revision != base_model.get("revision"):
        raise ModelBundleInvalid(
            "add_backbone_to_bundle model_id/revision does not match bundle base_model"
        )
    backbone_dir = bundle_dir / BACKBONE_DIRNAME
    if backbone_dir.exists():
        raise ModelBundleInvalid(f"{backbone_dir} already exists -- refuse to overwrite")
    tokenizer_hash = manifest["identities"].get("tokenizer_artifact_set_hash")
    if tokenizer_hash:
        actual_tokenizer_hash = _tokenizer_artifact_set_hash(
            snapshot_dir, model_id=model_id, revision=revision
        )
        if not _sha_eq(actual_tokenizer_hash, tokenizer_hash):
            raise ModelBundleInvalid(
                "snapshot_dir tokenizer_artifact_set_hash does not match the bundle's own "
                "identities.tokenizer_artifact_set_hash -- refusing to carry a backbone that "
                "disagrees with the rest of the bundle's identity"
            )

    files: dict[str, str] = {}
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_name = str(path.relative_to(snapshot_dir))
        dest = backbone_dir / relative_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files[relative_name] = _sha256_file(dest)

    manifest["identities"]["backbone"] = {
        "source_model_id": model_id,
        "source_revision": revision,
        "files": files,
    }
    _write_json_atomic(bundle_dir / MANIFEST_FILENAME, manifest)


def _write_json_atomic(path: Path, doc: dict[str, Any]) -> None:
    import json as _json
    import os as _os

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(_json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _os.replace(tmp, path)


def verify_bundle(
    bundle_dir: Path | str,
    *,
    expected_manifest_sha256: str | None = None,
    expected_precision_spec_hash: str | None = None,
    hf_cache_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Full torch-free verification of a model bundle directory (design D7).

    Refuses (raises `ModelBundleInvalid`) on:
    - a manifest SHA-256 that does not match `expected_manifest_sha256`
      (`UNFLATTEN_MODEL_BUNDLE_SHA256`, the D7 self-reference pin);
    - any file named in `manifest["files"]` whose actual SHA-256/size does
      not match, or that is missing, or any file present that is not named;
    - an ancestry sidecar that does not verify against the checkpoint's own
      bytes on disk, or whose identity disagrees with the manifest's own
      `identities` (see module docstring for what this can and cannot see
      without torch);
    - loop-guard v2 caps whose SHA-256 disagrees with
      `identities.loop_guard_v2_caps_sha256`, or whose `source.
      target_identity` does not bind `identities.eval_cache_identity`
      (design D5/G4);
    - a missing Qwen pinned-hashes record, a base-weights pin that disagrees
      with the manifest's own `base_model`, or any missing/mutated declared
      Qwen weight or processor asset in the offline HF snapshot;
    - (when `expected_precision_spec_hash` is given) a bundle whose
      `identities.precision.spec_hash` (bf16 sentinel when the field is
      absent, see `precision_spec_hash_from_bundle`) does not match --
      the mechanism that refuses loading a quantized bundle as bf16 or a
      bf16 bundle as a specific quantized variant.

    Returns the verified manifest.
    """
    bundle_dir = Path(bundle_dir)
    manifest = load_manifest(bundle_dir)

    if expected_precision_spec_hash is not None:
        actual_precision_hash = precision_spec_hash_from_bundle(bundle_dir)
        if actual_precision_hash != expected_precision_spec_hash:
            raise ModelBundleInvalid(
                f"model bundle precision spec_hash {actual_precision_hash!r} != expected "
                f"{expected_precision_spec_hash!r} -- refusing to load a bundle built at a "
                "different precision than requested"
            )

    actual_manifest_sha256 = manifest_sha256(bundle_dir)
    if expected_manifest_sha256 is not None and not _sha_eq(
        actual_manifest_sha256, expected_manifest_sha256
    ):
        raise ModelBundleInvalid(
            f"model bundle manifest sha256 {actual_manifest_sha256!r} != pinned "
            f"{expected_manifest_sha256!r} (UNFLATTEN_MODEL_BUNDLE_SHA256)"
        )

    missing_required = sorted(set(REQUIRED_FILES) - set(manifest["files"]))
    if missing_required:
        raise ModelBundleInvalid(
            f"model bundle manifest is missing required file(s): {', '.join(missing_required)}"
        )
    _verify_file_set(bundle_dir, manifest["files"])
    _verify_ancestry(bundle_dir, manifest["identities"])
    _verify_caps(bundle_dir, manifest["identities"])
    _verify_decode_policy(bundle_dir, manifest["identities"])
    base_model = dict(manifest["base_model"])
    base_model["tokenizer_artifact_set_hash"] = manifest["identities"].get(
        "tokenizer_artifact_set_hash"
    )
    # DEC-050 §3: a bundle that carries its own `backbone/` copy is fully
    # self-contained and must not require an offline HF cache snapshot --
    # `_verify_backbone_dir` below is then the sole base-weight identity
    # check. Absent `identities.backbone`, behaviour is unchanged.
    if manifest["identities"].get("backbone") is None:
        _verify_base_model_pin(bundle_dir, base_model, hf_cache_dir=hf_cache_dir)
    _verify_backbone_dir(bundle_dir, manifest["identities"], manifest["base_model"])

    return manifest
