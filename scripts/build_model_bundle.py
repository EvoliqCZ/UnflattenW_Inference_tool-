"""Build and verify a D7 self-contained model bundle (plan T8).

`build` reads one already-trusted training checkpoint (its own ancestry
sidecar already verified, design §35.1/§11) and re-packages it as a
diagnostic-service model bundle: a NEW weights-only checkpoint file with
`optimizer_state_dict` (and every other training-only key) dropped, its
ancestry sidecar re-finalized for the NEW bytes (same `ancestry_hash`, new
`checkpoint_sha256`/`checkpoint_id`), plus copies of the component lock,
text allowlist, loop-guard v2 caps and the frozen `decode` policy block,
and a `manifest.json` naming every file's SHA-256. Nothing written
references `runs/` or any prod path.

`verify` re-checks an already-built bundle directory
(`unflatten.service.model_bundle.verify_bundle`, torch-free) and prints
the bundle manifest's own SHA-256 -- the value operators pin into
`UNFLATTEN_MODEL_BUNDLE_SHA256` (design D7 self-reference note).

    python scripts/build_model_bundle.py build \\
        --checkpoint runs/s4-v15b/checkpoint-step-000027915.pt \\
        --run-dir runs/s4-v15b --out /path/to/bundle
    python scripts/build_model_bundle.py verify /path/to/bundle
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from unflatten.model import checkpoint_ancestry as ca  # noqa: E402
from unflatten.model import composite_vocab  # noqa: E402
from unflatten.model.candidate_loader import (  # noqa: E402
    QWEN_CANDIDATE_MODEL_ID,
    QWEN_CANDIDATE_REVISION,
)
from unflatten.model.capability_profile import (  # noqa: E402
    capability_profile_hash_from_component_lock,
    resolve_capability_profile_from_component_lock,
)
from unflatten.model.component_lock import text_tokenizer_identity  # noqa: E402
from unflatten.model.loop_guard_v2_caps import CAPS_PATH, parse_loop_guard_v2_caps  # noqa: E402
from unflatten.model.manifests import (  # noqa: E402
    component_lock_hash,
    validate_exact_text_s1_component_lock,
)
from unflatten.model.quantization.spec import QuantizationSpec, bf16_spec  # noqa: E402
from unflatten.model.s4_selection_policy import (  # noqa: E402
    load_s4_selection_policy,
    s4_selection_policy_hash,
)
from unflatten.service.model_bundle import (  # noqa: E402
    ALLOWLIST_FILENAME,
    BUNDLE_SCHEMA_VERSION,
    CAPS_FILENAME,
    CHECKPOINT_FILENAME,
    COMPONENT_LOCK_FILENAME,
    DECODE_POLICY_FILENAME,
    MANIFEST_FILENAME,
    PINNED_HASHES_FILENAME,
    ModelBundleInvalid,
    approved_s4_decode_policy,
    manifest_sha256,
    verify_bundle,
)

_DEFAULT_SELECTION_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "unflatten"
    / "model"
    / "policies"
    / "s4_selection_policy.v15b.json"
)

_HEADLESS_DEFAULTS_POLICY_ID = "headless-defaults.v0"

#: The keys a bundle's re-packaged checkpoint carries -- weights-only,
#: never a training-only key (design D6/D7).
_BUNDLE_STATE_KEYS = ("model_state_dict", "cache_identity", "typed_head_config_hash", "step")


class ModelBundleBuildError(RuntimeError):
    """The source checkpoint or one of its companion files cannot be
    re-packaged into a model bundle."""


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _load_and_copy_json(source: Path, dest: Path) -> dict[str, Any]:
    try:
        text = source.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleBuildError(f"cannot read {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelBundleBuildError(f"{source} must hold a JSON object")
    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return payload


def _validate_pinned_hashes(payload: dict[str, Any]) -> None:
    """Refuse a bundle whose Qwen asset declarations are incomplete."""
    for key in ("revision", "weights_set_sha256", "processor_set_sha256"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ModelBundleBuildError(f"pinned-hashes file is missing non-empty {key!r}")
    for family in ("weights", "processor"):
        assets = payload.get(family)
        if not isinstance(assets, dict) or not assets:
            raise ModelBundleBuildError(
                f"pinned-hashes file {family!r} must be a non-empty object"
            )
        if any(
            not isinstance(name, str) or not isinstance(sha, str)
            for name, sha in assets.items()
        ):
            raise ModelBundleBuildError(f"pinned-hashes file {family!r} has an invalid asset entry")


def _verify_checkpoint_semantics(bundle_dir: Path, manifest: dict[str, Any]) -> None:
    """Torch-consuming CLI-only checkpoint inspection after byte verification."""
    import torch

    state = torch.load(bundle_dir / CHECKPOINT_FILENAME, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ModelBundleInvalid("bundle checkpoint payload is not a mapping")
    allowed = {*_BUNDLE_STATE_KEYS, "ancestry_hash"}
    if "visual_feature_tap" in state:
        allowed.add("visual_feature_tap")
    if set(state) != allowed:
        raise ModelBundleInvalid(
            "bundle checkpoint contains keys outside the weights-only contract"
        )
    identities = manifest["identities"]
    if state.get("visual_feature_tap") != identities.get("visual_feature_tap"):
        raise ModelBundleInvalid(
            "bundle checkpoint visual_feature_tap does not match manifest identities"
        )
    for key in ("ancestry_hash", "cache_identity", "typed_head_config_hash", "step"):
        if key == "cache_identity":
            expected = identities["eval_cache_identity"]
        elif key == "step":
            expected = identities["source_step"]
        else:
            expected = identities[key]
        if state.get(key) != expected:
            raise ModelBundleInvalid(f"bundle checkpoint {key} does not match manifest identities")


def _repackage_checkpoint(source_checkpoint: Path, dest_checkpoint: Path) -> dict[str, Any]:
    """Load the source checkpoint on the CPU, drop every training-only key
    (design D6), and write the weights-only result to `dest_checkpoint`
    through a `.tmp` sibling and `os.replace` (never a partial file at the
    final path)."""
    import torch

    state = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ModelBundleBuildError("source checkpoint payload is not a mapping")
    missing = [
        key for key in ("model_state_dict", "cache_identity", "ancestry_hash") if key not in state
    ]
    if missing:
        raise ModelBundleBuildError(f"source checkpoint is missing key(s): {', '.join(missing)}")
    if not state.get("typed_head_config_hash"):
        raise ModelBundleBuildError(
            "source checkpoint carries no typed_head_config_hash -- the diagnostic "
            "worker requires typed_decode and cannot bundle a checkpoint without it"
        )
    new_state = {key: state[key] for key in _BUNDLE_STATE_KEYS}
    new_state["ancestry_hash"] = state["ancestry_hash"]
    # DEC-056: a DeepStack checkpoint keeps its tap (absent on every v15b
    # checkpoint, so their bundles keep exactly their old key set).
    if state.get("visual_feature_tap"):
        new_state["visual_feature_tap"] = state["visual_feature_tap"]

    tmp = dest_checkpoint.with_name(dest_checkpoint.name + ".tmp")
    torch.save(new_state, tmp)
    os.replace(tmp, dest_checkpoint)
    return new_state


def _finalize_ancestry(
    source_checkpoint: Path, new_checkpoint: Path, *, embedded_ancestry_hash: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the source sidecar against the source checkpoint's own bytes
    (self-consistency: the sidecar is asked to agree with itself, not with
    an outside authority this script does not have), then re-finalize the
    SAME document for the newly written checkpoint's bytes."""
    source_sidecar_path = ca.sidecar_path(source_checkpoint)
    try:
        raw_sidecar = json.loads(source_sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBundleBuildError(f"cannot read {source_sidecar_path}: {exc}") from exc
    checkpoint_block = raw_sidecar.get("checkpoint")
    if not isinstance(checkpoint_block, dict):
        raise ModelBundleBuildError(f"{source_sidecar_path} carries no 'checkpoint' block")
    try:
        source_doc = ca.verify_sidecar(
            source_checkpoint,
            expect_base_ancestry_hash=checkpoint_block.get("base_ancestry_hash"),
            expect_s1_run_manifest_hash=checkpoint_block.get("s1_run_manifest_hash"),
            embedded_ancestry_hash=embedded_ancestry_hash,
        )
    except ca.CheckpointAncestryInvalid as exc:
        raise ModelBundleBuildError(f"source checkpoint ancestry sidecar refused: {exc}") from exc

    new_doc = copy.deepcopy(source_doc)
    new_doc = ca.finalize_checkpoint_document(new_doc, new_checkpoint)
    ca.write_sidecar(new_doc, new_checkpoint)
    return source_doc, new_doc


def _precision_identity(precision: QuantizationSpec | None) -> dict[str, Any]:
    spec = precision or bf16_spec()
    return spec.to_dict() | {"spec_hash": spec.spec_hash}


def build(
    *,
    checkpoint_path: Path,
    component_lock_path: Path,
    allowlist_path: Path,
    caps_path: Path,
    selection_policy_path: Path,
    pinned_hashes_path: Path | None,
    out_dir: Path,
    precision: QuantizationSpec | None = None,
) -> Path:
    """Build a self-contained model bundle at `out_dir` (must not exist).

    `precision` (default `None` -> the bf16 sentinel spec) records what
    precision the checkpoint at `checkpoint_path` was already produced at --
    this function never quantizes anything itself; it only pins the
    identity so `verify_bundle`/`load_candidate_runtime` can refuse to load
    a bundle at the wrong precision (owner requirement item 3)."""
    if out_dir.exists():
        raise ModelBundleBuildError(f"{out_dir} already exists -- refuse to overwrite a bundle")
    out_dir.mkdir(parents=True)

    new_checkpoint = out_dir / CHECKPOINT_FILENAME
    new_state = _repackage_checkpoint(checkpoint_path, new_checkpoint)
    source_doc, new_doc = _finalize_ancestry(
        checkpoint_path, new_checkpoint, embedded_ancestry_hash=new_state["ancestry_hash"]
    )

    component_lock = _load_and_copy_json(component_lock_path, out_dir / COMPONENT_LOCK_FILENAME)
    issues = validate_exact_text_s1_component_lock(component_lock)
    if issues:
        raise ModelBundleBuildError(f"component lock invalid: {issues[0].message}")

    allowlist = _load_and_copy_json(allowlist_path, out_dir / ALLOWLIST_FILENAME)
    composite_vocab.validate_qwen_text_allowlist(allowlist)

    caps_payload = _load_and_copy_json(caps_path, out_dir / CAPS_FILENAME)
    caps = parse_loop_guard_v2_caps(caps_payload, origin=str(caps_path))

    selection_policy = load_s4_selection_policy(selection_policy_path)
    selection_hash = s4_selection_policy_hash(selection_policy)
    approved_decode, approved_selection_hash = approved_s4_decode_policy()
    if selection_policy["decode"] != approved_decode or selection_hash != approved_selection_hash:
        raise ModelBundleBuildError(
            "build requires the approved frozen S4-v15b decode policy and selection_policy_hash"
        )
    decode_policy_doc = {
        "decode": selection_policy["decode"],
        "selection_policy_hash": selection_hash,
        "source": {
            "policy_version": selection_policy.get("policy_version"),
            "path": selection_policy_path.name,
        },
    }
    _write_json(out_dir / DECODE_POLICY_FILENAME, decode_policy_doc)

    if pinned_hashes_path is None:
        raise ModelBundleBuildError("build requires --pinned-hashes (directly, or via --run-dir)")
    pinned_hashes = _load_and_copy_json(pinned_hashes_path, out_dir / PINNED_HASHES_FILENAME)
    _validate_pinned_hashes(pinned_hashes)

    components_by_kind = {entry["kind"]: entry for entry in component_lock["components"]}
    try:
        tokenizer_hash = text_tokenizer_identity(component_lock)
    except KeyError as exc:
        raise ModelBundleBuildError(f"component lock: {exc}") from exc
    if tokenizer_hash != allowlist.get("tokenizer_artifact_set_hash"):
        raise ModelBundleBuildError(
            "component lock text_tokenizer identity does not match the allowlist's "
            f"tokenizer_artifact_set_hash: {tokenizer_hash!r} vs "
            f"{allowlist.get('tokenizer_artifact_set_hash')!r}"
        )
    vocab_hash = components_by_kind["composite_vocabulary"]["semantic_hash"]
    allowlist_hash = composite_vocab.qwen_text_allowlist_hash(allowlist)
    capability_profile = resolve_capability_profile_from_component_lock(component_lock)
    capability_profile_hash = capability_profile_hash_from_component_lock(component_lock)
    lock_hash = component_lock_hash(component_lock)

    checkpoint_block = new_doc["checkpoint"]
    identities = {
        "checkpoint_sha256": checkpoint_block["checkpoint_sha256"],
        "checkpoint_id": checkpoint_block["checkpoint_id"],
        "ancestry_hash": new_doc["ancestry_hash"],
        "base_ancestry_hash": checkpoint_block["base_ancestry_hash"],
        "source_checkpoint_sha256": source_doc["checkpoint"]["checkpoint_sha256"],
        "source_checkpoint_id": source_doc["checkpoint"]["checkpoint_id"],
        "source_run_id": source_doc["checkpoint"]["run_id"],
        "source_step": source_doc["checkpoint"]["step"],
        "source_s1_run_manifest_hash": source_doc["checkpoint"]["s1_run_manifest_hash"],
        "component_lock_hash": lock_hash,
        "tokenizer_artifact_set_hash": tokenizer_hash,
        "qwen_text_allowlist_hash": allowlist_hash,
        "composite_vocabulary_hash": vocab_hash,
        "capability_profile_id": capability_profile.profile_id,
        "capability_profile_hash": capability_profile_hash,
        "loop_guard_v2_caps_sha256": caps.sha256,
        "eval_cache_identity": dict(new_state["cache_identity"]),
        "selection_policy_hash": selection_hash,
        "typed_head_config_hash": new_state["typed_head_config_hash"],
        "headless_defaults_policy_id": _HEADLESS_DEFAULTS_POLICY_ID,
        "precision": _precision_identity(precision),
    }
    if new_state.get("visual_feature_tap"):
        identities["visual_feature_tap"] = new_state["visual_feature_tap"]
    base_model = {
        "model_id": QWEN_CANDIDATE_MODEL_ID,
        "revision": QWEN_CANDIDATE_REVISION,
        "weights_set_sha256": pinned_hashes["weights_set_sha256"],
        "processor_set_sha256": pinned_hashes["processor_set_sha256"],
    }

    files: dict[str, Any] = {}
    for name in out_dir.iterdir():
        if name.name == MANIFEST_FILENAME:
            continue
        files[name.name] = {
            "sha256": ca.file_sha256(name),
            "size_bytes": name.stat().st_size,
        }
    manifest = {
        "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
        "files": files,
        "identities": identities,
        "base_model": base_model,
    }
    _write_json(out_dir / MANIFEST_FILENAME, manifest)
    return out_dir


def _build_command(args: argparse.Namespace) -> int:
    run_dir: Path | None = args.run_dir
    component_lock_path = args.component_lock or (
        run_dir / COMPONENT_LOCK_FILENAME if run_dir else None
    )
    allowlist_path = args.allowlist or (
        run_dir / "qwen_text_allowlist.json" if run_dir else None
    )
    pinned_hashes_path = args.pinned_hashes
    if pinned_hashes_path is None and run_dir is not None:
        candidate = run_dir / "qwen-pinned-hashes.json"
        pinned_hashes_path = candidate if candidate.exists() else None
    if component_lock_path is None or allowlist_path is None:
        raise SystemExit(
            "build requires --component-lock and --allowlist (directly, or via --run-dir)"
        )
    precision = None
    if args.precision != "bf16" or args.kv_cache != "bf16":
        precision = QuantizationSpec(decoder_precision=args.precision, kv_cache_dtype=args.kv_cache)
    try:
        out_dir = build(
            checkpoint_path=args.checkpoint,
            component_lock_path=component_lock_path,
            allowlist_path=allowlist_path,
            caps_path=args.caps,
            selection_policy_path=args.selection_policy,
            pinned_hashes_path=pinned_hashes_path,
            out_dir=args.out,
            precision=precision,
        )
    except (ModelBundleBuildError, ModelBundleInvalid) as exc:
        print(f"build refused: {exc}", file=sys.stderr)
        return 1
    print(f"model bundle written to {out_dir}")
    print(f"manifest sha256: {manifest_sha256(out_dir)}")
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    try:
        manifest = verify_bundle(
            args.bundle_dir,
            expected_manifest_sha256=args.expected_sha256,
            hf_cache_dir=args.hf_cache_dir,
        )
        _verify_checkpoint_semantics(args.bundle_dir, manifest)
    except ModelBundleInvalid as exc:
        print(f"verify refused: {exc}", file=sys.stderr)
        return 1
    print(f"manifest sha256: {manifest_sha256(args.bundle_dir)}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/verify a D7 diagnostic model bundle")
    sub = parser.add_subparsers(dest="command", required=True)

    build_parser = sub.add_parser("build", help="build a new model bundle")
    build_parser.add_argument(
        "--checkpoint", type=Path, required=True, help="source training checkpoint (no default)"
    )
    build_parser.add_argument("--run-dir", type=Path, default=None)
    build_parser.add_argument("--component-lock", type=Path, default=None)
    build_parser.add_argument("--allowlist", type=Path, default=None)
    build_parser.add_argument("--pinned-hashes", type=Path, default=None)
    build_parser.add_argument("--caps", type=Path, default=CAPS_PATH)
    build_parser.add_argument(
        "--selection-policy", type=Path, default=_DEFAULT_SELECTION_POLICY_PATH
    )
    build_parser.add_argument("--out", type=Path, required=True)
    build_parser.add_argument(
        "--precision",
        choices=("bf16", "int8", "fp8", "int4"),
        default="bf16",
        help="decoder precision the checkpoint at --checkpoint was already produced at "
        "-- this command never quantizes anything itself, it only pins the identity "
        "(owner requirement item 2, 2026-09-27)",
    )
    build_parser.add_argument(
        "--kv-cache",
        choices=("bf16", "fp8"),
        default="bf16",
        help="KV cache storage dtype identity to pin alongside --precision",
    )
    build_parser.set_defaults(func=_build_command)

    verify_parser = sub.add_parser("verify", help="verify an already-built model bundle")
    verify_parser.add_argument("bundle_dir", type=Path)
    verify_parser.add_argument("--expected-sha256", default=None)
    verify_parser.add_argument("--hf-cache-dir", type=Path, default=None)
    verify_parser.set_defaults(func=_verify_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
