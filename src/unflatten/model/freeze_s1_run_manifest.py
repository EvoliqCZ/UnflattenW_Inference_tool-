"""Freeze one `s1_run_manifest` (plus the `qwen_text_allowlist` it binds)
against a built exact-text cache (design §8, plan `spec-s1-run-manifest`
task 5).

This is the CLI that turns a cache's `ExactTextCacheIdentity` into a
concrete, hash-frozen run: it reads the identity the cache was actually
built with, refuses if the live allowlist producer (`build_exact_text_cache
._real_binding`) no longer reproduces the allowlist the cache is bound to,
derives `required_admission` from `gradient_bearing`, and writes
`s1_run_manifest.json` and `qwen_text_allowlist.json` into `out_dir`.

Ships no model, no gradient, no run execution -- only identity reads and
two JSON writes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unflatten.model import build_exact_text_cache, run_binding
from unflatten.model import stage_policy as sp
from unflatten.model.capability_profile import (
    CapabilityProfileHashUnresolved,
    resolve_capability_profile_by_hash,
)
from unflatten.model.exact_text_cache import (
    CacheExpectation,
    ExactTextCacheIncompatible,
    ExactTextCacheReader,
)
from unflatten.model.hashing import make_id

#: This module is the sole owner of this domain (RFC005 §44.1's `run_`
#: derivation needs artefacts that do not exist yet; this is a deliberately
#: different prefix/domain so the two can never be confused).
_RUN_ID_DOMAIN = "rfc005.s1-run-id/v1"

RUN_ID_PREFIX = "s1run"

#: Filenames this CLI writes under `--out-dir` -- the shared contract other
#: freeze CLIs (`freeze_stage_gate_report.py`) import rather than
#: re-hardcoding, so the names cannot drift apart again.
S1_RUN_MANIFEST_FILENAME = "s1_run_manifest.json"
QWEN_TEXT_ALLOWLIST_FILENAME = "qwen_text_allowlist.json"
#: The frozen cohort expectation (DEC-MP-4) written beside the manifest
#: that pins its hash. A separate file for the same reason the allowlist is
#: one: the manifest pins WHICH payload, the payload itself can be large.
EXACT_TEXT_CACHE_EXPECTATION_FILENAME = "exact_text_cache_expectation.json"


def freeze(
    *,
    cache_dir: Path,
    component_lock: Mapping[str, Any],
    run_class: str,
    gradient_bearing: bool,
    out_dir: Path,
    program_id: str,
    stage_policy: Mapping[str, Any],
    expectation: CacheExpectation | None = None,
    run_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze an `s1_run_manifest` for the cache at `cache_dir`, writing it
    and the bound `qwen_text_allowlist` payload under `out_dir`.

    Refuses (raising `ExactTextCacheIncompatible`) if: the cache itself
    fails its family/completeness/admission checks (`ExactTextCacheReader`);
    the live allowlist producer no longer reproduces the allowlist the
    cache is bound to; the supplied `component_lock` does not hash to the
    identity's `component_lock_hash`; the supplied `stage_policy` document
    does not hash to the identity's `stage_policy_hash`; or
    `out_dir/s1_run_manifest.json` already exists.

    `expectation` is the pinned cohort (DEC-MP-4): the expected page key
    set `E` and approved exclusions `X` this run may read. It is passed
    straight to `ExactTextCacheReader`, so freezing REFUSES an expectation
    the cache at `cache_dir` does not cover -- a manifest is never frozen
    against a cohort that was already known to be unreadable. A
    gradient-bearing freeze without one is refused by
    `RunManifestBindingVerifier`, which `build_s1_run_manifest` runs over
    its own output.

    `program_id` and the `stage_policy` document name the run (1.2.0,
    #17 C): `stage_id` is derived from the document, never typed.
    `run_config` is the run's own budget (#19 B / #21 C): a
    gradient-bearing freeze without one, or one whose `steps x
    sampler.batch_size` exceeds the cohort, is refused by the same
    verifier.
    """
    identity = ExactTextCacheReader(
        cache_dir, gradient_bearing=gradient_bearing, expectation=expectation
    ).identity

    binding, _tokenizer, _width = build_exact_text_cache._real_binding(None)
    if binding.qwen_text_allowlist_hash != identity.qwen_text_allowlist_hash:
        raise ExactTextCacheIncompatible(
            "the live allowlist producer does not reproduce the allowlist "
            "this cache bound"
        )

    required_admission = "strict" if gradient_bearing else identity.admission

    # Resolved from the cache identity's own `capability_profile_hash`
    # (DEC-024 wiring), never an import-time global: the manifest must
    # freeze the profile the cache was actually built under.
    try:
        capability_profile = resolve_capability_profile_by_hash(identity.capability_profile_hash)
    except CapabilityProfileHashUnresolved as exc:
        raise ExactTextCacheIncompatible(
            f"cache identity capability profile refused: {exc}"
        ) from exc

    provisional = run_binding.build_s1_run_manifest(
        run_id="s1run_pending",
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        required_admission=required_admission,
        component_lock=component_lock,
        cache_identity=identity,
        program_id=program_id,
        stage_policy=stage_policy,
        allowlist=binding.allowlist,
        capability_profile=capability_profile,
        expectation=expectation,
        run_config=run_config,
    )
    run_id = make_id(
        RUN_ID_PREFIX,
        _RUN_ID_DOMAIN,
        {k: v for k, v in provisional.items() if k not in ("run_id", "s1_run_manifest_hash")},
    )
    manifest = run_binding.build_s1_run_manifest(
        run_id=run_id,
        run_class=run_class,
        gradient_bearing=gradient_bearing,
        required_admission=required_admission,
        component_lock=component_lock,
        cache_identity=identity,
        program_id=program_id,
        stage_policy=stage_policy,
        allowlist=binding.allowlist,
        capability_profile=capability_profile,
        expectation=expectation,
        run_config=run_config,
    )

    out_dir = Path(out_dir)
    manifest_path = out_dir / S1_RUN_MANIFEST_FILENAME
    if manifest_path.exists():
        raise ExactTextCacheIncompatible(
            f"{manifest_path} already exists -- refusing to overwrite a "
            "frozen run manifest"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (out_dir / QWEN_TEXT_ALLOWLIST_FILENAME).write_text(
        json.dumps(binding.allowlist, indent=1, sort_keys=True) + "\n"
    )
    if expectation is not None:
        (out_dir / EXACT_TEXT_CACHE_EXPECTATION_FILENAME).write_text(
            json.dumps(expectation.payload(), indent=1, sort_keys=True) + "\n"
        )

    return {
        "run_id": manifest["run_id"],
        "s1_run_manifest_hash": manifest["s1_run_manifest_hash"],
        "out_dir": str(out_dir),
        "head_registry_hash": identity.head_registry_hash,
        "exact_text_cache_expectation_hash": manifest[
            "exact_text_cache_expectation_hash"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an s1_run_manifest (+ qwen_text_allowlist) against a "
            "built exact-text cache."
        )
    )
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--component-lock", type=Path, required=True)
    parser.add_argument("--run-class", choices=("discovery", "promotion"), required=True)
    parser.add_argument("--gradient-bearing", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--expectation",
        type=Path,
        default=None,
        help="frozen cohort expectation JSON -- {expected_page_keys: "
        "[[document_bundle_id, page_index], ...], approved_exclusions: [...]}. "
        "Required for --gradient-bearing (DEC-MP-4): the cache is read under "
        "it and the manifest pins its hash",
    )
    parser.add_argument(
        "--program-id",
        required=True,
        help="the training program this run belongs to, e.g. "
        "'unflatten-training-v0/s1-canary@1' (1.2.0, #17 C)",
    )
    parser.add_argument(
        "--stage-policy",
        type=Path,
        default=build_exact_text_cache.STAGE_POLICY_PATH,
        help="frozen stage policy document the manifest's stage_id and the "
        "identity's stage_policy_hash derive from (1.2.0, #17 C)",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=None,
        help="frozen run budget JSON -- {steps, sampler: {seed, order, "
        "batch_size}, augmentation: {kind}, optimizer: {algorithm, "
        "lr_micro, schedule, grad_clip_micro}}. Required for "
        "--gradient-bearing (1.2.0, #21 C): the manifest pins it and the "
        "exposure cap (steps x batch <= cohort, #19 B) is enforced on it",
    )
    args = parser.parse_args(argv)

    if args.gradient_bearing and args.run_config is None:
        parser.error(
            "--run-config is required for --gradient-bearing: a run that "
            "spends optimizer steps must freeze its budget in the manifest "
            "(1.2.0, #21 C)"
        )

    component_lock = json.loads(args.component_lock.read_text())
    expectation = (
        CacheExpectation.from_payload(json.loads(args.expectation.read_text()))
        if args.expectation is not None
        else None
    )
    run_config = json.loads(args.run_config.read_text()) if args.run_config is not None else None
    summary = freeze(
        cache_dir=args.cache_dir,
        component_lock=component_lock,
        run_class=args.run_class,
        gradient_bearing=args.gradient_bearing,
        out_dir=args.out_dir,
        program_id=args.program_id,
        stage_policy=sp.load_stage_policy(args.stage_policy),
        expectation=expectation,
        run_config=run_config,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
