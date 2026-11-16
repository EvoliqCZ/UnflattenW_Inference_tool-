from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from unflatten.service import model_variants as mv

_POLICY = {
    "fp8_min_compute_capability": [8, 9],
    "safety_margin_ratio": 1.15,
    "non_fp8_kv_overhead_ratio": 1.2,
    "static_kv_reserve_mib": 648.0,
    "tiers": [
        {"decoder_precision": "bf16", "kv_capable_dtype": "bf16", "measured_peak_mib": 10000.0},
        {
            "decoder_precision": "int8_or_fp8",
            "kv_capable_dtype": "fp8",
            "measured_peak_mib": 6000.0,
        },
        {"decoder_precision": "int4", "kv_capable_dtype": "fp8", "measured_peak_mib": 5000.0},
    ],
}


def test_estimated_peak_vram_mib_applies_safety_margin_for_matching_tier() -> None:
    assert mv.estimated_peak_vram_mib(
        "bf16", "bf16", policy=_POLICY, kv_cache_mode="bf16_dynamic"
    ) == pytest.approx(11500.0)


def test_estimated_peak_vram_mib_applies_kv_fallback_overhead() -> None:
    # fp8 tier requested with a bf16 KV cache: never-measured combination, inflated.
    expected = 6000.0 * 1.15 * 1.2
    assert mv.estimated_peak_vram_mib(
        "fp8", "bf16", policy=_POLICY, kv_cache_mode="bf16_dynamic"
    ) == pytest.approx(expected)


def test_estimated_peak_vram_mib_int4_maps_to_int4_tier() -> None:
    assert mv.estimated_peak_vram_mib(
        "int4", "fp8", policy=_POLICY, kv_cache_mode="bf16_dynamic"
    ) == pytest.approx(5000.0 * 1.15)


def test_variant_fits_true_when_free_vram_covers_estimate() -> None:
    # 5000 * 1.15 + 648 static KV reserve = 6398
    assert mv.variant_fits("int4", "fp8", free_vram_mib=6500.0, policy=_POLICY) is True


def test_variant_fits_false_when_free_vram_short() -> None:
    assert mv.variant_fits("bf16", "bf16", free_vram_mib=4000.0, policy=_POLICY) is False


def test_variant_fits_false_for_fp8_below_min_compute_capability() -> None:
    assert (
        mv.variant_fits(
            "fp8", "fp8", free_vram_mib=1_000_000.0, compute_capability=(7, 5), policy=_POLICY
        )
        is False
    )


def test_variant_fits_true_for_fp8_at_min_compute_capability() -> None:
    assert (
        mv.variant_fits(
            "fp8", "fp8", free_vram_mib=1_000_000.0, compute_capability=(8, 9), policy=_POLICY
        )
        is True
    )


def test_is_variant_downloaded_true_for_local_bundle_variant() -> None:
    variant = {"variant_id": "a", "bundle": "/some/dir"}
    assert mv.is_variant_downloaded(variant, cache_dir=Path("/unused")) == (True, None)


def test_is_variant_downloaded_false_when_cache_dir_absent(tmp_path: Path) -> None:
    variant = {
        "variant_id": "b",
        "source": {"kind": "hf", "repo_id": "org/repo", "revision": "a" * 40},
    }
    downloaded, size = mv.is_variant_downloaded(variant, cache_dir=tmp_path / "no-such-cache")
    assert downloaded is False
    assert size is None


def test_is_variant_downloaded_true_when_revision_present_in_scanned_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variant = {
        "variant_id": "b",
        "source": {"kind": "hf", "repo_id": "org/repo", "revision": "a" * 40},
    }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    raw = b'{"files":{"weights":{"size_bytes":4}},"identities":{}}'
    (snapshot / "manifest.json").write_bytes(raw)
    (snapshot / "weights").write_bytes(b"abcd")
    variant["source"]["bundle_manifest_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    fake_revision = SimpleNamespace(commit_hash="a" * 40, size_on_disk=1234, snapshot_path=snapshot)
    fake_repo = SimpleNamespace(repo_id="org/repo", revisions=[fake_revision])
    fake_info = SimpleNamespace(repos=[fake_repo])
    monkeypatch.setattr(mv, "_scan_cache_dir", lambda cache_dir: fake_info)

    downloaded, size = mv.is_variant_downloaded(variant, cache_dir=tmp_path)

    assert downloaded is True
    assert size == 1234


def test_partial_hf_snapshot_is_not_downloaded_until_all_declared_files_exist(
    monkeypatch,
    tmp_path,
):
    import hashlib
    import json

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    metadata = {
        "files": {"checkpoint.weights.pt": {"size_bytes": 4}},
        "identities": {"backbone": {"files": {"config.json": "unused-hash"}}},
    }
    manifest = json.dumps(metadata).encode()
    (snapshot / "manifest.json").write_bytes(manifest)
    source = {
        "kind": "hf",
        "repo_id": "org/repo",
        "revision": "a" * 40,
        "bundle_manifest_sha256": "sha256:" + hashlib.sha256(manifest).hexdigest(),
    }
    revision = SimpleNamespace(commit_hash="a" * 40, size_on_disk=1234, snapshot_path=snapshot)
    info = SimpleNamespace(repos=[SimpleNamespace(repo_id="org/repo", revisions=[revision])])
    monkeypatch.setattr(mv, "_scan_cache_dir", lambda cache_dir: info)
    variant = {"variant_id": "v", "source": source}
    assert mv.is_variant_downloaded(variant, cache_dir=tmp_path) == (False, None)
    (snapshot / "checkpoint.weights.pt").write_bytes(b"abcd")
    (snapshot / "backbone").mkdir()
    (snapshot / "backbone/config.json").write_text("{}")
    assert mv.is_variant_downloaded(variant, cache_dir=tmp_path) == (True, 1234)
    (snapshot / "checkpoint.weights.pt").write_bytes(b"partial")
    assert mv.is_variant_downloaded(variant, cache_dir=tmp_path) == (False, None)


def test_resolve_variant_bundle_returns_local_bundle_path_unchanged(tmp_path: Path) -> None:
    variant = {"variant_id": "a", "bundle": "/some/dir"}

    bundle_dir, manifest_sha256 = mv.resolve_variant_bundle(variant, cache_dir=tmp_path)

    assert bundle_dir == Path("/some/dir")
    assert manifest_sha256 is None


def test_resolve_variant_bundle_downloads_hf_source_and_returns_pinned_sha256(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variant = {
        "variant_id": "b",
        "source": {
            "kind": "hf",
            "repo_id": "org/repo",
            "revision": "a" * 40,
            "bundle_manifest_sha256": "sha256:" + "b" * 64,
        },
    }
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(mv, "_snapshot_download", fake_snapshot_download)

    bundle_dir, manifest_sha256 = mv.resolve_variant_bundle(variant, cache_dir=tmp_path)

    assert bundle_dir == tmp_path / "snapshot"
    assert manifest_sha256 == "sha256:" + "b" * 64
    assert seen["repo_id"] == "org/repo"
    assert seen["revision"] == "a" * 40
    assert seen["cache_dir"] == tmp_path


def test_variant_fit_reason_fp8_on_sm86_compute_capability() -> None:
    assert (
        mv.variant_fit_reason(
            "fp8", "fp8", free_vram_mib=1_000_000.0, compute_capability=(8, 6), policy=_POLICY
        )
        == "compute_capability"
    )


def test_variant_fit_reason_bf16_on_12gib_vram() -> None:
    # Short of the bf16 tier's 10000 * 1.15 = 11500 MiB requirement.
    assert mv.variant_fit_reason("bf16", "bf16", free_vram_mib=11000.0, policy=_POLICY) == "vram"


def test_variant_fit_reason_none_when_it_fits() -> None:
    assert mv.variant_fit_reason("int4", "fp8", free_vram_mib=6500.0, policy=_POLICY) is None


@pytest.mark.parametrize(
    ("precision", "kv_cache", "free_vram_mib", "compute_capability"),
    [
        ("bf16", "bf16", 11500.0, None),
        ("bf16", "bf16", 4000.0, None),
        ("fp8", "fp8", 1_000_000.0, (7, 5)),
        ("fp8", "fp8", 1_000_000.0, (8, 9)),
    ],
)
def test_variant_fits_equals_fit_reason_is_none(
    precision: str,
    kv_cache: str,
    free_vram_mib: float,
    compute_capability: tuple[int, int] | None,
) -> None:
    fits = mv.variant_fits(
        precision,
        kv_cache,
        free_vram_mib=free_vram_mib,
        compute_capability=compute_capability,
        policy=_POLICY,
    )
    reason = mv.variant_fit_reason(
        precision,
        kv_cache,
        free_vram_mib=free_vram_mib,
        compute_capability=compute_capability,
        policy=_POLICY,
    )
    assert fits == (reason is None)


_VARIANTS = [
    {"variant_id": "fp8", "precision": "fp8", "kv_cache": "fp8", "bundle": "/data/fp8"},
    {"variant_id": "bf16", "precision": "bf16", "kv_cache": "bf16", "bundle": "/data/bf16"},
    {"variant_id": "int4", "precision": "int4", "kv_cache": "fp8", "bundle": "/data/int4"},
]


def test_resolve_default_user_then_config_then_auto() -> None:
    config = {"default_variant": "bf16", "variants": _VARIANTS}

    assert mv.resolve_default_variant(config, user_default="int4", model_state=None) == (
        "int4",
        "user",
    )
    assert mv.resolve_default_variant(config, user_default=None, model_state=None) == (
        "bf16",
        "config",
    )
    assert mv.resolve_default_variant(config, user_default="does-not-exist", model_state=None) == (
        "bf16",
        "config",
    )


def test_resolve_auto_without_model_state_none() -> None:
    config = {"default_variant": "auto", "variants": _VARIANTS}

    assert mv.resolve_default_variant(config, user_default=None, model_state=None) == (
        None,
        None,
    )


def test_resolve_auto_largest_fitting_precision_tiebreak() -> None:
    config = {"default_variant": "auto", "variants": _VARIANTS}
    # 7600 MiB fits the fp8/int8 tier (6000 * 1.15 + 648 static KV reserve =
    # 7548) and the int4 tier (5000 * 1.15 + 648 = 6398), but not bf16
    # (10000 * 1.15 + 648 = 12148).
    model_state = {"total_vram_mib": 7600.0, "compute_capability": (8, 9)}

    variant_id, source = mv.resolve_default_variant(
        config, user_default=None, model_state=model_state, policy=_POLICY
    )

    assert (variant_id, source) == ("fp8", "auto")


def test_resolve_auto_nothing_fits_none() -> None:
    config = {"default_variant": "auto", "variants": _VARIANTS}
    model_state = {"total_vram_mib": 100.0, "compute_capability": (8, 9)}

    assert mv.resolve_default_variant(
        config, user_default=None, model_state=model_state, policy=_POLICY
    ) == (None, None)


def test_variant_download_state_matrix() -> None:
    variant = {"variant_id": "a", "bundle": "/some/dir"}

    absent = mv.variant_download_state(
        {"variant_id": "b", "source": {"kind": "hf", "repo_id": "org/repo", "revision": "a" * 40}},
        cache_dir=Path("/does/not/exist"),
        active_command=None,
        last_command=None,
    )
    assert absent == {
        "state": "absent",
        "bytes_done": None,
        "bytes_total": None,
        "error_code": None,
        "command_id": None,
    }

    ready = mv.variant_download_state(
        variant, cache_dir=Path("/unused"), active_command=None, last_command=None
    )
    assert ready["state"] == "ready"
    assert ready["command_id"] is None

    active_queued = mv.variant_download_state(
        variant,
        cache_dir=Path("/unused"),
        active_command={"command_id": "c1", "state": "queued", "phase": None},
        last_command=None,
    )
    assert active_queued == {
        "state": "downloading",
        "bytes_done": None,
        "bytes_total": None,
        "error_code": None,
        "command_id": "c1",
    }

    active_running = mv.variant_download_state(
        variant,
        cache_dir=Path("/unused"),
        active_command={
            "command_id": "c2",
            "state": "running",
            "phase": "verifying",
            "bytes_done": 512,
            "bytes_total": 1024,
        },
        last_command=None,
    )
    assert active_running == {
        "state": "verifying",
        "bytes_done": 512,
        "bytes_total": 1024,
        "error_code": None,
        "command_id": "c2",
    }

    failed = mv.variant_download_state(
        {"variant_id": "b", "source": {"kind": "hf", "repo_id": "org/repo", "revision": "a" * 40}},
        cache_dir=Path("/does/not/exist"),
        active_command=None,
        last_command={
            "command_id": "c3",
            "state": "failed",
            "error_code": "MODEL_CHECKPOINT_UNVERIFIED",
        },
    )
    assert failed == {
        "state": "failed",
        "bytes_done": None,
        "bytes_total": None,
        "error_code": "MODEL_CHECKPOINT_UNVERIFIED",
        "command_id": "c3",
    }


def _write_hf_cache_revision(repo_dir: Path, revision: str, content: bytes) -> None:
    import hashlib

    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    blob_hash = hashlib.sha256(content).hexdigest()
    blob_path = blobs / blob_hash
    blob_path.write_bytes(content)
    (snapshot / "config.json").symlink_to(Path("..") / ".." / "blobs" / blob_hash)


def test_delete_variant_snapshot_only_pinned_revision(tmp_path: Path) -> None:
    from huggingface_hub import scan_cache_dir

    repo_dir = tmp_path / "models--unflatten--v15b"
    (repo_dir / "refs").mkdir(parents=True)
    keep_revision = "b" * 40
    doomed_revision = "a" * 40
    _write_hf_cache_revision(repo_dir, doomed_revision, b"doomed-revision-content")
    _write_hf_cache_revision(repo_dir, keep_revision, b"kept-revision-content")

    variant = {
        "variant_id": "remote-int4",
        "source": {
            "kind": "hf",
            "repo_id": "unflatten/v15b",
            "revision": doomed_revision,
            "bundle_manifest_sha256": "sha256:" + "c" * 64,
        },
    }

    mv.delete_variant_snapshot(variant, cache_dir=tmp_path)

    info = scan_cache_dir(tmp_path)
    remaining = {revision.commit_hash for repo in info.repos for revision in repo.revisions}
    assert remaining == {keep_revision}


def test_delete_variant_snapshot_rejects_local_bundle_variant(tmp_path: Path) -> None:
    variant = {"variant_id": "a", "bundle": "/some/dir"}
    with pytest.raises(mv.ModelVariantsInvalid, match="bundle"):
        mv.delete_variant_snapshot(variant, cache_dir=tmp_path)


def test_resolve_variant_bundle_honours_subfolder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    variant = {
        "variant_id": "b",
        "source": {
            "kind": "hf",
            "repo_id": "org/repo",
            "revision": "a" * 40,
            "bundle_manifest_sha256": "sha256:" + "b" * 64,
            "subfolder": "v15b-fp8",
        },
    }
    monkeypatch.setattr(mv, "_snapshot_download", lambda **kwargs: str(tmp_path / "snapshot"))

    bundle_dir, _ = mv.resolve_variant_bundle(variant, cache_dir=tmp_path)

    assert bundle_dir == tmp_path / "snapshot" / "v15b-fp8"


def test_estimated_peak_vram_mib_adds_static_kv_reserve_by_default() -> None:
    """Owner decision 2026-09-29: the service decodes on a static KV cache by
    default, preallocated for `max_steps` -- the estimate carries that
    reserve on top of the margin-inflated measured (dynamic-cache) peak."""
    assert mv.SERVICE_DEFAULT_KV_CACHE_MODE == "static_graph"
    assert mv.estimated_peak_vram_mib("bf16", "bf16", policy=_POLICY) == pytest.approx(
        11500.0 + 648.0
    )


def test_estimated_peak_vram_mib_bf16_dynamic_mode_has_no_static_reserve() -> None:
    assert mv.estimated_peak_vram_mib(
        "bf16", "bf16", policy=_POLICY, kv_cache_mode="bf16_dynamic"
    ) == pytest.approx(11500.0)


def test_variant_fits_counts_the_static_kv_reserve() -> None:
    # int4: 5000 * 1.15 = 5750 fits 6000 MiB on a dynamic cache, but not
    # with the 648 MiB static reserve on top (6398).
    assert (
        mv.variant_fits(
            "int4", "fp8", free_vram_mib=6000.0, policy=_POLICY, kv_cache_mode="bf16_dynamic"
        )
        is True
    )
    assert mv.variant_fits("int4", "fp8", free_vram_mib=6000.0, policy=_POLICY) is False


def test_real_policy_static_kv_reserve_is_max_steps_of_bf16_kv() -> None:
    """The reserve is the static buffer's worst-case excess over a dynamic
    cache: `max_steps` (4608, every v15b bundle) positions of bf16 K and V
    over Qwen3-VL-4B's 36 layers x 8 KV heads x head_dim 128."""
    from unflatten.model.quantization.auto_policy import load_auto_policy

    per_token_bytes = 36 * 2 * 8 * 128 * 2
    assert load_auto_policy()["static_kv_reserve_mib"] == pytest.approx(
        4608 * per_token_bytes / (1024 * 1024)
    )
    assert load_auto_policy()["static_kv_reserve_mib"] == pytest.approx(648.0)


# Owner request 2026-09-29: unpublished/planned catalogue variants are never
# a default -- not from a user override, not from the config, not from auto.
_UNPUBLISHED_BF16 = {
    "variant_id": "bf16-hf",
    "precision": "bf16",
    "kv_cache": "bf16",
    "availability": "unpublished",
}
_PLANNED_MOE = {"variant_id": "moe", "precision": None, "kv_cache": None, "availability": "planned"}
#: The unpublished bf16 entry comes FIRST, so config order would pick it over
#: the published bf16 on the size tie if availability were not checked.
_NON_PUBLISHED_VARIANTS = [_UNPUBLISHED_BF16, *_VARIANTS, _PLANNED_MOE]


@pytest.mark.parametrize("user_default", ["bf16-hf", "moe"])
def test_resolve_default_variant_user_override_on_non_published_is_ignored(
    user_default: str,
) -> None:
    config = {"default_variant": "int4", "variants": _NON_PUBLISHED_VARIANTS}

    assert mv.resolve_default_variant(config, user_default=user_default, model_state=None) == (
        "int4",
        "config",
    )


def test_resolve_default_variant_auto_skips_non_published_variants() -> None:
    # 20000 MiB fits every tier; the unpublished bf16 entry ties the
    # published bf16 on size (and precedes it) but must never win, and the
    # planned entry (no precision) must never be sized at all.
    config = {"default_variant": "auto", "variants": _NON_PUBLISHED_VARIANTS}
    model_state = {"total_vram_mib": 20000.0, "compute_capability": (8, 9)}

    assert mv.resolve_default_variant(
        config, user_default=None, model_state=model_state, policy=_POLICY
    ) == ("bf16", "auto")


def test_resolve_default_variant_auto_with_only_non_published_returns_none() -> None:
    config = {"default_variant": "auto", "variants": [_UNPUBLISHED_BF16, _PLANNED_MOE]}
    model_state = {"total_vram_mib": 20000.0, "compute_capability": (8, 9)}

    assert mv.resolve_default_variant(
        config, user_default=None, model_state=model_state, policy=_POLICY
    ) == (None, None)


def test_static_graph_mode_carries_the_same_static_kv_reserve() -> None:
    """`static_graph` preallocates the same `StaticKVCache` buffer (plus a
    small graph pool), so it needs at least the static reserve."""
    assert mv.static_kv_reserve_mib("static_graph", _POLICY) == pytest.approx(648.0)
