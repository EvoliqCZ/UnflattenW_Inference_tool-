from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import pytest

from unflatten.dataset.artifact_ref import ArtifactRef
from unflatten.service.artifact_store import (
    DOWNLOADABLE_ROLES,
    ArtifactStore,
    ArtifactStoreError,
)
from unflatten.service.upload_staging import stage_upload

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_ROLE_CHUNKS = {
    "prediction_ir": [b'{"nodes":', b"[]}"],
    "source_pdf": [b"%PDF-", b"1.7"],
    "page_png": [b"\x89PNG", b"rest"],
    "diagnostic_docx": [b"PK\x03\x04", b"docx"],
}


def _accept(_artifact: ArtifactRef, publish: object) -> bool:
    publish()
    return True


def _stage(
    store: ArtifactStore, *, principal: str = "principal_alpha", role: str = "prediction_ir"
) -> ArtifactRef:
    return store.stage(
        principal_id=principal,
        job_id="job_1234567890abcdef",
        execution_id="execution_1234567890abcdef",
        role=role,
        mime_type={
            "prediction_ir": "application/json",
            "source_pdf": "application/pdf",
            "page_png": "image/png",
            "diagnostic_docx": _DOCX_MIME,
        }[role],
        chunks=_ROLE_CHUNKS[role],
    )


def test_artifact_store_stage_streams_hash_size_and_uses_restrictive_modes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = _stage(store)

    assert staged.sha256 == f"sha256:{hashlib.sha256(b'{"nodes":[]}').hexdigest()}"
    assert staged.size_bytes == 12
    path = store.root / staged.ref
    assert path.stat().st_mode & 0o777 == 0o600
    assert all(
        parent.stat().st_mode & 0o777 == 0o700
        for parent in path.parents
        if parent == store.root or store.root in parent.parents
    )


@pytest.mark.parametrize("failure", [RuntimeError("disconnect"), asyncio.CancelledError()])
def test_async_staging_cleans_partial_bytes_on_failure_or_cancellation(
    tmp_path: Path, failure: BaseException
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    async def chunks():
        yield b"%PDF-partial"
        raise failure

    async def stage() -> None:
        await stage_upload(
            store,
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            execution_id="ingress_1234567890abcdef",
            chunks=chunks(),
            max_bytes=1024,
        )

    expected = (
        asyncio.CancelledError
        if isinstance(failure, asyncio.CancelledError)
        else ArtifactStoreError
    )
    with pytest.raises(expected):
        asyncio.run(stage())
    assert list(store.root.rglob("*.staged")) == []


def test_artifact_store_publish_makes_bytes_visible_only_inside_fence(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = _stage(store)
    observed: list[bool] = []

    def fence(artifact: ArtifactRef, publish: object) -> bool:
        observed.append((store.root / artifact.ref).exists())
        publish()
        observed.append((store.root / artifact.ref).exists())
        return True

    published = store.publish(staged, fence)

    assert observed == [False, True]
    assert "/attempts/" not in published.ref
    assert (
        store.read_verified(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            role="prediction_ir",
            artifact=published,
        )
        == b'{"nodes":[]}'
    )


def test_artifact_store_rejected_fence_removes_staged_bytes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = _stage(store)

    with pytest.raises(ArtifactStoreError, match="fence") as error:
        store.publish(staged, lambda _artifact, _publish: False)

    assert error.value.code == "ARTIFACT_FENCE_REJECTED"
    assert not (store.root / staged.ref).exists()


@pytest.mark.parametrize("role", ["source_pdf", "page_png"])
def test_artifact_store_source_and_raster_are_never_downloadable(tmp_path: Path, role: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    published = store.publish(_stage(store, role=role), _accept)

    with pytest.raises(ArtifactStoreError) as error:
        store.read_downloadable(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            role=role,
            artifact=published,
            expires_at=200.0,
            now=100.0,
        )

    assert error.value.code == "ARTIFACT_NOT_DOWNLOADABLE"


def test_artifact_store_accepts_auth_contract_opaque_principal_ids(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    published = store.publish(_stage(store, principal="team-a"), _accept)

    assert (
        store.read_downloadable(
            principal_id="team-a",
            job_id="job_1234567890abcdef",
            role="prediction_ir",
            artifact=published,
            expires_at=200.0,
            now=100.0,
        )
        == b'{"nodes":[]}'
    )


def test_artifact_store_expired_prediction_is_not_read(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    published = store.publish(_stage(store), _accept)

    with pytest.raises(ArtifactStoreError) as error:
        store.read_downloadable(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            role="prediction_ir",
            artifact=published,
            expires_at=200.0,
            now=200.0,
        )

    assert error.value.code == "ARTIFACT_EXPIRED"


def test_artifact_store_cross_principal_read_is_hidden(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    published = store.publish(_stage(store), _accept)

    with pytest.raises(ArtifactStoreError) as error:
        store.read_verified(
            principal_id="principal_beta",
            job_id="job_1234567890abcdef",
            role="prediction_ir",
            artifact=published,
        )

    assert error.value.code == "ARTIFACT_NOT_FOUND"


@pytest.mark.parametrize("ref", ["../escape", "principals/alpha/../../escape", "/etc/passwd"])
def test_artifact_store_traversal_reference_is_rejected(tmp_path: Path, ref: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    artifact = object.__new__(ArtifactRef)
    object.__setattr__(artifact, "ref", ref)
    object.__setattr__(artifact, "sha256", "sha256:" + "0" * 64)
    object.__setattr__(artifact, "size_bytes", 0)
    object.__setattr__(artifact, "mime_type", "application/json")

    with pytest.raises(ArtifactStoreError):
        store.read_verified(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            role="prediction_ir",
            artifact=artifact,
        )


def test_artifact_store_symlink_swap_and_integrity_mismatch_fail_closed(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    published = store.publish(_stage(store), _accept)
    path = store.root / published.ref
    outside = tmp_path / "outside"
    outside.write_bytes(b'{"nodes":[]}')
    path.unlink()
    path.symlink_to(outside)

    with pytest.raises(ArtifactStoreError) as error:
        store.read_verified(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            role="prediction_ir",
            artifact=published,
        )
    assert error.value.code == "ARTIFACT_PATH_UNSAFE"


def test_artifact_store_modified_bytes_fail_hash_verification(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    published = store.publish(_stage(store), _accept)
    (store.root / published.ref).write_bytes(b"same length!!")

    with pytest.raises(ArtifactStoreError) as error:
        store.read_verified(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            role="prediction_ir",
            artifact=published,
        )
    assert error.value.code in {"ARTIFACT_SIZE_MISMATCH", "ARTIFACT_HASH_MISMATCH"}


def test_artifact_store_tampered_staging_never_reaches_fence(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = _stage(store)
    (store.root / staged.ref).write_bytes(b"tampered")
    fence_called = False

    def fence(_artifact: ArtifactRef, _publish: object) -> bool:
        nonlocal fence_called
        fence_called = True
        return True

    with pytest.raises(ArtifactStoreError) as error:
        store.publish(staged, fence)

    assert error.value.code == "ARTIFACT_SIZE_MISMATCH"
    assert fence_called is False


def test_artifact_store_cleanup_removes_only_old_invisible_attempt_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    old = _stage(store)
    recent = _stage(store)
    old_path = store.root / old.ref
    os.utime(old_path, (10, 10))

    assert store.cleanup_orphans(older_than=20) == 1
    assert not old_path.exists()
    assert (store.root / recent.ref).exists()


def _accept_many(_artifacts: list, publish: object) -> bool:
    publish()
    return True


def test_downloadable_roles_covers_prediction_ir_and_diagnostic_docx() -> None:
    assert DOWNLOADABLE_ROLES == frozenset(
        {"prediction_ir", "diagnostic_docx", "source_preview", "docx_preview"}
    )


def test_read_downloadable_accepts_diagnostic_docx(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    published = store.publish(_stage(store, role="diagnostic_docx"), _accept)

    assert (
        store.read_downloadable(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            role="diagnostic_docx",
            artifact=published,
            expires_at=200.0,
            now=100.0,
        )
        == b"PK\x03\x04docx"
    )


def test_publish_many_publishes_both_artifacts_with_mime_and_restrictive_mode(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged_ir = _stage(store, role="prediction_ir")
    staged_docx = _stage(store, role="diagnostic_docx")
    observed_roles: list[str] = []

    def fence(artifacts: list[ArtifactRef], publish: object) -> bool:
        observed_roles.extend(a.mime_type for a in artifacts)
        publish()
        return True

    published = store.publish_many([staged_ir, staged_docx], fence)

    assert [a.mime_type for a in published] == [
        "application/json",
        _DOCX_MIME,
    ]
    assert observed_roles == ["application/json", _DOCX_MIME]
    for artifact in published:
        path = store.root / artifact.ref
        assert path.exists()
        assert path.stat().st_mode & 0o777 == 0o600


def test_publish_many_calls_publish_callback_exactly_once(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = [_stage(store, role="prediction_ir"), _stage(store, role="diagnostic_docx")]
    calls: list[int] = []

    def fence(_artifacts: list[ArtifactRef], publish: object) -> bool:
        publish()
        calls.append(1)
        with pytest.raises(ArtifactStoreError) as error:
            publish()
        assert error.value.code == "ARTIFACT_FENCE_PROTOCOL"
        return True

    store.publish_many(staged, fence)

    assert calls == [1]


def test_publish_many_rejects_duplicate_role(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = [_stage(store, role="prediction_ir"), _stage(store, role="prediction_ir")]

    with pytest.raises(ArtifactStoreError) as error:
        store.publish_many(staged, _accept_many)

    assert error.value.code == "ARTIFACT_ROLE_DUPLICATE"


def test_publish_many_allow_repeated_roles_publishes_both_distinctly(tmp_path: Path) -> None:
    """DEC-044 Dodatok D §1 (B1): a multi-page job publishes one
    `prediction_ir` per page in the SAME atomic call as the merged docx."""
    store = ArtifactStore(tmp_path / "artifacts")
    staged = [_stage(store, role="prediction_ir"), _stage(store, role="prediction_ir")]

    published = store.publish_many(staged, _accept_many, allow_repeated_roles=True)

    assert len(published) == 2
    assert published[0].ref != published[1].ref
    for ref in published:
        assert (store.root / ref.ref).is_file()


def test_publish_many_second_replace_failure_leaves_no_target_and_cleans_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged_ir = _stage(store, role="prediction_ir")
    staged_docx = _stage(store, role="diagnostic_docx")

    real_replace = os.replace
    call_count = {"n": 0}

    def flaky_replace(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("ENOSPC simulated")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    with pytest.raises(ArtifactStoreError) as error:
        store.publish_many([staged_ir, staged_docx], _accept_many)

    assert error.value.code == "ARTIFACT_PUBLICATION_FAILED"
    assert error.value.role == "diagnostic_docx"
    assert isinstance(error.value.__cause__, OSError)
    assert not (store.root / staged_ir.ref).exists()
    assert not (store.root / staged_docx.ref).exists()
    assert list(store.root.rglob("artifact_*.bin")) == []
    assert list(store.root.rglob("*.staged")) == []


def test_publish_many_rejected_fence_removes_all_staged_bytes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = [_stage(store, role="prediction_ir"), _stage(store, role="diagnostic_docx")]

    with pytest.raises(ArtifactStoreError) as error:
        store.publish_many(staged, lambda _artifacts, _publish: False)

    assert error.value.code == "ARTIFACT_FENCE_REJECTED"
    for ref in staged:
        assert not (store.root / ref.ref).exists()
