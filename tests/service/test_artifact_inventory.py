from __future__ import annotations

from pathlib import Path

import pytest

from unflatten.service.artifact_store import ArtifactStore, ArtifactStoreError

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _accept(_artifact, publish) -> bool:
    publish()
    return True


def test_published_artifacts_finds_diagnostic_docx_with_docx_mime(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = store.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id="execution_1234567890abcdef",
        role="diagnostic_docx",
        mime_type=_DOCX_MIME,
        chunks=[b"PK\x03\x04docx"],
    )
    store.publish(staged, _accept)

    [record] = store.published_artifacts_for_retention()

    assert record.role == "diagnostic_docx"
    assert record.artifact.mime_type == _DOCX_MIME


def test_diagnostic_docx_role_refuses_a_wrong_mime(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ArtifactStoreError) as error:
        store.stage(
            principal_id="principal_alpha",
            job_id="job_1234567890abcdef",
            execution_id="execution_1234567890abcdef",
            role="diagnostic_docx",
            mime_type="application/json",
            chunks=[b"PK\x03\x04docx"],
        )

    assert error.value.code == "ARTIFACT_MIME_INVALID"
