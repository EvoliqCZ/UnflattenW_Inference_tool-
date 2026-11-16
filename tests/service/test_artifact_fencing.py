from __future__ import annotations

from pathlib import Path

import pytest

from unflatten.service.artifact_store import ArtifactStore, ArtifactStoreError


def test_publish_callback_must_execute_atomic_publication(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    staged = store.stage(
        principal_id="principal_alpha",
        job_id="job_1234567890abcdef",
        execution_id="execution_stale00000001",
        role="prediction_ir",
        mime_type="application/json",
        chunks=[b"{}"],
    )

    with pytest.raises(ArtifactStoreError) as error:
        store.publish(staged, lambda _artifact, _publish: True)

    assert error.value.code == "ARTIFACT_FENCE_PROTOCOL"
    assert not (store.root / staged.ref).exists()
