"""Freeze / load the generation-0 checkpoint ancestry base for an RL run.

Wraps ``freeze_checkpoint_ancestry.freeze`` (which does the cross-checks of
``checkpoint_ancestry.build_base_ancestry``): the output directory is a function
parameter, an existing ``checkpoint_ancestry_base.json`` is never overwritten,
the result is re-read and verified with ``ca.verify_base``. For a warm start from
a parent checkpoint the parent's release must be declared in ``also_exposed``
(DEC-006 transitive exposure) -- ``build_base_ancestry`` refuses otherwise.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_RELEASE_MANIFEST_NAME = "release-manifest.json"


def freeze_ancestry_base(
    parent_checkpoint: Path | None,
    parent_run_manifest: Path | None,
    out_dir: Path,
    *,
    pinned_hashes: Path,
    component_lock: Path,
    s0_report: Path,
    release_manifest: Path,
    release_name: str,
    store_name: str,
    base_model_decision: Path,
    dataset_decision: Path,
    distribution_decision: Path | None = None,
    also_exposed: Sequence[tuple[Path, str, str]] = (),
    expected_revision: str | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Write ``out_dir/checkpoint_ancestry_base.json`` and return its document.

    ``release_manifest`` is the ``release-manifest.json`` file of the release being
    trained on; ``also_exposed`` is ``(release_dir, release_name, store_name)`` per
    further exposure (the parent's release for a warm start).
    """
    from unflatten.model import checkpoint_ancestry as ca
    from unflatten.model import freeze_checkpoint_ancestry as fca

    release_manifest = Path(release_manifest)
    if release_manifest.name != _RELEASE_MANIFEST_NAME:
        raise ca.CheckpointAncestryInvalid(
            f"release_manifest must be a file named {_RELEASE_MANIFEST_NAME!r}, "
            f"got {release_manifest.name!r}"
        )
    out_path = Path(out_dir) / fca.CHECKPOINT_ANCESTRY_BASE_FILENAME
    if out_path.exists():
        raise ca.CheckpointAncestryInvalid(
            f"{out_path} already exists -- refusing to overwrite a frozen base ancestry"
        )
    if expected_revision is None or model_id is None:
        from unflatten.model.candidate_loader import (
            QWEN_CANDIDATE_MODEL_ID,
            QWEN_CANDIDATE_REVISION,
        )

        expected_revision = expected_revision or QWEN_CANDIDATE_REVISION
        model_id = model_id or QWEN_CANDIDATE_MODEL_ID
    fca.freeze(
        pinned_hashes=pinned_hashes,
        component_lock=component_lock,
        s0_report=s0_report,
        release_dir=release_manifest.parent,
        release_name=release_name,
        store_name=store_name,
        base_model_decision=base_model_decision,
        dataset_decision=dataset_decision,
        distribution_decision=distribution_decision,
        parent_checkpoint=parent_checkpoint,
        parent_run_manifest=parent_run_manifest,
        also_exposed_releases=[(Path(d), n, s) for d, n, s in also_exposed],
        out_dir=Path(out_dir),
        expected_revision=expected_revision,
        model_id=model_id,
    )
    return _load_verified(out_path)


def _load_verified(path: Path) -> dict[str, Any]:
    from unflatten.model import checkpoint_ancestry as ca

    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ca.CheckpointAncestryInvalid(f"ancestry base {path} is unreadable: {exc}") from exc
    if not isinstance(doc, dict):
        raise ca.CheckpointAncestryInvalid(f"ancestry base {path} is not a JSON object")
    ca.verify_base(doc)
    return doc


def make_ancestry_base_port(path: Path) -> Callable[[], Mapping[str, Any]]:
    """Load + verify an already frozen base once; the port returns a copy per call."""
    doc = _load_verified(Path(path))

    def ancestry_base() -> Mapping[str, Any]:
        return copy.deepcopy(doc)

    return ancestry_base
