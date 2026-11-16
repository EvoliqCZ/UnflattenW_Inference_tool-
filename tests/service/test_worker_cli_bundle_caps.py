"""The worker decodes with the loop guard v2 caps the BUNDLE carries and pins
(verify_bundle checks the file's sha256 against the manifest), not whatever
caps file sits in the code tree."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from unflatten.model import loop_guard_v2_caps
from unflatten.service import worker_cli

_E2E_BUNDLE = Path(__file__).resolve().parents[1] / "data" / "e2e" / "model-bundle"


@pytest.fixture(autouse=True)
def _restore_caps_path(monkeypatch):
    monkeypatch.setattr(loop_guard_v2_caps, "CAPS_PATH", loop_guard_v2_caps.CAPS_PATH)


def test_bind_bundle_loop_guard_caps_points_runtime_at_the_bundle_file(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    shutil.copy2(_E2E_BUNDLE / "loop_guard_v2_caps.json", bundle / "loop_guard_v2_caps.json")
    before = loop_guard_v2_caps.CAPS_PATH

    worker_cli.bind_bundle_loop_guard_caps(bundle)

    assert loop_guard_v2_caps.CAPS_PATH == bundle / "loop_guard_v2_caps.json"
    assert loop_guard_v2_caps.CAPS_PATH != before


def test_bind_bundle_loop_guard_caps_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="loop_guard_v2_caps.json"):
        worker_cli.bind_bundle_loop_guard_caps(tmp_path)
