"""Unit tests for the LibreOffice open probe with a mocked subprocess
(RFC002 V5 "application open probe") — exercised independently of
whether LibreOffice is actually installed in the CI environment.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from unflatten.renderer import probe as probe_module
from unflatten.renderer.probe import libreoffice_open_probe, libreoffice_pdf_bytes, probe_available


def _fake_completed(returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def _outdir_from_cmd(cmd: list[str]) -> Path:
    return Path(cmd[cmd.index("--outdir") + 1])


def test_open_probe_reports_missing_input(tmp_path):
    issue = libreoffice_open_probe(tmp_path / "does-not-exist.docx")
    assert issue is not None
    assert issue.code == "OPEN_PROBE_INPUT_MISSING"


def test_open_probe_success_when_pdf_produced(tmp_path):
    docx_path = tmp_path / "in.docx"
    docx_path.write_bytes(b"fake-docx-bytes")

    def fake_run(cmd, **kwargs):
        # Simulate libreoffice writing the expected PDF as a side effect,
        # into whatever unique --outdir this call was given.
        outdir = _outdir_from_cmd(cmd)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "in.pdf").write_bytes(b"%PDF-1.4 fake")
        return _fake_completed(returncode=0)

    with patch("unflatten.renderer.probe.subprocess.run", side_effect=fake_run):
        issue = libreoffice_open_probe(docx_path)
    assert issue is None


def test_open_probe_stages_each_call_in_a_unique_directory(tmp_path):
    docx_path = tmp_path / "in.docx"
    docx_path.write_bytes(b"fake-docx-bytes")
    seen_outdirs: list[Path] = []

    def fake_run(cmd, **kwargs):
        outdir = _outdir_from_cmd(cmd)
        seen_outdirs.append(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "in.pdf").write_bytes(b"%PDF-1.4 fake")
        return _fake_completed(returncode=0)

    with patch("unflatten.renderer.probe.subprocess.run", side_effect=fake_run):
        libreoffice_open_probe(docx_path)
        libreoffice_open_probe(docx_path)

    assert len(seen_outdirs) == 2
    assert seen_outdirs[0] != seen_outdirs[1]
    # Cleaned up in `finally` — nothing left behind after the calls.
    assert not seen_outdirs[0].exists()
    assert not seen_outdirs[1].exists()


def test_open_probe_failure_when_returncode_nonzero(tmp_path):
    docx_path = tmp_path / "in.docx"
    docx_path.write_bytes(b"fake-docx-bytes")

    with patch(
        "unflatten.renderer.probe.subprocess.run",
        return_value=_fake_completed(returncode=1, stderr="boom"),
    ):
        issue = libreoffice_open_probe(docx_path)
    assert issue is not None
    assert issue.code == "OPEN_PROBE_FAILED"
    assert issue.details["stderr"] == "boom"


def test_open_probe_failure_when_pdf_missing_despite_zero_returncode(tmp_path):
    docx_path = tmp_path / "in2.docx"
    docx_path.write_bytes(b"fake-docx-bytes")

    with patch(
        "unflatten.renderer.probe.subprocess.run", return_value=_fake_completed(returncode=0)
    ):
        issue = libreoffice_open_probe(docx_path)
    assert issue is not None
    assert issue.code == "OPEN_PROBE_FAILED"


def test_open_probe_timeout_reported_as_issue(tmp_path):
    docx_path = tmp_path / "in.docx"
    docx_path.write_bytes(b"fake-docx-bytes")

    with patch(
        "unflatten.renderer.probe.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="libreoffice", timeout=60),
    ):
        issue = libreoffice_open_probe(docx_path)
    assert issue is not None
    assert issue.code == "OPEN_PROBE_TIMEOUT"


def test_open_probe_unavailable_when_binary_missing(tmp_path):
    docx_path = tmp_path / "in.docx"
    docx_path.write_bytes(b"fake-docx-bytes")

    with patch(
        "unflatten.renderer.probe.subprocess.run", side_effect=FileNotFoundError()
    ):
        issue = libreoffice_open_probe(docx_path)
    assert issue is not None
    assert issue.code == "OPEN_PROBE_UNAVAILABLE"


def test_probe_available_is_cached(monkeypatch):
    monkeypatch.setattr(probe_module, "_available", True)
    assert probe_module.probe_available() is True


def test_probe_available_false_when_libreoffice_not_on_path(monkeypatch):
    monkeypatch.setattr(probe_module, "_available", None)
    with patch("unflatten.renderer.probe.shutil.which", return_value=None):
        assert probe_module.probe_available() is False
    monkeypatch.setattr(probe_module, "_available", None)


@pytest.mark.slow
@pytest.mark.skipif(not probe_available(), reason="LibreOffice open probe unavailable here.")
class TestLibreofficePdfBytes:
    """Real LibreOffice conversion — no mocking, guarded by
    `probe_available()` per RUNBOOK/CLAUDE.md convention.
    """

    def _minimal_docx(self, tmp_path: Path) -> Path:
        from unflatten.renderer.api import RendererOptions
        from unflatten.renderer.compiler import compile_docx

        ast = {
            "ast_version": "0.1.0",
            "profile": "word-core-v0",
            "document_id": "probe-pdf-bytes-test",
            "metadata": {"language": None, "title": None, "subject": None},
            "resources": [],
            "sections": [
                {
                    "id": "s1",
                    "type": "section",
                    "break_type": "next_page",
                    "page": {
                        "width_mpt": 595276,
                        "height_mpt": 841890,
                        "orientation": "portrait",
                    },
                    "margins": {
                        "top_mpt": 72000,
                        "right_mpt": 72000,
                        "bottom_mpt": 72000,
                        "left_mpt": 72000,
                        "gutter_mpt": 0,
                    },
                    "columns": {
                        "count": 1,
                        "equal_width": True,
                        "gap_mpt": 18000,
                        "widths_mpt": [],
                    },
                    "children": [],
                }
            ],
            "extensions": {},
        }
        docx_path = tmp_path / "in.docx"
        compile_docx(ast, docx_path, RendererOptions())
        return docx_path

    def test_returns_pdf_bytes_for_known_good_docx(self, tmp_path):
        docx_path = self._minimal_docx(tmp_path)
        pdf_bytes, issue = libreoffice_pdf_bytes(docx_path)
        assert issue is None
        assert pdf_bytes is not None
        assert pdf_bytes.startswith(b"%PDF-")

    def test_uses_the_process_profile_dir(self, tmp_path):
        docx_path = self._minimal_docx(tmp_path)
        seen_profiles: list[str] = []
        real_run = probe_module._run_convert

        def spy_run_convert(input_path, outdir, profile_dir):
            seen_profiles.append(str(profile_dir))
            return real_run(input_path, outdir, profile_dir)

        with patch.object(probe_module, "_run_convert", side_effect=spy_run_convert):
            pdf_bytes, issue = libreoffice_pdf_bytes(docx_path)

        assert issue is None
        assert pdf_bytes is not None
        assert seen_profiles == [str(probe_module._get_profile_dir())]

    def test_libreoffice_profile_paths_stay_distinct(self):
        """Guard for the existing distinct-profile-dir invariant this
        change must not break (CLAUDE.md: all three LibreOffice call
        paths must resolve to distinct profile directories).
        """
        import os

        from unflatten.libreoffice import REGISTERED_CALL_PATHS, registered_profile_dirs

        paths = registered_profile_dirs(process_id=os.getpid())
        assert set(paths) == set(REGISTERED_CALL_PATHS)
        assert len(set(paths.values())) == len(REGISTERED_CALL_PATHS) == 5
