from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from unflatten.eval.s2_gate import sealed
from unflatten.service import docx_stage as ds


def _fake_runtime(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "tokenizer": object(),
        "decode_policy": SimpleNamespace(
            typed_decode=True, region_decode=True, loop_guard="v2", max_steps=4608,
            selection_policy_hash="sha256:selection", prefill_chunk_tokens=None,
        ),
        "eval_cache_identity": {"dataset_release_root_hash": "sha256:release"},
        "component_lock_hash": "sha256:lock",
        "selection_policy_hash": "sha256:selection",
        "vocabulary": object(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestInMemoryRasterSource:
    def test_serves_the_pinned_bundle_at_page_zero(self) -> None:
        source = ds.InMemoryRasterSource("job_abc", b"png-bytes")
        assert source.read("job_abc", 0) == b"png-bytes"

    def test_refuses_a_foreign_bundle_id(self) -> None:
        source = ds.InMemoryRasterSource("job_abc", b"png-bytes")
        with pytest.raises(ValueError, match="foreign bundle"):
            source.read("job_other", 0)

    def test_refuses_a_nonzero_page_index(self) -> None:
        source = ds.InMemoryRasterSource("job_abc", b"png-bytes")
        with pytest.raises(ValueError, match="page at index 0"):
            source.read("job_abc", 1)

    def test_refuses_a_bool_page_index(self) -> None:
        source = ds.InMemoryRasterSource("job_abc", b"png-bytes")
        with pytest.raises(ValueError, match="page at index 0"):
            source.read("job_abc", True)


class TestDiagnosticDocxCapability:
    def test_true_when_every_member_present(self) -> None:
        assert ds.diagnostic_docx_capability(_fake_runtime()) is True

    @pytest.mark.parametrize(
        "overrides",
        [
            {"tokenizer": None},
            {"eval_cache_identity": {}},
            {"component_lock_hash": ""},
            {"selection_policy_hash": ""},
        ],
    )
    def test_false_when_a_member_is_flipped(self, overrides: dict[str, object]) -> None:
        assert ds.diagnostic_docx_capability(_fake_runtime(**overrides)) is False

    @pytest.mark.parametrize(
        "decode_policy",
        [
            SimpleNamespace(typed_decode=False, region_decode=True, loop_guard="v2"),
            SimpleNamespace(typed_decode=True, region_decode=False, loop_guard="v2"),
            SimpleNamespace(typed_decode=True, region_decode=True, loop_guard="v1"),
            SimpleNamespace(typed_decode=True, region_decode=True, loop_guard=None),
        ],
    )
    def test_false_when_decode_policy_member_is_flipped(self, decode_policy) -> None:
        assert ds.diagnostic_docx_capability(_fake_runtime(decode_policy=decode_policy)) is False

    @pytest.mark.parametrize("policy_module", ["hd", "tg", "sd", "rim"])
    def test_false_when_any_applied_policy_status_is_not_ratified(
        self, monkeypatch: pytest.MonkeyPatch, policy_module: str
    ) -> None:
        """DEC-044 Dodatok B: a PROPOSED policy must never ship -- readiness
        is the conjunction over EVERY policy the service applies."""
        monkeypatch.setattr(getattr(ds, policy_module), "POLICY_STATUS", "PROPOSED")
        assert ds.diagnostic_docx_capability(_fake_runtime()) is False

    def test_service_applies_exactly_the_ratified_chain_policies(self) -> None:
        assert [m.POLICY_ID for m in ds.SERVICE_CHAIN_POLICIES] == [
            "headless-defaults.v0",
            "run-typography-snap.v0",
            "image-region-snap.v0",
            "tab-stops-from-regions.v0",
            "table-geometry.v0",
            "story-kind-dedup.v0",
            "leader-fill.v1",
            "recovered-image-materialization.v0",
        ]
        assert all(m.POLICY_STATUS == "RATIFIED" for m in ds.SERVICE_CHAIN_POLICIES)

    def test_false_rather_than_raise_on_a_missing_attribute(self) -> None:
        assert ds.diagnostic_docx_capability(SimpleNamespace()) is False

    def test_true_under_the_loop_guard_v3_worker_override(self) -> None:
        """DEC-055: v3 wraps v2 unchanged, so a `--loop-guard v3` worker keeps
        the diagnostic Word output."""
        policy = SimpleNamespace(typed_decode=True, region_decode=True, loop_guard="v3")
        assert ds.diagnostic_docx_capability(_fake_runtime(decode_policy=policy)) is True


class TestLoopGuardDetail:
    def test_none_when_no_loop_guard_issue_and_no_chunking(self) -> None:
        assert ds._loop_guard_detail(()) is None

    def test_none_when_prefill_chunk_tokens_explicitly_none(self) -> None:
        assert ds._loop_guard_detail((), prefill_chunk_tokens=None) is None

    def test_prefill_chunk_tokens_alone_produces_a_detail(self) -> None:
        import json

        detail = ds._loop_guard_detail((), prefill_chunk_tokens=1024)
        assert json.loads(detail) == {"prefill_chunk_tokens": 1024}

    def test_loop_guard_issue_alone_is_unchanged_by_the_new_parameter(self) -> None:
        import json

        issues = ({"kind": "loop_guard_v1", "counters": {"ban_tab": 2}},)
        detail = ds._loop_guard_detail(issues)
        assert json.loads(detail) == {
            "loop_guard": "v1",
            "loop_guard_counters": {"ban_tab": 2},
        }

    def test_loop_guard_v3_issue_names_the_policy_and_arm(self) -> None:
        import json

        issues = (
            {
                "kind": "loop_guard_v3",
                "counters": {"unit_cap": 1},
                "policy_sha256": "sha256:v3",
                "arm": "r1",
                "caps_sha256": "sha256:caps",
                "interventions": [{"step": 18, "rule": "unit_cap"}],
            },
        )
        assert json.loads(ds._loop_guard_detail(issues)) == {
            "loop_guard": "v3",
            "loop_guard_counters": {"unit_cap": 1},
            "loop_guard_v3_policy_sha256": "sha256:v3",
            "loop_guard_v3_arm": "r1",
            "loop_guard_caps_sha256": "sha256:caps",
            "loop_guard_interventions": [{"step": 18, "rule": "unit_cap"}],
        }

    def test_loop_guard_issue_and_prefill_chunk_tokens_both_present(self) -> None:
        import json

        issues = ({"kind": "loop_guard_v1", "counters": {"ban_tab": 2}},)
        detail = ds._loop_guard_detail(issues, prefill_chunk_tokens=512)
        assert json.loads(detail) == {
            "loop_guard": "v1",
            "loop_guard_counters": {"ban_tab": 2},
            "prefill_chunk_tokens": 512,
        }


def _page(prediction_ir_hash: str = "sha256:ir") -> SimpleNamespace:
    return SimpleNamespace(prediction_ir_hash=prediction_ir_hash, node_states=())


def _row(
    *,
    page_outcome: str,
    envelope: object | None = None,
    policy_counts: dict | None = None,
    geometry_counts: dict | None = None,
    story_dedup_counts: dict | None = None,
    leader_fill_counts: dict | None = None,
    recovered_image_counts: dict | None = None,
    run_typography_snap_counts: dict | None = None,
    image_region_snap_counts: dict | None = None,
    tab_stops_counts: dict | None = None,
    leader_fill_v1_counts: dict | None = None,
    profile: str | None = "word-core-v0",
) -> SimpleNamespace:
    """A `ChainRow` stand-in with every policy the service applies switched
    on: `run.process_model_page` reports an applied policy as a dict (`{}`
    when it had nothing to edit), `None` only when it was off."""
    return SimpleNamespace(
        page_outcome=page_outcome,
        envelope=envelope,
        policy_counts=policy_counts if policy_counts is not None else {},
        geometry_counts=geometry_counts if geometry_counts is not None else {},
        story_dedup_counts=(
            story_dedup_counts if story_dedup_counts is not None else {}
        ),
        leader_fill_counts=leader_fill_counts if leader_fill_counts is not None else {},
        recovered_image_counts=(
            recovered_image_counts if recovered_image_counts is not None else {}
        ),
        run_typography_snap_counts=(
            run_typography_snap_counts if run_typography_snap_counts is not None else {}
        ),
        image_region_snap_counts=(
            image_region_snap_counts if image_region_snap_counts is not None else {}
        ),
        tab_stops_counts=tab_stops_counts if tab_stops_counts is not None else {},
        leader_fill_v1_counts=(
            leader_fill_v1_counts if leader_fill_v1_counts is not None else {}
        ),
        profile=profile,
    )


def _envelope(
    *, ast_status: str, renderer_status: str, docx_path: Path | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        axes=SimpleNamespace(ast_status=ast_status),
        renderer=SimpleNamespace(status=renderer_status, docx_path=docx_path),
    )


def _kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "raw": SimpleNamespace(issues=()),
        "prepared": SimpleNamespace(visual_features=None),
        "png_bytes": b"png",
        "runtime": _fake_runtime(),
        "job_id": "job_abc",
        "work_dir": tmp_path,
        "max_docx_bytes": 1_000_000,
        "expected_prediction_ir_hash": "sha256:ir",
    }
    values.update(overrides)
    return values


class TestBuildDiagnosticDocx:
    def test_success_reads_bytes_and_reports_ratified_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        docx_path = tmp_path / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="terminated",
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
                ),
                policy_counts={"default:run:/style/font_family": 1},
                geometry_counts={"table_geometry:tblw_from_grid": 1},
                recovered_image_counts={},
                profile="word-visual-v1",
            ),
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.produced is True
        assert result.docx_bytes == b"docx-bytes"
        assert result.reason_codes == ("HEADLESS_DEFAULTS_V0_APPLIED",)
        assert result.docx_chain == {
            "policy_id": "headless-defaults.v0",
            "policy_status": "RATIFIED",
            "page_outcome": "terminated",
            "policy_counts": {"default:run:/style/font_family": 1},
            "policies": [
                {
                    "policy_id": "headless-defaults.v0",
                    "policy_status": "RATIFIED",
                    "counts": {"default:run:/style/font_family": 1},
                },
                {
                    "policy_id": "run-typography-snap.v0",
                    "policy_status": "RATIFIED",
                    "counts": {},
                },
                {
                    "policy_id": "image-region-snap.v0",
                    "policy_status": "RATIFIED",
                    "counts": {},
                },
                {
                    "policy_id": "tab-stops-from-regions.v0",
                    "policy_status": "RATIFIED",
                    "counts": {},
                },
                {
                    "policy_id": "table-geometry.v0",
                    "policy_status": "RATIFIED",
                    "counts": {"table_geometry:tblw_from_grid": 1},
                },
                {
                    "policy_id": "story-kind-dedup.v0",
                    "policy_status": "RATIFIED",
                    "counts": {},
                },
                {
                    "policy_id": "leader-fill.v1",
                    "policy_status": "RATIFIED",
                    "counts": {},
                },
                {
                    "policy_id": "recovered-image-materialization.v0",
                    "policy_status": "RATIFIED",
                    "counts": {},
                },
            ],
            "profile": "word-visual-v1",
            "ast_status": "strict_valid",
            "renderer_status": "render_ready",
        }

    def test_seal_page_detail_carries_prefill_chunk_tokens_from_the_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """W0-11: a job decoded with chunked prefill must be distinguishable
        from one decoded without it -- `runtime.decode_policy.
        prefill_chunk_tokens` reaches `seal_page`'s `detail`."""
        import json

        seen: dict[str, object] = {}

        def _capture_seal_page(**kwargs):
            seen.update(kwargs)
            return _page()

        monkeypatch.setattr(ds, "seal_page", _capture_seal_page)
        docx_path = tmp_path / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="terminated",
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
                ),
            ),
        )
        runtime = _fake_runtime(
            decode_policy=SimpleNamespace(
                typed_decode=True, region_decode=True, loop_guard="v2", max_steps=4608,
                selection_policy_hash="sha256:selection", prefill_chunk_tokens=1024,
            )
        )
        ds.build_diagnostic_docx(**_kwargs(tmp_path, runtime=runtime))
        assert json.loads(seen["detail"])["prefill_chunk_tokens"] == 1024

    def test_chain_runs_with_every_service_policy_switched_on(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEC-044 Dodatok B/D, DEC-077: the service has ONE ratified chain -- all
        eight opt-in policies on, never per request; leader-fill.v1 in place of v0."""
        seen: list[dict[str, object]] = []

        def _capture(page, **kwargs):
            seen.append(kwargs)
            return _row(page_outcome="max_steps_exceeded")

        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(ds, "process_model_page", _capture)
        ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert len(seen) == 1
        assert seen[0]["headless_defaults"] is True
        assert seen[0]["run_typography_snap"] is True
        assert seen[0]["image_region_snap"] is True
        assert seen[0]["tab_stops_from_regions"] is True
        assert seen[0]["table_geometry"] is True
        assert seen[0]["story_dedup"] is True
        assert seen[0]["leader_fill_v1"] is True
        assert "leader_fill" not in seen[0]  # v0 is not requested alongside v1
        assert seen[0]["recovered_image_materialization"] is True

    @pytest.mark.parametrize(
        "row_overrides",
        [
            {"policy_counts": None},
            {"geometry_counts": None},
            {"story_dedup_counts": None},
            {"leader_fill_v1_counts": None},
            {"recovered_image_counts": None},
            {"run_typography_snap_counts": None},
            {"image_region_snap_counts": None},
            {"tab_stops_counts": None},
        ],
    )
    def test_requested_policy_that_did_not_run_is_a_chain_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row_overrides: dict
    ) -> None:
        """`None` on a `ChainRow` means "policy off" (`run.py`). The service
        requested it, so `None` is a broken chain -- never reported as "ran
        with nothing to edit"."""
        docx_path = tmp_path / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        row = SimpleNamespace(
            page_outcome="terminated",
            envelope=_envelope(
                ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
            ),
            policy_counts={},
            geometry_counts={},
            story_dedup_counts={},
            leader_fill_counts={},
            leader_fill_v1_counts={},
            recovered_image_counts={},
            run_typography_snap_counts={},
            image_region_snap_counts={},
            tab_stops_counts={},
            profile="word-core-v0",
        )
        for key, value in row_overrides.items():
            setattr(row, key, value)
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(ds, "process_model_page", lambda *_a, **_kw: row)
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.produced is False
        assert result.reason_codes == ("DOCX_CHAIN_ERROR",)
        assert result.docx_chain["error_class"] == "ValueError"

    def test_not_terminated_row_still_carries_policies_and_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="max_steps_exceeded",
                geometry_counts={"table_geometry:table_without_cells": 1},
            ),
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.reason_codes == ("DOCX_PAGE_NOT_TERMINATED",)
        assert "policy_counts" not in result.docx_chain  # empty headless counts stay omitted
        assert result.docx_chain["profile"] == "word-core-v0"
        assert [p["policy_id"] for p in result.docx_chain["policies"]] == [
            "headless-defaults.v0",
            "run-typography-snap.v0",
            "image-region-snap.v0",
            "tab-stops-from-regions.v0",
            "table-geometry.v0",
            "story-kind-dedup.v0",
            "leader-fill.v1",
            "recovered-image-materialization.v0",
        ]
        assert result.docx_chain["policies"][0]["counts"] == {}
        assert result.docx_chain["policies"][4]["counts"] == {
            "table_geometry:table_without_cells": 1
        }
        # the ERRATA-011 pre-assembly policies, story-dedup, leader-fill.v1 and
        # recovered-image found nothing to edit
        assert all(
            result.docx_chain["policies"][i]["counts"] == {} for i in (1, 2, 3, 5, 6, 7)
        )

    def _approximation_row(self, tmp_path: Path, codes: dict, *, write: bool = True):
        if write:
            (tmp_path / "job_abc.docx").write_bytes(b"docx-bytes")
        return _row(
            page_outcome="terminated",
            envelope=SimpleNamespace(
                axes=SimpleNamespace(ast_status="strict_valid"),
                renderer=SimpleNamespace(
                    status="approximation_required", docx_path=None,
                    renderer_gate_report={"issue_codes_by_severity": codes},
                ),
            ),
            policy_counts={"default:run:/style/font_family": 1},
        )

    def test_font_substitution_only_is_published_with_its_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Owner 2026-09-26 (DEC-044 Dodatok C): Calibri -> Carlito is a
        metric-compatible substitution; the diagnostic Word is published and
        the substitution is named, never hidden."""
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        row = self._approximation_row(tmp_path, {"warning": ["FONT_FALLBACK", "FONT_FALLBACK"]})
        monkeypatch.setattr(ds, "process_model_page", lambda *_a, **_kw: row)

        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))

        assert result.produced is True
        assert result.docx_bytes == b"docx-bytes"
        assert result.reason_codes == ("HEADLESS_DEFAULTS_V0_APPLIED", "DOCX_FONT_SUBSTITUTED")
        assert result.docx_chain["renderer_status"] == "approximation_required"
        assert result.docx_chain["renderer_approximations"] == ["FONT_FALLBACK"]

    @pytest.mark.parametrize(
        "codes",
        [
            {"warning": ["FONT_FALLBACK", "IMAGE_DOWNSAMPLED"]},
            {"warning": ["FONT_FALLBACK"], "error": ["TABLE_CLIPPED"]},
            {"warning": []},
            {},
        ],
    )
    def test_any_other_approximation_remains_withheld(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, codes: dict
    ) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        row = self._approximation_row(tmp_path, codes)
        monkeypatch.setattr(ds, "process_model_page", lambda *_a, **_kw: row)

        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))

        assert result.produced is False
        assert result.reason_codes == ("DOCX_RENDER_NOT_READY",)

    def test_font_substitution_without_the_written_file_is_withheld(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        row = self._approximation_row(tmp_path, {"warning": ["FONT_FALLBACK"]}, write=False)
        monkeypatch.setattr(ds, "process_model_page", lambda *_a, **_kw: row)

        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))

        assert result.produced is False
        assert result.reason_codes == ("DOCX_RENDER_NOT_READY",)

    def test_docx_read_error_is_a_chain_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        docx_path = tmp_path / "out.docx"
        docx_path.write_bytes(b"docx-bytes")
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="terminated",
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
                ),
            ),
        )
        monkeypatch.setattr(Path, "read_bytes", lambda _path: (_ for _ in ()).throw(OSError("I/O")))

        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))

        assert result.produced is False
        assert result.reason_codes == ("DOCX_CHAIN_ERROR",)
        assert result.docx_chain["error_class"] == "OSError"

    @pytest.mark.parametrize("page_outcome", ["PageStreamInvalid", "AssemblyInvariantViolation"])
    def test_page_stream_invalid_outcomes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, page_outcome: str
    ) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds, "process_model_page", lambda *_a, **_kw: _row(page_outcome=page_outcome)
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.produced is False
        assert result.docx_bytes is None
        assert result.reason_codes == ("DOCX_PAGE_STREAM_INVALID",)
        assert result.docx_chain["page_outcome"] == page_outcome

    def test_not_terminated_outcome(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds, "process_model_page", lambda *_a, **_kw: _row(page_outcome="max_steps_exceeded")
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.reason_codes == ("DOCX_PAGE_NOT_TERMINATED",)

    def test_ast_not_strict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="terminated",
                envelope=_envelope(ast_status="failed", renderer_status="not_run"),
            ),
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.reason_codes == ("DOCX_AST_NOT_STRICT",)
        assert result.docx_chain["ast_status"] == "failed"

    def test_render_not_ready(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="terminated",
                envelope=_envelope(ast_status="strict_valid", renderer_status="failed"),
            ),
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.reason_codes == ("DOCX_RENDER_NOT_READY",)

    def test_output_too_large(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        docx_path = tmp_path / "big.docx"
        docx_path.write_bytes(b"x" * 20)
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="terminated",
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
                ),
            ),
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path, max_docx_bytes=10))
        assert result.produced is False
        assert result.reason_codes == ("DOCX_OUTPUT_TOO_LARGE",)

    def test_docx_path_outside_work_dir_is_a_chain_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outside = tmp_path.parent / "outside.docx"
        outside.write_bytes(b"escaped")
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_page",
            lambda *_a, **_kw: _row(
                page_outcome="terminated",
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=outside
                ),
            ),
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.produced is False
        assert result.reason_codes == ("DOCX_CHAIN_ERROR",)
        outside.unlink()

    def test_runtime_error_is_caught_with_partial_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sealed_page = sealed.PageDecodeResult(
            document_bundle_id="job_abc",
            outcome="terminated",
            prediction_ir_hash="sha256:ir",
            node_states=(
                sealed.NodeStateProjection(
                    local_node_id="r1", slot_kind="opened_node", node_type="run",
                    parent_local_node_id=None, open_event_index=1,
                ),
                # A table with no cells: `table-geometry.v0` counts it, so the
                # P8 partial path must show the geometry policy's counts too.
                sealed.NodeStateProjection(
                    local_node_id="t1", slot_kind="opened_node", node_type="table",
                    parent_local_node_id=None, open_event_index=2,
                ),
            ),
        )
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: sealed_page)

        def _boom(*_a, **_kw):
            raise RuntimeError("builder exploded")

        monkeypatch.setattr(ds, "process_model_page", _boom)
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.produced is False
        assert result.reason_codes == ("DOCX_CHAIN_ERROR",)
        assert result.docx_chain["error_class"] == "RuntimeError"
        assert result.docx_chain["page_outcome"] == "error"
        assert result.docx_chain["policy_counts"]["default:run:/style/font_family"] == 1
        policies = {p["policy_id"]: p for p in result.docx_chain["policies"]}
        assert list(policies) == [
            "headless-defaults.v0",
            "run-typography-snap.v0",
            "image-region-snap.v0",
            "tab-stops-from-regions.v0",
            "table-geometry.v0",
            "story-kind-dedup.v0",
            "leader-fill.v1",
            "recovered-image-materialization.v0",
        ]
        assert all(p["policy_status"] == "RATIFIED" for p in policies.values())
        assert policies["headless-defaults.v0"]["counts"] == result.docx_chain["policy_counts"]
        assert policies["table-geometry.v0"]["counts"] == {
            "table_geometry:table_without_cells": 1
        }
        # Story-dedup and recovered-image materialization both run on the
        # assembled document, which the failed chain never produced: on,
        # nothing edited.
        assert policies["story-kind-dedup.v0"]["counts"] == {}
        assert policies["leader-fill.v1"]["counts"] == {}
        assert policies["image-region-snap.v0"]["counts"] == {}
        assert policies["recovered-image-materialization.v0"]["counts"] == {}
        assert "profile" not in result.docx_chain

    def test_exception_before_seal_page_is_caught_without_partial_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(**_kw):
            raise ValueError("checkpoint bindings drifted")

        monkeypatch.setattr(ds, "seal_page", _boom)
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.reason_codes == ("DOCX_CHAIN_ERROR",)
        assert result.docx_chain["error_class"] == "ValueError"
        assert "policy_counts" not in result.docx_chain
        assert "policies" not in result.docx_chain  # no page: no policy ever ran
        assert "profile" not in result.docx_chain

    def test_prediction_ir_hash_mismatch_is_a_chain_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page("sha256:different"))
        called = []
        monkeypatch.setattr(
            ds, "process_model_page", lambda *_a, **_kw: called.append(True)
        )
        result = ds.build_diagnostic_docx(**_kwargs(tmp_path))
        assert result.reason_codes == ("DOCX_CHAIN_ERROR",)
        assert called == []  # the chain never ran once the hash disagreed


def _multi_row(
    *,
    page_outcome: str = "terminated",
    envelope: object | None = None,
    policy_counts: dict | None = None,
    geometry_counts: dict | None = None,
    story_dedup_counts: dict | None = None,
    profile: str | None = "word-core-v0",
    assembler_summary: dict | None = None,
    page_outcomes: tuple[str, ...] = ("terminated", "terminated"),
) -> SimpleNamespace:
    return SimpleNamespace(
        page_outcome=page_outcome,
        envelope=envelope,
        policy_counts=policy_counts if policy_counts is not None else {},
        geometry_counts=geometry_counts if geometry_counts is not None else {},
        story_dedup_counts=story_dedup_counts if story_dedup_counts is not None else {},
        leader_fill_counts={},
        leader_fill_v1_counts={},
        run_typography_snap_counts={},
        tab_stops_counts={},
        recovered_image_counts={},
        image_region_snap_counts={},
        profile=profile,
        assembler_summary=assembler_summary,
        page_outcomes=page_outcomes,
    )


def _attempt(index: int, *, seconds: float = 1.0, accepted_tokens: tuple = (1, 2, 3)) -> object:
    return ds.PageAttempt(
        index=index, seconds=seconds,
        raw=SimpleNamespace(issues=(), accepted_tokens=accepted_tokens),
        prepared=SimpleNamespace(visual_features=None),
    )


class TestBuildDiagnosticDocxMulti:
    def test_requires_at_least_one_attempt(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one page"):
            ds.build_diagnostic_docx_multi(
                [], runtime=_fake_runtime(), job_id="job_m", work_dir=tmp_path,
                max_docx_bytes=1_000_000,
            )

    def test_success_reads_bytes_and_reports_two_succeeded_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        docx_path = tmp_path / "out.docx"
        docx_path.write_bytes(b"multi-docx-bytes")
        seen_pages = []
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_pages",
            lambda pages, **_kw: (seen_pages.extend(pages), _multi_row(
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
                ),
                assembler_summary={"joins_accepted": 1, "joins_refused": 0, "refusal_codes": {}},
            ))[1],
        )

        result = ds.build_diagnostic_docx_multi(
            [_attempt(0, accepted_tokens=(1, 2)), _attempt(1, accepted_tokens=(1, 2, 3, 4))],
            runtime=_fake_runtime(), job_id="job_m", work_dir=tmp_path, max_docx_bytes=1_000_000,
        )

        assert result.produced is True
        assert result.docx_bytes == b"multi-docx-bytes"
        assert result.docx_chain["assembler"] == {
            "joins_accepted": 1, "joins_refused": 0, "refusal_codes": {},
        }
        assert len(seen_pages) == 2
        assert [p.state for p in result.pages] == ["succeeded", "succeeded"]
        assert [p.decode_steps for p in result.pages] == [2, 4]
        assert [p.index for p in result.pages] == [0, 1]

    def test_page_that_never_terminated_becomes_a_failed_placeholder_not_a_drop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        docx_path = tmp_path / "out.docx"
        docx_path.write_bytes(b"bytes")
        seen_pages = []
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds,
            "process_model_pages",
            lambda pages, **_kw: (seen_pages.extend(pages), _multi_row(
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
                ),
            ))[1],
        )
        failed = ds.PageAttempt(index=1, seconds=0.5, error_code="MODEL_DECODE_MAX_STEPS")

        result = ds.build_diagnostic_docx_multi(
            [_attempt(0), failed], runtime=_fake_runtime(), job_id="job_m", work_dir=tmp_path,
            max_docx_bytes=1_000_000,
        )

        assert [p.state for p in result.pages] == ["succeeded", "failed"]
        assert result.pages[1].error_code == "MODEL_DECODE_MAX_STEPS"
        assert seen_pages[1].outcome == "MODEL_DECODE_MAX_STEPS"
        assert result.produced is True

    def test_seal_page_exception_fails_only_that_page(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        docx_path = tmp_path / "out.docx"
        docx_path.write_bytes(b"bytes")

        def _seal_page(*, document_bundle_id: str, **_kw):
            if document_bundle_id.endswith("#1"):
                raise ValueError("boom")
            return _page()

        monkeypatch.setattr(ds, "seal_page", _seal_page)
        monkeypatch.setattr(
            ds,
            "process_model_pages",
            lambda pages, **_kw: _multi_row(
                envelope=_envelope(
                    ast_status="strict_valid", renderer_status="render_ready", docx_path=docx_path
                ),
            ),
        )

        result = ds.build_diagnostic_docx_multi(
            [_attempt(0), _attempt(1)], runtime=_fake_runtime(), job_id="job_m",
            work_dir=tmp_path, max_docx_bytes=1_000_000,
        )

        assert [p.state for p in result.pages] == ["succeeded", "failed"]
        assert result.pages[1].error_code == "DOCX_CHAIN_ERROR"
        assert result.produced is True

    def test_row_not_terminated_reports_docx_page_not_terminated_but_keeps_pages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds, "process_model_pages", lambda pages, **_kw: _multi_row(page_outcome="error")
        )

        result = ds.build_diagnostic_docx_multi(
            [_attempt(0), _attempt(1)], runtime=_fake_runtime(), job_id="job_m",
            work_dir=tmp_path, max_docx_bytes=1_000_000,
        )

        assert result.produced is False
        assert result.reason_codes == ("DOCX_PAGE_NOT_TERMINATED",)
        assert len(result.pages) == 2

    def test_only_applies_the_multi_page_policies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ds, "seal_page", lambda **_kw: _page())
        monkeypatch.setattr(
            ds, "process_model_pages", lambda pages, **_kw: _multi_row(page_outcome="error")
        )

        result = ds.build_diagnostic_docx_multi(
            [_attempt(0)], runtime=_fake_runtime(), job_id="job_m",
            work_dir=tmp_path, max_docx_bytes=1_000_000,
        )

        assert [p["policy_id"] for p in result.docx_chain["policies"]] == [
            "headless-defaults.v0", "run-typography-snap.v0", "image-region-snap.v0",
            "tab-stops-from-regions.v0", "table-geometry.v0", "story-kind-dedup.v0",
            "leader-fill.v1", "recovered-image-materialization.v0",
        ]

    def test_ast_failure_reports_specific_codes_and_pointers(self) -> None:
        envelope = _envelope(ast_status="failed", renderer_status="not_run")
        envelope.ast = SimpleNamespace(failure_report={"issues": [
            {"code": "RESOURCE_SLOT_UNRESOLVED", "node_property_pointer": "/resource_ref"},
            {"code": "RESOURCE_SLOT_UNRESOLVED", "node_property_pointer": "/resource_ref"},
            {"code": "CANONICALIZATION_REFUSED", "ast_pointer": "/sections/0"},
        ]})
        chain = ds._docx_chain_from_row(_multi_row(envelope=envelope),
                                         policies=ds.MULTI_PAGE_CHAIN_POLICIES)
        assert chain["failure_codes"] == [
            "CANONICALIZATION_REFUSED|/sections/0", "RESOURCE_SLOT_UNRESOLVED|/resource_ref",
        ]

    def test_renderer_failure_reports_underlying_gate_codes(self) -> None:
        envelope = _envelope(ast_status="strict_valid", renderer_status="failed")
        envelope.renderer.failure_report = {"issues": [{
            "code": "RENDERER_GATE_FAILED", "node_property_pointer": None,
            "details": {"renderer_codes": ["TABLE_WIDTH_OVERFLOW", "LIMIT_EXCEEDED_NODES"]},
        }]}
        chain = ds._docx_chain_from_row(_multi_row(envelope=envelope),
                                        policies=ds.MULTI_PAGE_CHAIN_POLICIES)
        assert "TABLE_WIDTH_OVERFLOW|/" in chain["failure_codes"]
        assert "LIMIT_EXCEEDED_NODES|/" in chain["failure_codes"]
