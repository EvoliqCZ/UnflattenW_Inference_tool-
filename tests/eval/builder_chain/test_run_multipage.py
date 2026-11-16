from __future__ import annotations

from pathlib import Path

import pytest

from unflatten.eval.builder_chain import gold, run
from unflatten.eval.builder_chain import sealed_adapter as sa
from unflatten.eval.s2_gate import sealed


def _terminated_page(document_bundle_id: str) -> sealed.PageDecodeResult:
    return sealed.PageDecodeResult(
        document_bundle_id=document_bundle_id, outcome="terminated", accepted_tokens=(),
        node_states=(sealed.NodeStateProjection("n0", "opened_node", "section", None, 0),),
    )


def test_process_model_pages_requires_at_least_one_page() -> None:
    with pytest.raises(ValueError, match="at least one page"):
        run.process_model_pages([], vocabulary=None, tokenizer=None, document_id="doc")


def test_process_model_pages_never_drops_a_failed_page_and_merges_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """3 pages: terminated, a decode that never terminated (max_steps_exceeded),
    terminated again -- the merged document must still build and render, and
    every INPUT page's own outcome must survive on the row, in order."""
    canonical = gold.load("valid/minimal_paragraph.json")

    def fake(page, **kw):
        return gold.gold_page(canonical, ordinal=kw["page_input"].page_ordinal)

    monkeypatch.setattr(sa, "page_hypothesis", fake)
    pages = [
        _terminated_page("bundle#0"),
        sealed.PageDecodeResult(document_bundle_id="bundle#1", outcome="max_steps_exceeded"),
        _terminated_page("bundle#2"),
    ]
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    row = run.process_model_pages(
        pages, vocabulary=None, tokenizer=None, document_id="job_multi", render_dir=render_dir,
    )

    assert row.page_outcome == "terminated"
    assert row.page_outcomes == ("terminated", "max_steps_exceeded", "terminated")
    assert row.envelope is not None
    assert row.envelope.axes.ast_status == "strict_valid"
    assert (render_dir / "job_multi.docx").is_file()
    assert row.assembler_summary is not None
    assert row.assembler_summary["joins_accepted"] + row.assembler_summary["joins_refused"] > 0


def test_process_model_pages_page_stream_invalid_becomes_a_placeholder_not_a_drop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A page that fails to PARSE (real code path, no mocking): an unknown
    typed-prediction node id makes `sa.page_hypothesis` raise
    `PageStreamInvalid`; the row still merges every page, with that one
    outcome named after the exception, never aborting the whole document."""
    canonical = gold.load("valid/minimal_paragraph.json")
    original = sa.page_hypothesis

    def fake(page, **kw):
        if page.document_bundle_id == "bundle#0":
            return gold.gold_page(canonical, ordinal=kw["page_input"].page_ordinal)
        return original(page, **kw)

    monkeypatch.setattr(run.sa, "page_hypothesis", fake)
    good = _terminated_page("bundle#0")
    bad = sealed.PageDecodeResult(
        document_bundle_id="bundle#1", outcome="terminated", accepted_tokens=(),
        node_states=(sealed.NodeStateProjection("n1", "opened_node", "run", None, 1),),
        typed_predictions=(
            {"local_node_id": "n999", "node_type": "run", "pointer": "/style/bold",
             "head_id": "h", "value": True},
        ),
    )
    render_dir = tmp_path / "render"
    render_dir.mkdir()

    row = run.process_model_pages(
        [good, bad], vocabulary=None, tokenizer=None, document_id="job_bad",
        render_dir=render_dir,
    )

    assert row.page_outcomes[0] == "terminated"
    assert row.page_outcomes[1] == "PageStreamInvalid"
    assert row.page_outcome == "terminated"
    assert row.envelope is not None
