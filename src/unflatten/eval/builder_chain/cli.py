"""`builder-chain-report` CLI (design §2.6, plan P7/U26).

`--corpus` runs the gold arm over the pinned `gold.GOLD_FIXTURES`.
`--sealed <artefact.json> --checkpoint-id <id>` runs the model arm,
guarded by R12 (DEC-038: S2 is retired) -- `--checkpoint-id` is mandatory
whenever `--sealed` is given, and both the mismatch and the retired-id
refusals are real exceptions, never swallowed into an exit code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unflatten.eval.builder_chain import report, run
from unflatten.eval.builder_chain import sealed_adapter as sa
from unflatten.eval.builder_chain import table_geometry as tg
from unflatten.eval.builder_chain import visual_metrics as vm
from unflatten.eval.builder_chain.raster_source import PinnedRasterSource
from unflatten.eval.s2_gate import sealed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="builder-chain-report")
    parser.add_argument(
        "--corpus", action="store_true", help="run the gold arm over the pinned corpus fixtures"
    )
    parser.add_argument("--sealed", type=Path, default=None, help="a sealed decode artefact JSON")
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument(
        "--page-raster-source",
        type=Path,
        nargs=2,
        metavar=("RELEASE_DIR", "STORE_DIR"),
        default=None,
        help="release and content-addressed store that pin model-arm page rasters",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--render-dir", type=Path, default=None,
        help="compile each strict-valid AST to DOCX here, so the renderer axis is measured "
        "(without it every row is not_run|not_requested)",
    )
    parser.add_argument(
        "--headless-defaults", action="store_true",
        help="apply the opt-in `headless-defaults.v0` policy (RATIFIED, RFC004-ERRATA-003) to "
        "each page's typed predictions before the strict build; recorded and counted in the "
        "report",
    )
    parser.add_argument(
        "--table-geometry", action="store_true",
        help="apply the opt-in `table-geometry.v0` policy (RATIFIED, RFC004-ERRATA-004 E2) "
        "right after headless-defaults (which it requires): one coherent table geometry "
        "derived from the grid; recorded and counted in the report",
    )
    parser.add_argument(
        "--table-geometry-column-source", choices=tg.COLUMN_SOURCES,
        default=tg.DEFAULT_COLUMN_SOURCE,
        help="which head supplies the column widths for table-geometry.v0 (part of the "
        "hashed policy record)",
    )
    parser.add_argument(
        "--story-dedup", action="store_true",
        help="apply the opt-in `story-kind-dedup.v0` policy (RATIFIED, RFC004-ERRATA-004 E5) "
        "after headless-defaults (which it requires): one header/footer per kind and section, "
        "duplicates re-kinded or dropped; recorded and counted",
    )
    parser.add_argument(
        "--language-detect", action="store_true",
        help="apply the opt-in `language-detect.v0` policy (PROPOSED, RFC004-ERRATA-007) "
        "after headless-defaults (which it requires): deterministically fill a run's null "
        "/language from its own, its paragraph's or the page's text; recorded and counted",
    )
    parser.add_argument(
        "--runaway-row-repair", action="store_true",
        help="apply the opt-in `runaway-row-repair.v0` policy (PROPOSED, RFC004-ERRATA-008) "
        "after headless-defaults (which it requires) and before table-geometry: a runaway "
        "row's trailing empty cells are dropped back to the table's modal count; "
        "recorded and counted",
    )
    parser.add_argument(
        "--page-geometry-from-raster", action="store_true",
        help="apply the opt-in `page-geometry-from-raster.v0` policy (PROPOSED, "
        "RFC004-ERRATA-008): reconcile the section's page size/orientation against the "
        "300 DPI raster before table-geometry; recorded and counted",
    )
    parser.add_argument(
        "--recovered-image-materialization", action="store_true",
        help="apply the opt-in `recovered-image-materialization.v0` policy (RATIFIED, "
        "RFC004-ERRATA-004 E1) after resource recovery: a render-recovered image is shown "
        "uncropped, unrotated, at its pixel size on the 300 DPI raster; recorded and counted",
    )
    parser.add_argument(
        "--page-number-fields", action="store_true",
        help="apply the opt-in `page-number-fields.v0` policy (PROPOSED, RFC004-ERRATA-009) "
        "after story-dedup: a header/footer paragraph matching a closed page-number pattern "
        "gets a real PAGE/NUMPAGES field; recorded and counted",
    )
    parser.add_argument(
        "--leader-fill", action="store_true",
        help="apply the `leader-fill.v0` policy (RATIFIED, RFC004-ERRATA-010, DEC-059) "
        "after autolink: a DEC-036 canonical fill leader in a body paragraph becomes a tab "
        "to a stop with the matching leader, filling the line; recorded and counted",
    )
    parser.add_argument(
        "--image-region-snap", action="store_true",
        help="apply the opt-in `image-region-snap.v0` policy (PROPOSED, RFC004-ERRATA-011 "
        "draft) before resource recovery: a predicted image region is tightened onto the "
        "ink the 300 DPI raster shows there; recorded and counted",
    )
    parser.add_argument(
        "--run-typography-snap", action="store_true",
        help="apply the opt-in `run-typography-snap.v0` policy (PROPOSED, RFC004-ERRATA-011 "
        "E2) after headless-defaults: run font size onto the 0.5 pt grid and one size per "
        "paragraph, sub-point character spacing to zero; recorded and counted",
    )
    parser.add_argument(
        "--tab-stops-from-regions", action="store_true",
        help="apply the opt-in `tab-stops-from-regions.v0` policy (PROPOSED, RFC004-ERRATA-011 "
        "E3) before table-geometry: a TAB paragraph's stops are derived from the regions of "
        "the runs that follow its TAB groups; recorded and counted",
    )
    parser.add_argument(
        "--leader-fill-v1", action="store_true",
        help="apply the opt-in `leader-fill.v1` policy (PROPOSED, RFC004-ERRATA-011 E4) in "
        "place of --leader-fill (never both): leaders after an existing tab get that tab's "
        "stop, >= 3 repeated leader groups collapse to two; recorded and counted",
    )
    parser.add_argument(
        "--autolink", action="store_true",
        help="apply the opt-in `autolink.v0` policy (PROPOSED, RFC004-ERRATA-009): a run's "
        "URL/e-mail span becomes a real `hyperlink` node; recorded and counted",
    )
    parser.add_argument(
        "--visual-metrics", action="store_true",
        help="attach additive, UNHASHED visual evidence (visual_metrics.annotate_rows/"
        "aggregate) as a per-row `visual` key and a top-level `visual` aggregate; requires "
        "--render-dir; never enters report_hash's core (docs/plans/"
        "spec-visual-chain-metrics-design.md)",
    )
    return parser


def _run_gold(args: argparse.Namespace) -> dict:
    rows = run.run_gold_arm(render_dir=args.render_dir)
    return report.build_report(rows, arm="gold")


def _run_model(args: argparse.Namespace) -> dict:
    if args.checkpoint_id is None:
        raise SystemExit("--checkpoint-id is required with --sealed (DEC-038 R12)")
    if args.table_geometry and not args.headless_defaults:
        raise SystemExit(
            "--table-geometry requires --headless-defaults (it consumes the derived grid)"
        )
    if args.story_dedup and not args.headless_defaults:
        raise SystemExit("--story-dedup requires --headless-defaults (one chain order)")
    if args.language_detect and not args.headless_defaults:
        raise SystemExit("--language-detect requires --headless-defaults (one chain order)")
    if args.runaway_row_repair and not args.headless_defaults:
        raise SystemExit("--runaway-row-repair requires --headless-defaults (one chain order)")
    artefact = sealed.artefact_from_json(args.sealed.read_text())
    # R12: refuse before touching the vocabulary/tokenizer at all.
    sa.refuse_unless_checkpoint_ok(artefact, args.checkpoint_id)
    if args.page_raster_source is None:
        raise SystemExit("--page-raster-source RELEASE_DIR STORE_DIR is required with --sealed")
    vocabulary, tokenizer = sa.build_vocabulary_and_tokenizer(artefact)
    try:
        decoded_release_root_hash = artefact.eval_cache_identity["dataset_release_root_hash"]
    except KeyError as exc:
        raise SystemExit(
            "sealed artefact has no decoded input dataset_release_root_hash"
        ) from exc
    raster_source = PinnedRasterSource(
        *args.page_raster_source,
        expected_release_root_hash=decoded_release_root_hash,
    )
    rows = run.run_model_arm(
        artefact,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        raster_source=raster_source,
        render_dir=args.render_dir,
        headless_defaults=args.headless_defaults,
        table_geometry=args.table_geometry,
        table_geometry_column_source=args.table_geometry_column_source,
        recovered_image_materialization=args.recovered_image_materialization,
        story_dedup=args.story_dedup,
        language_detect=args.language_detect,
        page_number_fields=args.page_number_fields,
        autolink=args.autolink,
        page_geometry_from_raster=args.page_geometry_from_raster,
        runaway_row_repair=args.runaway_row_repair,
        leader_fill=args.leader_fill,
        image_region_snap=args.image_region_snap,
        run_typography_snap=args.run_typography_snap,
        tab_stops_from_regions=args.tab_stops_from_regions,
        leader_fill_v1=args.leader_fill_v1,
    )
    return report.build_report(
        rows,
        arm="model",
        artefact_sha256=_sha256(args.sealed),
        eval_cache_identity=artefact.eval_cache_identity,
        checkpoint_id=args.checkpoint_id,
        role=artefact.role,
        headless_defaults=args.headless_defaults,
        table_geometry=args.table_geometry,
        table_geometry_column_source=args.table_geometry_column_source,
        page_geometry_from_raster=args.page_geometry_from_raster,
        runaway_row_repair=args.runaway_row_repair,
        leader_fill=args.leader_fill,
        image_region_snap=args.image_region_snap,
        run_typography_snap=args.run_typography_snap,
        tab_stops_from_regions=args.tab_stops_from_regions,
        leader_fill_v1=args.leader_fill_v1,
        recovered_image_materialization=args.recovered_image_materialization,
        story_dedup=args.story_dedup,
        language_detect=args.language_detect,
        page_number_fields=args.page_number_fields,
        autolink=args.autolink,
    )


def _attach_visual_metrics(doc: dict, args: argparse.Namespace) -> None:
    """Additive, UNHASHED evidence per docs/plans/spec-visual-chain-metrics-design.md
    ("Wiring for the chain report"): a per-row `visual` key and a top-level `visual`
    aggregate. Never touches `core`/`report_hash`."""
    if args.render_dir is None:
        raise SystemExit("--visual-metrics requires --render-dir (its DOCX files)")
    sealed_json = json.loads(args.sealed.read_text()) if args.sealed is not None else None
    records = vm.annotate_rows(doc["rows"], args.render_dir, sealed=sealed_json)
    by_bundle = {record["document_bundle_id"]: record for record in records}
    for row in doc["rows"]:
        bundle = row.get("document_bundle_id") or row.get("item_id")
        row["visual"] = by_bundle.get(bundle)
    doc["visual"] = vm.aggregate(records)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.render_dir is not None:
        args.render_dir.mkdir(parents=True, exist_ok=True)
    if args.sealed is not None:
        doc = _run_model(args)
    elif args.corpus:
        doc = _run_gold(args)
    else:
        raise SystemExit("one of --corpus or --sealed is required")

    if args.visual_metrics:
        _attach_visual_metrics(doc, args)

    text = json.dumps(doc, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.write_text(text)
    else:
        print(text)
    return 0


__all__ = ["build_arg_parser", "main"]
