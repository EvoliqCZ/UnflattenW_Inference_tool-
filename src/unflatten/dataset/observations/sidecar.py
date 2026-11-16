"""Observation Sidecar builder + validator (RFC001 §37, RFC003 §24–§25).

The sidecar binds ONE Canonical AST to ONE rendered view's geometry —
it is per-RenderView under the frozen-pixels rule (RFC003 §22.2):
cross-run PDF bytes from the same DOCX may differ, so the sidecar binds
to the *stored* PDF artifact, never to "whatever a re-render would
produce". Same PDF bytes + same renderer meta ⇒ byte-identical sidecar
(all geometry is exact integer arithmetic over pdftotext's decimal
strings).

``mapping_confidence`` is integer ppm; ``1_000_000`` is emitted only
for deterministic exact matches (RFC003) — which is the only kind of
match ``pdf_text_alignment`` produces; anything weaker is a typed
issue, not a lower-confidence box.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

import unflatten.dataset.observations.frames as frames
import unflatten.dataset.observations.images as images
import unflatten.dataset.observations.stories as stories
from unflatten.ast.hashing import ast_hash
from unflatten.ast.issues import Issue
from unflatten.dataset.observations.aligner import resolve_aligner_policy
from unflatten.dataset.observations.pdf_text import extract_pdf_words, pt_text_to_px
from unflatten.dataset.observations.reading_order import pick_reading_order

SIDECAR_SCHEMA_VERSION = "0.4.0"
#: written only by a build under an aligner policy (`aligner_policy`): adds
#: `field_observation_v1` (a `field` node observed by its computed result)
#: and the top-level `aligner_policy` id. Without a policy the builder keeps
#: writing 0.4.0, byte for byte.
POLICY_SIDECAR_SCHEMA_VERSION = "0.5.0"
MAPPING_METHOD = "pdf_text_alignment"
#: per-observation method of a `text_frame` node's own box — derived as
#: the union of its content's observations (`frames.frame_observations`),
#: hence RFC003 §24.4 `geometry_inferred`, not a text or image match
FRAME_MAPPING_METHOD = "geometry_inferred"
EXACT_MATCH_CONFIDENCE_PPM = 1_000_000

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "observation_sidecar.schema.json"

_RENDERER_META_KEYS = ("name", "version", "dpi", "font_pack_hash")


class ObservationSidecarInvalid(Exception):
    """A built sidecar failed :func:`validate_observation_sidecar`.

    Deliberately NOT a :class:`ValueError` subclass: callers such as
    ``scripts/aligner_corpus_gate.py`` catch ``ValueError`` to mean "this
    document could not be re-derived, skip it" — an invalid sidecar is a
    named defect, not a skip.
    """

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__(f"observation sidecar failed validation: {[i.code for i in issues]}")


@dataclass
class ObservationSidecarBuild:
    """Builder result: the sidecar dict (JCS-serializable), alignment
    issues, and text-node coverage counts (paragraphs/runs with text).

    ``story_placements_by_page`` and ``stories_glyphless`` /
    ``stories_glyphless_share`` are DEC-033 §3 / R5 build-report
    counters, not part of the sidecar schema — callers (m2b's
    ``observation.coverage``) read them off the build result the same
    way they already read ``matched_text_nodes``. The ``frame_*``
    counters are the DEC-033 §7 equivalent, from
    :mod:`unflatten.dataset.observations.frames`.
    """

    sidecar: dict[str, Any]
    issues: list[Issue]
    matched_text_nodes: int
    total_text_nodes: int
    #: distinct header/footer stories placed on each page, indexed by
    #: page_index
    story_placements_by_page: list[int]
    #: header/footer stories with no matchable glyphs (image-only
    #: header, empty footer)
    stories_glyphless: int
    #: ``stories_glyphless`` as a fraction of this document's pages
    #: (0.0 when the document has no pages)
    stories_glyphless_share: float
    #: ``sum(story_placements_by_page)`` -- the store-gate axis 10
    #: numerator (design §2, `spec-page-standalone-targets`)
    story_placements: int = 0
    #: pages whose section carries a header/footer, and how many of
    #: those pages actually placed one (`stories.story_page_coverage`)
    story_pages_covered: int = 0
    story_pages_total: int = 0
    #: pages excluded from `story_pages_total` because their only
    #: applicable story is glyphless, or because no story applies to
    #: them at all (`stories.story_page_coverage`) -- informational,
    #: no threshold (R5-style)
    story_pages_glyphless_only: int = 0
    story_pages_not_applicable: int = 0
    #: ``OBSERVATION_STORY_UNPLACED`` issue count
    stories_unplaced: int = 0
    #: text_frame nodes in the document, and how many were placed on a
    #: page (DEC-033 §7)
    frames_total: int = 0
    frames_placed: int = 0
    #: glyph-bearing frame runs, and how many of them landed inside a
    #: placed frame
    frame_runs_total: int = 0
    frame_runs_observed: int = 0
    #: matched characters across observed frame runs
    frame_text_scalars: int = 0
    #: text_frame nodes that received their own (derived) observation —
    #: at least `frames_placed`, plus image-only frames whose image the
    #: image pass observed
    frames_observed: int = 0
    image_nodes_total: int = 0
    image_nodes_observed: int = 0
    image_nodes_observed_by_size: int = 0
    image_nodes_observed_by_one_sided_corridor: int = 0
    image_nodes_observed_by_story_pages: int = 0
    image_nodes_transformed: int = 0
    image_nodes_ambiguous: int = 0
    image_nodes_unplaced: int = 0
    image_placements_unexplained: int = 0
    image_extraction_failed: int = 0
    residual_claimed: int = 0
    residual_refused_ambiguous: int = 0
    residual_refused_contested: int = 0
    residual_not_found: int = 0
    residual_rounds: int = 0
    residual_invariant_skips: int = 0
    note_marker_matches: int = 0
    reading_order_switched_pages: list[int] = field(default_factory=list)

    def coverage(self) -> dict[str, int]:
        """Stable build-report projection shared by m2b and the factory."""
        return {
            "matched_text_nodes": self.matched_text_nodes,
            "total_text_nodes": self.total_text_nodes,
            "frames_total": self.frames_total,
            "frames_placed": self.frames_placed,
            "frames_observed": self.frames_observed,
            "frame_runs_observed": self.frame_runs_observed,
            "frame_runs_unobserved": self.frame_runs_total - self.frame_runs_observed,
            "story_placements": self.story_placements,
            "story_pages_covered": self.story_pages_covered,
            "story_pages_total": self.story_pages_total,
            "story_pages_glyphless_only": self.story_pages_glyphless_only,
            "story_pages_not_applicable": self.story_pages_not_applicable,
            "stories_glyphless": self.stories_glyphless,
            "stories_unplaced": self.stories_unplaced,
            "residual_claimed": self.residual_claimed,
            "residual_refused_ambiguous": self.residual_refused_ambiguous,
            "residual_refused_contested": self.residual_refused_contested,
            "residual_not_found": self.residual_not_found,
            "residual_rounds": self.residual_rounds,
            "residual_invariant_skips": self.residual_invariant_skips,
            "note_marker_matches": self.note_marker_matches,
            "image_nodes_total": self.image_nodes_total,
            "image_nodes_observed": self.image_nodes_observed,
            "image_nodes_observed_by_size": self.image_nodes_observed_by_size,
            "image_nodes_observed_by_one_sided_corridor": (
                self.image_nodes_observed_by_one_sided_corridor
            ),
            "image_nodes_observed_by_story_pages": self.image_nodes_observed_by_story_pages,
            "image_nodes_transformed": self.image_nodes_transformed,
            "image_nodes_ambiguous": self.image_nodes_ambiguous,
            "image_nodes_unplaced": self.image_nodes_unplaced,
            "image_placements_unexplained": self.image_placements_unexplained,
            "image_extraction_failed": self.image_extraction_failed,
        }

    def issue_codes(self) -> list[str]:
        """Sorted, de-duplicated codes of every alignment issue — the
        projection m2b (and the synthetic factory) store verbatim in
        ``observation.issue_codes`` (moved here from m2b so a change to
        it moves ``derivation_hash``: sidecar.py is in
        ``_DERIVATION_MODULES``, m2b.py is not)."""
        return sorted({i.code for i in self.issues})

    def issue_details(self) -> list[dict[str, Any]]:
        """``{code, node_id, message}`` per alignment issue, in the order
        raised — the projection m2b stores verbatim in
        ``observation.issue_details``."""
        return [{"code": i.code, "node_id": i.node_id, "message": i.message} for i in self.issues]


def build_observation_sidecar(
    ast: dict[str, Any],
    pdf_bytes: bytes,
    renderer_meta: dict[str, Any],
    *,
    aligner_policy: str | None = None,
) -> ObservationSidecarBuild:
    """Build the Observation Sidecar for ``ast`` rendered as ``pdf_bytes``.

    ``renderer_meta`` must carry exactly ``name``/``version`` (strings),
    ``dpi`` (positive int) and ``font_pack_hash`` — recorded verbatim in
    the sidecar. Alignment failures become typed issues (mismatched
    nodes simply have no observations); pdftotext failures raise
    :class:`~unflatten.dataset.observations.pdf_text.PdfTextExtractionError`.

    ``aligner_policy`` (an `aligner.ALIGNER_POLICIES` id, e.g.
    ``"v16-align.v1"``) opts into the v16 alignment rules; the sidecar then
    records the id and is written as schema 0.5.0. ``None`` (default) is
    today's sidecar, byte for byte.
    """
    policy = resolve_aligner_policy(aligner_policy)
    missing = [k for k in _RENDERER_META_KEYS if k not in renderer_meta]
    unexpected = [k for k in renderer_meta if k not in _RENDERER_META_KEYS]
    if missing or unexpected:
        raise ValueError(
            f"renderer_meta must have exactly {_RENDERER_META_KEYS}; "
            f"missing={missing} unexpected={unexpected}"
        )
    dpi = renderer_meta["dpi"]
    if not isinstance(dpi, int) or isinstance(dpi, bool) or dpi < 1:
        raise ValueError(f"renderer_meta['dpi'] must be a positive integer, got {dpi!r}")

    pdf_pages = extract_pdf_words(pdf_bytes, order="layout")
    raw_pages = extract_pdf_words(pdf_bytes, order="raw")
    pick_kwargs = {} if policy is None else {"policy": policy}
    pick = pick_reading_order(ast, pdf_pages, raw_pages, **pick_kwargs)
    alignment = pick.alignment

    image_result = images.ImageObservation([], [], {})
    extraction: dict[str, str] | None = None
    image_node_count = images.image_node_count(ast)
    if image_node_count:
        try:
            image_pages = images.extract_pdf_images(pdf_bytes)
        except images.PdfImageExtractionError as exc:
            image_result.issues.append(
                Issue(
                    code="PDF_IMAGE_EXTRACTION_FAILED",
                    severity="warning",
                    node_id=None,
                    path="/",
                    message=str(exc),
                )
            )
        else:
            image_result = images.observe_images(
                ast,
                pdf_pages,
                image_pages,
                alignment.fragments,
                images.story_text_pages(ast, alignment.fragments),
            )
            extraction = {
                "pdftotext": images.poppler_version("pdftotext"),
                "pdftohtml": images.poppler_version("pdftohtml"),
            }

    # DEC-033 §3: a header/footer node is observed on every page it
    # renders on, so its observations are page-scoped chains — computed
    # from the AST alone (`page_scoped_node_ids`), not from how many
    # pages this alignment actually placed it on.
    page_scoped_ids = (
        stories.page_scoped_node_ids(ast, story_frames=True)
        if policy is not None and policy.story_frames
        else stories.page_scoped_node_ids(ast)
    )
    repeated_header_containers: set[str] = set()
    if policy is not None and policy.repeat_header_rows:
        page_scoped_ids = page_scoped_ids | stories.repeated_header_node_ids(ast)
        repeated_header_containers = {
            node["id"]
            for table, _path, count in stories._repeating_tables(ast)
            for row in table["rows"][:count]
            for node in (row, *row["cells"])
        }
    story_id_sets = stories.section_story_id_sets(ast)

    observations_by_page: dict[int, list[dict[str, Any]]] = {
        page.page_index: [] for page in pdf_pages
    }
    for fragment in alignment.fragments:
        observation = {
            "node_id": fragment.node_id,
            "bbox_ppm": list(fragment.bbox_ppm),
            "char_start": fragment.char_start,
            "char_end": fragment.char_end,
            "fragment_index": fragment.fragment_index,
            "continuation_prev": fragment.continuation_prev,
            "continuation_next": fragment.continuation_next,
            "mapping_confidence": EXACT_MATCH_CONFIDENCE_PPM,
        }
        if fragment.node_id in page_scoped_ids:
            observation["chain_scope"] = "page"
        if fragment.node_id in repeated_header_containers:
            # Row/cell character offsets describe synthetic descendant text,
            # not text owned by the container. Its per-page union is one local
            # chain, even when different cells were observed on different
            # pages. Preserve its measured box and character count; actual
            # paragraph/run text ranges remain unchanged.
            observation.update(
                char_start=0,
                char_end=fragment.char_end - fragment.char_start,
                fragment_index=0,
                continuation_prev=False,
                continuation_next=False,
            )
        observations_by_page[fragment.page_index].append(observation)
    for fragment in image_result.observations:
        observation = {
            "node_id": fragment.node_id,
            "bbox_ppm": list(fragment.bbox_ppm),
            "fragment_index": fragment.fragment_index,
            "continuation_prev": fragment.continuation_prev,
            "continuation_next": fragment.continuation_next,
            "mapping_confidence": EXACT_MATCH_CONFIDENCE_PPM,
            "mapping_method": "image_object_match",
        }
        if fragment.node_id in page_scoped_ids:
            observation["chain_scope"] = "page"
        observations_by_page[fragment.page_index].append(observation)
    # a `field` node observed by its computed result (policy only): no
    # char range, one closed chain per page it renders on
    for fragment in alignment.field_fragments if policy is not None else ():
        observations_by_page[fragment.page_index].append(
            {
                "node_id": fragment.node_id,
                "bbox_ppm": list(fragment.bbox_ppm),
                "fragment_index": 0,
                "continuation_prev": False,
                "continuation_next": False,
                "mapping_confidence": EXACT_MATCH_CONFIDENCE_PPM,
                "mapping_method": MAPPING_METHOD,
                "chain_scope": "page",
            }
        )
    # the text_frame node's own box: the union of its content's
    # observations above (text and image), char-less like an image
    # observation, tagged `geometry_inferred` (frames module docstring)
    frame_fragments = frames.frame_observations(
        ast, [*alignment.fragments, *image_result.observations], page_scoped_ids
    )
    for fragment in frame_fragments:
        observation = {
            "node_id": fragment.node_id,
            "bbox_ppm": list(fragment.bbox_ppm),
            "fragment_index": fragment.fragment_index,
            "continuation_prev": fragment.continuation_prev,
            "continuation_next": fragment.continuation_next,
            "mapping_confidence": EXACT_MATCH_CONFIDENCE_PPM,
            "mapping_method": FRAME_MAPPING_METHOD,
        }
        if fragment.node_id in page_scoped_ids:
            observation["chain_scope"] = "page"
        observations_by_page[fragment.page_index].append(observation)

    story_placements_by_page = [
        sum(
            1
            for owned in story_id_sets.values()
            if owned.intersection(obs["node_id"] for obs in observations_by_page[page.page_index])
        )
        for page in pdf_pages
    ]
    stories_glyphless, _section_stories_total = stories.glyphless_section_story_count(ast)
    stories_glyphless_share = stories_glyphless / len(pdf_pages) if pdf_pages else 0.0
    (
        story_pages_covered,
        story_pages_total,
        story_pages_glyphless_only,
        story_pages_not_applicable,
    ) = stories.story_page_coverage(ast, observations_by_page)

    sidecar = {
        "observation_sidecar_schema_version": (
            SIDECAR_SCHEMA_VERSION if policy is None else POLICY_SIDECAR_SCHEMA_VERSION
        ),
        "ast_hash": ast_hash(ast),
        "mapping_method": MAPPING_METHOD,
        "renderer": {key: renderer_meta[key] for key in _RENDERER_META_KEYS},
        "pages": [
            {
                "page_index": page.page_index,
                "width_px": pt_text_to_px(page.width_pt_text, dpi),
                "height_px": pt_text_to_px(page.height_pt_text, dpi),
                "observations": observations_by_page[page.page_index],
            }
            for page in pdf_pages
        ],
    }
    if extraction is not None:
        sidecar["extraction"] = extraction
    if policy is not None:
        sidecar["aligner_policy"] = policy.policy_id
    issues = list(alignment.issues) + list(pick.issues) + image_result.issues
    for page in pdf_pages:
        if page.zero_area_words_dropped:
            issues.append(
                Issue(
                    code="PDF_ZERO_AREA_WORDS_DROPPED",
                    severity="warning",
                    node_id=None,
                    path=f"/pages/{page.page_index}",
                    message=f"{page.zero_area_words_dropped} words",
                )
            )
    build = ObservationSidecarBuild(
        sidecar=sidecar,
        issues=issues,
        matched_text_nodes=alignment.matched_text_nodes,
        total_text_nodes=alignment.total_text_nodes,
        story_placements_by_page=story_placements_by_page,
        stories_glyphless=stories_glyphless,
        stories_glyphless_share=stories_glyphless_share,
        story_placements=sum(story_placements_by_page),
        story_pages_covered=story_pages_covered,
        story_pages_total=story_pages_total,
        story_pages_glyphless_only=story_pages_glyphless_only,
        story_pages_not_applicable=story_pages_not_applicable,
        stories_unplaced=sum(1 for i in issues if i.code == "OBSERVATION_STORY_UNPLACED"),
        frames_total=alignment.frames_total,
        frames_placed=alignment.frames_placed,
        frame_runs_total=alignment.frame_runs_total,
        frame_runs_observed=alignment.frame_runs_observed,
        frame_text_scalars=alignment.frame_text_scalars,
        frames_observed=len({fragment.node_id for fragment in frame_fragments}),
        image_nodes_total=image_result.counters.get("image_nodes_total", 0),
        image_nodes_observed=image_result.counters.get("image_nodes_observed", 0),
        image_nodes_observed_by_size=image_result.counters.get(
            "image_nodes_observed_by_size", 0
        ),
        image_nodes_observed_by_one_sided_corridor=image_result.counters.get(
            "image_nodes_observed_by_one_sided_corridor", 0
        ),
        image_nodes_observed_by_story_pages=image_result.counters.get(
            "image_nodes_observed_by_story_pages", 0
        ),
        image_nodes_transformed=image_result.counters.get("image_nodes_transformed", 0),
        image_nodes_ambiguous=image_result.counters.get("image_nodes_ambiguous", 0),
        image_nodes_unplaced=image_result.counters.get("image_nodes_unplaced", 0),
        image_placements_unexplained=image_result.counters.get(
            "image_placements_unexplained", 0
        ),
        image_extraction_failed=int(
            any(issue.code == "PDF_IMAGE_EXTRACTION_FAILED" for issue in image_result.issues)
        ),
        residual_claimed=alignment.residual_claimed,
        residual_refused_ambiguous=alignment.residual_refused_ambiguous,
        residual_refused_contested=alignment.residual_refused_contested,
        residual_not_found=alignment.residual_not_found,
        residual_rounds=alignment.residual_rounds,
        residual_invariant_skips=alignment.residual_invariant_skips,
        note_marker_matches=alignment.note_marker_matches,
        reading_order_switched_pages=list(pick.switched_pages),
    )
    if build.image_extraction_failed:
        build.image_nodes_total = image_node_count
    validation_issues = validate_observation_sidecar(sidecar)
    if validation_issues:
        raise ObservationSidecarInvalid(validation_issues)
    return build


@lru_cache(maxsize=1)
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _pointer(path: list[Any]) -> str:
    if not path:
        return "/"
    return "/" + "/".join(str(p) for p in path)


def validate_observation_sidecar(doc: Any) -> list[Issue]:
    """Schema pass (Draft 2020-12, closed) then semantic pass; never
    raises on invalid documents — returns Issues (empty means valid)."""
    issues = [
        Issue(
            code="SCHEMA_VALIDATION_ERROR",
            severity="error",
            node_id=None,
            path=_pointer(list(error.absolute_path)),
            message=error.message,
        )
        for error in _validator().iter_errors(doc)
    ]
    if issues:
        return issues
    return _semantic_issues(doc)


def _semantic_issues(doc: dict[str, Any]) -> list[Issue]:
    """Chain-level checks, run per fragment chain.

    A chain is normally one per ``node_id`` (`chain_scope` absent or
    ``"document"``): the node's fragments partition its text across the
    whole document. A header/footer node under DEC-033 §3 is
    ``chain_scope: "page"`` instead — each page it renders on is its
    OWN closed chain, so the chain key there is ``(node_id,
    page_index)``, not ``node_id`` alone. A node mixing both scopes
    across its observations is a build defect, not a valid chain of
    either kind.
    """
    issues: list[Issue] = []
    chains: dict[tuple[str, int | None], list[tuple[int, dict[str, Any], str]]] = {}
    scopes_by_node: dict[str, set[str]] = {}
    for i, page in enumerate(doc["pages"]):
        if page["page_index"] != i:
            issues.append(
                Issue(
                    code="OBSERVATION_PAGE_INDEX_INVALID",
                    severity="error",
                    node_id=None,
                    path=f"/pages/{i}/page_index",
                    message=f"page_index must be contiguous from 0 (expected {i}, "
                    f"got {page['page_index']}).",
                )
            )
        for j, obs in enumerate(page["observations"]):
            path = f"/pages/{i}/observations/{j}"
            x1, y1, x2, y2 = obs["bbox_ppm"]
            if x1 >= x2 or y1 >= y2:
                issues.append(
                    Issue(
                        code="OBSERVATION_BBOX_INVALID",
                        severity="error",
                        node_id=obs["node_id"],
                        path=f"{path}/bbox_ppm",
                        message="bbox_ppm must satisfy x1 < x2 and y1 < y2 — a "
                        "zero-area box points at no pixels.",
                    )
                )
            # image observation (RFC003 §24.3/§25) or frame observation
            # (schema 0.4.0): no char range
            char_less = "char_start" not in obs
            if not char_less and obs["char_start"] >= obs["char_end"]:
                issues.append(
                    Issue(
                        code="OBSERVATION_CHAR_RANGE_INVALID",
                        severity="error",
                        node_id=obs["node_id"],
                        path=f"{path}/char_start",
                        message="char range must be a non-empty half-open [start, end).",
                    )
                )
            scope = obs.get("chain_scope", "document")
            scopes_by_node.setdefault(obs["node_id"], set()).add(scope)
            chain_key = (obs["node_id"], i if scope == "page" else None)
            chains.setdefault(chain_key, []).append((i, obs, path))
    for node_id, scopes in scopes_by_node.items():
        if len(scopes) > 1:
            issues.append(
                Issue(
                    code="OBSERVATION_CHAIN_SCOPE_MIXED",
                    severity="error",
                    node_id=node_id,
                    path="/",
                    message="a node's observations must use one chain_scope "
                    f"throughout; found {sorted(scopes)}.",
                )
            )
    for (node_id, _page_key), entries in chains.items():
        fragments = sorted(entries, key=lambda e: e[1]["fragment_index"])
        indices = [e[1]["fragment_index"] for e in fragments]
        last = len(fragments) - 1
        chain_ok = indices == list(range(len(fragments))) and all(
            e[1]["continuation_prev"] == (k > 0) and e[1]["continuation_next"] == (k < last)
            for k, e in enumerate(fragments)
        )
        if not chain_ok:
            issues.append(
                Issue(
                    code="OBSERVATION_CONTINUATION_INVALID",
                    severity="error",
                    node_id=node_id,
                    path=fragments[0][2],
                    message="fragment_index must be contiguous from 0 with consistent "
                    "continuation_prev/continuation_next flags.",
                )
            )
            continue
        if "char_start" not in fragments[0][1]:
            # image or frame observation chain: no char range to
            # partition, only chain contiguity (checked above) applies.
            continue
        # RFC001 §37.2: the fragments' half-open char ranges must form a
        # contiguous partition starting at 0 — fragment 0 starts at 0 and
        # each fragment's char_end equals the next fragment's char_start.
        starts_at_zero = fragments[0][1]["char_start"] == 0
        contiguous = all(
            fragments[k][1]["char_end"] == fragments[k + 1][1]["char_start"]
            for k in range(len(fragments) - 1)
        )
        if not (starts_at_zero and contiguous):
            issues.append(
                Issue(
                    code="OBSERVATION_CHAR_PARTITION_INVALID",
                    severity="error",
                    node_id=node_id,
                    path=fragments[0][2],
                    message="fragment char ranges must partition the node text: "
                    "start at 0 and be contiguous across the chain.",
                )
            )
    return issues
