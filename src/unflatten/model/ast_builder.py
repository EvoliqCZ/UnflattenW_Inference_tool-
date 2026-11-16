"""AST Builder, strict mode (RFC004 §33, design §4, T3).

Turns an `AssembledDocumentHypothesis` into a Canonical AST dict. Node
types, order and run text come from the hypothesis unchanged (§33.2-§33.3);
each typed value is written at its RFC001 JSON pointer. Nothing unknown is
filled in: a required value the hypothesis does not carry is an
`UNRESOLVED_REQUIRED_PROPERTY` naming its pointer, and the build fails
(§33.6: `0`, `""` and `null` never mean "unknown").

Repeated containers (`paragraph /layout/tabs`, `section /columns/
widths_mpt`, `table /columns`) take their element count from the typed
value at the container pointer itself; element values arrive at indexed
pointers (`/layout/tabs/0/position_mpt`). `paragraph /layout/tabs`'s count
comes from a model head (`typed_decode.decode_typed`'s `CardinalityHead`
step, design `spec-typed-heads-missing-families` §4.1/§4.2, T2);
`table /columns` and `section /columns/widths_mpt` have no count head --
their count is a registered, deterministic derivation
(`table.columns.from_grid.v0`, `section.columns.widths_mpt.length.v0`,
DEC-037) applied after typed values are written, never predicted.

The output is plain RFC001 dicts, not `unflatten.ast.model.Document`,
which has no `endnotes` field.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from unflatten.assembly.assemble import AssembledDocumentHypothesis, AssembledNode
from unflatten.ast.canonicalizer import _CHILDREN_KEY, CanonicalizationError, canonicalize
from unflatten.model import builder_report as br
from unflatten.model import derivations as dv
from unflatten.model import resource_binder as rb
from unflatten.model.ast_required import required_leaf_pointers, union_arm_leaves
from unflatten.model.typed_property_pointer_map import repeated_containers, union_containers

UNRESOLVED_REQUIRED_PROPERTY = "UNRESOLVED_REQUIRED_PROPERTY"
NOTE_REF_UNBOUND = "NOTE_REF_UNBOUND"
REPEATED_INDEX_OUT_OF_RANGE = "REPEATED_INDEX_OUT_OF_RANGE"
STORY_UNRESOLVED = "STORY_UNRESOLVED"
ASSEMBLER_PROPERTY_CONFLICT = "ASSEMBLER_PROPERTY_CONFLICT"
CANONICALIZATION_REFUSED = "CANONICALIZATION_REFUSED"

#: Design §2.5/§3 P2: `_build_raw`'s own version, embedded in every
#: Builder Report (`builder_report.build_report(builder_version=...)`).
BUILDER_VERSION = "0.2.0"

_IMAGE_TYPES = frozenset({"inline_image", "floating_image"})
_NOTE_TYPES = {"footnote": "footnotes", "endnote": "endnotes"}
#: Supplied by the builder itself, never by a typed value.
_BUILDER_POINTERS = frozenset({"/resource_ref", "/note_id"})
_MISSING = object()


@dataclass(frozen=True, slots=True)
class BuilderIssue:
    code: str
    node_type: str | None
    pointer: str | None
    source: str | None  # assembled id "a<k>", when the issue is one node's
    detail: str = ""
    #: RFC001 tree id "b<n>" (design §2.5, plan P4), when the issue is
    #: scoped to an already-built node -- `review_candidate.py`'s only use
    #: of this field is to locate where to write an imputed default. `None`
    #: for issues with no single owning node (`ASSEMBLER_PROPERTY_CONFLICT`,
    #: `STORY_UNRESOLVED`).
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class BuildResult:
    status: Literal["built", "failed"]
    ast: dict[str, Any] | None  # canonical; None when failed
    issues: tuple[BuilderIssue, ...]
    assets: Mapping[str, bytes] = dataclasses.field(default_factory=dict)  # path -> bytes
    report: Mapping[str, Any] = dataclasses.field(default_factory=dict)  # Builder Report, design §3


@dataclass(frozen=True, slots=True)
class RawBuild:
    """`_build_raw`'s own result: the tree exactly as `_Builder` produced it
    (not yet gated on issues, not yet canonicalized) plus the Builder Report
    computed over it. `build_strict` = this + the issue gate + `canonicalize`
    (design §2 P2: "Žiadne iné zmeny správania nie sú")."""

    tree: dict[str, Any]
    issues: tuple[BuilderIssue, ...]
    binding: rb.Binding
    report: Mapping[str, Any]


def _build_raw(
    document: AssembledDocumentHypothesis,
    *,
    profile: str,
    ast_version: str,
    document_id: str,
    recovered: Mapping[str, rb.RecoveredResource] | None = None,
) -> RawBuild:
    """`recovered` maps an image's assembled id to its recovered bytes; an
    image without one is an abstention (`RESOURCE_SLOT_UNRESOLVED`)."""
    builder = _Builder()
    sections = [builder.node(section) for section in document.sections]
    builder.attach_stories(document)
    notes: dict[str, list[dict[str, Any]]] = {}
    for note in document.notes:
        notes.setdefault(_NOTE_TYPES[note.node_type], []).append(builder.node(note))
    ast: dict[str, Any] = {
        "ast_version": ast_version,
        "profile": profile,
        "document_id": document_id,
        "metadata": {"language": None, "title": None, "subject": None},
        "resources": [],
        "sections": sections,
        "extensions": {},
        **notes,
    }
    # Snapshot node_counts/predicted_by_pointer BEFORE the derivation
    # post-pass runs: "predicted" must count what the hypothesis itself
    # carried, never a value the post-pass filled in (design §3;
    # /note_id is a `_BUILDER_POINTERS` exclusion either way, so bind_note_refs
    # running after this snapshot changes nothing it counts).
    node_counts = _node_counts(builder.built)
    predicted_by_pointer = _predicted_by_pointer(builder.built)
    builder.bind_note_refs(ast)
    derivation_rows, discarded_rows = builder.apply_derivations(ast)
    binding = builder.bind_resources(recovered or {})
    ast["resources"] = list(binding.resources)
    for conflict in document.conflicts:
        builder.issue(ASSEMBLER_PROPERTY_CONFLICT, None, conflict.pointer, conflict.assembled_id)
    for story_set in document.story_sets:
        if story_set.issue is not None:
            builder.issue(STORY_UNRESOLVED, story_set.story_type, None, story_set.section_id)
    for node, assembled in builder.built:
        builder.check_required(node, assembled.assembled_id)
    report = br.build_report(
        builder_version=BUILDER_VERSION,
        node_counts=node_counts,
        predicted_by_pointer=predicted_by_pointer,
        derivations=derivation_rows,
        discarded_inputs=discarded_rows,
        unresolved=[_issue_row(i) for i in builder.issues if i.code != ASSEMBLER_PROPERTY_CONFLICT],
        conflicts=[_issue_row(i) for i in builder.issues if i.code == ASSEMBLER_PROPERTY_CONFLICT],
    )
    return RawBuild(tree=ast, issues=tuple(builder.issues), binding=binding, report=report)


def build_strict(
    document: AssembledDocumentHypothesis,
    *,
    profile: str,
    ast_version: str,
    document_id: str,
    recovered: Mapping[str, rb.RecoveredResource] | None = None,
) -> BuildResult:
    """`recovered` maps an image's assembled id to its recovered bytes; an
    image without one is an abstention (`RESOURCE_SLOT_UNRESOLVED`)."""
    raw = _build_raw(
        document,
        profile=profile,
        ast_version=ast_version,
        document_id=document_id,
        recovered=recovered,
    )
    if raw.issues:
        return BuildResult(status="failed", ast=None, issues=raw.issues, report=raw.report)
    try:
        canonical = canonicalize(raw.tree)
    except CanonicalizationError as exc:
        issues = tuple(
            BuilderIssue(CANONICALIZATION_REFUSED, None, i.path, None, f"{i.code}: {i.message}")
            for i in exc.issues
        )
        return BuildResult(status="failed", ast=None, issues=issues, report=raw.report)
    return BuildResult(
        status="built", ast=canonical, issues=(), assets=raw.binding.assets, report=raw.report
    )


def _issue_row(issue: BuilderIssue) -> dict[str, Any]:
    return {
        "source": issue.source,
        "node_type": issue.node_type,
        "pointer": issue.pointer,
        "code": issue.code,
        "detail": issue.detail,
    }


def _node_counts(built: list[tuple[dict[str, Any], AssembledNode]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node, _source in built:
        counts[node["type"]] = counts.get(node["type"], 0) + 1
    return counts


def _predicted_by_pointer(built: list[tuple[dict[str, Any], AssembledNode]]) -> dict[str, int]:
    """One count per `"<node_type><pointer>"` a required leaf pointer was
    fully present at (design §3 `predicted_by_pointer`). `_build_raw` calls
    this on `builder.built` BEFORE `apply_derivations` runs (`ast_builder.py`
    `_build_raw`'s own snapshot-ordering comment), so every present required
    leaf counted here came from the hypothesis itself, never from a
    derivation the post-pass filled in afterwards."""
    counts: dict[str, int] = {}
    for node, _source in built:
        node_type = node["type"]
        unions = {c for nt, c in union_containers() if nt == node_type}
        for pointer in sorted(required_leaf_pointers(node_type) - _BUILDER_POINTERS):
            for _ in _present(node, pointer, unions):
                key = f"{node_type}{pointer}"
                counts[key] = counts.get(key, 0) + 1
    return counts


def _present(node: dict[str, Any], pointer: str, unions: set[str]) -> Iterator[str]:
    """The dual of `_Builder._missing`: yields `pointer` once per element
    that IS present, instead of once per element that is missing."""
    for container in unions:
        if pointer.startswith(container + "/") and pointer != container + "/mode":
            mode = _get(node, container + "/mode")
            if mode is _MISSING:
                return
            arm = union_arm_leaves(node["type"], container).get(mode, frozenset())
            if pointer not in arm:
                return
    if "/*/" in pointer:
        container, rest = pointer.split("/*/", 1)
        elements = _get(node, container)
        if elements is _MISSING:
            return
        for element in elements:
            if _get(element, "/" + rest) is not _MISSING:
                yield pointer
        return
    if _get(node, pointer) is not _MISSING:
        yield pointer


class _Builder:
    def __init__(self) -> None:
        self.issues: list[BuilderIssue] = []
        self.built: list[tuple[dict[str, Any], AssembledNode]] = []
        self.images: list[tuple[dict[str, Any], str]] = []
        self._next = 0
        #: Indexed elements of a repeated container the hypothesis carried
        #: without the container's own count, keyed by `(node["id"],
        #: container)`. Not dropped (U4): the derivation post-pass
        #: (`apply_derivations`) is the only other place that pops an entry,
        #: once it has derived a count to materialize them against.
        self._stash: dict[tuple[str, str], dict[str, Any]] = {}

    def issue(
        self, code: str, node_type: str | None, pointer: str | None, source: str | None,
        detail: str = "", node_id: str | None = None,
    ) -> None:
        self.issues.append(BuilderIssue(code, node_type, pointer, source, detail, node_id))

    def node(self, assembled: AssembledNode) -> dict[str, Any]:
        self._next += 1
        node_type = assembled.node_type
        out: dict[str, Any] = {"id": f"b{self._next}", "type": node_type}
        if node_type == "run":
            out["text"] = assembled.text or ""
        counts: dict[str, Any] = {}
        containers = {c for nt, c in repeated_containers() if nt == node_type}
        for pointer, value in assembled.typed.items():
            if pointer in containers:
                counts[pointer] = value
            elif "*" not in pointer.split("/"):
                _write(out, pointer, value)
        for container in containers:
            self._materialize(out, container, counts.get(container), assembled)
        if node_type in _IMAGE_TYPES:
            self.images.append((out, assembled.assembled_id))
        key = _CHILDREN_KEY.get(node_type)
        if key is not None:
            out[key] = [self.node(child) for child in assembled.children]
        self.built.append((out, assembled))
        return out

    def _materialize(
        self, node: dict[str, Any], container: str, count: Any, assembled: AssembledNode
    ) -> None:
        elements = _get(node, container)
        if count is None:
            # Indexed elements without a count are not a list: the container
            # stays unresolved and `check_required` names it. Not dropped
            # (U4): stashed for the derivation post-pass, which is the only
            # other place that may still supply a count.
            if elements is not _MISSING:
                _pop(node, container)
                self._stash[(node["id"], container)] = elements
            return
        by_index = {} if elements is _MISSING else elements
        self._fill_container(node, container, by_index, count, assembled)

    def _fill_container(
        self,
        node: dict[str, Any],
        container: str,
        by_index: Mapping[str, Any],
        count: int,
        assembled: AssembledNode,
        *,
        report_stray_as_discarded: bool = False,
    ) -> list[str]:
        """Write `container`'s `count`-length list from `by_index`'s
        elements, keyed by decimal string index.

        A missing SCALAR leaf element (the whole container pointer is
        itself a required leaf, e.g. `section /columns/widths_mpt` --
        `default_policy.py`'s docstring on why `table /columns` is not
        one) is `UNRESOLVED_REQUIRED_PROPERTY` naming its index, never a
        silent `{}` placeholder (U5: `check_required`'s own `/*/` handling
        already catches a missing OBJECT element's own sub-pointer, so `{}`
        stays correct there). Indices at or past `count` are stray: either
        `REPEATED_INDEX_OUT_OF_RANGE` (a builder-materialized container
        whose own typed count disagreed with its own typed indices) or, when
        `report_stray_as_discarded` (the derivation post-pass, U6: the
        model made no count claim at all, so its leftover indices are not
        "out of range" against a count it never gave), returned for the
        caller to report as `discarded_inputs`.
        """
        stray = sorted(k for k in by_index if not (k.isdigit() and int(k) < count))
        discarded: list[str] = []
        if stray:
            if report_stray_as_discarded:
                discarded = stray
            else:
                self.issue(REPEATED_INDEX_OUT_OF_RANGE, node["type"], container,
                           assembled.assembled_id, f"indices {stray} with count {count}")
        scalar_leaf = container in required_leaf_pointers(node["type"])
        elements: list[Any] = []
        for i in range(count):
            key = str(i)
            if key in by_index:
                elements.append(by_index[key])
            elif scalar_leaf:
                self.issue(UNRESOLVED_REQUIRED_PROPERTY, node["type"], f"{container}/{i}",
                           assembled.assembled_id, node_id=node["id"])
                elements.append(None)
            else:
                elements.append({})
        _write(node, container, elements)
        return discarded

    def apply_derivations(
        self, ast: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """RFC004 §27.2 post-pass (design §2.3/§3 plan P3): for every built
        node and every `derivations.DERIVATIONS` record that applies to its
        type, fill `record.output_pointer` when the hypothesis left it
        missing and `record.precondition` holds -- never when it is already
        present (R11: predicted wins, derivation not computed). Runs after
        `bind_note_refs` (marker_text needs bound `note_id`s) and before
        `bind_resources`/`check_required` (plan P3 step 3)."""
        note_kind_and_order: dict[str, tuple[str, int]] = {}
        for index, note in enumerate(ast.get("footnotes", ()), start=1):
            note_kind_and_order[note["id"]] = ("footnote", index)
        for index, note in enumerate(ast.get("endnotes", ()), start=1):
            note_kind_and_order[note["id"]] = ("endnote", index)

        # `self.built` is postorder (a node is appended after its children);
        # node ids are assigned in preorder (`self._next` incremented before
        # a node's children are built), so sorting by numeric id recovers
        # preorder without a second tree walk.
        ordered = sorted(self.built, key=lambda t: int(t[0]["id"][1:]))
        by_page_floating: dict[int, list[str]] = {}
        for node, assembled in ordered:
            if node["type"] in ("floating_image", "text_frame") and assembled.sources:
                by_page_floating.setdefault(assembled.sources[0][0], []).append(node["id"])

        containers = repeated_containers()
        derivation_rows: list[dict[str, Any]] = []
        discarded_rows: list[dict[str, Any]] = []
        for node, assembled in ordered:
            page_ordinal = assembled.sources[0][0] if assembled.sources else None
            ctx = dv.DerivationContext(
                node=node,
                assembled_id=assembled.assembled_id,
                page_ordinal=page_ordinal,
                page_floating_preorder=(
                    tuple(by_page_floating.get(page_ordinal, ()))
                    if page_ordinal is not None
                    else ()
                ),
                note_kind_and_order=note_kind_and_order,
            )
            for record in dv.DERIVATIONS:
                node_types = (
                    record.node_type if isinstance(record.node_type, tuple) else (record.node_type,)
                )
                if node["type"] not in node_types:
                    continue
                if _get(node, record.output_pointer) is not _MISSING:
                    continue  # R11: a predicted value always wins.
                if not record.precondition(ctx):
                    continue
                value = record.fn(ctx)
                row: dict[str, Any] = {
                    "source": assembled.assembled_id,
                    "node_id": node["id"],
                    "node_type": node["type"],
                    "pointer": record.output_pointer,
                    "derivation_id": record.derivation_id,
                    "version": record.version,
                    "input_pointers": list(record.input_pointers),
                }
                if (node["type"], record.output_pointer) in containers:
                    by_index = self._stash.pop((node["id"], record.output_pointer), {})
                    discarded = self._fill_container(
                        node, record.output_pointer, by_index, value, assembled,
                        report_stray_as_discarded=True,
                    )
                    for key in discarded:
                        discarded_rows.append({
                            "source": assembled.assembled_id,
                            "node_id": node["id"],
                            "node_type": node["type"],
                            "pointer": f"{record.output_pointer}/{key}",
                            "detail": f"discarded index {key}: {record.derivation_id} "
                                      f"derived length {value}",
                        })
                    row["value_length"] = value
                else:
                    _write(node, record.output_pointer, value)
                    row["value"] = value
                derivation_rows.append(row)

        # `self._stash` entries left after the loop above belong to a
        # container no `DERIVATIONS` record ever names (e.g. `paragraph
        # /layout/tabs` -- DEC-037 Q5 has no count source for it, unlike
        # `section /columns/widths_mpt`). Report them as `discarded_inputs`
        # too (suggestion, verify FAIL): the container is still correctly
        # `UNRESOLVED_REQUIRED_PROPERTY` (`check_required` runs after this),
        # but the model's own indexed elements must not silently vanish
        # from the Builder Report along the way.
        by_node_id = {node["id"]: (node, assembled) for node, assembled in ordered}
        for (node_id, container), by_index in self._stash.items():
            node, assembled = by_node_id[node_id]
            for key in sorted(by_index):
                discarded_rows.append({
                    "source": assembled.assembled_id,
                    "node_id": node_id,
                    "node_type": node["type"],
                    "pointer": f"{container}/{key}",
                    "detail": f"discarded index {key}: no derivation for {container}",
                })
        return derivation_rows, discarded_rows

    def attach_stories(self, document: AssembledDocumentHypothesis) -> None:
        section_by_source = {
            source.assembled_id: built for built, source in self.built if built["type"] == "section"
        }
        for story_set in document.story_sets:
            section = section_by_source[story_set.section_id]
            for story in story_set.stories:
                for assembled in story.nodes:
                    built = self.node(assembled)
                    built["kind"] = story.kind
                    section.setdefault(f"{story_set.story_type}s", []).append(built)

    def bind_resources(self, recovered: Mapping[str, rb.RecoveredResource]) -> rb.Binding:
        """RFC004 §28.4: one binder record per image, slot = its node id. No
        NodeVisualAligner yet (DEC-037 bod 3): absent -> abstained."""
        records = [
            dataclasses.replace(recovered[source], slot=node["id"])
            if source in recovered else rb.RecoveredResource.abstained(node["id"])
            for node, source in self.images
        ]
        binding = rb.bind(records)
        by_slot = {node["id"]: (node, source) for node, source in self.images}
        for issue in binding.issues:
            node, source = by_slot[issue.slot]
            self.issue(issue.code, node["type"], "/resource_ref", source, issue.detail,
                       node_id=node["id"])
        for slot, resource_id in binding.refs.items():
            by_slot[slot][0]["resource_ref"] = resource_id
        return binding

    def bind_note_refs(self, ast: Mapping[str, Any]) -> None:
        """k-th note_ref in reading order -> k-th note in page order
        (design §10: `/note_id` is the builder's). Unequal counts leave
        every ref unbound."""
        refs = [n for n in _preorder(ast) if n["type"] == "note_ref"]
        notes = [n for key in ("footnotes", "endnotes") for n in ast.get(key, ())]
        notes.sort(key=lambda n: self._order(n))
        if len(refs) != len(notes):
            for ref in refs:
                self.issue(NOTE_REF_UNBOUND, "note_ref", "/note_id", None,
                           f"{len(refs)} note refs, {len(notes)} notes", node_id=ref["id"])
            return
        for ref, note in zip(refs, notes, strict=True):
            ref["note_id"] = note["id"]

    def _order(self, node: Mapping[str, Any]) -> int:
        return int(node["id"][1:])

    def check_required(self, node: dict[str, Any], source: str) -> None:
        node_type = node["type"]
        unions = {c for nt, c in union_containers() if nt == node_type}
        reported: set[str] = set()
        for pointer in sorted(required_leaf_pointers(node_type) - _BUILDER_POINTERS):
            for missing in self._missing(node, pointer, unions):
                if missing not in reported:
                    reported.add(missing)
                    self.issue(UNRESOLVED_REQUIRED_PROPERTY, node_type, missing, source,
                               node_id=node["id"])

    def _missing(self, node: dict[str, Any], pointer: str, unions: set[str]) -> Iterator[str]:
        for container in unions:
            if pointer.startswith(container + "/") and pointer != container + "/mode":
                mode = _get(node, container + "/mode")
                if mode is _MISSING:
                    return
                arm = union_arm_leaves(node["type"], container).get(mode, frozenset())
                if pointer not in arm:
                    return
        if "/*/" in pointer:
            container, rest = pointer.split("/*/", 1)
            elements = _get(node, container)
            if elements is _MISSING:
                yield container
                return
            for index, element in enumerate(elements):
                if _get(element, "/" + rest) is _MISSING:
                    yield f"{container}/{index}/{rest}"
            return
        if _get(node, pointer) is _MISSING:
            yield pointer


def _write(node: dict[str, Any], pointer: str, value: Any) -> None:
    parts = pointer.strip("/").split("/")
    target = node
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value


def _pop(node: dict[str, Any], pointer: str) -> None:
    parts = pointer.strip("/").split("/")
    target = node
    for part in parts[:-1]:
        target = target[part]
    del target[parts[-1]]


def _get(node: Any, pointer: str) -> Any:
    value = node
    for part in pointer.strip("/").split("/"):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return _MISSING
    return value


def _preorder(ast: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    def walk(node: dict[str, Any]) -> Iterator[dict[str, Any]]:
        yield node
        for key in ("headers", "footers", _CHILDREN_KEY.get(node["type"], "")):
            for child in node.get(key) or ():
                yield from walk(child)

    for section in ast["sections"]:
        yield from walk(section)
