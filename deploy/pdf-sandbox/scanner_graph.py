from __future__ import annotations

import base64
import math
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import scanner_fonts as fonts
import scanner_geometry as geometry
from scanner_content import scan_content
from scanner_state import ContentState, GraphicsState, Visibility

_MAX_FORM_DEPTH = 32
_MAX_PAGE_TREE_DEPTH = 64
_FormInvoker = Callable[[str, GraphicsState, bool], Visibility]
_ReferencePath = tuple[str, tuple[str, ...]]


def classify_object_graph(
    graph: object, *, max_image_width: int, max_image_height: int
) -> dict[str, int]:
    objects = dict(_object_items(graph))
    index = _object_index(graph)
    metadata = _c2pa_metadata_objects(index)
    ef_streams = {ref for value in objects.values() for node in _mapping_nodes(value)
                  for ref in _ef_references(node)}
    result = {
        "object_count": len(objects),
        "javascript_action_objects": 0,
        "attachment_objects": 0,
        "c2pa_metadata_objects": 0,
        "annotation_objects": 0,
        "extreme_image_objects": 0,
    }
    for key, value in objects.items():
        reference = key.removeprefix("obj:")
        flattened = list(_mapping_nodes(value))
        keys = {str(key) for node in flattened for key in node}
        scalar_values = {
            item for node in flattened for item in node.values()
            if isinstance(item, str) and item.startswith("/")
        }
        if _carries_action(flattened, keys, scalar_values, index):
            result["javascript_action_objects"] += 1
        if (keys & {"/EmbeddedFiles", "/EF", "/AF"} or scalar_values & {
            "/Filespec", "/FileSpec", "/EmbeddedFile", "/FileAttachment"
        } or reference in ef_streams):
            result["attachment_objects"] += 1
            if reference in metadata:
                result["c2pa_metadata_objects"] += 1
        if "/Annot" in scalar_values or "/Annots" in keys:
            result["annotation_objects"] += 1
        for node in flattened:
            if node.get("/Subtype") != "/Image":
                continue
            width, height = node.get("/Width"), node.get("/Height")
            if (
                type(width) is not int
                or type(height) is not int
                or width <= 0
                or height <= 0
                or width > max_image_width
                or height > max_image_height
                or width * height > max_image_width * max_image_height
            ):
                result["extreme_image_objects"] += 1
    return result

def _ef_references(value: Mapping[str, Any]) -> set[str]:
    ef = value.get("/EF")
    return ({ref for ref in ef.values() if isinstance(ref, str)}
            if isinstance(ef, Mapping) else set())

def _reference_paths(value: object, path: tuple[str, ...] = ()) -> Iterator[_ReferencePath]:
    if isinstance(value, str):
        yield value, path
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _reference_paths(child, (*path, str(key)))
    elif isinstance(value, list):
        for child in value:
            yield from _reference_paths(child, path)

def _c2pa_specs(index: Mapping[str, Any]) -> dict[str, set[str]]:
    candidates = {}
    for ref, value in index.items():
        if (not isinstance(value, Mapping) or value.get("/Type") not in {"/Filespec", "/FileSpec"}
                or value.get("/AFRelationship") != "/C2PA_Manifest"):
            continue
        ef = value.get("/EF")
        streams = _ef_references(value)
        if (not isinstance(ef, Mapping) or not ef or set(ef) - {"/F", "/UF"}
                or len(streams) != len(set(ef.values()))):
            continue
        if all(_is_metadata_stream(index.get(stream)) for stream in streams):
            candidates[ref] = streams
    # An EF stream may be used only by its own declared metadata Filespec.
    for ref, streams in list(candidates.items()):
        for owner, value in index.items():
            if any(target in streams and (owner != ref or path[:1] != ("/EF",))
                   for target, path in _reference_paths(value)):
                del candidates[ref]
                break
    return candidates

def _is_metadata_stream(value: object) -> bool:
    if not isinstance(value, Mapping) or not isinstance(value.get("stream"), Mapping):
        return False
    stream = value["stream"]
    dictionary = stream.get("dict")
    if not isinstance(dictionary, Mapping) or not isinstance(stream.get("data"), str):
        return False
    if dictionary.get("/Type") == "/EmbeddedFile":
        return True
    # The observed Content Credentials writer nests its stream dictionary under /F.
    nested = dictionary.get("/F")
    return (set(dictionary) == {"/F"} and isinstance(nested, Mapping)
            and nested.get("/Subtype") == "u:application/c2pa")

def _filespec_list(value: object, index: Mapping[str, Any]) -> set[str]:
    value = _resolve(value, index)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ValueError("associated files are not references")
    return set(value)

def _name_tree(
    value: object, index: Mapping[str, Any], active: frozenset[str] = frozenset()
) -> tuple[set[str], set[str]]:
    refs = {value} if isinstance(value, str) else set()
    if refs & active or len(active) >= 64:
        raise ValueError("embedded files name tree is cyclic or too deep")
    node = _resolve(value, index)
    if not isinstance(node, Mapping) or set(node) - {"/Names", "/Kids", "/Limits"}:
        raise ValueError("embedded files name tree is invalid")
    names = node.get("/Names", [])
    if (not isinstance(names, list) or len(names) % 2
            or not all(isinstance(item, str) for item in names)):
        raise ValueError("embedded files names are invalid")
    targets = set(names[1::2])
    kids = node.get("/Kids", [])
    if not isinstance(kids, list):
        raise ValueError("embedded files children are invalid")
    for child in kids:
        child_targets, child_refs = _name_tree(child, index, active | refs)
        targets |= child_targets
        refs |= child_refs
    if not targets:
        raise ValueError("embedded files name tree is empty")
    return targets, refs

def _c2pa_declarations(index: Mapping[str, Any]) -> dict[str, set[str]]:
    declarations = {}
    for ref, value in index.items():
        if not isinstance(value, Mapping):
            continue
        try:
            if value.get("/Type") == "/Catalog":
                if "/AF" in value:
                    declarations.setdefault(ref, set()).update(_filespec_list(value["/AF"], index))
                names_ref = value.get("/Names")
                names = _resolve(names_ref, index)
                if isinstance(names, Mapping) and "/EmbeddedFiles" in names:
                    targets, trees = _name_tree(names["/EmbeddedFiles"], index)
                    declarations.setdefault(ref, set()).update(targets)
                    for tree in trees:
                        declarations.setdefault(tree, set()).update(targets)
                    if isinstance(names_ref, str):
                        declarations.setdefault(names_ref, set()).update(targets)
            elif value.get("/Subtype") == "/FileAttachment" and isinstance(value.get("/FS"), str):
                declarations[ref] = {value["/FS"]}
        except ValueError:
            # Invalid containers remain counted as ordinary attachments.
            declarations.pop(ref, None)
    return declarations

def _c2pa_metadata_objects(index: Mapping[str, Any]) -> set[str]:
    candidates = _c2pa_specs(index)
    declarations = _c2pa_declarations(index)
    declared = set().union(*declarations.values()) if declarations else set()
    specs = set(candidates) & declared
    streams = set().union(*(candidates[ref] for ref in specs)) if specs else set()
    metadata = specs | streams
    for ref, targets in declarations.items():
        if targets and targets <= specs:
            metadata.add(ref)
    # No nested attachment marker can be hidden inside a whitelisted object.
    for ref in list(metadata):
        value = index[ref]
        for node in _mapping_nodes(value):
            if ("/EF" in node and (ref not in specs or node is not value)
                    or "/AF" in node and (ref not in declarations or node is not value)
                    or node.get("/Type") in {"/Filespec", "/FileSpec"}
                    and (ref not in specs or node is not value)
                    or node.get("/Type") == "/EmbeddedFile" and ref not in streams
                    or node.get("/Subtype") == "/FileAttachment"
                    and (node is not value or node.get("/FS") not in specs)
                    or "/EmbeddedFiles" in node and ref not in declarations):
                metadata.discard(ref)
                break
    return metadata

def c2pa_filespec_references(graph: object) -> set[str]:
    """Known declared, exclusively referenced C2PA Filespecs for warning matching."""
    index = _object_index(graph)
    metadata = _c2pa_metadata_objects(index)
    return {ref for ref in metadata if isinstance(index[ref], Mapping)
            and index[ref].get("/Type") == "/FileSpec"}


_ACTION_SUBTYPES = frozenset({"/JavaScript", "/Launch", "/SubmitForm", "/ImportData", "/GoToR"})


def _carries_action(
    flattened: list[Mapping[str, Any]],
    keys: set[str],
    scalar_values: set[str],
    index: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Whether one object is (or triggers) an executable action.

    `/JS`, `/AA` (additional actions) and an action subtype anywhere in the
    object count as before. `/A` and `/OpenAction` count only when their
    value IS an action of a dangerous subtype: a tagged PDF's structure
    elements carry `/A` attribute dictionaries (ISO 32000-1 14.7.6, `/O`
    owner, no `/S`) and the catalog's `/OpenAction` is usually a page
    destination array -- 26.9. every LibreOffice/Word PDF was refused as
    PDF_ACTIVE_CONTENT for those alone. A reference that does not resolve
    stays counted (fail closed)."""
    if keys & {"/AA", "/JS"} or scalar_values & _ACTION_SUBTYPES:
        return True
    for node in flattened:
        for key in ("/A", "/OpenAction"):
            if key in node and _is_dangerous_action(node[key], index):
                return True
    return False


def _is_dangerous_action(value: object, index: Mapping[str, Mapping[str, Any]]) -> bool:
    if isinstance(value, list):
        return False  # a destination array, never an action
    if isinstance(value, str):
        if value.startswith("/"):
            return value in _ACTION_SUBTYPES
        resolved = index.get(value)
        if resolved is None:
            return True  # unresolvable reference: refuse rather than guess
        value = resolved
    if not isinstance(value, Mapping):
        return True
    subtype = value.get("/S")
    if subtype in _ACTION_SUBTYPES or "/JS" in value:
        return True
    if "/Next" in value:  # an action chain: any dangerous link poisons it
        nxt = value["/Next"]
        chain = nxt if isinstance(nxt, list) else [nxt]
        return any(_is_dangerous_action(v, index) for v in chain)
    return False


def classify_graph_text(graph: object) -> tuple[str, int, int]:
    try:
        objects = _object_index(graph)
        pages = [
            value
            for value in objects.values()
            if isinstance(value, Mapping) and value.get("/Type") == "/Page"
        ]
        if not pages:
            raise ValueError("page object absent")
        visibility = Visibility()
        for page in pages:
            visibility.merge(_scan_page(page, objects))
        return visibility.result()
    except (TypeError, ValueError):
        return ("ambiguous", 0, 1)


def _scan_page(page: Mapping[str, Any], objects: Mapping[str, Any]) -> Visibility:
    resources = _resolved_mapping(page.get("/Resources", {}), objects)
    graphics = GraphicsState(page_box=_page_box(page, objects))
    streams = list(_content_streams(page.get("/Contents"), objects))
    invoker = _form_invoker(objects, resources, frozenset(), 0)
    font_metrics = _font_metrics_provider(objects, resources)
    return scan_content(b"\n".join(streams), graphics, False, invoker, font_metrics)


def _font_metrics_provider(
    objects: Mapping[str, Any], resources: Mapping[str, Any]
) -> Callable[[str], object]:
    """A per-resources, memoised `Tf` name -> `WidthTable | None` lookup."""
    cache: dict[str, object] = {}

    def resolve(name: str) -> object:
        if name not in cache:
            cache[name] = _resolve_font_widths(name, resources, objects)
        return cache[name]

    return resolve


def _resolve_font_widths(
    name: str, resources: Mapping[str, Any], objects: Mapping[str, Any]
) -> object:
    try:
        font = _resolve(_resolved_mapping(resources.get("/Font", {}), objects).get(name), objects)
    except ValueError:
        return None
    if not isinstance(font, Mapping):
        return None
    return fonts.build_width_table(font, lambda value: _resolve(value, objects))


def _page_box(
    page: Mapping[str, Any], objects: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    """The visible page rectangle: the inherited /MediaBox cut by any /CropBox."""
    node: object = page
    box = None
    for _ in range(_MAX_PAGE_TREE_DEPTH):
        if not isinstance(node, Mapping):
            break
        if box is None and "/MediaBox" in node:
            box = _rect(_resolve(node["/MediaBox"], objects))
        if "/CropBox" in node:
            box = geometry.intersect(box, _rect(_resolve(node["/CropBox"], objects)))
            if "/MediaBox" in node:
                break
        if box is not None and "/Parent" not in node:
            break
        node = _resolve(node.get("/Parent"), objects)
    if box is None or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("page box is absent or empty")
    return box


def _rect(value: object) -> tuple[float, float, float, float]:
    if not _valid_bbox(value):
        raise ValueError("page rectangle is invalid")
    return tuple(float(item) for item in value)  # type: ignore[return-value, union-attr]


def _form_invoker(
    objects: Mapping[str, Any],
    resources: Mapping[str, Any],
    active: frozenset[str],
    depth: int,
) -> _FormInvoker:
    def invoke(name: str, graphics: GraphicsState, optional: bool) -> Visibility:
        xobjects = _resolved_mapping(resources.get("/XObject", {}), objects)
        reference = xobjects.get(name)
        reference_key = reference if isinstance(reference, str) else f"direct:{id(reference)}"
        target = _resolve(reference, objects)
        if not isinstance(target, Mapping):
            return Visibility(ambiguous=True)
        dictionary = _stream_dictionary(target)
        if dictionary.get("/Subtype") == "/Image":
            region = geometry.image_paint_region(graphics)
            if region is None:
                return Visibility(unbounded_paint=True)
            return Visibility(paints=[region])
        if (
            dictionary.get("/Subtype") != "/Form"
            or depth >= _MAX_FORM_DEPTH
            or reference_key in active
        ):
            return Visibility(ambiguous=True)
        return _scan_form(
            target, dictionary, objects, resources, graphics, optional,
            active | {reference_key}, depth,
        )

    return invoke


def _scan_form(
    target: Mapping[str, Any],
    dictionary: Mapping[str, Any],
    objects: Mapping[str, Any],
    inherited_resources: Mapping[str, Any],
    graphics: GraphicsState,
    optional: bool,
    active: frozenset[str],
    depth: int,
) -> Visibility:
    matrix = dictionary.get("/Matrix", [1, 0, 0, 1, 0, 0])
    parsed_matrix = _matrix(matrix)
    if parsed_matrix is None:
        return Visibility(ambiguous=True)
    determinant = parsed_matrix[0] * parsed_matrix[3] - parsed_matrix[1] * parsed_matrix[2]
    graphics.ctm_ambiguous |= abs(determinant) <= 1e-12 or parsed_matrix != (1, 0, 0, 1, 0, 0)
    if "/BBox" in dictionary:
        bbox = dictionary["/BBox"]
        if not _valid_bbox(bbox):
            return Visibility(ambiguous=True)
        clip_state = ContentState(graphics)
        geometry.clip_to_rect(clip_state, geometry.transform_rect(_rect(bbox), graphics.ctm))
    optional |= "/OC" in dictionary
    resources = _resolved_mapping(dictionary.get("/Resources", inherited_resources), objects)
    invoker = _form_invoker(objects, resources, active, depth + 1)
    font_metrics = _font_metrics_provider(objects, resources)
    return scan_content(_stream_data(target), graphics, optional, invoker, font_metrics)


def _matrix(value: object) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) != 6:
        return None
    parsed = tuple(_bounded_number(item) for item in value)
    if any(item is None for item in parsed):
        return None
    return tuple(item for item in parsed if item is not None)


def _valid_bbox(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    parsed = tuple(_bounded_number(item) for item in value)
    return (
        all(item is not None for item in parsed)
        and float(parsed[0]) < float(parsed[2])
        and float(parsed[1]) < float(parsed[3])
    )


def _bounded_number(value: object) -> float | None:
    if type(value) not in {int, float}:
        return None
    try:
        parsed = float(value)
    except OverflowError:
        return None
    return parsed if math.isfinite(parsed) and abs(parsed) <= 1_000_000 else None


def _content_streams(contents: object, objects: Mapping[str, Any]) -> Iterator[bytes]:
    entries = contents if isinstance(contents, list) else [contents]
    for entry in entries:
        resolved = _resolve(entry, objects)
        if not isinstance(resolved, Mapping):
            raise ValueError("page content stream is invalid")
        yield _stream_data(resolved)


def _stream_data(value: Mapping[str, Any]) -> bytes:
    stream = value.get("stream")
    if not isinstance(stream, Mapping) or not isinstance(stream.get("data"), str):
        raise ValueError("stream data is not inline base64")
    return base64.b64decode(stream["data"], validate=True)


def _stream_dictionary(value: Mapping[str, Any]) -> Mapping[str, Any]:
    stream = value.get("stream")
    if not isinstance(stream, Mapping) or not isinstance(stream.get("dict"), Mapping):
        raise ValueError("stream dictionary is invalid")
    return stream["dict"]


def _resolved_mapping(value: object, objects: Mapping[str, Any]) -> Mapping[str, Any]:
    resolved = _resolve(value, objects)
    if not isinstance(resolved, Mapping):
        raise ValueError("resource dictionary is invalid")
    return resolved


def _resolve(value: object, objects: Mapping[str, Any]) -> object:
    return objects.get(value, value) if isinstance(value, str) else value


def _object_index(graph: object) -> dict[str, Any]:
    """Every indirect object by reference, dictionaries and bare values alike."""
    return {
        key.removeprefix("obj:"): value
        for key, value in _object_items(graph, mappings_only=False)
    }


def _object_items(graph: object, *, mappings_only: bool = True) -> Iterator[tuple[str, Any]]:
    if not isinstance(graph, Mapping) or not isinstance(graph.get("qpdf"), list):
        raise ValueError("qpdf JSON root is invalid")
    for entry in graph["qpdf"]:
        if not isinstance(entry, Mapping):
            continue
        for raw_key, wrapped in entry.items():
            key = str(raw_key)
            if not key.startswith("obj:") or not isinstance(wrapped, Mapping):
                continue
            value = wrapped.get("value", wrapped)
            if isinstance(value, Mapping) or not mappings_only:
                yield key, value


def _mapping_nodes(value: object) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _mapping_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _mapping_nodes(child)
