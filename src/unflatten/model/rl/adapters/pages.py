"""``open_pages`` / ``read_selection`` ports of the S4.5 real run (DP4, DP7, DP8; torch-free).

``make_open_pages`` closes over the release, the store, the strict
gradient-bearing cache reader, the frozen selection, the exclusion registry and
the frozen S4 StageDataMatrix, and returns the ``RealRunPorts.open_pages``
callable. Per call, in this order, each step a named refusal:

1. the registry is verified against the release (``verify_exclusion_registry``);
2. the cache is strict-admitted (``require_strict_admission``) and the reader's
   pinned expectation is the one injected;
3. ``rights_privacy_refs`` are re-derived from the release (pending privacy
   refused) and must all appear in the manifest's ``rights_privacy_refs``;
4. ``load_train_pages`` reads the hash-verified split records with the
   store-derived ``bundle_source`` and the registry's lineages as exclusion;
5. the StageDataMatrix hash is computed from the validated frozen file.

Page keys are ``expectation.target_keys()`` (pinned order, ``k > 0`` included).
``RlPageAccessors.cache_row`` is a KEYED lookup: ``key -> index`` over that
order, and the row handed back must carry the key it was asked for -- a reader
that ignored its expectation fails there, not as a silently shifted page.

``make_read_selection`` re-reads the frozen selection from disk so that
``assert_not_reweighted`` compares the file with what is in memory.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from unflatten.model.rl.data import (
    PageKey,
    RlDataRefused,
    RlPageAccessors,
    load_train_pages,
)
from unflatten.model.rl.selection import FrozenSelection, selection_from_document

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _keyed_cache_row(reader: Any, keys: tuple[PageKey, ...]) -> Callable[[PageKey], Any]:
    index = {key: i for i, key in enumerate(keys)}
    if len(index) != len(keys):
        raise RlDataRefused("expectation.target_keys() holds a duplicate page key")

    def cache_row(key: PageKey) -> Any:
        i = index.get((key[0], key[1]))
        if i is None:
            raise RlDataRefused(f"page {key} is not in the pinned expectation (E \\ X)")
        row = reader[i]
        if (row.document_bundle_id, row.page_index) != (key[0], key[1]):
            raise RlDataRefused(
                f"cache row {i} is page ({row.document_bundle_id!r}, {row.page_index}), "
                f"asked for {key}: the reader does not address expectation.target_keys()"
            )
        return row

    return cache_row


def _matrix_hash(matrix: Path | str) -> str:
    """``str`` = an already-verified ``sha256:<64 hex>``; ``Path`` = the frozen
    ``stage_data_matrix.json``, validated and hashed here. Nothing else."""
    if isinstance(matrix, Path):
        from unflatten.model.stage_data_matrix import (
            StageDataMatrixInvalid,
            load_stage_data_matrix,
            stage_data_matrix_hash,
        )

        try:
            return stage_data_matrix_hash(load_stage_data_matrix(matrix))
        except StageDataMatrixInvalid as exc:
            raise RlDataRefused(f"stage data matrix {matrix} is refused: {exc}") from exc
    if isinstance(matrix, str) and _HASH_RE.match(matrix):
        return matrix
    raise RlDataRefused(
        f"stage data matrix must be a Path or a sha256:<64 hex> string, got {matrix!r}"
    )


def _require_expectation(cache_reader: Any, expectation: Any) -> tuple[PageKey, ...]:
    keys = tuple((b, k) for b, k in expectation.target_keys())
    pinned = getattr(cache_reader, "expectation", None)
    if pinned is None or tuple(tuple(p) for p in pinned.target_keys()) != keys:
        raise RlDataRefused(
            "cache reader's pinned expectation is not the injected expectation: its index "
            "would not address expectation.target_keys()"
        )
    if not keys:
        raise RlDataRefused("expectation.target_keys() is empty: no page to open")
    return keys


def _restrict_to_selected_pages(
    keys: tuple[PageKey, ...], selection: FrozenSelection
) -> tuple[PageKey, ...]:
    """Page-level selection: keep only the selected pages, in cache order. Lineage-level
    selections (no ``pages``) pass ``keys`` through untouched."""
    chosen = selection.selected_pages()
    if not chosen:
        return keys
    missing = sorted(chosen - {(b, int(k)) for b, k in keys})
    if missing:
        raise RlDataRefused(
            f"{len(missing)} selected page(s) are not in the cache expectation: {missing[:5]}"
        )
    return tuple(key for key in keys if (key[0], int(key[1])) in chosen)


def make_open_pages(
    release_dir: Path | str,
    store_dir: Path | str,
    cache_reader: Any,
    expectation: Any,
    exclusion_registry_doc: Mapping[str, Any],
    selection: FrozenSelection,
    matrix: Path | str,
    *,
    artefact_root: Path | str | None = None,
    decisions_dir: Path | str | None = None,
    gold_row: Callable[[PageKey], Any] | None = None,
    document_reference: Callable[[PageKey], Any] | None = None,
    page_image_path: Callable[[PageKey], Path] | None = None,
) -> Callable[[Mapping[str, Any]], Any]:
    """The ``open_pages`` port. ``matrix`` is a ``Path`` to the frozen
    ``stage_data_matrix.json`` (validated and hashed per call) or an
    already-verified ``sha256:<64 hex>`` (format-checked here, eagerly)."""
    if not isinstance(matrix, Path):
        _matrix_hash(matrix)
    release, store = Path(release_dir), Path(store_dir)

    def open_pages(dataset_manifest: Mapping[str, Any]) -> Any:
        from unflatten.dataset.samples import release_root_hash
        from unflatten.model.rl.adapters import exclusions as ex
        from unflatten.model.rl.adapters.admission import (
            AdmissionRefused,
            require_strict_admission,
            rights_privacy_refs,
        )
        from unflatten.model.rl.adapters.release import load_bundle_source
        from unflatten.model.rl.real_run_ports import OpenedPages

        ex.verify_exclusion_registry(
            exclusion_registry_doc,
            release_root_hash=release_root_hash(release),
            release_dir=release,
            artefact_root=artefact_root,
            decisions_dir=ex.DEFAULT_DECISIONS_DIR if decisions_dir is None else decisions_dir,
        )
        require_strict_admission(cache_reader)
        keys = _require_expectation(cache_reader, expectation)
        load_keys = _restrict_to_selected_pages(keys, selection)
        derived = rights_privacy_refs(release, cache_reader, dataset_manifest["lineage_ids"])
        missing = sorted(set(derived) - set(dataset_manifest["rights_privacy_refs"]))
        if missing:
            raise AdmissionRefused(
                f"dataset manifest rights_privacy_refs lack {len(missing)} derived ref(s): "
                f"{missing}"
            )
        train = load_train_pages(
            release,
            dataset_manifest,
            bundle_source=load_bundle_source(store),
            page_keys=load_keys,
            cached_page_keys=frozenset(load_keys),
            accessors=RlPageAccessors(
                cache_row=_keyed_cache_row(cache_reader, keys),
                gold_row=gold_row,
                document_reference=document_reference,
            ),
            page_image_path=page_image_path,
            registry_exclude_lineage_ids=ex.exclude_lineage_ids(exclusion_registry_doc),
        )
        return OpenedPages(
            train,
            selection,
            _matrix_hash(matrix),
            exclusion_registry_hash=exclusion_registry_doc[ex.HASH_FIELD],
        )

    return open_pages


def make_read_selection(path: Path | str) -> Callable[[], FrozenSelection]:
    """The ``read_selection`` port: the frozen selection as it is on disk now."""
    selection_path = Path(path)

    def read_selection() -> FrozenSelection:
        try:
            doc = json.loads(selection_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise RlDataRefused(f"frozen selection {selection_path} is unreadable: {exc}") from exc
        return selection_from_document(doc)

    return read_selection
