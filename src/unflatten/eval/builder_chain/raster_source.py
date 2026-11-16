"""Fail-closed page-raster lookup for the builder-chain model arm.

The release artifact index, rather than the mutable render ledger, is the
authority for model-arm pixels.  This module deliberately returns bytes only
after both the release pin and the content-addressed store object verify.
"""

from __future__ import annotations

from pathlib import Path

from unflatten.dataset.gates.store import StoreError, read_object
from unflatten.dataset.samples import _pinned_rows, _release_manifest, _verify_ref


class PageRasterRefused(ValueError):
    """A model page cannot be tied to one pinned page raster."""


class PinnedRasterSource:
    """Resolve page PNGs exclusively through a release's pinned artifact index."""

    def __init__(
        self,
        release_dir: Path | str,
        store_dir: Path | str,
        *,
        expected_release_root_hash: str,
    ) -> None:
        self.release_dir = Path(release_dir)
        self.store_dir = Path(store_dir)
        manifest = _release_manifest(self.release_dir)
        self.release_root_hash = manifest.get("release_root_hash")
        self.require_decoded_input_provenance(expected_release_root_hash)
        self._pages: dict[tuple[str, int], dict] = {}
        #: Bundles the release indexes without a page list (never rendered):
        #: refused only when read, never for the whole release (24.9., r6 has 5).
        self._without_pages: set[str] = set()
        for row in _pinned_rows(self.release_dir, manifest, "artifact_index"):
            bundle_id = row.get("document_bundle_id")
            if not isinstance(bundle_id, str) or not bundle_id:
                raise PageRasterRefused("artifact index contains a missing document_bundle_id")
            artifacts = row.get("artifacts")
            pages = artifacts.get("pages") if isinstance(artifacts, dict) else None
            if not isinstance(pages, list):
                self._without_pages.add(bundle_id)
                continue
            for page_index, ref in enumerate(pages):
                key = (bundle_id, page_index)
                if key in self._pages:
                    raise PageRasterRefused(
                        f"{bundle_id}/pages/{page_index}: duplicate artifact pin"
                    )
                try:
                    self._pages[key] = _verify_ref(
                        self.store_dir, ref, f"{bundle_id}/pages/{page_index}"
                    )
                except Exception as exc:
                    raise PageRasterRefused(
                        f"{bundle_id}/pages/{page_index}: invalid pinned raster"
                    ) from exc

    def read(self, document_bundle_id: str, page_index: int) -> bytes:
        """Read one indexed raster, checking bundle, page, hash, and size."""
        if not isinstance(page_index, int) or isinstance(page_index, bool) or page_index < 0:
            raise PageRasterRefused(f"{document_bundle_id}: invalid page index {page_index!r}")
        if document_bundle_id in self._without_pages:
            raise PageRasterRefused(f"{document_bundle_id}: artifact index has no page list")
        ref = self._pages.get((document_bundle_id, page_index))
        if ref is None:
            raise PageRasterRefused(
                f"{document_bundle_id}/pages/{page_index}: no pinned raster in this release"
            )
        try:
            return read_object(self.store_dir, ref)
        except StoreError as exc:
            raise PageRasterRefused(
                f"{document_bundle_id}/pages/{page_index}: {exc.code}"
            ) from exc

    def require_decoded_input_provenance(self, expected_release_root_hash: object) -> None:
        """Refuse releases other than the one that produced decoded features."""
        if self.release_root_hash != expected_release_root_hash:
            raise PageRasterRefused(
                "raster release does not match decoded input provenance: "
                f"expected {expected_release_root_hash!r}, got {self.release_root_hash!r}"
            )
