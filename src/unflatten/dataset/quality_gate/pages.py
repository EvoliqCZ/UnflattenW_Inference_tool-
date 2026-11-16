"""Where the gate reads pages from: the built exact-text cache (the rebuild
chain's input to training) or a gold-token extract of one (the audit's
`gold-tokens-<split>.jsonl`, same streams, used for local runs).

Both yield `GatePage` rows whose `tokens` are the page's packed ids
truncated to its loss units -- exactly the stream the model is trained on.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GatePage:
    bundle: str
    page: int
    tokens: list[int]
    #: typed / region channels, only when read from cache shards
    #: (G-C renders the typed targets; empty for a gold-token extract)
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def page_id(self) -> str:
        return f"{self.bundle}#{self.page}"


@dataclass(frozen=True)
class CacheShard:
    """A read-only work descriptor: deserialize pages in the scan worker."""

    path: Path
    with_extra: bool = False
    max_pages: int | None = None


def iter_gold_jsonl(path: Path) -> Iterator[GatePage]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tokens = row["packed_token_ids"]
            if len(tokens) != row["seq_len"]:
                raise ValueError(f"{row['document_bundle_id']}: seq_len != stream length")
            extra = {
                k: row[k]
                for k in (
                    "typed_targets",
                    "typed_cardinalities",
                    "region_targets",
                    "gallery_input_provenance",
                )
                if k in row
            }
            yield GatePage(row["document_bundle_id"], int(row["page_index"]), tokens, extra)


def cache_shards(cache_dir: Path) -> list[Path]:
    shards = sorted(Path(cache_dir).glob("shard-*.pt"))
    if not shards:
        raise FileNotFoundError(f"{cache_dir}: no shard-*.pt files")
    return shards


def iter_cache_shard(path: Path, *, with_extra: bool = False) -> Iterator[GatePage]:
    """Rows of one exact-text cache shard, memory-mapped (the audit's
    `origins` reader): tokens are cut to `len(loss_units)`."""
    import torch

    for data in torch.load(path, mmap=True, weights_only=False):
        n = len(data["loss_units"])
        extra = (
            {
                k: data[k]
                for k in (
                    "typed_targets",
                    "typed_cardinalities",
                    "region_targets",
                    "gallery_input_provenance",
                )
                if k in data
            }
            if with_extra
            else {}
        )
        yield GatePage(
            data["document_bundle_id"],
            int(data["page_index"]),
            data["packed_token_ids"][:n].tolist(),
            extra,
        )


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def skipped_keys(manifest: Mapping[str, Any]) -> set[tuple[str, int]]:
    """`(bundle, page)` of every page the cache build refused."""
    return {(s["document_bundle_id"], int(s["page_index"])) for s in manifest["page_skip_rows"]}
