"""RFC004 §15.2 / RFC004-ERRATA-005 E5 post-encoder VisualCompressor contract.

``VisualMemory`` here is the per-page bag of already-encoded visual tokens
that sits at the §15 boundary (between ``BackboneAdapter`` and the Word
Expert Adapter) — a plain data contract, distinct from
:class:`unflatten.model.visual_memory.VisualMemory`, which is the trainable
``nn.Module`` that projects those tokens into decoder cross-attention memory.
Naming follows the RFC; the two live in different modules and are never
imported together under the same name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class VisualMemory:
    """One batch of per-page visual tokens, RFC004 §15.2 compressor input.

    ``tokens`` is ``(batch, tokens, d_model)``. ``key_padding_mask`` is
    ``(batch, tokens)``, True == padding (matches
    :class:`unflatten.model.visual_memory.MemoryBatch`'s convention).
    ``page_ordinals`` is ``(batch,)`` and must stay monotonically consistent
    with the input contract's page order (§15.2 "page order").
    """

    tokens: torch.Tensor
    key_padding_mask: torch.Tensor
    page_ordinals: torch.Tensor

    def __post_init__(self) -> None:
        if self.tokens.dim() != 3:
            raise ValueError(
                f"tokens must be (batch, tokens, d_model), got {tuple(self.tokens.shape)}"
            )
        batch, n_tokens, _ = self.tokens.shape
        if tuple(self.key_padding_mask.shape) != (batch, n_tokens):
            raise ValueError(
                f"key_padding_mask must be {(batch, n_tokens)}, "
                f"got {tuple(self.key_padding_mask.shape)}"
            )
        if tuple(self.page_ordinals.shape) != (batch,):
            raise ValueError(
                f"page_ordinals must be ({batch},), got {tuple(self.page_ordinals.shape)}"
            )


@dataclass(frozen=True)
class BudgetReport:
    """RFC004-ERRATA-005 E5: compression ratio, token counts, bypass count, policy version."""

    original_token_count: int
    compressed_token_count: int
    bypassed_region_count: int
    policy_version: str

    def __post_init__(self) -> None:
        if self.original_token_count < 0 or self.compressed_token_count < 0:
            raise ValueError("token counts must be >= 0")
        if self.compressed_token_count > self.original_token_count:
            raise ValueError(
                "compressed_token_count "
                f"({self.compressed_token_count}) cannot exceed original_token_count "
                f"({self.original_token_count})"
            )
        if self.bypassed_region_count < 0:
            raise ValueError("bypassed_region_count must be >= 0")
        if not self.policy_version:
            raise ValueError("policy_version must be a non-empty string")

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count == 0:
            return 1.0
        if not self.compressed_token_count:
            return 0.0
        return self.original_token_count / self.compressed_token_count


@dataclass(frozen=True)
class CompressedVisualMemory:
    """RFC004-ERRATA-005 E5, the ``adaptive-document-visual-memory-v0`` profile."""

    coarse_memory: VisualMemory
    protected_detail: VisualMemory
    risk_map: torch.Tensor
    retention_map: torch.Tensor
    budget_report: BudgetReport


@dataclass(frozen=True)
class OriginalTokenAlignment:
    """``expand_alignment()`` output: compressed-token claims in original coordinates.

    ``compressed_index[i]`` maps to every original token index in
    ``original_indices[i]`` (a many-to-one merge, §15.2 "mapping na original
    tokens/regions"). Every original token index in ``[0, original_token_count)``
    must appear in exactly one group — no gaps, no orphans (RFC005 §21.2).
    """

    original_token_count: int
    compressed_index: tuple[int, ...]
    original_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.compressed_index) != len(self.original_indices):
            raise ValueError("compressed_index and original_indices must have the same length")
        seen: set[int] = set()
        for group in self.original_indices:
            for index in group:
                if index in seen:
                    raise ValueError(
                        f"original token {index} claimed by more than one compressed token"
                    )
                seen.add(index)
        if seen != set(range(self.original_token_count)):
            missing = sorted(set(range(self.original_token_count)) - seen)
            raise ValueError(f"original tokens missing from alignment (orphans): {missing}")


def retention_map_to_alignment(retention_map: torch.Tensor) -> OriginalTokenAlignment:
    """Expand a dense ``retention_map`` (§15.2 "retention_map") into groups.

    ``retention_map`` is 1D, ``retention_map[j] = i`` meaning original token
    ``j`` is retained/merged into compressed token ``i``. This is the shared
    implementation behind every :meth:`VisualCompressor.expand_alignment`.
    """
    if retention_map.dim() != 1:
        raise ValueError(f"retention_map must be 1D, got shape {tuple(retention_map.shape)}")
    original_token_count = retention_map.shape[0]
    groups: dict[int, list[int]] = {}
    for original_index, compressed_index_t in enumerate(retention_map.tolist()):
        compressed_index = int(compressed_index_t)
        groups.setdefault(compressed_index, []).append(original_index)
    ordered_keys = sorted(groups)
    return OriginalTokenAlignment(
        original_token_count=original_token_count,
        compressed_index=tuple(ordered_keys),
        original_indices=tuple(tuple(groups[key]) for key in ordered_keys),
    )


class VisualCompressor(Protocol):
    """RFC004 §15.2 protocol."""

    def compress(self, memory: VisualMemory) -> CompressedVisualMemory: ...

    def expand_alignment(self, retention_map: torch.Tensor) -> OriginalTokenAlignment: ...
