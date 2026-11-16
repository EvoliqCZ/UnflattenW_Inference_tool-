"""Scaling curve: measure how much data the decoder actually needs.

The "how many pages do we need" question has been answered twice so far
by estimate — once at 500 000 pages from analogy with other fine-tuning
work, and once at ~3 000 documents from RFC003's coverage minima. Neither
is a measurement of *this* model on *this* task, and RFC005 is explicit
that sample count guarantees nothing.

A scaling curve answers it directly: train the same decoder on
increasing prefixes of the cache and watch where validation stops
improving. On cached features that is minutes per point, which makes it
far cheaper than acquiring data on a guess — and the answer decides
whether the next step is another 80 000 downloads or none at all.

Subsets are **prefixes, not samples**. Every larger run must contain
every smaller one, otherwise consecutive points differ by which documents
were drawn as well as by how many, and the curve measures sampling noise.
"""

from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


def subset_sizes(available: int, *, smallest: int = 2500) -> list[int]:
    """Doubling sizes from `smallest` up to `available`.

    Doubling rather than linear steps because saturation shows up on a
    log axis: four points spanning 8x say more than four spanning 2x.
    """
    if available <= smallest:
        return [available]
    sizes: list[int] = []
    size = smallest
    while size < available:
        sizes.append(size)
        size *= 2
    sizes.append(available)
    return sizes


class _Prefix(Dataset):
    """The first `n` items of `dataset`, in order."""

    def __init__(self, dataset: Dataset, n: int) -> None:
        self.dataset = dataset
        self.n = min(n, len(dataset))  # type: ignore[arg-type]
        # Shard geometry passes through: a prefix of a sharded cache is
        # still sharded, and a curve point that forgot it would read the
        # cache once per sample instead of once per epoch. Index 0..n-1
        # maps through unchanged, so the prefix property is untouched.
        self.shard_size = getattr(dataset, "shard_size", None)
        self.shard_window = getattr(dataset, "shard_window", None)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> Any:
        if index >= self.n:
            raise IndexError(index)
        return self.dataset[index]


def take_subset(dataset: Dataset, n: int) -> Dataset:
    """First `n` items, so each curve point contains the previous one."""
    return _Prefix(dataset, n)
