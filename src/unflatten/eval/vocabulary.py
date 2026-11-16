"""Measured vocabulary of `/layout/spacing/after_mpt` over a named sample.

Design `categorical-word-properties` §1 reports the vocabulary as a
*measurement* (register entry M1, `release-v6`), not as a fixed list, and
§1.2 is explicit that the mechanism is corpus-independent while the
vocabulary is corpus-bound. This module therefore never ships a
vocabulary: it derives one from documents the caller names, and records
the sample size alongside it so the number can never be cited without
its provenance.

The rule is "every distinct value observed", which is a count, not a
cutoff. No coverage threshold is applied anywhere here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

POINTER = "/layout/spacing/after_mpt"


def spacing_after_values(ast: dict[str, Any]) -> Iterator[int]:
    """Yield `layout.spacing.after_mpt` of every paragraph in `ast`.

    Paragraphs nested in table cells count: the property is a paragraph
    property wherever the paragraph sits.
    """
    for node in walk_nodes(ast):
        if node.get("type") != "paragraph":
            continue
        value = node.get("layout", {}).get("spacing", {}).get("after_mpt")
        if isinstance(value, int):
            yield value


def walk_nodes(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every dict in `node`, preorder (AST nodes and their subobjects)."""
    if isinstance(node, dict):
        yield node
        for child in node.values():
            yield from walk_nodes(child)
    elif isinstance(node, list):
        for child in node:
            yield from walk_nodes(child)


@dataclass(frozen=True)
class ValueStat:
    """One observed value, with both counts RFC003 §42 distinguishes."""

    value_mpt: int
    node_count: int
    document_count: int
    share: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_mpt": self.value_mpt,
            "node_count": self.node_count,
            "document_count": self.document_count,
            "share": self.share,
        }


@dataclass(frozen=True)
class VocabularyMeasurement:
    """The vocabulary observed in one sample, with the sample recorded."""

    pointer: str
    document_count: int
    node_count: int
    values: tuple[ValueStat, ...]

    def stat(self, value_mpt: int) -> ValueStat:
        for stat in self.values:
            if stat.value_mpt == value_mpt:
                return stat
        raise KeyError(f"{value_mpt} was not observed in this sample")

    def coverage_of_top(self, n: int) -> float:
        """Share of nodes carried by the `n` most frequent values."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return sum(stat.share for stat in self.values[:n])

    def vocabulary_mpt(self) -> tuple[int, ...]:
        """Observed values, ascending — the input to a cost matrix."""
        return tuple(sorted(stat.value_mpt for stat in self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pointer": self.pointer,
            "document_count": self.document_count,
            "node_count": self.node_count,
            "distinct_values": len(self.values),
            # the design reports "podiel top-8" next to the distinct
            # count (§1.0, M1 and M5): a headline number must be
            # readable off the tool's own output, not recomputed by hand
            "coverage_of_top_8": self.coverage_of_top(8),
            "vocabulary_mpt": list(self.vocabulary_mpt()),
            "values": [stat.to_dict() for stat in self.values],
        }


def measure_vocabulary(
    documents: Iterable[tuple[str, dict[str, Any]]],
) -> VocabularyMeasurement:
    """Measure the vocabulary over `(document_id, ast)` pairs.

    Raises `ValueError` when the sample carries no paragraph with the
    property: an empty vocabulary is a broken sample, never a result.
    """
    node_counts: Counter[int] = Counter()
    doc_ids: defaultdict[int, set[str]] = defaultdict(set)
    documents_seen: set[str] = set()
    for document_id, ast in documents:
        documents_seen.add(document_id)
        for value in spacing_after_values(ast):
            node_counts[value] += 1
            doc_ids[value].add(document_id)
    total = sum(node_counts.values())
    if total == 0:
        raise ValueError(f"sample contains no paragraph carrying {POINTER}")
    stats = tuple(
        ValueStat(
            value_mpt=value,
            node_count=count,
            document_count=len(doc_ids[value]),
            share=count / total,
        )
        # ties broken by value so the ordering is deterministic
        for value, count in sorted(node_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    return VocabularyMeasurement(
        pointer=POINTER,
        document_count=len(documents_seen),
        node_count=total,
        values=stats,
    )
