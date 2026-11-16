"""RFC003 structural stream -> RFC004 token stream bridge (§18.3).

``encode_structural`` deterministically maps a Canonical Target Bundle
structural stream (``unflatten.dataset.targets`` format, ops
``open``/``text``/``close``) to the decoder event alphabet — the exact
inverse of the §18.3 ``StructuralEventProjector`` table:

- container ``open``/``close`` -> ``<TYPE>_OPEN`` / ``<TYPE>_CLOSE``,
- leaf-control ``open`` + immediate ``close`` -> one leaf token,
- ``text`` -> ``TEXT_BEGIN``, ``TEXT_END``,
- final ``END_DOCUMENT`` termination claim (owns no structural event).

This is the proof bridge that dataset targets are grammar-legal: the
replay test steps every corpus bundle's token stream through the
Grammar Controller with zero violations. Node types with no token at all
(``word-extended-v2``: text boxes, shapes, equations) raise
``ValueError``. Whether a node type that HAS a token may appear is the
Grammar Controller's profile decision, not this projection's — the
replay above is what enforces it.
"""

from __future__ import annotations

from typing import Any

from unflatten.model.events_vocab import (
    CLOSE_TOKEN,
    END_DOCUMENT,
    LEAF_TOKEN,
    OPEN_TOKEN,
    TEXT_BEGIN,
    TEXT_END,
)


def encode_structural(structural: dict[str, Any]) -> list[str]:
    """Map a target-bundle structural stream to V0 decoder tokens."""
    tokens: list[str] = []
    leaf_open_indices: set[int] = set()
    node_type_by_open_index: dict[int, str] = {}

    for event in structural["events"]:
        op = event["op"]
        if op == "open":
            node_type = event["node_type"]
            node_type_by_open_index[event["index"]] = node_type
            if node_type in LEAF_TOKEN:
                leaf_open_indices.add(event["index"])
                tokens.append(LEAF_TOKEN[node_type])
            elif node_type in OPEN_TOKEN:
                tokens.append(OPEN_TOKEN[node_type])
            else:
                raise ValueError(
                    f"node type {node_type!r} has no token in the event "
                    f"alphabet (event index {event['index']})"
                )
        elif op == "text":
            tokens.append(TEXT_BEGIN)
            tokens.append(TEXT_END)
        elif op == "close":
            open_index = event["open_index"]
            if open_index in leaf_open_indices:
                continue  # leaf controls are single tokens
            tokens.append(CLOSE_TOKEN[node_type_by_open_index[open_index]])
        else:
            raise ValueError(f"unknown structural op {op!r}")

    tokens.append(END_DOCUMENT)
    return tokens
