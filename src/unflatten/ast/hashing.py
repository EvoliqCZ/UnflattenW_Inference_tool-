"""Canonical AST hash (RFC001 §35.2, ratified by ERRATA-001 E1).

Implements the `rfc001-hash-projection-v0` algorithm:
1. deep-copy the document,
2. remove the root `metadata` key,
3. serialize via canonical JSON (RFC001 §35.1 — the no-float JCS subset),
4. `ast_hash = sha256:<64 lowercase hex>` of those bytes.

Profile extensions and this projection (DEC-003, measured 2026-09-03)
--------------------------------------------------------------------
`profile` is INSIDE the projection — only `metadata` is removed — so the
profile string alone changes the hash of every document. Measured over
18 documents (the 13 P0 corpus fixtures, `minimal_valid.json` and the 4
pinned vectors in `tests/data/hash_vectors.json`):

| variant                                            | `ast_hash` moved |
|----------------------------------------------------|-----------------|
| `word-visual-v1` keys optional, omitted when empty  | **0 / 18**      |
| the same keys required with empty arrays            | **18 / 18**     |
| `profile` flipped to `"word-visual-v1"`             | **18 / 18**     |

The third row is why DEC-003 §3's original reading — "optional keys cost
65-70 % of `document_bundle_id` instead of 100 %" — does NOT hold: once a
regenerated store labels its documents `word-visual-v1`, 100 % of hashes
move whatever the key policy is. Optional-when-empty is therefore chosen
for a different reason: it makes the profile extension provably ADDITIVE,
so every pinned golden for a document without the new nodes still passes
and stays a regression net through F1-F7. Do not re-derive the identity
saving; it is not there.

`ast_hash` (semantic identity of the document content) and
`ast_file_hash` (raw SHA-256 of a file's bytes on disk) are two distinct
fields per ERRATA-001 E1 and MUST NEVER be assumed equal — an unchanged
document re-serialized with different whitespace/key order has a stable
`ast_hash` but a different `ast_file_hash`.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json


def ast_hash(doc: dict[str, Any]) -> str:
    """Compute the semantic `ast_hash` of a Canonical AST document.

    Excludes the root `metadata` key per ERRATA-001 E1; does not mutate
    `doc`.
    """
    projected = copy.deepcopy(doc)
    projected.pop("metadata", None)
    payload = to_canonical_json(projected)
    digest = hashlib.sha256(payload).hexdigest()
    return f"sha256:{digest}"


def ast_file_hash(data: bytes) -> str:
    """Compute the raw SHA-256 of file bytes. Distinct from `ast_hash`."""
    digest = hashlib.sha256(data).hexdigest()
    return f"sha256:{digest}"
