"""Fragment identity (RFC003 §26): `fragment_` + H("rfc003.fragment/v1", ...).

LOCAL copy of the `unflatten.dataset.identity` hashing style (same
`H(domain, payload) = SHA-256(UTF-8(JCS([domain, payload])))` contract,
same `sha256:`/`<prefix>_<64hex>` shapes). Kept local per task scope so
`dataset/identity.py` stays untouched; NOTE for future consolidation:
fold `fragment_id` into `dataset/identity.py` alongside the other §12
ID constructors.

The identity payload binds, per RFC003 §26:

- the document: `fragment_schema_version`, `ast_hash`,
  `document_bundle_id`;
- the authorities the fragment is a view OF: `target_bundle_hash` and
  `observation_sidecar_hash` — without these the cache has nothing to
  cryptographically bind target and mask authority to;
- the producer: `builder` (`name`, `version`, `config_hash`, the last
  covering the window policy the builder ran under);
- the window: `window_policy`, `sequence_index`, ordered `page_indices`;
- the RENDER BINDING: the fragment's frozen page records
  (`page_index`/`width_px`/`height_px`) plus a domain hash over the
  sidecar's `renderer` block, so two renders of the same AST at
  different DPI never share fragment identities;
- the owned projections: domain hashes over `slices` and
  `control_event_ownership`.

`task_view_id` is in the payload but is always `None` today: the
pipeline materializes no TaskView artifacts, and binding a fabricated ID
would assert a relationship nobody can verify — the exact failure this
closure exists to remove. It is an explicit null rather than an omitted
key so that binding a real TaskView later necessarily changes the
identity. `lineage_root_id`, `render_view_id` and the
`fragment_policy_*` triple are absent for the same reason: no such
artifact exists to bind to. `fragment_policy_hash` in particular must
equal a Component Lock entry (§26.5), and there is no Component Lock.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from unflatten.ast.canonical_json import to_canonical_json
from unflatten.dataset.enums import WINDOW_POLICY

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_BUNDLE_ID_RE = re.compile(r"^bundle_[0-9a-f]{64}$")
_TASK_VIEW_ID_RE = re.compile(r"^task_[0-9a-f]{64}$")

#: Schema version of the envelope this identity is computed for; a
#: schema change is a derivation change, so it is identity-bearing.
FRAGMENT_SCHEMA_VERSION = "0.2.0"


def _domain_hash(domain: str, payload: Any) -> str:
    return "sha256:" + hashlib.sha256(to_canonical_json([domain, payload])).hexdigest()


def fragment_id(
    *,
    ast_hash: str,
    document_bundle_id: str | None,
    task_view_id: str | None,
    target_bundle_hash: str,
    observation_sidecar_hash: str,
    builder: dict[str, Any],
    window_policy: str,
    sequence_index: int,
    page_indices: list[int],
    pages: list[dict[str, Any]],
    renderer: dict[str, Any],
    slices: list[dict[str, Any]],
    control_event_ownership: list[dict[str, Any]],
) -> str:
    """`fragment_<64hex>` over the `rfc003.fragment/v1` domain."""
    if not isinstance(ast_hash, str) or not _SHA256_RE.fullmatch(ast_hash):
        raise ValueError(f"ast_hash must be 'sha256:<64 lowercase hex>', got {ast_hash!r}")
    if document_bundle_id is not None and not _BUNDLE_ID_RE.fullmatch(document_bundle_id):
        raise ValueError(
            f"document_bundle_id must be 'bundle_<64 lowercase hex>' or None, "
            f"got {document_bundle_id!r}"
        )
    if task_view_id is not None and not _TASK_VIEW_ID_RE.fullmatch(task_view_id):
        raise ValueError(
            f"task_view_id must be 'task_<64 lowercase hex>' or None, "
            f"got {task_view_id!r}"
        )
    for name, value in (
        ("target_bundle_hash", target_bundle_hash),
        ("observation_sidecar_hash", observation_sidecar_hash),
    ):
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise ValueError(f"{name} must be 'sha256:<64 lowercase hex>', got {value!r}")
    if set(builder) != {"name", "version", "config_hash"}:
        raise ValueError(
            f"builder must carry exactly name/version/config_hash, got {sorted(builder)}"
        )
    if window_policy not in WINDOW_POLICY:
        raise ValueError(f"unknown window_policy {window_policy!r}")
    if isinstance(sequence_index, bool) or not isinstance(sequence_index, int) or (
        sequence_index < 0
    ):
        raise ValueError(f"sequence_index must be a non-negative int, got {sequence_index!r}")
    digest = _domain_hash(
        "rfc003.fragment/v1",
        {
            "fragment_schema_version": FRAGMENT_SCHEMA_VERSION,
            "ast_hash": ast_hash,
            # identity closure: the document bundle, the authorities this
            # fragment is a view of, and the builder that produced it.
            # `task_view_id` is null until TaskView artifacts exist.
            "document_bundle_id": document_bundle_id,
            "task_view_id": task_view_id,
            "target_bundle_hash": target_bundle_hash,
            "observation_sidecar_hash": observation_sidecar_hash,
            "builder": dict(sorted(builder.items())),
            "window_policy": window_policy,
            "sequence_index": sequence_index,
            "page_indices": list(page_indices),
            # render binding: page pixel dims + renderer meta (D4 must_fix 1)
            "pages": [
                {
                    "page_index": page["page_index"],
                    "width_px": page["width_px"],
                    "height_px": page["height_px"],
                }
                for page in pages
            ],
            "renderer_hash": _domain_hash("rfc003.fragment-renderer/v1", renderer),
            "owned_projections": {
                "slices_hash": _domain_hash("rfc003.fragment-slices/v1", slices),
                "control_event_ownership_hash": _domain_hash(
                    "rfc003.fragment-controls/v1", control_event_ownership
                ),
            },
        },
    )
    return "fragment_" + digest.removeprefix("sha256:")
