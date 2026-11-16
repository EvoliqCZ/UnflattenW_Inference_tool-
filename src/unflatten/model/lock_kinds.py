"""Component Lock kinds for the remaining RFC004 §39.2 components (W2-3).

`schemas/component_lock.schema.json` already carries the kinds
`reasoning_module`/`reasoning_gate`/`pre_encoder_compressor`/
`visual_compressor` (RFC004-ERRATA-005 E3, `unflatten.model.reasoning.lock`).
This module adds six more §39.2-named component kinds that had no kind
value at all yet: the node visual aligner, the fragment assembler and its
policy, an opt-in chain policy (`headless-defaults.v0` and similar,
RFC004-ERRATA-003/004), the confidence calibrator, the speculative decoder,
and the acceptance policy.

As in `reasoning/lock.py`: adding enum members to an already-emitted schema
file does not change the identity of any existing lock -- a lock's
`component_lock_hash` hashes the canonical JSON of the lock *document*,
never the schema file's bytes. A lock that used none of these six kinds
before this change still canonicalizes to the same bytes after it, so
existing runs and their pinned hashes are untouched; only a NEW run's lock
document may now declare one of these kinds.
"""

from __future__ import annotations

#: The six kinds this module adds to the closed `component.kind` enum.
NEW_LOCK_KINDS = frozenset(
    {
        "node_visual_aligner",
        "assembler_policy",
        "chain_policy",
        "confidence_calibrator",
        "speculative_decoder",
        "acceptance_policy",
    }
)
