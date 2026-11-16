"""RFC006 §27.5/§33.5 — patch journal, compare-and-swap commit, and replay.

Simplified to an in-memory, single-process journal: no crash-recovery
storage layer exists in this package yet, so "crash before/after commit"
(§50.10) is modeled as "does `commit` observe a consistent head", not as
a real process-crash test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from unflatten.ast.hashing import ast_hash
from unflatten.refinement.patch_applier import apply_patch_set
from unflatten.refinement.patch_language import ASTPatchSet


class HeadConflictError(RuntimeError):
    """`commit`'s PatchSet does not start from the journal's current head
    (RFC006 §27 CommitReceipt compare-and-swap)."""


class TamperedJournalError(RuntimeError):
    """A journal entry's recorded hash chain does not match its
    PatchSet/candidate — the journal was edited outside `commit`."""


@dataclass(frozen=True)
class JournalEntry:
    seq: int
    patch_set: ASTPatchSet
    candidate_ast_hash: str
    receipt_hash: str


@dataclass
class PatchJournal:
    """Append-only sequence of committed patches, starting from a fixed
    genesis AST."""

    genesis_ast_hash: str
    entries: list[JournalEntry] = field(default_factory=list)

    @property
    def head_ast_hash(self) -> str:
        return self.entries[-1].candidate_ast_hash if self.entries else self.genesis_ast_hash

    def commit(self, current_ast: dict[str, Any], patch_set: ASTPatchSet) -> dict[str, Any]:
        """Apply `patch_set` to `current_ast` and append it, iff
        `current_ast`'s hash equals the journal's current head
        (compare-and-swap)."""
        current_hash = ast_hash(current_ast)
        if current_hash != self.head_ast_hash:
            raise HeadConflictError(
                f"journal head is {self.head_ast_hash}, commit was attempted against {current_hash}"
            )
        candidate, _ = apply_patch_set(current_ast, patch_set)
        candidate_hash = ast_hash(candidate)
        entry = JournalEntry(
            seq=len(self.entries),
            patch_set=patch_set,
            candidate_ast_hash=candidate_hash,
            receipt_hash=_receipt_hash(len(self.entries), patch_set, candidate_hash),
        )
        self.entries.append(entry)
        return candidate

    def replay(self, genesis_ast: dict[str, Any]) -> dict[str, Any]:
        """Reapply every committed entry from `genesis_ast` in order.

        Raises `HeadConflictError` if `genesis_ast`'s hash does not match
        `genesis_ast_hash`, and `TamperedJournalError` if any entry's
        recorded `receipt_hash` or `candidate_ast_hash` no longer matches
        what re-applying its PatchSet produces.
        """
        if ast_hash(genesis_ast) != self.genesis_ast_hash:
            raise HeadConflictError(
                f"replay genesis ast_hash {ast_hash(genesis_ast)} != "
                f"journal genesis {self.genesis_ast_hash}"
            )
        current = genesis_ast
        for entry in self.entries:
            expected_receipt = _receipt_hash(entry.seq, entry.patch_set, entry.candidate_ast_hash)
            if expected_receipt != entry.receipt_hash:
                raise TamperedJournalError(
                    f"entry {entry.seq} receipt_hash does not match its contents"
                )
            candidate, _ = apply_patch_set(current, entry.patch_set)
            if ast_hash(candidate) != entry.candidate_ast_hash:
                raise TamperedJournalError(
                    f"entry {entry.seq} replay produced {ast_hash(candidate)}, "
                    f"journal recorded {entry.candidate_ast_hash}"
                )
            current = candidate
        return current


def _receipt_hash(seq: int, patch_set: ASTPatchSet, candidate_ast_hash: str) -> str:
    payload = (
        b"rfc006.commit-receipt.v0\0"
        + str(seq).encode("ascii")
        + b"\0"
        + patch_set.patch_set_hash.encode("utf-8")
        + b"\0"
        + candidate_ast_hash.encode("utf-8")
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
