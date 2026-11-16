"""Deterministic leakage fingerprints + pinned policy (RFC003 §40–§41).

Per-document fingerprints, all derived from the canonical AST and the
resource manifest — integer/string only, JCS-serializable (the canonical
serializer rejects floats, so similarities and thresholds are integer
ppm):

- ``exact_text_hash`` — domain hash over the visible-text event stream
  (RFC003 §28 stream E), with RUN_TEXT events carrying their span text.
  Projection: ``[{"event", "text"?}]`` in stream order (node ids and
  span ids excluded — they are positional under canonical preorder ids
  and would not add content information).
- ``minhash`` — 64-slot MinHash signature over k=5 word shingles of the
  concatenated RUN_TEXT words. Permutation parameters derive from
  sha256 of a pinned domain string + permutation index (NO ``random``
  module anywhere): ``d = sha256(f"{PERMUTATION_DOMAIN}:{i}")``,
  ``a = d[:8] % (P-1) + 1``, ``b = d[8:16] % P`` with P = 2^61 - 1.
  An empty shingle set yields the sentinel signature ``[P] * 64``.
- ``structure_hash`` — domain hash over the node-type open/text/close
  event stream WITHOUT any text content (near-duplicate layout signal,
  §41.2; it is never a join edge by itself — see the allocator).
- ``resource_sha256s`` — sorted set of media hashes from the resource
  manifest (resource-sharing leakage signal, §41.1/§41.3).

`LeakageFingerprintPolicy` pins the parameters and thresholds; its JCS
bytes hash (`policy_hash`, domain-separated per §41.2) invalidates every
split when any parameter changes. Per the pinned thresholds a MinHash
jaccard estimate >= 0.85 (850000 ppm) is a near-duplicate join edge and
the ambiguous band [0.70, 0.85) takes the RFC `ambiguous_pair_action:
quarantine` path. Note (documented decision): "90 % shared words" under
k=5 shingling lands around jaccard 0.78, i.e. in the ambiguous band —
the join edge fires from roughly 95 % word overlap upward.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from unflatten.dataset.identity import domain_hash
from unflatten.dataset.targets.events import derive_event_streams

FINGERPRINT_SCHEMA_VERSION = "0.1.0"

SHINGLE_K = 5
NUM_PERMUTATIONS = 64
#: Pinned domain string seeding the MinHash permutations (never random).
PERMUTATION_DOMAIN = "rfc003.leakage-minhash-permutation/v1"

_PRIME = (1 << 61) - 1  # Mersenne prime 2^61 - 1
#: Signature slot value for an empty shingle set.
EMPTY_SLOT = _PRIME

EXACT_TEXT_DOMAIN = "rfc003.fingerprint.visible-text/v1"
STRUCTURE_DOMAIN = "rfc003.fingerprint.structure/v1"
POLICY_DOMAIN = "rfc003.leakage-fingerprint-policy/v1"


@dataclass(frozen=True)
class LeakageFingerprintPolicy:
    """Immutable leakage-fingerprint-policy (RFC003 §41.2), integer ppm."""

    policy_schema_version: str = "0.1.0"
    policy_id: str = "leakage-word-core-v0"
    shingle_k: int = SHINGLE_K
    num_permutations: int = NUM_PERMUTATIONS
    permutation_domain: str = PERMUTATION_DOMAIN
    #: minhash jaccard >= this ppm => near-duplicate join edge
    near_duplicate_threshold_ppm: int = 850_000
    #: [ambiguous_low, near_duplicate) ppm => ambiguous pair
    ambiguous_low_threshold_ppm: int = 700_000
    ambiguous_pair_action: str = "quarantine"

    def to_dict(self) -> dict[str, Any]:
        """JCS-safe policy projection — never contains `policy_hash`."""
        return {
            "leakage_fingerprint_policy_schema_version": self.policy_schema_version,
            "policy_id": self.policy_id,
            "shingle_k": self.shingle_k,
            "num_permutations": self.num_permutations,
            "permutation_domain": self.permutation_domain,
            "near_duplicate_threshold_ppm": self.near_duplicate_threshold_ppm,
            "ambiguous_low_threshold_ppm": self.ambiguous_low_threshold_ppm,
            "ambiguous_pair_action": self.ambiguous_pair_action,
        }

    def policy_hash(self) -> str:
        """`H(POLICY_DOMAIN, to_dict())` — the hash field excludes itself."""
        return domain_hash(POLICY_DOMAIN, self.to_dict())


DEFAULT_LEAKAGE_POLICY = LeakageFingerprintPolicy()


@lru_cache(maxsize=1)
def _permutations() -> tuple[tuple[int, int], ...]:
    params = []
    for index in range(NUM_PERMUTATIONS):
        digest = hashlib.sha256(f"{PERMUTATION_DOMAIN}:{index}".encode("ascii")).digest()
        a = int.from_bytes(digest[:8], "big") % (_PRIME - 1) + 1
        b = int.from_bytes(digest[8:16], "big") % _PRIME
        params.append((a, b))
    return tuple(params)


def _shingle_value(words: tuple[str, ...]) -> int:
    digest = hashlib.sha256("\x1f".join(words).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def word_shingles(words: tuple[str, ...], k: int = SHINGLE_K) -> frozenset[int]:
    """Hashed k-word shingles; fewer than k words form one shingle."""
    if not words:
        return frozenset()
    if len(words) < k:
        return frozenset({_shingle_value(words)})
    return frozenset(
        _shingle_value(words[i : i + k]) for i in range(len(words) - k + 1)
    )


def minhash_signature(shingles: frozenset[int]) -> list[int]:
    """64-slot MinHash signature; sentinel `[EMPTY_SLOT] * 64` when empty."""
    if not shingles:
        return [EMPTY_SLOT] * NUM_PERMUTATIONS
    return [min((a * v + b) % _PRIME for v in shingles) for a, b in _permutations()]


def minhash_similarity_ppm(sig_a: list[int], sig_b: list[int]) -> int:
    """Estimated jaccard similarity in integer ppm (matching slots / 64).

    A sentinel (empty-shingle) signature carries no text evidence, so any
    comparison involving one is 0 ppm — textless documents must never
    chain into one component through MinHash (exact_text / structure /
    resource edges still apply)."""
    if len(sig_a) != NUM_PERMUTATIONS or len(sig_b) != NUM_PERMUTATIONS:
        raise ValueError("minhash signatures must have exactly 64 slots")
    if EMPTY_SLOT in (sig_a[0], sig_b[0]):
        return 0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches * 1_000_000 // NUM_PERMUTATIONS


def build_fingerprint(
    canonical_ast: dict[str, Any], resource_sha256s: list[str]
) -> dict[str, Any]:
    """Deterministic leakage fingerprint record for one canonical AST."""
    streams = derive_event_streams(canonical_ast)
    spans = {s["text_span_id"]: s["text"] for s in streams["text"]["spans"]}

    text_events: list[dict[str, Any]] = []
    words: list[str] = []
    for event in streams["text"]["visible_text_events"]:
        if event["event"] == "RUN_TEXT":
            text = spans[event["text_span_id"]]
            text_events.append({"event": "RUN_TEXT", "text": text})
            words.extend(text.split())
        else:
            text_events.append({"event": event["event"]})

    type_by_open_index = {
        event["index"]: event["node_type"]
        for event in streams["structural"]["events"]
        if event["op"] == "open"
    }
    structure_events: list[str] = []
    for event in streams["structural"]["events"]:
        if event["op"] == "open":
            structure_events.append(f"open:{event['node_type']}")
        elif event["op"] == "text":
            structure_events.append("text")
        else:
            structure_events.append(f"close:{type_by_open_index[event['open_index']]}")

    shingles = word_shingles(tuple(words))
    return {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "exact_text_hash": domain_hash(EXACT_TEXT_DOMAIN, text_events),
        "structure_hash": domain_hash(STRUCTURE_DOMAIN, structure_events),
        "minhash": minhash_signature(shingles),
        "resource_sha256s": sorted(set(resource_sha256s)),
        "shingle_count": len(shingles),
    }
