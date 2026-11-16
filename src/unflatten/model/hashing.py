"""RFC004 §3.1 domain-separated semantic hashing.

``domain_hash(domain, payload) = SHA-256(UTF-8(domain) || b"\\x00" || CBOR(payload))``
where ``CBOR`` is the deterministic ``rfc8949-core-det-nfc-int-v0`` profile
(:mod:`unflatten.model.cbor`). The domain prefix is the *exact* UTF-8 byte
sequence of the domain string including the trailing NUL separator; the
domain itself is not NFC-normalized and must not contain an embedded NUL
(which would make the framing ambiguous).

Serializations mirror the RFC003 identity layer:
- hash fields: ``sha256:<64 lowercase hex>``
- IDs: ``<prefix>_<64 lowercase hex>`` (same digest, no ``sha256:``)

Domain separation note: RFC004-native identities hash the §3.1 CBOR bytes;
RFC003 dataset identities (:mod:`unflatten.dataset.identity`) hash RFC 8785
JCS bytes. The two serialization worlds produce different digests for the
same logical payload by construction and must never be mixed.

``make_release_id``-style composite helpers beyond
:func:`unflatten.model.manifests.model_release_id` are deferred to later
A-phases.
"""

from __future__ import annotations

import hashlib
from typing import Any

from unflatten.model.cbor import encode


def domain_hash(domain: str, payload: Any) -> str:
    """``H(domain, payload)`` serialized as ``sha256:<64 lowercase hex>``."""
    return f"sha256:{_digest_hex(domain, payload)}"


def make_id(prefix: str, domain: str, payload: Any) -> str:
    """``<prefix>_<64hex>`` from the same digest as :func:`domain_hash`."""
    return f"{prefix}_{_digest_hex(domain, payload)}"


def _digest_hex(domain: str, payload: Any) -> str:
    if not isinstance(domain, str) or not domain:
        raise ValueError(f"domain must be a non-empty string, got {domain!r}")
    if "\x00" in domain:
        raise ValueError(f"domain must not contain an embedded NUL, got {domain!r}")
    data = domain.encode("utf-8") + b"\x00" + encode(payload)
    return hashlib.sha256(data).hexdigest()
