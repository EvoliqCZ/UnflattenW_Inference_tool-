"""RFC004 §11.6 input fingerprint — audit identity over the model-safe input."""

from __future__ import annotations

import hashlib
from typing import Any

from unflatten.model.hashing import domain_hash
from unflatten.model.input_contract import ModelSafeInput

_DOMAIN = "rfc004.input-fingerprint.v0"


def input_fingerprint(msi: ModelSafeInput) -> str:
    """``sha256:<64hex>`` over the canonical-CBOR projection of ``msi``.

    Audit identity ONLY — it binds logs, caches and eval records to exact
    decoded inputs. It is never a model feature: the fingerprint must not
    enter embeddings, the router, the decoder prompt or typed heads.

    The projection covers profile, input mode, declared physical scale,
    ordered plane geometry + raw pixel SHA-256s, aligned validity-mask
    hashes, the tile→page transforms, and the locale hint (or null).
    Process-local handles, filenames and audit metadata are excluded.
    """
    aligned = sorted(range(len(msi.planes)), key=lambda i: msi.planes[i].page_ordinal)
    planes = [msi.planes[i] for i in aligned]
    projection: dict[str, Any] = {
        "profile": msi.profile,
        "input_mode": msi.input_mode,
        "scale": (
            {"status": "known", "dpi": msi.scale.dpi}
            if msi.scale.status == "known"
            else {"status": "unknown"}
        ),
        "planes": [
            {
                "page_ordinal": plane.page_ordinal,
                "width_px": plane.width_px,
                "height_px": plane.height_px,
                "pixels_sha256": hashlib.sha256(plane.pixels).hexdigest(),
            }
            for plane in planes
        ],
        "mask_sha256s": [
            None
            if msi.masks[i].bitmap is None
            else hashlib.sha256(msi.masks[i].bitmap).hexdigest()
            for i in aligned
        ],
        "transforms": [
            {
                "xx": msi.transforms[i].xx,
                "xy": msi.transforms[i].xy,
                "yx": msi.transforms[i].yx,
                "yy": msi.transforms[i].yy,
                "tx": msi.transforms[i].tx,
                "ty": msi.transforms[i].ty,
                "denom": msi.transforms[i].denom,
            }
            for i in aligned
        ],
        "locale_hint": msi.locale_hint.value if msi.locale_hint is not None else None,
    }
    return domain_hash(_DOMAIN, projection)
