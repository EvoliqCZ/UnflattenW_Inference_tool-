"""RFC004 §46.1 Cross-capability gate (W2-2).

Exact product auto-accept requires all six conditions to hold; this module
is a single pure boolean function over the closed status vocabularies
already defined elsewhere (`capability_manifest.CAPABILITY_STATUS`,
`result_axes.AST_STATUS`/`RENDERER_STATUS`/`ACCEPTANCE_STATUS`). It performs
no I/O and changes nothing about `promotion.py`/`result_axes.py`; those
modules would call it, not the other way around, if this gate is wired
into a real acceptance decision -- that wiring is out of scope here.
"""

from __future__ import annotations

from unflatten.model.capability_manifest import CAPABILITY_STATUS
from unflatten.model.result_axes import ACCEPTANCE_STATUS, AST_STATUS, RENDERER_STATUS

#: §46.1's own closed renderer-capability vocabulary (RFC002's per-feature
#: capability, distinct from RFC004's model capability and from
#: `result_axes.RENDERER_STATUS`, which is the *run outcome*, not the
#: renderer's declared capability for the feature in question).
RENDERER_CAPABILITY = frozenset({"exact", "approximation", "unsupported"})


def cross_capability_gate(
    *,
    model_capability: str,
    profile_permits_feature: bool,
    renderer_capability: str,
    ast_status: str,
    renderer_status: str,
    acceptance_status: str,
) -> bool:
    """§46.1: exact product auto-accept requires every one of:

    ``model capability = strict AND RFC001 profile permits feature AND
    RFC002 renderer capability = exact AND ast.status = strict_valid AND
    renderer.status = render_ready AND acceptance.status = auto_accept``.

    Raises `ValueError` for any argument outside its closed vocabulary --
    an unknown status must never silently evaluate as "gate closed" via a
    plain `==` miss, since that is indistinguishable from a real refusal.
    """
    if model_capability not in CAPABILITY_STATUS:
        raise ValueError(f"model_capability must be one of {sorted(CAPABILITY_STATUS)}")
    if renderer_capability not in RENDERER_CAPABILITY:
        raise ValueError(f"renderer_capability must be one of {sorted(RENDERER_CAPABILITY)}")
    if ast_status not in AST_STATUS:
        raise ValueError(f"ast_status must be one of {sorted(AST_STATUS)}")
    if renderer_status not in RENDERER_STATUS:
        raise ValueError(f"renderer_status must be one of {sorted(RENDERER_STATUS)}")
    if acceptance_status not in ACCEPTANCE_STATUS:
        raise ValueError(f"acceptance_status must be one of {sorted(ACCEPTANCE_STATUS)}")

    return (
        model_capability == "strict"
        and profile_permits_feature
        and renderer_capability == "exact"
        and ast_status == "strict_valid"
        and renderer_status == "render_ready"
        and acceptance_status == "auto_accept"
    )
