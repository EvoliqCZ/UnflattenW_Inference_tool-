"""Gold reference assembly for the S4 readiness gate (measurement only).

M-self-ref: the reference in :mod:`unflatten.eval.assembly.reference` may
import only ``unflatten.model.boundary_projector`` from the model layer —
never ``unflatten.assembly`` — so the reference stays independent of the
assembler it judges.
"""

from unflatten.eval.assembly.reference import (
    ATTRIBUTION_KEYS,
    EXCLUDED_KEYS,
    DocumentReference,
    ReferenceError,
    Seam,
    SeamLabel,
    build,
)

__all__ = [
    "ATTRIBUTION_KEYS",
    "DocumentReference",
    "EXCLUDED_KEYS",
    "ReferenceError",
    "Seam",
    "SeamLabel",
    "build",
]
