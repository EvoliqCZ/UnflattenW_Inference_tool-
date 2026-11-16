"""Sparse Word Expert Adapter -- RFC004 §22/§23, dormant per DEC-047.

Not imported by `candidate_model`, `expert_adapter.py`, or any construction
path. See `adapter.py` for the module docstring covering the five DEC-047
conditions this package satisfies.
"""

from unflatten.model.sparse_expert_adapter.adapter import (
    MissingExpertWeightError,
    RoutingResult,
    SparseExpertAdapter,
)
from unflatten.model.sparse_expert_adapter.manifest import (
    ExpertRole,
    SparseExpertAdapterManifest,
    SparseExpertAdapterManifestError,
)
from unflatten.model.sparse_expert_adapter.reasoning_integration import (
    DenseAdapterNotSupportedError,
    SparseExpertAdapterWithReasoning,
    SparseReasoningResult,
)

__all__ = [
    "DenseAdapterNotSupportedError",
    "ExpertRole",
    "MissingExpertWeightError",
    "RoutingResult",
    "SparseExpertAdapter",
    "SparseExpertAdapterManifest",
    "SparseExpertAdapterManifestError",
    "SparseExpertAdapterWithReasoning",
    "SparseReasoningResult",
]
