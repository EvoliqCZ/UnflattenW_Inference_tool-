"""RFC004 §15 / RFC004-ERRATA-005 optional visual compression boundary.

Two independent, separately-gated slots (RFC004-ERRATA-005 E6 rule 1 — neither
is a prerequisite for the other):

- ``visual_compressor`` (post-encoder, RFC004 §15 / ERRATA-005 E5): compresses
  already-encoded :class:`~unflatten.model.visual_compression.contracts.VisualMemory`.
- ``pre_encoder_compressor`` (RFC004-ERRATA-005 E4): plans per-region resolution
  before the page ever reaches the vision encoder.

Both default to ``bypass``, which is the product path and the control arm of
every ablation (§15.1, E4 rule 1).
"""
