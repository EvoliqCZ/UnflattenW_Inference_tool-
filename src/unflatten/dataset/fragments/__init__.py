"""Page Fragment Envelope (RFC003 §26).

A fragment is a DERIVED view over the page ranges of one rendered
document — bound to the full Canonical AST, the Canonical Target Bundle
and the Observation Sidecar. It never adds a ``Page`` node to the AST
(RFC003 §26/§64.3): natural page transitions are continuation metadata
only, while an explicit ``page_break`` stays an AST node.

Like the target bundle, fragments are never hand-fixed independently of
their sources — fix the AST/sidecar and re-derive.
"""
