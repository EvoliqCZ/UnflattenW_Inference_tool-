"""RFC007 (PROPOSED) document-aware level router -- DEC-047 dormant code.

Variant B of `docs/rfc/RFC007-document-aware-deepstack-a-vizualne-smerovanie-v0.1.md`:
the multi-level visual features the v16 on-the-fly ViT produces (the final
merged tokens plus the three native DeepStack levels) are kept as SEPARATE
query-side memories, and a small module in our reconstruction branch reads
from them at every decode position, weighting the levels by the grammar
state (innermost open AST container, inside text or not, inside a table or
not). Its output is added to the dense-adapted hidden state, like
`DenseExpertAdapter`; the frozen backbone and native DeepStack are never
touched (DEC-056).

DEC-047 conditions, as this package meets them:

1. default off -- the architecture variant's `level_router` slot defaults to
   disabled, and every trainer/runtime switch defaults to absent;
2. step 0 == dense -- the reader's output projection is zero-initialised
   (`module.DocumentLevelRouter`);
3. no hashed identity moved -- nothing here imports the trainable-group
   registry, head registry, loss-unit schema or any v15b policy; the one
   manifest key and checkpoint key are written only when the router is on;
4. topology as policy fields -- levels, memory width, heads, grammar-state
   buckets, temperature and attach point are `policy.LevelRouterPolicy`
   fields;
5. no gradient -- every trainer refuses a real step without an on-disk
   ratification record (`ratification.require_level_router_ratification`).
"""
