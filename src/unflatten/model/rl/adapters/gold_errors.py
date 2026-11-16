"""Named refusals of the gold adapter (torch-free)."""

from __future__ import annotations

__all__ = [
    "GoldChainFailed",
    "GoldPageRefused",
    "GoldReferenceRefused",
    "GoldRefused",
    "GoldTypedMappingFailed",
]


class GoldRefused(ValueError):
    """The gold of a page cannot be produced truthfully. ``counter`` names it."""

    counter = "gold_refused"


class GoldChainFailed(GoldRefused):
    """The gold stream did not build a strict-valid AST (``BuilderReportInvalid``,
    an invalid page stream, a non-strict build): the page has no usable gold."""

    counter = "gold_chain_failed"


class GoldTypedMappingFailed(GoldRefused):
    """A typed target cannot be placed on a gold AST node (anchor or type mismatch,
    or the decoder body preorder is not the AST body preorder)."""

    counter = "gold_typed_mapping_failed"


class GoldReferenceRefused(GoldRefused):
    """The AST has content the reference cannot represent (duplicate id, ...)."""

    counter = "gold_reference_refused"


class GoldPageRefused(GoldRefused):
    """The page is not addressable: key outside the expectation, row mismatch,
    unresolved source / cell, a row that has no page image."""

    counter = "gold_page_refused"
