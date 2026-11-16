"""Gold screen of a frozen selection (RFC005-ERRATA-010 strategy §8 (c), design P8).

Runs ``gold.make_build_gold`` over the pages of the selected lineages and records,
by name, every page whose gold chain does not end in a strict-valid AST. No model,
no GPU, no render: ``render_dir`` is ``None``, so LibreOffice never starts.

The chain flags are the reward policy's pinned set -- the same set the candidate
will take -- and are never a parameter of their own. A PROPOSED reward policy is
allowed (the screen runs before ratification) but must verify. Torch-free at import
time; the heavy parts sit behind ``gold``'s own lazy imports.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

from unflatten.model.rl.adapters import gold
from unflatten.model.rl.data import RlPage, RlVisualSource
from unflatten.model.rl.policies import verify_reward_policy

__all__ = ["GoldScreenRefused", "lineages_without_pass", "screen_selection", "screen_with_counters"]

TRAIN_SPLIT = "train"


class GoldScreenRefused(ValueError):
    """The screen cannot be run on what it was given (nothing to screen, unreadable world)."""


def selected_pages(
    expectation: Any, selection: Any, bundle_lineage: Mapping[str, str]
) -> list[RlPage]:
    """``expectation.target_keys()`` whose bundle belongs to a selected lineage, in pinned
    order. ``GoldBuilder`` reads only ``page_key``; the rest is the train page shape."""
    wanted = set(selection.lineage_ids())
    chosen = selection.selected_pages()  # empty = every page of the lineage
    return [
        RlPage(
            page_key=(bundle, int(k)),
            lineage_root_id=bundle_lineage[bundle],
            split_name=TRAIN_SPLIT,
            visual=RlVisualSource(kind="on_the_fly"),
        )
        for bundle, k in expectation.target_keys()
        if bundle_lineage.get(bundle) in wanted and (not chosen or (bundle, int(k)) in chosen)
    ]


def screen_with_counters(
    *,
    reader: Any,
    expectation: Any,
    selection: Any,
    reward_policy: Mapping[str, Any],
    vocabulary: Any,
    tokenizer: Any,
    store_dir: Path | str,
    bundle_lineage: Mapping[str, str],
    bundle_source: Mapping[str, str],
    fetch: Callable[[Path, Mapping[str, Any]], bytes] | None = None,
) -> tuple[gold.GoldScreen, Counter[str]]:
    """:func:`screen_selection` plus the builder's counters (``built``, one per refusal
    kind, ``typed_outside_body``)."""
    verify_reward_policy(reward_policy)
    pages = selected_pages(expectation, selection, bundle_lineage)
    if not pages:
        raise GoldScreenRefused(
            "no page of the cache belongs to a selected lineage: nothing to screen"
        )
    build = gold.make_build_gold(
        store_dir=store_dir,
        reader=reader,
        expectation=expectation,
        vocabulary=vocabulary,
        tokenizer=tokenizer,
        chain_flags=dict(reward_policy["chain_policy_flags"]),
        bundle_source=bundle_source,
        cells=gold.s4_cells_by_source(),
        fetch=fetch,
    )
    return gold.screen_pages(build, pages), build.counters


def screen_selection(**kwargs: Any) -> gold.GoldScreen:
    """Pages that build gold, and the named refusal of each that does not. Keyword
    arguments as :func:`screen_with_counters`."""
    return screen_with_counters(**kwargs)[0]


def lineages_without_pass(
    screen: gold.GoldScreen, bundle_lineage: Mapping[str, str], selected: Collection[str]
) -> list[str]:
    """Selected lineages with no page that passed -- the ones ``build-dataset-manifest``
    would refuse."""
    covered = {bundle_lineage[b] for b, _k in screen.passed if b in bundle_lineage}
    return sorted(set(selected) - covered)
