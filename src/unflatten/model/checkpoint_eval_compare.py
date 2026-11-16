"""Paired comparison of checkpoints teacher-forced on the SAME pinned eval pages.

`s4_train --eval-checkpoints` scores each checkpoint page by page; this module turns
those per-page losses into a paired difference with its own standard error. Pairing is
by page: the eval pages are fixed, so the spread between checkpoints comes from the
weights snapshot, and an unpaired mean would hide a page-level shift.

Numbers only -- `distinguishable` says the 95 % interval of the mean difference excludes
zero; it never selects or promotes a checkpoint (RFC005 §12.2: teacher-forced loss is a
monitoring diagnostic, free-running decode is the selection signal).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_Z95 = 1.96


def paired_summary(
    base: Sequence[float | None], other: Sequence[float | None]
) -> dict[str, Any]:
    """`other - base` over the pages where BOTH have a value (a missing value is never
    encoded as zero). Negative `mean_diff` = `other` has the lower loss."""
    if len(base) != len(other):
        raise ValueError(f"length mismatch: {len(base)} != {len(other)}")
    pairs = [(a, b) for a, b in zip(base, other, strict=True) if a is not None and b is not None]
    n = len(pairs)
    if n == 0:
        return {
            "n": 0, "mean_base": None, "mean_other": None, "mean_diff": None,
            "sd_diff": None, "se_diff": None, "ci95": None,
            "wins": 0, "losses": 0, "ties": 0, "distinguishable": None,
        }
    diffs = [b - a for a, b in pairs]
    mean_diff = sum(diffs) / n
    sd = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    half = _Z95 * se
    return {
        "n": n,
        "mean_base": sum(a for a, _ in pairs) / n,
        "mean_other": sum(b for _, b in pairs) / n,
        "mean_diff": mean_diff,
        "sd_diff": sd,
        "se_diff": se,
        "ci95": [mean_diff - half, mean_diff + half],
        "wins": sum(1 for d in diffs if d < 0),
        "losses": sum(1 for d in diffs if d > 0),
        "ties": sum(1 for d in diffs if d == 0),
        "distinguishable": abs(mean_diff) > half if n > 1 else None,
    }


def build_report(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Each member: `{"name", "step", "pages": [page_id...], "per_page": [{"total",
    "families": {name: value|None}}...]}`. Every member after the first is compared
    to the first, on the total and on each family."""
    if len(members) < 2:
        raise ValueError("need at least two checkpoints to compare")
    names = [m["name"] for m in members]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate member names: {names}")
    pages = list(members[0]["pages"])
    for m in members[1:]:
        if list(m["pages"]) != pages:
            raise ValueError(
                f"{m['name']!r} was not scored on the same pages (in the same order) as "
                f"{members[0]['name']!r}"
            )

    def totals(m: Mapping[str, Any]) -> list[float | None]:
        return [row["total"] for row in m["per_page"]]

    def family(m: Mapping[str, Any], name: str) -> list[float | None]:
        return [row["families"].get(name) for row in m["per_page"]]

    base = members[0]
    family_names = sorted({f for m in members for row in m["per_page"] for f in row["families"]})
    comparisons: dict[str, Any] = {}
    for m in members[1:]:
        comparisons[m["name"]] = {
            "total": paired_summary(totals(base), totals(m)),
            "families": {
                f: paired_summary(family(base, f), family(m, f)) for f in family_names
            },
        }
    return {
        "baseline": base["name"],
        "n_pages": len(pages),
        "members": {
            m["name"]: {
                "step": m.get("step"),
                "mean_total": _mean(totals(m)),
                "mean_families": {f: _mean(family(m, f)) for f in family_names},
            }
            for m in members
        },
        "comparisons": comparisons,
    }


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
