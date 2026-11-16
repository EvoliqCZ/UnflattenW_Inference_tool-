"""Assembly policy identity (assembler design §3, §10).

Skeleton for T1: the rule classes and the policy hash. Rule parameters
join the hashed payload as T2/T6 define them; T6 freezes version 1.0.0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from unflatten.assembly.hypothesis import RuleView
from unflatten.model.hashing import domain_hash

_POLICY_DOMAIN = "unflatten.assembly-policy.v0"

#: Unfrozen until T6 (design §10 M1).
ASSEMBLY_POLICY_VERSION = "0.0.0"

#: One id per rule class (design §5), in the G-A1 reporting order.
RULE_IDS: tuple[str, ...] = ("SEC", "PARA", "TABLE", "ROW", "LIST", "ITEM", "RUN", "STORY")


#: Join keys per rule (design §5): a pointer matching a pattern that is
#: KNOWN on both pieces and differs forbids the join. `*` stands for one
#: index segment; a trailing `/*` matches the whole subtree. `RUN` keys
#: on every run pointer (the canonicalizer's adjacency rule, §5.4).
JOIN_KEYS: dict[str, tuple[str, ...]] = {
    "SEC": ("/page/*", "/margins/*", "/columns/count", "/columns/equal_width"),
    "PARA": (
        "/layout/alignment",
        "/layout/direction",
        "/layout/indent/left_mpt",
        "/layout/indent/right_mpt",
        "/layout/spacing/line/*",
        "/layout/tabs/*",
        "/style/*",
    ),
    "TABLE": ("#card:/columns", "/columns/*/width_mpt", "/layout/*", "/style/*"),
    "LIST": ("/list_kind", "/marker/format", "/marker/suffix", "/layout/*"),
    "ROW": (),
    "ITEM": (),
    "RUN": ("*",),
}

#: `SEC` positive predicate: same rendered page size, within this many px.
PAGE_SIZE_TOLERANCE_PX = 2

type SeamPredicate = Callable[[RuleView, RuleView], bool]


@dataclass(frozen=True, slots=True)
class JoinPolicy:
    """Which rule classes may join at all, and the positive predicate of
    the two variant rules (design §5.3/§5.5). A `None` predicate is
    variant A, never join. `ITEM` reuses the `PARA` predicate (§5.6)."""

    name: str
    enabled: frozenset[str]
    para: SeamPredicate | None = None
    row: SeamPredicate | None = None

    def __post_init__(self) -> None:
        unknown = sorted(self.enabled - set(RULE_IDS))
        if unknown:
            raise ValueError(f"JoinPolicy {self.name!r}: unknown rule ids {unknown}")
        for rule, predicate in (("PARA", self.para), ("ROW", self.row)):
            if rule in self.enabled and predicate is None:
                raise ValueError(f"JoinPolicy {self.name!r}: {rule} enabled without a predicate")


#: B0, "adjacency only" (design §10): all variant A.
POLICY_B0 = JoinPolicy(name="B0", enabled=frozenset({"SEC", "TABLE", "LIST", "ITEM", "RUN"}))

#: B00, "never join anything".
POLICY_B00 = JoinPolicy(name="B00", enabled=frozenset())


def assembly_policy_hash() -> str:
    return domain_hash(
        _POLICY_DOMAIN,
        {
            "version": ASSEMBLY_POLICY_VERSION,
            "rule_ids": list(RULE_IDS),
            "join_keys": {rule: list(keys) for rule, keys in JOIN_KEYS.items()},
            "page_size_tolerance_px": PAGE_SIZE_TOLERANCE_PX,
        },
    )
