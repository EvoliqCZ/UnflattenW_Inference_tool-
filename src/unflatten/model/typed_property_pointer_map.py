"""RFC004 §24/§26 (DEC-021) + RFC004-ERRATA-001 §26.9 — the closed
`(node_type, pointer)` typed-property surface, its schema-derived enum
domains, and the two *derived* structure surfaces a typed head needs.

Promoted from `diagnostics/typed_property_head/{pointer_map,enum_domain}.py`
and widened: 29 paragraph pointers on one head become **183** across six
heads, and the value types the diagnostic implicitly dropped are now
counted and recorded (:data:`DEFERRED`, 28) rather than vanishing.

Flat module, not a package: `test_m1_design_invariants.py` globs
`src/unflatten/model/*.py` **non-recursively**, so a package would be
invisible to the §11 inventory guard — the exact property the diagnostic
boundary exploited to stay outside it. Promotion means being seen.

Two things are *derived* here rather than enumerated, because a list
drifts from the schema silently and a derivation cannot:

**Repeated containers** (3) — group a node's emitted leaves by the prefix
before the first ``*`` segment of the normalized pointer. Deriving all
three is simpler than special-casing two and needs no list to maintain.

**Union containers** (8) — walk **each node type's property tree**
(following ``$ref``, descending ``properties`` and array ``items``,
cycle-guarded), never the ``$defs`` list: `table:/layout/width` is an
**inline** ``oneOf`` that no ``$defs`` scan reaches. A container is a
typed-property union iff

  (i) the resolved schema is a ``oneOf`` whose every arm is an object
      carrying a ``const``-valued property, **and**
  (ii) it has at least one leaf in the property head registry.

Clause (i) alone is wrong in both directions: it misses the inline
`table:/layout/width`, and it admits the 15 node-polymorphism
``/children/*`` containers, which carry **zero** typed leaves and are the
tree decoder's structural child dispatch, not properties. **Clause (ii) is
what excludes them** — and its cost, stated rather than discovered: it
makes a *schema* surface depend on the *registry*, so the union set moves
when `_HEAD_ASSIGNMENTS` moves and not only when the schema does. The
tests pin both clauses as membership so the two causes fail differently.

**Root set — a self-applying predicate, not a taste.** A node type is a
``$defs`` entry whose ``properties.type`` carries a ``const``: the same
discriminant the ``/children/*`` polymorphism dispatches on. It yields
**22** node types. ``$defs/anchor`` fails it and is deliberately not a
root — it is a shared property *contract*, not a node
(RFC004-ERRATA-001 E3), which is exactly why one head serves it on two
node types. Rooting the walk there as well adds `anchor:/{horizontal,
vertical,wrap}` and gives 26/8/18 instead of 23/8/15; that difference, and
nothing else, is why two careful derivations of this number disagreed
before the predicate was written down.

Torch-free. Mints no hash domain. **Must not derive the property head
registry hash** — neither by calling its producer nor by importing its
module constant: this module *consumes* an injected identity and has no
business deriving one. Enforced project-wide by
`test_policy_producers.py::TestHeadRegistryHashProducer`, which allows
exactly one derivation site in `src/` and it is not here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from unflatten.ast.schema_validator import _load_schema
from unflatten.dataset.targets.properties import PROPERTY_FIELDS, pointer_registry
from unflatten.model.property_head_registry import head_for_pointer, registry_projection

#: Value types a typed head predicts today.
_FAMILIES: frozenset[str] = frozenset({"integer", "enum", "boolean"})

#: Value types deliberately deferred, and *recorded* rather than dropped:
#: colour is RFC004 §25's own three-way design (string), and an object is
#: not a head's target at all. Task 8 stores these on the row too, so a
#: later §25 head does not need a rebuild to reach them.
_DEFERRED_TYPES: frozenset[str] = frozenset({"string", "object"})


class UnrecognisedValueTypeError(ValueError):
    """`pointer_registry()` returned a value type that is neither a
    predicted family nor a recorded deferral. Registry/schema drift, not a
    case to route around — raised rather than silently dropped."""


class UnknownEnumPointerError(ValueError):
    """`(node_type, pointer)` is not a registered enum pointer in the
    canonical AST schema — never a domain to guess."""


class UnionContainerInvalid(ValueError):
    """A `oneOf` passes clause (i) but has no single property that is
    `const`-valued in every arm with distinct values, so its arms cannot be
    named. Refused rather than defaulted to a positional index: an arm the
    consumer cannot name is an arm it cannot supervise."""


def _classify() -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    registry = pointer_registry()
    family: dict[tuple[str, str], str] = {}
    deferred: dict[tuple[str, str], str] = {}
    for entry in registry_projection()["mapped"]:
        key = (entry["node_type"], entry["node_property_pointer"])
        value_type = registry[key[0]][key[1]]
        if value_type in _FAMILIES:
            family[key] = value_type
        elif value_type in _DEFERRED_TYPES:
            deferred[key] = value_type
        else:
            raise UnrecognisedValueTypeError(
                f"{key!r} has value_type {value_type!r}, which is neither a "
                f"predicted family ({sorted(_FAMILIES)}) nor a recorded "
                f"deferral ({sorted(_DEFERRED_TYPES)})"
            )
    return family, deferred


#: `(node_type, pointer) -> value_type` for every assigned pointer a typed
#: head predicts. 155 = integer 79 + enum 57 + boolean 19 [measured].
FAMILY: dict[tuple[str, str], str]

#: The same, for the 28 pointers whose head is deferred (string 24,
#: object 4). Present so the deferral is *counted*, not implicit.
DEFERRED: dict[tuple[str, str], str]

FAMILY, DEFERRED = _classify()


def head_for_target(node_type: str, pointer: str) -> str:
    """The head that predicts `(node_type, pointer)`.

    Delegates to `property_head_registry.head_for_pointer` rather than
    re-deriving: two registries that silently drift apart is how a property
    gets predicted twice or never.
    """
    return head_for_pointer(node_type, pointer)


# ---------------------------------------------------------------------------
# Repeated containers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def repeated_containers() -> frozenset[tuple[str, str]]:
    """The array-valued containers whose element count is itself a typed
    target: `paragraph:/layout/tabs`, `section:/columns/widths_mpt`,
    `table:/columns` [measured].

    Derived uniformly — the prefix before the first ``*`` segment — rather
    than listed. `table:/columns` carries `table_plan/grid_column_count` as
    a *tensor identity* and not a stored target, so special-casing it would
    mean maintaining a list of two; deriving all three needs no list at
    all.
    """
    return frozenset(
        (node_type, pointer[: pointer.index("/*")])
        for node_type, pointer in (*FAMILY, *DEFERRED)
        if "/*" in pointer
    )


# ---------------------------------------------------------------------------
# Union containers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _defs() -> dict[str, Any]:
    return _load_schema()["$defs"]


def _resolve(subschema: Any) -> Any:
    """Follow `$ref` to a concrete subschema, guarding against a `$ref`
    cycle rather than recursing until the stack ends."""
    seen: set[str] = set()
    while isinstance(subschema, dict) and "$ref" in subschema:
        ref = subschema["$ref"].removeprefix("#/$defs/")
        if ref in seen:
            return subschema
        seen.add(ref)
        subschema = _defs()[ref]
    return subschema


def _arm_domain(subschema: dict[str, Any]) -> tuple[str, ...] | None:
    """Clause (i): the arm names of a `oneOf` whose every arm is an object
    carrying a `const`-valued property, or `None` if it is not one.

    The arm *name* is the discriminant's value. The discriminant is the one
    property that is `const` in every arm with values distinct across arms
    -- derived, because `table:/layout/width`'s `auto` arm carries a second
    `const` (`value_mpt: 0`) and a rule that took "the first const" would
    name that arm wrongly.
    """
    arms = subschema.get("oneOf")
    if not isinstance(arms, list) or not arms:
        return None
    per_arm: list[dict[str, Any]] = []
    for arm in arms:
        resolved = _resolve(arm)
        if not isinstance(resolved, dict):
            return None
        properties = resolved.get("properties")
        if not isinstance(properties, dict):
            return None
        consts = {
            name: value["const"]
            for name, value in properties.items()
            if isinstance(value, dict) and "const" in value
        }
        if not consts:
            return None
        per_arm.append(consts)
    shared = set(per_arm[0]).intersection(*(set(arm) for arm in per_arm))
    discriminants = [
        name
        for name in sorted(shared)
        if len({arm[name] for arm in per_arm}) == len(per_arm)
    ]
    if len(discriminants) != 1:
        raise UnionContainerInvalid(
            "a oneOf passing clause (i) has "
            f"{len(discriminants)} candidate discriminants ({discriminants}); "
            "its arms cannot be named, and a positional index is not a name"
        )
    return tuple(str(arm[discriminants[0]]) for arm in per_arm)


def _walk_node_type(node_type: str) -> dict[str, tuple[str, ...]]:
    """Clause-(i) containers reachable from one node type's property tree.

    Descends `properties` and array `items`, following `$ref`, and **stops
    at a union**: a clause-(i) container is recorded, not entered. Without
    that stop the walk re-enters the node polymorphism under
    `/children/*` and enumerates the whole document tree.
    """
    found: dict[str, tuple[str, ...]] = {}

    def descend(subschema: Any, pointer: str, seen: frozenset[int]) -> None:
        subschema = _resolve(subschema)
        if not isinstance(subschema, dict):
            return
        arms = _arm_domain(subschema)
        if arms is not None:
            found[pointer] = arms
            return
        if id(subschema) in seen:
            return
        seen = seen | {id(subschema)}
        if isinstance(subschema.get("oneOf"), list):
            for arm in subschema["oneOf"]:
                resolved = _resolve(arm)
                if isinstance(resolved, dict) and resolved.get("type") == "null":
                    continue
                descend(arm, pointer, seen)
            return
        if isinstance(subschema.get("properties"), dict):
            for name, prop in subschema["properties"].items():
                descend(prop, f"{pointer}/{name}", seen)
            return
        if "items" in subschema:
            descend(subschema["items"], f"{pointer}/*", seen)

    for name, prop in _defs()[node_type].get("properties", {}).items():
        descend(prop, f"/{name}", frozenset())
    return found


@lru_cache(maxsize=1)
def node_types() -> tuple[str, ...]:
    """The root set, by the self-applying predicate: a `$defs` entry whose
    `properties.type` carries a `const`. **22** [measured]; `anchor` is not
    among them (module docstring)."""
    return tuple(
        sorted(
            name
            for name, definition in _defs().items()
            if isinstance(definition, dict)
            and isinstance(definition.get("properties"), dict)
            and isinstance(definition["properties"].get("type"), dict)
            and "const" in definition["properties"]["type"]
        )
    )


def _predicate_roots() -> frozenset[str]:
    """Bound here rather than inside the functions below, which take a
    parameter of the same name."""
    return frozenset(node_types())


@lru_cache(maxsize=1)
def union_candidate_containers(
    *, node_types: frozenset[str] | None = None
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Clause (i) only — every `const`-armed `oneOf` reachable from the
    root set, **23** at the predicate root set [measured].

    Exposed separately from :func:`union_containers` so a change in the
    *schema* and a change in the *registry* fail as different assertions:
    clause-(i) membership moving is a schema cause, clause-(i) stable while
    the 8 move is a registry cause.
    """
    roots = _predicate_roots() if node_types is None else node_types
    return {
        (node_type, pointer): arms
        for node_type in sorted(roots)
        for pointer, arms in _walk_node_type(node_type).items()
    }


@lru_cache(maxsize=1)
def union_containers(
    *, node_types: frozenset[str] | None = None
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Clause (i) **and** clause (ii) — the **8** typed-property unions
    [measured], each mapped to its arm domain in schema order.

    26 assigned leaves in total, **20 of them on `typed_heads.anchor`** —
    the whole of the second-largest head. An earlier reading of this
    surface said "3 unions, including `row:/layout/height`"; `row/layout/
    height` is a flat object requiring both `mode` and `value_mpt` in all
    three modes and is genuinely not a union, but six anchor unions were
    missed entirely. Having a `mode` field is not the same as being a
    union.
    """
    assigned = (*FAMILY, *DEFERRED)
    return {
        key: arms
        for key, arms in union_candidate_containers(node_types=node_types).items()
        if any(
            node_type == key[0] and pointer.startswith(key[1] + "/")
            for node_type, pointer in assigned
        )
    }


# ---------------------------------------------------------------------------
# Enum domains -- from the schema, never from observed data
# ---------------------------------------------------------------------------


def enum_domain(node_type: str, pointer: str) -> tuple[str, ...]:
    """The full, schema-defined set of legal values for an enum pointer.

    Schema-derived on purpose: the diagnostic predecessor built an enum
    head's classifier from the FIRST batch's observed values and froze
    `num_classes` there, so the same data in a different batch order
    produced a different head, and a legal value not yet seen raised. The
    schema is the one source that cannot depend on order.

    `pointer` is the node-relative, normalized JSON Pointer (e.g.
    `"/layout/alignment"`). Returned in schema order; a caller needing a
    deterministic label index sorts it itself.

    The shared colour palette (`typed_head_config.COLOR_PALETTE_V0`/
    `palette_domain`, design `spec-typed-heads-missing-families` §5.1)
    respects this function's REASON, not its source: it is a pinned,
    sorted, hashed literal, not something re-derived from a batch's order.
    It must NOT be read through this function: `color`'s `oneOf` has one
    branch with a `const` (`"auto"`) and one bare `pattern` branch (an
    open hex string, never an `enum`), so `_collect_values` below silently
    registers a colour pointer with domain `("auto",)` -- not a raise, and
    not the real 733-hex palette either (MEASURED,
    `tests/model/test_typed_property_pointer_map.py::TestEnumDomain::
    test_enum_domain_never_carries_the_real_palette_for_a_color_pointer`).
    The palette domain is configuration, never schema, and never this
    function's return value.
    """
    domains = _enum_domains().get(node_type, {})
    if pointer not in domains:
        raise UnknownEnumPointerError(
            f"{(node_type, pointer)!r} is not a registered enum pointer in the "
            "canonical AST schema"
        )
    return domains[pointer]


@lru_cache(maxsize=1)
def _enum_domains() -> dict[str, dict[str, tuple[str, ...]]]:
    registry: dict[str, dict[str, tuple[str, ...]]] = {}
    for node_type, fields in PROPERTY_FIELDS.items():
        collected: dict[str, list[str]] = {}
        node_def = _defs()[node_type]
        for field in fields:
            _collect_values(node_def["properties"][field], f"/{field}", collected)
        registry[node_type] = {
            pointer: tuple(values) for pointer, values in collected.items()
        }
    return registry


def _collect_values(
    subschema: Any, pointer: str, out: dict[str, list[str]]
) -> None:
    """Mirrors `properties._walk_schema`'s traversal but records the legal
    values rather than the `"enum"` marker. `oneOf` branches are **unioned
    into the same pointer's domain**: `line/mode` is a `const` per branch,
    and its real domain is all four, not whichever branch is walked last.
    """
    subschema = _resolve(subschema)
    if not isinstance(subschema, dict):
        return
    if isinstance(subschema.get("properties"), dict):
        for name, prop in subschema["properties"].items():
            _collect_values(prop, f"{pointer}/{name}", out)
        return
    if "items" in subschema:
        _collect_values(subschema["items"], f"{pointer}/*", out)
        return
    if isinstance(subschema.get("oneOf"), list):
        for branch in subschema["oneOf"]:
            resolved = _resolve(branch)
            if isinstance(resolved, dict) and resolved.get("type") == "null":
                continue
            _collect_values(branch, pointer, out)
        return
    values = list(subschema.get("enum", ()))
    if "const" in subschema:
        values.append(subschema["const"])
    if not values:
        # A leaf with neither: not an enum pointer. Returning without
        # touching `out` matters -- a `setdefault` here would create an
        # empty domain, and a later `del` would discard the values a
        # sibling `oneOf` branch had already contributed to this pointer.
        return
    collected = out.setdefault(pointer, [])
    for value in values:
        if value not in collected:
            collected.append(value)
