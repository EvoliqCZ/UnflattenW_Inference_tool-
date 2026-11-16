"""Stage-0 probe labels: does the frozen backbone carry document signal?

The Stage-0 probe asks four questions of a frozen vision backbone, and
these labels are how each is scored:

- **structure** — how many paragraphs and runs are on this page?
- **table** — is there a table on it?
- **style** — what is the typical font size, and are runs bolded?
- **spacing** — what is the typical paragraph spacing?

If small linear heads on frozen features cannot beat a constant baseline
on these, the backbone does not encode document structure and the answer
is to move the feature tap, unfreeze partially, or swap the backbone —
not to train a bigger decoder on top of it.

Two rules the extraction follows, both load-bearing:

- Labels are **page-local**. The target bundle is document-level, so the
  fragment's slices decide which nodes count; a table on page 3 must not
  label page 1.
- A missing value is `None`, **never** a numeric zero (RFC005: every loss
  is masked by target availability). A page with no runs has no font
  size — it does not have font size zero. The same rule governs the
  presence and count labels: a fragment that owns no structural event is
  a page whose nodes the observation geometry could not place, not a
  blank page, so `has_table`/`paragraph_count`/`run_count` are `None`
  there rather than `False`/`0`. The same holds when the page owns its
  events but supervises none of them (every slice `unresolved`): the
  predicate is supervision, never ownership.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

#: RFC001 pointer for paragraph spacing-after. Named rather than inlined
#: because getting it wrong is silent: an unknown pointer simply matches
#: nothing and the signal reports "no labelled examples" instead of an
#: error. The first version of this module used `/layout/space_after_mpt`,
#: which does not exist, and the head was dead until a real run showed
#: zero labels.
PARAGRAPH_SPACING_POINTER = "/layout/spacing/after_mpt"

#: Version of the LABEL DERIVATION, not of the label names. Bump it
#: whenever what a label means changes — the withdrawal rules, the
#: supervision predicate, the pointer set. It is recorded in the probe's
#: feature cache identity, because the trap this closes is a cache of
#: labels computed under the previous semantics being reused silently
#: under the same flags. 0.2.0: presence and count labels are `None` on a
#: page that owns no supervised structural event (previously `False`/`0`).
#: 0.3.0: font size and spacing are the page's MODAL value, not its mean
#: (and are named `modal_*`); a tie leaves the page unlabelled.
LABELS_VERSION = "0.3.0"

#: Names of every label this module produces, in report order.
PROBE_LABEL_NAMES: tuple[str, ...] = (
    "has_table",
    "has_list",
    "has_image",
    "paragraph_count",
    "run_count",
    "bold_run_ratio",
    "modal_font_size_mpt",
    "modal_space_after_mpt",
)


@dataclass(frozen=True)
class ProbeLabels:
    """Per-page probe targets. `None` means "not labelled for this page"."""

    has_table: bool | None
    has_list: bool | None
    has_image: bool | None
    paragraph_count: int | None
    run_count: int | None
    bold_run_ratio: float | None
    modal_font_size_mpt: float | None
    modal_space_after_mpt: float | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _known_values(
    typed: Mapping[str, Any], node_ids: Iterable[str], pointer: str
) -> list[Any]:
    """Values of `pointer` across `node_ids`, skipping fields that are
    unknown or whose supervision was WITHDRAWN.

    `known_mask` and `supervision_mask` answer different questions:
    "we know what the document said" versus "this is safe to learn
    from". Reading only the first made both mask projections inert here —
    a run whose requested face the pack substituted (`observable_mask:
    false`) was labelled with the face the pixels never showed, and a
    node on an imputed page (RFC003 §29.3) would be labelled the same
    way. An explicit `supervision_mask: false` therefore excludes the
    field.

    An ABSENT `supervision_mask` is undetermined, not false: V0 assesses
    observability for `/style/font_family` only, so requiring `true`
    would discard every label this probe reads. Same predicate as
    `model.build_cache`.
    """
    values: list[Any] = []
    for node_id in node_ids:
        entry = typed.get(node_id)
        if not entry:
            continue
        for field in entry.get("fields", ()):
            if field.get("node_property_pointer") != pointer:
                continue
            if not field.get("known_mask") or field.get("value") is None:
                continue
            if field.get("supervision_mask") is False:
                continue
            values.append(field["value"])
    return values


def fragment_owns_supervised_structure(fragment: Mapping[str, Any]) -> bool:
    """Does this fragment carry any SUPERVISED structural event?

    Ownership is not supervision (RFC003 §29.3, `fragments.envelope`).
    A page whose slices are all `unresolved` still owns its events —
    the event partition has to be complete — but supervises none of
    them, and counting its nodes would label a placement nobody derived.
    A page that owns nothing at all could not be placed by the geometry
    in the first place. Either way the structure labels are missing, not
    zero: labelling them `no table, no paragraphs` would teach the probe
    that a full page of text is empty.

    Delegates to :func:`~unflatten.dataset.fragments.envelope.supervised_event_ranges`
    so the predicate cannot drift from the contract it enforces.
    """
    from unflatten.dataset.fragments.envelope import supervised_event_ranges

    return bool(supervised_event_ranges(dict(fragment)))


def _mode(values: list[Any]) -> float | None:
    """The page's most frequent value, or `None` when there is no unique one.

    These properties are CATEGORICAL with a short vocabulary — 25 distinct
    `font_size_mpt` and 26 distinct `spacing/after_mpt` values over the
    corpus, both quantized onto Word's half-point lattice
    (`renderer/units.py`). The mean this replaced was not a worse estimate
    of the page's typical value, it was a value that exists nowhere: a page
    of 11 pt body plus 9 pt footnotes averaged to 10.4 pt, which is in no
    document and in no vocabulary, and the spread that averaging
    manufactured (113 page-level values against 26 node-level ones) was
    what the head was then scored on.

    A tie is `None` — the page has no typical value, and the probe's rule
    is that an undetermined datum is dropped, never coerced. Breaking the
    tie by min/max would not just invent a determination, it would invent a
    biased one: the low side of these ties is disproportionately the
    dominant class (spacing `0`, 48.7 % of values), so a min-break would
    quietly enrich the majority class it is supposed to be measuring
    against.
    """
    if not values:
        return None
    counts = Counter(values)
    ranked = counts.most_common(2)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return float(ranked[0][0])


def extract_labels(
    fragment: Mapping[str, Any], targets: Mapping[str, Any]
) -> ProbeLabels:
    """Probe labels for the page described by `fragment`.

    `fragment` is a RFC003 §26 page fragment; `targets` is the document's
    Canonical Target Bundle, read for the typed properties of the nodes
    the fragment places on this page.
    """
    by_type: dict[str, list[str]] = {}
    for entry in fragment.get("slices", ()):
        node_type = entry.get("node_type")
        node_id = entry.get("node_id")
        if node_type is None or node_id is None:
            continue
        by_type.setdefault(node_type, []).append(node_id)

    typed = targets.get("typed") or {}
    runs = by_type.get("run", [])
    paragraphs = by_type.get("paragraph", [])
    placed = fragment_owns_supervised_structure(fragment)

    bold = _known_values(typed, runs, "/style/bold")
    return ProbeLabels(
        has_table=("table" in by_type) if placed else None,
        has_list=("list" in by_type) if placed else None,
        has_image=("inline_image" in by_type) if placed else None,
        paragraph_count=len(paragraphs) if placed else None,
        run_count=len(runs) if placed else None,
        bold_run_ratio=(sum(1 for v in bold if v) / len(bold)) if bold else None,
        modal_font_size_mpt=_mode(_known_values(typed, runs, "/style/font_size_mpt")),
        modal_space_after_mpt=_mode(
            _known_values(typed, paragraphs, PARAGRAPH_SPACING_POINTER)
        ),
    )
