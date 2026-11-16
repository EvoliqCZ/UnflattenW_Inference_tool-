"""Opt-in chain policy `language-detect.v0` (PROPOSED, RFC004-ERRATA-007;
owner request 2026-09-27).

The model has no `/language` head (RFC004-ERRATA-003): `headless_defaults`
sets every run's `/language` to `null`, so a strict-valid page still gives
Word no language and spell-check is wrong on every run. This policy fills
that null, deterministically and without a network call, ML model or new
dependency: for each run it scores a small closed set of candidate BCP-47
languages by script, language-specific diacritics and a closed stopword
list, and only accepts the winner when its margin over the runner-up clears
`MARGIN_THRESHOLD` and the text is at least `MIN_CHARS` long. A text too
short or too ambiguous is left `None` -- this policy never guesses.

Detection runs at three levels, in order, per RFC004-ERRATA-007 §4: a run
too short to score on its own inherits its paragraph's decision; a
paragraph too short inherits the page's document-level decision (all run
text on the page, concatenated). `cascade_language` is the one function
that encodes this, reused by `apply` (over an already-parsed
`hypothesis.PageHypothesis`, run text and paragraph grouping both already
resolved there) and by `scripts/language_detect_vs_gold.py` (over gold AST
run text, so the measurement exercises the identical rule).

Chain order (owner request 2026-09-27): after `headless_defaults` (which
sets every run's `/language` to `null`) and before assembly -- `apply`
takes the `PageHypothesis` `sealed_adapter.page_hypothesis` builds from the
sealed page (so it already carries `headless_defaults`'s null default) and
returns an edited one, right before `assemble()`. Only a run whose current
typed `/language` is absent or `None` is ever touched (the model predicts
no such value today, so this is always true in practice; the check is the
same "never override a model value" contract `headless_defaults` uses).

LANGUAGES covers the 12 base languages the coordinator's 2026-09-27
measurement (1,500 documents, store `m2a-v11`, RFC004-ERRATA-007) found in
the admitted corpus: `en`, `es`, `nl`, `sk`, `pt`, `pl`, `cs`, `de`, `it`,
`fr`, `lt`, `sv`. Text alone cannot tell `en-US` from `en-GB`/`en-AU`, or
`pt-BR` from `pt-PT`, etc. -- a detected base language is always emitted
with `REGION_DEFAULTS`'s region, the single most frequent region for that
base in the same measurement. That region is a **default, not a
detection**: two documents both detected as `es` always get `es-ES` here
even though the measured corpus also carries `es-MX`/`es-CO`."""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from unflatten.assembly import hypothesis as hy

POLICY_ID = "language-detect.v0"
POLICY_STATUS = "RATIFIED"
ERRATUM = "docs/rfc/RFC004-ERRATA-007.md"
HEAD_ID = "policy." + POLICY_ID
COUNT_PREFIX = "language_detect:"

#: A run/paragraph/document text shorter than this (after `.strip()`) is
#: never scored on its own -- it inherits the next level up.
MIN_CHARS = 24
#: `(best_score - second_score) / best_score` must clear this before a
#: winner is accepted; otherwise the text is left `None`.
MARGIN_THRESHOLD = 0.34

RULES = (
    "document_fallback_detected.v0",
    "paragraph_detected_or_inherits_document.v0",
    "run_detected_or_inherits_paragraph.v0",
)


@dataclasses.dataclass(frozen=True, slots=True)
class LanguageProfile:
    #: BCP-47, region included (RFC001's own canonical fixtures always
    #: carry a region, e.g. `"sk-SK"`, never a bare `"sk"`).
    tag: str
    script: str
    diacritics: frozenset[str]
    stopwords: frozenset[str]


def _profile(
    tag: str, script: str, diacritics: str, stopwords: tuple[str, ...]
) -> LanguageProfile:
    return LanguageProfile(
        tag=tag, script=script, diacritics=frozenset(diacritics),
        stopwords=frozenset(w.lower() for w in stopwords),
    )


def _dedup(languages: tuple[LanguageProfile, ...]) -> tuple[LanguageProfile, ...]:
    """A stopword or diacritic every scored language's profile shares (e.g.
    Slovak and Czech both use `é`) adds the same weight to every score and
    only shrinks the winner's margin over the runner-up -- drop it from all
    of them so each score reflects only what actually discriminates."""
    def shared_of(attr: str) -> frozenset[str]:
        counts = Counter(item for profile in languages for item in getattr(profile, attr))
        return frozenset(item for item, count in counts.items() if count > 1)

    shared_stopwords, shared_diacritics = shared_of("stopwords"), shared_of("diacritics")
    return tuple(
        dataclasses.replace(
            profile,
            stopwords=profile.stopwords - shared_stopwords,
            diacritics=profile.diacritics - shared_diacritics,
        )
        for profile in languages
    )


#: Base language -> the single BCP-47 region emitted for it, chosen by the
#: coordinator's 2026-09-27 measurement (dominant-language-per-document
#: counts, RFC004-ERRATA-007): the most frequent region for that base
#: language in the admitted corpus. Text cannot discriminate a base
#: language's regions (`en-US` vs `en-GB` vs `en-AU`, ...) -- this is a
#: default applied on top of a detection, never itself detected.
REGION_DEFAULTS: Mapping[str, str] = MappingProxyType({
    "en": "en-US", "es": "es-ES", "pt": "pt-BR", "nl": "nl-NL", "de": "de-DE",
    "fr": "fr-FR", "it": "it-IT", "pl": "pl-PL", "cs": "cs-CZ", "sk": "sk-SK",
    "lt": "lt-LT", "sv": "sv-SE",
})

#: One profile per base language `REGION_DEFAULTS` covers (script,
#: diacritics, stopwords keyed by base language; the emitted `tag` is
#: `REGION_DEFAULTS[base]`). Every profile's script is `"Latin"`; `script`
#: is still a named field (not folded away) because a non-Latin candidate
#: (Cyrillic, Greek, ...) is the obvious next addition and must slot in
#: without a rescore of these twelve.
_RAW_LANGUAGES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("sk", "Latin", "áäčďéíĺľňóôŕšťúýž", (
        "a", "je", "sa", "na", "za", "so", "aj", "ale", "alebo", "pre", "pri",
        "podľa", "ktorý", "ktorá", "ktoré", "nie", "byť", "bol", "bola",
        "boli", "tento", "táto", "toto", "tieto", "však", "medzi", "ako",
        "kde", "keď", "číslo", "strana",
    )),
    ("cs", "Latin", "áčďéěíňóřšťůúýž", (
        "a", "je", "se", "na", "za", "nebo", "ale", "pro", "při", "podle",
        "který", "která", "které", "není", "být", "byl", "byla", "byly",
        "tento", "tato", "toto", "tyto", "však", "mezi", "jak", "kde",
        "když", "číslo", "strana", "také",
    )),
    ("de", "Latin", "äöüß", (
        "der", "die", "das", "und", "ist", "nicht", "mit", "auf", "für",
        "von", "zu", "im", "ein", "eine", "einer", "eines", "wird", "werden",
        "wurde", "auch", "oder", "als", "aus", "bei", "nach", "über",
        "seite",
    )),
    ("en", "Latin", "", (
        "the", "and", "is", "are", "of", "to", "in", "for", "on", "with",
        "as", "that", "this", "these", "those", "was", "were", "be",
        "been", "or", "not", "by", "from", "at", "page", "section",
    )),
    ("es", "Latin", "áéíóúñü¡¿", (
        "el", "los", "las", "que", "es", "por", "para", "su", "más", "pero",
        "también", "está", "muy", "esto", "esta", "estos", "estas", "cómo",
        "página", "sección", "número", "así", "entre", "según", "cuando",
        "porque", "aquí", "cuál", "dónde", "quién", "sino", "además",
        "mediante", "con", "una", "un", "en", "de",
    )),
    ("pt", "Latin", "áâãàçéêíóôõú", (
        "os", "que", "não", "com", "para", "mais", "mas", "também", "está",
        "são", "foi", "muito", "isso", "esta", "estes", "estas", "número",
        "página", "seção", "então", "entre", "um", "uma", "de", "em",
    )),
    ("nl", "Latin", "ëï", (
        "de", "het", "een", "van", "en", "is", "niet", "met", "op", "voor",
        "aan", "bij", "dat", "die", "deze", "dit", "worden", "wordt", "werd",
        "ook", "of", "als", "uit", "naar", "over", "pagina",
    )),
    ("fr", "Latin", "àâçèéêëîïôùûüÿœ", (
        "le", "la", "les", "des", "que", "et", "est", "pas", "avec", "sur",
        "pour", "de", "un", "une", "dans", "ce", "cette", "ces", "être",
        "sont", "était", "aussi", "ou", "par", "entre", "comme", "page",
    )),
    ("it", "Latin", "àèéìíîòóùú", (
        "il", "lo", "la", "gli", "le", "che", "e", "è", "non", "con", "su",
        "per", "di", "un", "una", "questo", "questa", "questi", "queste",
        "essere", "sono", "era", "anche", "come", "tra", "pagina", "sezione",
    )),
    ("pl", "Latin", "ąćęłńóśźż", (
        "i", "jest", "się", "na", "za", "czy", "ale", "lub", "dla", "przy",
        "według", "który", "która", "które", "nie", "być", "był", "była",
        "były", "ten", "ta", "to", "te", "jednak", "między", "jak", "gdzie",
        "kiedy", "numer", "strona", "także",
    )),
    ("lt", "Latin", "ąčęėįšųūž", (
        "yra", "ir", "su", "kad", "bet", "arba", "dėl", "prie", "pagal",
        "kuris", "kuri", "kurie", "nėra", "būti", "buvo", "šis", "ši", "šie",
        "tačiau", "tarp", "kaip", "kur", "kada", "puslapis", "numeris",
    )),
    ("sv", "Latin", "åäö", (
        "och", "är", "inte", "med", "på", "för", "av", "till", "ett", "en",
        "som", "detta", "denna", "dessa", "vara", "var", "också", "eller",
        "från", "vid", "efter", "över", "sida", "mellan",
    )),
)

LANGUAGES: tuple[LanguageProfile, ...] = _dedup(tuple(
    _profile(REGION_DEFAULTS[base], script, diacritics, stopwords)
    for base, script, diacritics, stopwords in _RAW_LANGUAGES
))

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_DIACRITIC_WEIGHT = 3
_STOPWORD_WEIGHT = 1


def policy_record() -> dict[str, Any]:
    """The literal policy, embedded in the hashed chain report. The RFC001
    CBOR/canonical-JSON profile forbids floats (`report.py`'s own
    `_assert_no_floats` precedent), so `MARGIN_THRESHOLD` is recorded as
    parts-per-million, not the float itself."""
    return {
        "policy_id": POLICY_ID,
        "status": POLICY_STATUS,
        "erratum": ERRATUM,
        "languages": [profile.tag for profile in LANGUAGES],
        "region_defaults": dict(sorted(REGION_DEFAULTS.items())),
        "min_chars": MIN_CHARS,
        "margin_threshold_ppm": round(MARGIN_THRESHOLD * 1_000_000),
        "rules": list(RULES),
    }


def _score(chars: frozenset[str], words: list[str], profile: LanguageProfile) -> int:
    diacritic_hits = len(chars & profile.diacritics)
    stopword_hits = sum(1 for word in words if word in profile.stopwords)
    return diacritic_hits * _DIACRITIC_WEIGHT + stopword_hits * _STOPWORD_WEIGHT


def detect_language(text: str) -> str | None:
    """The BCP-47 tag whose script/diacritics/stopwords best explain `text`,
    or `None` when `text` is shorter than `MIN_CHARS`, carries no word
    characters, or the winner's margin over the runner-up does not clear
    `MARGIN_THRESHOLD`. Never guesses."""
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return None
    lowered = stripped.lower()
    words = _WORD_RE.findall(lowered)
    if not words:
        return None
    chars = frozenset(lowered)
    scored = sorted(
        ((_score(chars, words, profile), profile.tag) for profile in LANGUAGES), reverse=True
    )
    best_score, best_tag = scored[0]
    if best_score <= 0:
        return None
    second_score = scored[1][0] if len(scored) > 1 else 0
    margin = (best_score - second_score) / best_score
    if margin < MARGIN_THRESHOLD:
        return None
    return best_tag


def cascade_language(*, run_text: str, paragraph_text: str, document_text: str) -> str | None:
    """The run's language: its own detection when it is long enough on its
    own, else its paragraph's (detected, or the document's when the
    paragraph itself is too short), else `None`."""
    if len(run_text.strip()) >= MIN_CHARS:
        return detect_language(run_text)
    if len(paragraph_text.strip()) >= MIN_CHARS:
        return detect_language(paragraph_text)
    return detect_language(document_text)


def _run_nodes(node: hy.PageNode) -> list[hy.PageNode]:
    if node.node_type == "run":
        return [node]
    return [run for child in node.children for run in _run_nodes(child)]


def _text_of(nodes: list[hy.PageNode]) -> str:
    return " ".join(node.text or "" for node in nodes)


def apply(hypothesis: hy.PageHypothesis) -> tuple[hy.PageHypothesis, Counter[str]]:
    """`hypothesis` with every run whose typed `/language` is absent or
    `None` filled in by `cascade_language`, and one count per run: either
    `runs_set` (plus `lang:<tag>`) or `runs_left_null_low_confidence`. The
    same `hypothesis` object, with no counts, when nothing was edited."""
    counts: Counter[str] = Counter()
    document_text = _text_of(
        [run for root in (*hypothesis.sections, *hypothesis.notes) for run in _run_nodes(root)]
    )

    def edit(node: hy.PageNode, paragraph_text: str) -> hy.PageNode:
        if node.node_type == "paragraph":
            paragraph_text = _text_of(_run_nodes(node))
        if node.node_type == "run":
            if node.typed.get("/language") is not None:
                return node
            detected = cascade_language(
                run_text=node.text or "", paragraph_text=paragraph_text,
                document_text=document_text,
            )
            if detected is None:
                counts[COUNT_PREFIX + "runs_left_null_low_confidence"] += 1
                return node
            typed = dict(node.typed)
            typed["/language"] = detected
            counts[COUNT_PREFIX + "runs_set"] += 1
            counts[COUNT_PREFIX + f"lang:{detected}"] += 1
            return dataclasses.replace(node, typed=MappingProxyType(typed))
        children = tuple(edit(child, paragraph_text) for child in node.children)
        if children == node.children:
            return node
        return dataclasses.replace(node, children=children)

    sections = tuple(edit(root, document_text) for root in hypothesis.sections)
    notes = tuple(edit(root, document_text) for root in hypothesis.notes)
    if sections == hypothesis.sections and notes == hypothesis.notes:
        return hypothesis, counts
    return dataclasses.replace(hypothesis, sections=sections, notes=notes), counts
