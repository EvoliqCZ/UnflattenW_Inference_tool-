"""Gate 10's text corpus, drawn from the cohort itself.

RFC005 §14.4 gate 10 round-trips text through `exact_text.encode_text` /
`decode_tokens`. Which strings it round-trips decides what the gate is a
claim about: an invented fixture makes gate 10 a statement about the
fixture, so the corpus is read from the SAME documents the feature cache
holds -- `target_bundle["text"]["spans"][*]["text"]`, through
`dataset.items.build_item`, which verifies every object's SHA-256 on read.

**Why this is a separate, optional source and not a shipped artefact.**
The cohort's text lives only in the ~62 GB object store; `release-v10`
carries none of it, and the candidate feature cache carries no document
text at all (measured: the highest non-negative id across its 150 entries
is 68, inside the structural range). Extracting a corpus into a shipped
artefact would put document text into the Colab payload and silently
reverse the owner's privacy decision that rests on the cache holding none.
So `s1_runner` reads this source when the store is present (locally) and
reports gate 10 `blocked`, naming the absent store, when it is not.

**The empty-set hazard is refused here, loudly.** An empty corpus makes
gate 10 report `pass` -- "0 fixture(s) round-tripped exactly" -- which is
a pass over nothing. Every way this function could return one (no cohort
document enumerated, no document carrying a text span) raises
`S1TextCorpusUnavailable` instead; `s1_runner` turns that into `blocked`.

This module authorizes no gradient update and writes nothing: it reads
the release and the store, and returns strings in memory.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from unflatten.dataset.items import build_item
from unflatten.dataset.samples import iter_page_samples

#: How many distinct strings gate 10 round-trips by default. Gate 10's
#: subject is the encode/decode contract, not corpus breadth -- and every
#: additional document is another object-store read.
DEFAULT_MAX_STRINGS = 256


class S1TextCorpusUnavailable(ValueError):
    """No cohort text corpus could be built without inventing one."""


@dataclass(frozen=True)
class CohortTextCorpus:
    """Gate 10's corpus plus the provenance its evidence must carry, so a
    reader can tell cohort text from a hand-written fixture."""

    texts: tuple[str, ...]
    #: Human-readable provenance, recorded verbatim in gate 10's evidence.
    source: str
    #: Cohort documents actually read (not the cohort's size).
    document_count: int
    #: Text spans seen in those documents, before dedup and the empty-string
    #: drop -- `len(texts)` alone cannot say how much was collapsed.
    span_count: int
    #: `max_strings` was reached, so the corpus is a prefix of the cohort's text.
    truncated: bool


def build_cohort_text_corpus(
    release_dir: Path | str,
    store_dir: Path | str,
    bundle_ids: Iterable[str],
    *,
    split: str,
    task_kind: str,
    max_strings: int = DEFAULT_MAX_STRINGS,
) -> CohortTextCorpus:
    """Read the text spans of `bundle_ids` out of `store_dir`.

    `split`/`task_kind` must be the ones the feature cache was built under
    (`feature_cache.CacheIdentity`), so the enumeration this walks is the
    same one the cohort came from rather than a second, independently
    chosen selection.

    Raises `S1TextCorpusUnavailable` rather than returning an empty or
    partly-invented corpus -- see the module docstring.
    """
    if max_strings <= 0:
        raise ValueError(f"max_strings must be positive, got {max_strings}")
    release_dir, store_dir = Path(release_dir), Path(store_dir)
    wanted = frozenset(bundle_ids)

    texts: list[str] = []
    seen: set[str] = set()
    documents: set[str] = set()
    span_count = 0
    truncated = False

    for sample in iter_page_samples(release_dir, store_dir, split, task_kind):
        if sample.document_bundle_id not in wanted:
            continue
        documents.add(sample.document_bundle_id)
        item = build_item(store_dir, sample)
        spans = (item.targets or {}).get("text", {}).get("spans", ())
        for span in spans:
            span_count += 1
            text = span.get("text") or ""
            if not text or text in seen:
                continue
            seen.add(text)
            texts.append(text)
            if len(texts) >= max_strings:
                truncated = True
                break
        if truncated:
            break

    if not documents:
        raise S1TextCorpusUnavailable(
            f"no cohort document of the {len(wanted)} requested was enumerated in "
            f"{release_dir} (split={split!r}, task_kind={task_kind!r}) -- gate 10 "
            "reports blocked rather than passing over an empty corpus"
        )
    if not texts:
        raise S1TextCorpusUnavailable(
            f"{len(documents)} cohort document(s) read from {store_dir} carried no "
            f"text span content ({span_count} span(s) seen) -- gate 10 reports "
            "blocked rather than passing over an empty corpus"
        )

    return CohortTextCorpus(
        texts=tuple(texts),
        source=(
            f"cohort text spans (target_bundle['text']['spans']) read via "
            f"dataset.items.build_item from {store_dir} "
            f"(split={split!r}, task_kind={task_kind!r}, "
            f"{len(documents)} document(s), {span_count} span(s) seen"
            f"{', truncated at max_strings' if truncated else ''})"
        ),
        document_count=len(documents),
        span_count=span_count,
        truncated=truncated,
    )
