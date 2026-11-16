"""`exact_text_roundtrip_corpus` producer -- a hash-bound, synthetic
Unicode/control fixture corpus (design §4.7, spec-roundtrip-corpus-c3).

Acceptance criterion 2 (design §16) needs evidence about a fixture corpus at
a **named `corpus_version`**, so shrinking or reordering the corpus after a
failure changes `cases_hash` and is therefore visible. This module owns:

* the ordered case table (:data:`CASES`, 15 categories, >= 2 cases each);
* the code-point hash projection and :func:`cases_hash` (domain
  `rfc004.exact-text-roundtrip-corpus/v1`) -- hashing code points rather
  than raw strings, because the deterministic CBOR profile
  (:mod:`unflatten.model.cbor`) NFC-normalizes text and rejects lone
  surrogates on encode, which would silently fold an NFD case onto its NFC
  twin and crash on the `invalid_surrogate` category outright;
* the closed 5-field descriptor (:func:`build_corpus_descriptor`,
  :func:`validate_corpus_descriptor`, :func:`load_corpus_descriptor`);
* :func:`check_corpus`, which runs every case through
  :mod:`unflatten.model.exact_text` and reports pass/fail per case plus the
  codec identity that produced the result;
* :func:`assert_monotonic`, which refuses a `corpus_version` bump that
  narrows the corpus or leaves its content unchanged.

The corpus is synthetic and independent of the object store (owner R2,
2026-09-07): it is test evidence bound in the Stage Gate Report (C5), never
in `s1_run_manifest` (closed 14-field schema) or the Component Lock, and it
does not unblock gate 10 (which keeps its cohort-text leg from the store).

Torch-free: this module and everything it imports
(`exact_text` -> `composite_vocab` -> `hashing`/`cbor`) declare only
dataclasses, a `NamedTuple` and pure functions -- no neural-network module,
no weights, no optimizer.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import jsonschema.protocols
import jsonschema.validators

from unflatten.model import exact_text as et
from unflatten.model.composite_vocab import CompositeVocabulary, QwenRef, TokenRef
from unflatten.model.hashing import domain_hash

_SCHEMA_DIR = Path(__file__).parent / "schemas"
_SCHEMA_NAME = "exact_text_roundtrip_corpus"

#: The frozen S1 descriptor document (design §4.7, plan `spec-roundtrip-corpus-c3`).
FROZEN_CORPUS_PATH = Path(__file__).parent / "corpora" / "s1_exact_text_roundtrip_corpus.json"

#: Design §4.9 registry row `rfc004.exact-text-roundtrip-corpus/v1`. Owned
#: here: no other module computes a `cases_hash`.
_CORPUS_DOMAIN = "rfc004.exact-text-roundtrip-corpus/v1"

CORPUS_SCHEMA_VERSION = "1.0.0"

#: The frozen v1.0.0 corpus version this module's `CASES` table implements.
CORPUS_VERSION = "1.0.0"


class RoundtripCorpusInvalid(ValueError):
    """The case table, a descriptor, or a `check_corpus` invocation is invalid.

    M1-internal (design §9.3); not an RFC004 registry code.
    """


class RoundtripCorpusMonotonicityViolation(RoundtripCorpusInvalid):
    """A `corpus_version` bump did not satisfy the monotonicity contract
    (design §4.7 decision 5): shrinking, reordering into a lower version, or
    bumping the version with unchanged content are all refused.
    """


class Case(NamedTuple):
    """One fixture: `category` from :data:`CATEGORY_KEYS`, `text` and
    whether it must `"roundtrip"` or `"reject"` (design §4.7 decision 1)."""

    category: str
    text: str
    expect: str  # "roundtrip" | "reject"


@cache
def _validator() -> jsonschema.protocols.Validator:
    with open(_SCHEMA_DIR / f"{_SCHEMA_NAME}.schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


#: Read from the schema's own `category_counts.required` list (design
#: decision 1) so a category added there forces this module to notice,
#: instead of the two silently drifting apart.
CATEGORY_KEYS: tuple[str, ...] = tuple(
    _validator().schema["properties"]["category_counts"]["required"]
)

#: Categories whose every case must `"reject"` (design §4.7: non-NFC input
#: and lone surrogates are not scalars). Every other category is positive.
NEGATIVE_CATEGORIES: frozenset[str] = frozenset({"non_nfc_reject", "invalid_surrogate"})

_BYTE_LEN_CATEGORIES: tuple[str, ...] = ("byte_len_1", "byte_len_2", "byte_len_3", "byte_len_4")

# ---------------------------------------------------------------------------
# The ordered case table -- 32 cases, 15 categories, >= 2 each (decision 1).
# ---------------------------------------------------------------------------

CASES: tuple[Case, ...] = (
    # ascii (3, including the empty string)
    Case("ascii", "", "roundtrip"),
    Case("ascii", "Hello, World! 123", "roundtrip"),
    Case("ascii", "The quick brown fox jumps.", "roundtrip"),
    # spaces (2): NBSP, EM SPACE
    Case("spaces", "a b", "roundtrip"),
    Case("spaces", "a b", "roundtrip"),
    # nfc_accents (2): already-NFC precomposed accents
    Case("nfc_accents", "café", "roundtrip"),
    Case("nfc_accents", "áéíóú", "roundtrip"),
    # non_nfc_reject (2): NFD decomposed forms of the same letters
    Case("non_nfc_reject", "é", "reject"),
    Case("non_nfc_reject", "Å", "reject"),
    # bidi (2)
    Case("bidi", "مرحبا", "roundtrip"),
    Case("bidi", "שלום", "roundtrip"),
    # cjk (2)
    Case("cjk", "你好世界", "roundtrip"),
    Case("cjk", "日本語", "roundtrip"),
    # emoji_zwj (2): ZWJ sequences
    Case("emoji_zwj", "\U0001f468‍\U0001f469‍\U0001f467", "roundtrip"),
    Case("emoji_zwj", "\U0001f469‍\U0001f4bb", "roundtrip"),
    # variation_selector (2)
    Case("variation_selector", "❤️", "roundtrip"),
    Case("variation_selector", "☃︎", "roundtrip"),
    # supplementary (2): astral-plane scalars
    Case("supplementary", "\U0001d49c", "roundtrip"),
    Case("supplementary", "\U00013000", "roundtrip"),
    # allowed_controls (3): encoder only forbids 0x09/0x0A/0x0D (design §6);
    # FF, ESC, DEL and NEL are legal Run substitutes at this codec layer.
    Case("allowed_controls", "a\x0cb", "roundtrip"),
    Case("allowed_controls", "a\x1bb", "roundtrip"),
    Case("allowed_controls", "a\x7fb\x85c", "roundtrip"),
    # invalid_surrogate (2): lone surrogates are not scalars
    Case("invalid_surrogate", "a\ud800b", "reject"),
    Case("invalid_surrogate", "\udfff", "reject"),
    # byte_len_1..4 (2 each): exactly one scalar of that UTF-8 byte length
    Case("byte_len_1", "A", "roundtrip"),
    Case("byte_len_1", "0", "roundtrip"),
    Case("byte_len_2", "é", "roundtrip"),
    Case("byte_len_2", "ñ", "roundtrip"),
    Case("byte_len_3", "€", "roundtrip"),
    Case("byte_len_3", "가", "roundtrip"),
    Case("byte_len_4", "😀", "roundtrip"),
    Case("byte_len_4", "𝄞", "roundtrip"),
)


def _validate_case_table(cases: tuple[Case, ...]) -> None:
    """Import-time self-check of :data:`CASES` (design §4.7 decision 1)."""
    counts: dict[str, int] = dict.fromkeys(CATEGORY_KEYS, 0)
    for index, case in enumerate(cases):
        if case.category not in CATEGORY_KEYS:
            raise RoundtripCorpusInvalid(
                f"case {index} has unknown category {case.category!r}; "
                f"expected one of {CATEGORY_KEYS}"
            )
        if case.expect not in ("roundtrip", "reject"):
            raise RoundtripCorpusInvalid(
                f"case {index} ({case.category}) has expect={case.expect!r}, "
                "expected 'roundtrip' or 'reject'"
            )
        expected_verdict = "reject" if case.category in NEGATIVE_CATEGORIES else "roundtrip"
        if case.expect != expected_verdict:
            raise RoundtripCorpusInvalid(
                f"case {index} ({case.category}) declares expect={case.expect!r}, "
                f"but category {case.category!r} must declare {expected_verdict!r}"
            )
        if case.category in _BYTE_LEN_CATEGORIES:
            wanted_len = int(case.category.removeprefix("byte_len_"))
            if len(case.text) != 1:
                raise RoundtripCorpusInvalid(
                    f"case {index} ({case.category}) must be exactly one scalar, "
                    f"got {len(case.text)}"
                )
            try:
                actual_len = len(case.text.encode("utf-8"))
            except UnicodeEncodeError as exc:
                raise RoundtripCorpusInvalid(
                    f"case {index} ({case.category}) is not valid UTF-8: {exc}"
                ) from exc
            if actual_len != wanted_len:
                raise RoundtripCorpusInvalid(
                    f"case {index} ({case.category}) encodes to {actual_len} UTF-8 "
                    f"bytes, expected {wanted_len}"
                )
        counts[case.category] += 1
    short = sorted(cat for cat in CATEGORY_KEYS if counts[cat] < 2)
    if short:
        raise RoundtripCorpusInvalid(
            f"categor(ies) with fewer than 2 cases: {', '.join(short)}"
        )


_validate_case_table(CASES)


# ---------------------------------------------------------------------------
# Hashing (code points, not strings -- module docstring)
# ---------------------------------------------------------------------------


def cases_projection(cases: tuple[Case, ...] = CASES) -> list[list[Any]]:
    """`[[category, [code points...], expect], ...]`, order preserved."""
    return [[case.category, [ord(ch) for ch in case.text], case.expect] for case in cases]


def cases_hash(cases: tuple[Case, ...] = CASES) -> str:
    return domain_hash(_CORPUS_DOMAIN, cases_projection(cases))


# ---------------------------------------------------------------------------
# Descriptor (design §4.7: exactly 5 schema fields, identity = cases_hash)
# ---------------------------------------------------------------------------


def validate_corpus_descriptor(descriptor: dict[str, Any]) -> None:
    errors = list(_validator().iter_errors(descriptor))
    if errors:
        raise RoundtripCorpusInvalid(
            f"schema validation failed: {errors[0].message} at "
            f"{list(errors[0].absolute_path)}"
        )


def build_corpus_descriptor(corpus_version: str) -> dict[str, Any]:
    """Build the 5-field descriptor for `CASES` at `corpus_version`, validated."""
    category_counts: dict[str, int] = dict.fromkeys(CATEGORY_KEYS, 0)
    for case in CASES:
        category_counts[case.category] += 1
    descriptor = {
        "exact_text_roundtrip_corpus_schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_version": corpus_version,
        "case_count": len(CASES),
        "cases_hash": cases_hash(),
        "category_counts": category_counts,
    }
    validate_corpus_descriptor(descriptor)
    return descriptor


def load_corpus_descriptor(path: Path | str) -> dict[str, Any]:
    """Read a frozen descriptor document from disk, validated on the way in."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RoundtripCorpusInvalid(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RoundtripCorpusInvalid(f"{path} is not valid JSON: {exc}") from exc
    validate_corpus_descriptor(payload)
    return payload


# ---------------------------------------------------------------------------
# check_corpus
# ---------------------------------------------------------------------------


def _uses_qwen(tokens: tuple[TokenRef, ...]) -> bool:
    return any(isinstance(token, QwenRef) for token in tokens)


def _failure(index: int, case: Case, detail: str) -> dict[str, Any]:
    return {
        "index": index,
        "category": case.category,
        "expect": case.expect,
        "detail": detail,
    }


def check_corpus(
    *,
    tokenizer: et.QwenTokenizerLike,
    vocabulary: CompositeVocabulary,
    budget_ops: int,
    descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run every case in `CASES` through `exact_text.encode_text`/`decode_tokens`.

    Positive cases must round-trip scalar-exact; negative cases must raise
    `ValueError` (`TextNotNfc`/`TextInvalidUnicode`/...) -- any other
    exception propagates rather than being swallowed. `byte_len_*`
    categories are always exercised through the byte fallback: the
    checker builds a byte-only view of `vocabulary` (its
    `qwen_text_allowlist` replaced with the empty set) for those
    categories specifically, so a vocabulary that could otherwise cover a
    short scalar with a native Qwen token cannot silently substitute a
    different route than the one the byte-fallback contract names
    (design §6 criterion 5). `descriptor`, if given, must already match
    the live `CASES` table -- a stale descriptor (its `cases_hash` no
    longer equal to `cases_hash()`) is refused rather than checked against
    a corpus that no longer exists.
    """
    if descriptor is None:
        descriptor = build_corpus_descriptor(CORPUS_VERSION)
    else:
        validate_corpus_descriptor(descriptor)
        current_hash = cases_hash()
        if descriptor["cases_hash"] != current_hash:
            raise RoundtripCorpusInvalid(
                f"descriptor cases_hash {descriptor['cases_hash']} does not match "
                f"the live CASES table hash {current_hash}; descriptor is stale"
            )

    byte_only_vocabulary = dataclasses.replace(vocabulary, qwen_text_allowlist=frozenset())

    #: The only two exceptions that constitute a correct "reject" of a
    #: negative case (design §4.7 decision 1: non-NFC input / lone
    #: surrogates are not scalars). Any other `ValueError` means the case
    #: was rejected for an unrelated reason and must not be counted as a
    #: pass.
    typed_reject_errors = (et.TextNotNfc, et.TextInvalidUnicode)

    failures: list[dict[str, Any]] = []
    used_qwen = False
    for index, case in enumerate(CASES):
        is_byte_len = case.category in _BYTE_LEN_CATEGORIES
        working_vocabulary = byte_only_vocabulary if is_byte_len else vocabulary
        try:
            tokens = et.encode_text(
                case.text,
                vocabulary=working_vocabulary,
                tokenizer=tokenizer,
                budget_ops=budget_ops,
            )
            decoded = et.decode_tokens(tokens, vocabulary=working_vocabulary, tokenizer=tokenizer)
        except ValueError as exc:
            if case.expect != "reject":
                failures.append(
                    _failure(index, case, f"raised {type(exc).__name__}: {exc}")
                )
            elif not isinstance(exc, typed_reject_errors):
                failures.append(
                    _failure(
                        index,
                        case,
                        f"rejected for the wrong reason: {type(exc).__name__}: {exc}",
                    )
                )
            continue
        if case.expect == "reject":
            failures.append(_failure(index, case, f"did not raise; decoded to {decoded!r}"))
            continue
        if decoded != case.text:
            failures.append(
                _failure(index, case, f"decoded {decoded!r} != expected {case.text!r}")
            )
            continue
        if is_byte_len and _uses_qwen(tokens):
            failures.append(
                _failure(index, case, "byte_len bucket not exercised via byte fallback")
            )
            continue
        if not is_byte_len and _uses_qwen(tokens):
            used_qwen = True

    return {
        "passed": not failures,
        "failures": failures,
        "checked_case_count": len(CASES),
        "codec_mode": "qwen" if used_qwen else "byte_fallback_only",
        "byte_len_route": "byte_fallback",
        "budget_ops": budget_ops,
        "tokenizer_artifact_set_hash": vocabulary.tokenizer_artifact_set_hash,
        "composite_vocabulary_hash": vocabulary.artifact_hash,
    }


# ---------------------------------------------------------------------------
# assert_monotonic
# ---------------------------------------------------------------------------


def _parse_semver(version: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in version.split("."))
    return (major, minor, patch)


def assert_monotonic(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse a `corpus_version` transition that narrows or silently changes
    the corpus (design §4.7 decision 5): equal version must carry identical
    content; a higher version must not decrease any category count or
    `case_count`, and must actually change `cases_hash`; a lower version is
    always refused.
    """
    validate_corpus_descriptor(previous)
    validate_corpus_descriptor(current)

    previous_version = _parse_semver(previous["corpus_version"])
    current_version = _parse_semver(current["corpus_version"])

    if current_version < previous_version:
        raise RoundtripCorpusMonotonicityViolation(
            f"corpus_version regressed from {previous['corpus_version']} to "
            f"{current['corpus_version']}"
        )

    if current_version == previous_version:
        if (
            current["cases_hash"] != previous["cases_hash"]
            or current["case_count"] != previous["case_count"]
            or current["category_counts"] != previous["category_counts"]
        ):
            raise RoundtripCorpusMonotonicityViolation(
                f"corpus_version {current['corpus_version']} is unchanged but "
                "cases_hash/case_count/category_counts differ from the previous "
                "descriptor at the same version"
            )
        return

    for category, previous_count in previous["category_counts"].items():
        current_count = current["category_counts"].get(category, 0)
        if current_count < previous_count:
            raise RoundtripCorpusMonotonicityViolation(
                f"category {category!r} count decreased from {previous_count} to "
                f"{current_count} at corpus_version {current['corpus_version']}"
            )
    if current["case_count"] < previous["case_count"]:
        raise RoundtripCorpusMonotonicityViolation(
            f"case_count decreased from {previous['case_count']} to "
            f"{current['case_count']} at corpus_version {current['corpus_version']}"
        )
    if current["cases_hash"] == previous["cases_hash"]:
        raise RoundtripCorpusMonotonicityViolation(
            f"corpus_version bumped from {previous['corpus_version']} to "
            f"{current['corpus_version']} but cases_hash did not change"
        )
