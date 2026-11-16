"""Golden fixtures corpus (RFC003 §31 P0_CONTRACT_GOLDEN seed).

Walks tests/data/corpus/valid/ and tests/data/corpus/invalid/, exercising
the full RFC001 contract: schema validation, semantic validation, and
canonicalization/hashing determinism.
"""

import json
from pathlib import Path

import pytest

from unflatten.ast.canonicalizer import canonicalize
from unflatten.ast.hashing import ast_hash
from unflatten.ast.schema_validator import validate_schema
from unflatten.ast.semantic_validator import validate_semantics

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
VALID_DIR = CORPUS_DIR / "valid"
INVALID_DIR = CORPUS_DIR / "invalid"

_BLOCKING = {"error", "fatal"}


def _valid_fixture_paths() -> list[Path]:
    return sorted(VALID_DIR.glob("*.json"))


def _invalid_fixture_paths() -> list[Path]:
    return sorted(p for p in INVALID_DIR.glob("*.json") if not p.name.endswith(".expected.json"))


def test_corpus_has_at_least_eight_valid_fixtures():
    assert len(_valid_fixture_paths()) >= 8


def test_corpus_has_at_least_eight_invalid_fixtures():
    assert len(_invalid_fixture_paths()) >= 8


@pytest.mark.parametrize("path", _valid_fixture_paths(), ids=lambda p: p.stem)
def test_valid_fixture_passes_schema_and_semantic_validation(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)

    schema_issues = validate_schema(doc)
    blocking_schema = [i for i in schema_issues if i.severity in _BLOCKING]
    assert blocking_schema == [], f"{path.name}: unexpected schema issues {blocking_schema}"

    semantic_issues = validate_semantics(doc)
    blocking_semantic = [i for i in semantic_issues if i.severity in _BLOCKING]
    assert blocking_semantic == [], f"{path.name}: unexpected semantic issues {blocking_semantic}"


@pytest.mark.parametrize("path", _valid_fixture_paths(), ids=lambda p: p.stem)
def test_valid_fixture_canonicalizes_successfully(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    canonicalize(doc)  # must not raise


@pytest.mark.parametrize("path", _valid_fixture_paths(), ids=lambda p: p.stem)
def test_valid_fixture_ast_hash_stable_across_two_runs(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    first = canonicalize(doc)
    second = canonicalize(json.loads(json.dumps(doc)))
    assert ast_hash(first) == ast_hash(second)


@pytest.mark.parametrize("path", _invalid_fixture_paths(), ids=lambda p: p.stem)
def test_invalid_fixture_reports_expected_issue_code_and_path_pairs(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    expected_path = path.with_suffix("").with_suffix(".expected.json")
    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)

    schema_issues = validate_schema(doc)
    semantic_issues = validate_semantics(doc) if not schema_issues else []
    all_pairs = {(i.code, i.path) for i in schema_issues} | {
        (i.code, i.path) for i in semantic_issues
    }

    for expected_issue in expected["expected_issues"]:
        pair = (expected_issue["code"], expected_issue["path"])
        assert pair in all_pairs, f"{path.name}: expected {pair} not found in {all_pairs}"
