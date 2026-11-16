import json
from pathlib import Path

import pytest

from unflatten.ast.hashing import ast_file_hash, ast_hash

DATA_DIR = Path(__file__).parent.parent / "data"


def test_ast_hash_returns_sha256_prefixed_hex(minimal_valid_doc_copy):
    result = ast_hash(minimal_valid_doc_copy)
    assert result.startswith("sha256:")
    assert len(result) == len("sha256:") + 64
    hex_part = result[len("sha256:") :]
    assert hex_part == hex_part.lower()
    int(hex_part, 16)  # raises if not valid hex


def test_ast_hash_is_deterministic(minimal_valid_doc_copy):
    assert ast_hash(minimal_valid_doc_copy) == ast_hash(minimal_valid_doc_copy)


def test_ast_hash_changes_when_content_field_changes(minimal_valid_doc_copy):
    before = ast_hash(minimal_valid_doc_copy)
    minimal_valid_doc_copy["sections"][0]["children"][0]["children"][0]["text"] = "Changed."
    after = ast_hash(minimal_valid_doc_copy)
    assert before != after


def test_ast_hash_unaffected_by_metadata_change(minimal_valid_doc_copy):
    before = ast_hash(minimal_valid_doc_copy)
    minimal_valid_doc_copy["metadata"]["title"] = "A brand new title"
    after = ast_hash(minimal_valid_doc_copy)
    assert before == after


def test_ast_hash_does_not_mutate_input(minimal_valid_doc_copy):
    original = json.loads(json.dumps(minimal_valid_doc_copy))
    ast_hash(minimal_valid_doc_copy)
    assert minimal_valid_doc_copy == original


def test_ast_file_hash_differs_from_ast_hash_for_same_document(minimal_valid_doc_copy):
    raw_bytes = json.dumps(minimal_valid_doc_copy).encode("utf-8")
    assert ast_file_hash(raw_bytes) != ast_hash(minimal_valid_doc_copy)


def test_ast_file_hash_returns_sha256_of_raw_bytes():
    import hashlib

    data = b'{"a": 1}'
    expected = f"sha256:{hashlib.sha256(data).hexdigest()}"
    assert ast_file_hash(data) == expected


@pytest.fixture
def hash_vectors() -> list[dict]:
    with open(DATA_DIR / "hash_vectors.json", encoding="utf-8") as f:
        return json.load(f)


def test_golden_hash_vectors_match_expected(hash_vectors):
    for vector in hash_vectors:
        computed = ast_hash(vector["document"])
        assert computed == vector["expected_ast_hash"], vector["description"]
