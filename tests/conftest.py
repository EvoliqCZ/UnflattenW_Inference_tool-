import copy
import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def minimal_valid_doc() -> dict:
    with open(DATA_DIR / "minimal_valid.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def minimal_valid_doc_copy(minimal_valid_doc) -> dict:
    return copy.deepcopy(minimal_valid_doc)
