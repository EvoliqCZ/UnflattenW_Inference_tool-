"""`ETag`/`If-None-Match` 304 revalidation and `Cache-Control` (design §3.0,
§6.2, T30 API A6). Named tests: `test_status_etag_if_none_match_returns_304`,
`test_jobs_list_etag_if_none_match_returns_304`, `test_models_etag_if_none_match_returns_304`."""

from __future__ import annotations

from pathlib import Path

from tests.service.test_api_endpoints import AUTH, _client, _submit


def test_status_etag_if_none_match_returns_304(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    status_url = f"/v1/conversions/{job_id}"

    first = client.get(status_url, headers=AUTH)
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert etag
    assert first.headers["cache-control"] == "private, no-cache"

    revalidated = client.get(status_url, headers={**AUTH, "If-None-Match": etag})

    assert revalidated.status_code == 304
    assert revalidated.headers["etag"] == etag
    assert revalidated.headers["cache-control"] == "private, no-cache"
    assert revalidated.content == b""


def test_status_etag_changes_when_state_changes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]
    status_url = f"/v1/conversions/{job_id}"
    before = client.get(status_url, headers=AUTH).headers["etag"]

    client.delete(status_url, headers=AUTH)

    after = client.get(status_url, headers=AUTH).headers["etag"]
    assert after != before
    stale = client.get(status_url, headers={**AUTH, "If-None-Match": before})
    assert stale.status_code == 200


def test_jobs_list_etag_if_none_match_returns_304(tmp_path: Path) -> None:
    client = _client(tmp_path)
    _submit(client)

    first = client.get("/v1/conversions", headers=AUTH)
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert first.headers["cache-control"] == "private, no-cache"

    revalidated = client.get("/v1/conversions", headers={**AUTH, "If-None-Match": etag})

    assert revalidated.status_code == 304
    assert revalidated.headers["etag"] == etag
    assert revalidated.headers["cache-control"] == "private, no-cache"
    assert revalidated.content == b""


def test_models_etag_if_none_match_returns_304(tmp_path: Path) -> None:
    client = _client(tmp_path)

    first = client.get("/v1/models", headers=AUTH)
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert first.headers["cache-control"] == "private, no-cache"

    revalidated = client.get("/v1/models", headers={**AUTH, "If-None-Match": etag})

    assert revalidated.status_code == 304
    assert revalidated.headers["etag"] == etag
    assert revalidated.headers["cache-control"] == "private, no-cache"
    assert revalidated.content == b""


def test_submission_response_cache_control_is_no_store(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = _submit(client)

    assert response.headers["cache-control"] == "no-store"


def test_cancel_response_cache_control_is_no_store(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_id = _submit(client).json()["job_id"]

    response = client.delete(f"/v1/conversions/{job_id}", headers=AUTH)

    assert response.headers["cache-control"] == "no-store"

