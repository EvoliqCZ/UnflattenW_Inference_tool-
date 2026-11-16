"""`GET /v1/conversions` (design §3.4, T30 API A6): cursor pagination,
`state` filtering, and 400 on a malformed `state`/`limit`/`cursor`."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from tests.service.test_api_endpoints import AUTH, TOKEN, _client, _submit


def _submit_many(client: TestClient, count: int) -> list[str]:
    return [
        _submit(client, **{"Idempotency-Key": f"request-{index}"}).json()["job_id"]
        for index in range(count)
    ]


def test_list_jobs_returns_newest_first_and_scopes_to_principal(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_ids = _submit_many(client, 3)

    response = client.get("/v1/conversions", headers=AUTH)

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert [item["job_id"] for item in body["items"]] == list(reversed(job_ids))
    assert body["next_cursor"] is None


def test_list_jobs_paginates_with_cursor_across_pages(tmp_path: Path) -> None:
    # `active_jobs_per_principal` caps concurrently queued jobs at 4 (policy
    # defaults), so every submission here is immediately cancelled -- the
    # list still orders by `created_at DESC, id DESC`, unaffected by state.
    client = _client(tmp_path)
    job_ids = []
    for index in range(5):
        job_id = _submit(client, **{"Idempotency-Key": f"request-{index}"}).json()["job_id"]
        job_ids.append(job_id)
        client.delete(f"/v1/conversions/{job_id}", headers=AUTH)

    first = client.get("/v1/conversions", params={"limit": 2}, headers=AUTH).json()
    assert [item["job_id"] for item in first["items"]] == list(reversed(job_ids))[:2]
    assert first["next_cursor"] is not None

    second = client.get(
        "/v1/conversions", params={"limit": 2, "cursor": first["next_cursor"]}, headers=AUTH
    ).json()
    assert [item["job_id"] for item in second["items"]] == list(reversed(job_ids))[2:4]
    assert second["next_cursor"] is not None

    third = client.get(
        "/v1/conversions", params={"limit": 2, "cursor": second["next_cursor"]}, headers=AUTH
    ).json()
    assert [item["job_id"] for item in third["items"]] == list(reversed(job_ids))[4:]
    assert third["next_cursor"] is None


def test_list_jobs_filters_by_state(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_ids = _submit_many(client, 2)
    client.delete(f"/v1/conversions/{job_ids[0]}", headers=AUTH)

    cancelled = client.get("/v1/conversions", params={"state": "cancelled"}, headers=AUTH).json()
    queued = client.get("/v1/conversions", params={"state": "queued"}, headers=AUTH).json()

    assert [item["job_id"] for item in cancelled["items"]] == [job_ids[0]]
    assert [item["job_id"] for item in queued["items"]] == [job_ids[1]]


def test_list_jobs_repeated_state_is_an_or_filter(tmp_path: Path) -> None:
    client = _client(tmp_path)
    job_ids = _submit_many(client, 2)
    client.delete(f"/v1/conversions/{job_ids[0]}", headers=AUTH)

    response = client.get(
        "/v1/conversions",
        params=[("state", "cancelled"), ("state", "queued")],
        headers=AUTH,
    )

    assert {item["job_id"] for item in response.json()["items"]} == set(job_ids)


def test_list_jobs_unknown_state_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/conversions", params={"state": "bogus"}, headers=AUTH)

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_list_jobs_out_of_range_limit_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/v1/conversions", params={"limit": 0}, headers=AUTH).status_code == 400
    assert client.get("/v1/conversions", params={"limit": 101}, headers=AUTH).status_code == 400
    assert (
        client.get("/v1/conversions", params={"limit": "abc"}, headers=AUTH).status_code == 400
    )


def test_list_jobs_malformed_cursor_returns_400(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.get("/v1/conversions", params={"cursor": "not-a-cursor"}, headers=AUTH)

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


def test_list_jobs_requires_authentication(tmp_path: Path) -> None:
    client = _client(tmp_path)

    assert client.get("/v1/conversions").status_code == 401


def test_list_jobs_excludes_other_principals_jobs(tmp_path: Path) -> None:
    import hashlib as _hashlib
    from dataclasses import replace

    from unflatten.service.auth import Principal, PrincipalStore

    client = _client(tmp_path)
    _submit_many(client, 1)
    other_token = "b" * 32
    principal_a_caps = frozenset(
        {"submit_diagnostic", "read_status", "download_prediction_ir",
         "cancel_own_job", "download_docx"}
    )
    two_principals = PrincipalStore(
        [
            (_hashlib.sha256(TOKEN.encode()).digest(), Principal("principal_a", principal_a_caps)),
            (
                _hashlib.sha256(other_token.encode()).digest(),
                Principal("principal_b", frozenset({"read_status", "submit_diagnostic"})),
            ),
        ]
    )
    client.app.state.dependencies = replace(
        client.app.state.dependencies, principal_store=two_principals
    )

    response = client.get(
        "/v1/conversions", headers={"Authorization": f"Bearer {other_token}"}
    )

    assert response.status_code == 200
    assert response.json()["items"] == []


def test_list_jobs_summary_shape_omits_artifacts_and_diagnostic_outcome(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)
    _submit_many(client, 1)

    body = client.get("/v1/conversions", headers=AUTH).json()

    assert set(body["items"][0]) == {
        "job_id", "state", "stage", "requested_output", "model_variant", "label",
        "input_kind", "created_at", "updated_at", "terminal_at", "expires_at",
        "pages_total", "pages_done", "docx_produced", "error_code",
    }
