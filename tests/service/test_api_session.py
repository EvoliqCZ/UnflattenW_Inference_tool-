"""T21 API A4 (design §2.3/§2.4): `GET/POST/DELETE /v1/session`,
`POST /v1/setup`, `PUT /v1/setup/password`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.service.test_api_endpoints import AUTH, PDF, TOKEN, _client

from unflatten.service.api import WebSettings
from unflatten.service.auth import Principal, PrincipalStore

PASSWORD = "correct horse battery"  # 22 code points, >= 12 minimum
NEW_PASSWORD = "another correct battery staple"
WEB_HEADERS = {
    "Content-Type": "application/json",
    "X-Unflatten-Client": "web",
    "Sec-Fetch-Site": "same-origin",
}
_ISO8601_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _web_settings(tmp_path: Path) -> WebSettings:
    return WebSettings(web_auth_path=tmp_path / "web-auth.json")


def _web_client(tmp_path: Path, **kwargs: object) -> TestClient:
    return _client(tmp_path, web=_web_settings(tmp_path), **kwargs)


def _setup(client: TestClient, password: str = PASSWORD):
    return client.post(
        "/v1/setup", json={"password": password}, headers={**AUTH, **WEB_HEADERS}
    )


# ---------------------------------------------------------------------------
# Setup/login/logout/whoami/password-change matrix
# ---------------------------------------------------------------------------


def test_session_get_before_setup_reports_setup_required_and_unauthenticated(
    tmp_path: Path,
) -> None:
    client = _web_client(tmp_path)

    response = client.get("/v1/session")

    assert response.status_code == 200
    body = response.json()
    assert body["setup_required"] is True
    assert body["authenticated"] is False
    assert body["principal_id"] is None
    assert body["capabilities"] is None
    assert body["auth_kind"] is None
    assert body["expires_at"] is None


def test_setup_requires_bearer_and_returns_session_with_cookie(tmp_path: Path) -> None:
    client = _web_client(tmp_path)

    response = _setup(client)

    assert response.status_code == 201
    body = response.json()
    assert body["authenticated"] is True
    assert body["setup_required"] is False
    assert body["principal_id"] == "principal_a"
    assert body["auth_kind"] == "session"
    assert body["capabilities"] == sorted(body["capabilities"])
    assert _ISO8601_UTC.match(body["expires_at"])
    assert "unflatten_session" in response.cookies


def test_setup_without_bearer_rejected_401(tmp_path: Path) -> None:
    client = _web_client(tmp_path)

    response = client.post("/v1/setup", json={"password": PASSWORD}, headers=WEB_HEADERS)

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_FAILED"


def test_setup_already_complete_returns_409(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    # A fresh, cookie-less client: `client` itself now also carries the
    # session cookie `_setup` returned, and sending both a cookie and a
    # bearer credential would (correctly) be rejected as ambiguous (400)
    # rather than exercising the already-complete path this test targets.
    bearer_only = _web_client(tmp_path)

    second = _setup(bearer_only)

    assert second.status_code == 409
    assert second.json()["code"] == "SETUP_ALREADY_COMPLETE"


def test_login_before_setup_returns_409_setup_required(tmp_path: Path) -> None:
    client = _web_client(tmp_path)

    response = client.post(
        "/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS
    )

    assert response.status_code == 409
    assert response.json()["code"] == "SETUP_REQUIRED"


def test_setup_login_whoami_password_change_logout_matrix(tmp_path: Path) -> None:
    same_origin_get = {"Sec-Fetch-Site": "same-origin"}
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    whoami = client.get("/v1/session", headers=same_origin_get)
    assert whoami.status_code == 200
    assert whoami.json()["authenticated"] is True
    assert whoami.json()["auth_kind"] == "session"

    # A second, cookie-less client logging in with the password.
    other = _web_client(tmp_path)
    login = other.post("/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS)
    assert login.status_code == 201
    assert login.json()["auth_kind"] == "session"

    wrong_password_client = _web_client(tmp_path)
    wrong_password = wrong_password_client.post(
        "/v1/session", json={"password": "not the password at all"}, headers=WEB_HEADERS
    )
    assert wrong_password.status_code == 401
    assert wrong_password.json()["code"] == "AUTHENTICATION_FAILED"

    changed = client.put(
        "/v1/setup/password",
        json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        headers=WEB_HEADERS,
    )
    assert changed.status_code == 204

    # The password change revoked every other session -- `other`'s cookie
    # is now a stale, revoked session, which is an invalid credential (401),
    # not "no credential" (200 authenticated=false).
    stale = other.get("/v1/session", headers=same_origin_get)
    assert stale.status_code == 401
    assert "unflatten_session=" in stale.headers["set-cookie"]
    assert "Max-Age=0" in stale.headers["set-cookie"]

    # `client`'s own session survives the password change.
    still_valid = client.get("/v1/session", headers=same_origin_get)
    assert still_valid.status_code == 200
    assert still_valid.json()["authenticated"] is True

    logout = client.delete("/v1/session", headers=WEB_HEADERS)
    assert logout.status_code == 204
    assert "Max-Age=0" in logout.headers["set-cookie"]

    after_logout = client.get("/v1/session", headers=same_origin_get)
    assert after_logout.status_code == 200
    assert after_logout.json()["authenticated"] is False


def test_password_change_wrong_current_password_rejected(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    response = client.put(
        "/v1/setup/password",
        json={"current_password": "wrong password entirely", "new_password": NEW_PASSWORD},
        headers=WEB_HEADERS,
    )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_FAILED"


def test_password_change_before_setup_returns_409(tmp_path: Path) -> None:
    client = _web_client(tmp_path)

    response = client.put(
        "/v1/setup/password",
        json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        headers=WEB_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["code"] == "SETUP_REQUIRED"


def test_delete_session_via_bearer_rejected_401(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    bearer_only = _web_client(tmp_path)

    response = bearer_only.delete(
        "/v1/session", headers={**AUTH, "X-Unflatten-Client": "web"}
    )

    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Credential ambiguity / cookie CSRF context (design §2.4, §11 bullet 3)
# ---------------------------------------------------------------------------


def test_cookie_and_bearer_together_returns_400(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    response = client.get("/v1/session", headers=AUTH)

    assert response.status_code == 400
    assert response.json()["code"] == "REQUEST_INVALID"


@pytest.mark.parametrize(
    "sec_fetch_site", ["same-site", "cross-site", None],
)
def test_cookie_request_rejects_disallowed_sec_fetch_site(
    tmp_path: Path, sec_fetch_site: str | None
) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    headers = {"X-Unflatten-Client": "web"}
    if sec_fetch_site is not None:
        headers["Sec-Fetch-Site"] = sec_fetch_site

    response = client.delete("/v1/session", headers=headers)

    assert response.status_code == 403
    assert response.json()["code"] == "WEB_REQUEST_REJECTED"


def test_cookie_without_client_header_returns_403_on_non_get(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    response = client.delete("/v1/session", headers={"Sec-Fetch-Site": "same-origin"})

    assert response.status_code == 403
    assert response.json()["code"] == "WEB_REQUEST_REJECTED"


def test_cookie_get_without_client_header_still_allowed(tmp_path: Path) -> None:
    """GET is a safe method: `X-Unflatten-Client` is only required for
    non-GET cookie requests (design §2.4)."""
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    response = client.get("/v1/session", headers={"Sec-Fetch-Site": "same-origin"})

    assert response.status_code == 200
    assert response.json()["authenticated"] is True


def test_invalid_session_401_clears_cookie_without_www_authenticate(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    client.cookies.set("unflatten_session", "not-a-real-session-token")

    response = client.get("/v1/session", headers={"Sec-Fetch-Site": "same-origin"})

    assert response.status_code == 401
    assert "WWW-Authenticate" not in response.headers
    assert "Max-Age=0" in response.headers["set-cookie"]


def test_cookie_without_client_header_returns_403_without_calling_asgi_receive(
    tmp_path: Path,
) -> None:
    """Mirrors `test_missing_auth_returns_without_calling_asgi_receive`: a
    cookie request missing `X-Unflatten-Client` on `/v1/conversions` (a
    non-GET route) must be rejected before the body is ever read."""
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    session_cookie = client.cookies["unflatten_session"]
    app = client.app

    messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        raise AssertionError("request body was read before the CSRF context check")

    async def send(message: dict[str, object]) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/conversions",
        "raw_path": b"/v1/conversions",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"cookie", f"unflatten_session={session_cookie}".encode()),
            (b"sec-fetch-site", b"same-origin"),
        ],
        "client": ("test", 1),
        "server": ("test", 80),
        "root_path": "",
    }

    asyncio.run(app(scope, receive, send))

    assert messages[0]["status"] == 403


# ---------------------------------------------------------------------------
# Rate limiting: only 401 login failures count, and only after the context
# check (design §2.6, §11 bullet 17).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "method", "extra_headers", "ok_body"),
    [
        ("/v1/session", "post", {}, {"password": PASSWORD}),
        ("/v1/setup", "post", AUTH, {"password": PASSWORD}),
        (
            "/v1/setup/password",
            "put",
            AUTH,
            {"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        ),
    ],
)
def test_context_rejections_do_not_feed_limiter(
    tmp_path: Path,
    path: str,
    method: str,
    extra_headers: dict[str, str],
    ok_body: dict[str, str],
) -> None:
    """Item 6: the same context-rejection-doesn't-feed-the-limiter contract
    (design §2.6/§11 bullet 17) holds for all three login/setup/password
    routes, not only `POST /v1/session`."""
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    fresh = _web_client(tmp_path)
    rejected_variants = [
        {},  # missing every context header
        {
            "Content-Type": "text/plain",
            "X-Unflatten-Client": "web",
            "Sec-Fetch-Site": "same-origin",
        },
        {"Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"},
        {"Content-Type": "application/json", "X-Unflatten-Client": "web"},
        {
            "Content-Type": "application/json",
            "X-Unflatten-Client": "web",
            "Sec-Fetch-Site": "same-site",
        },
        {
            "Content-Type": "application/json",
            "X-Unflatten-Client": "web",
            "Sec-Fetch-Site": "cross-site",
        },
    ]

    for headers in rejected_variants:
        response = fresh.request(
            method, path, json={"password": "wrong"}, headers={**extra_headers, **headers}
        )
        assert response.status_code == 403
        assert response.json()["code"] == "WEB_REQUEST_REJECTED"

    # A properly-contexted request afterwards must never see 429 -- the
    # rejections above must not have fed the limiter. Its own outcome
    # (200/201/204/409) depends on the route/fixture state, which is not
    # what this test is about.
    ok = fresh.request(method, path, json=ok_body, headers={**extra_headers, **WEB_HEADERS})
    assert ok.status_code != 429


def test_sixth_login_failure_rate_limited_with_retry_after(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    fresh = _web_client(tmp_path)

    for _ in range(5):
        response = fresh.post(
            "/v1/session", json={"password": "wrong password"}, headers=WEB_HEADERS
        )
        assert response.status_code == 401

    limited = fresh.post("/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS)

    assert limited.status_code == 429
    assert limited.json()["code"] == "RATE_LIMITED"
    body_retry_after = limited.json()["retry_after_seconds"]
    assert body_retry_after >= 1
    # Item 2: the HTTP `Retry-After` header must carry the same value as the
    # body's `retry_after_seconds`, not just the JSON body.
    assert limited.headers["Retry-After"] == str(body_retry_after)


# ---------------------------------------------------------------------------
# Item 1: `WebAuthFile.verify()` raising `WebAuthInvalid` (design §11 bullet 1).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_password", ["short", "x" * 300])
def test_login_password_shape_invalid_is_authentication_failure(
    tmp_path: Path, bad_password: str
) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    fresh = _web_client(tmp_path)

    response = fresh.post("/v1/session", json={"password": bad_password}, headers=WEB_HEADERS)

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_FAILED"


def test_login_password_shape_invalid_feeds_rate_limiter(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    fresh = _web_client(tmp_path)

    for _ in range(5):
        response = fresh.post("/v1/session", json={"password": "short"}, headers=WEB_HEADERS)
        assert response.status_code == 401

    limited = fresh.post("/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS)

    assert limited.status_code == 429
    assert limited.json()["code"] == "RATE_LIMITED"


@pytest.mark.parametrize("bad_password", ["short", "x" * 300])
def test_password_change_current_password_shape_invalid_is_authentication_failure(
    tmp_path: Path, bad_password: str
) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    response = client.put(
        "/v1/setup/password",
        json={"current_password": bad_password, "new_password": NEW_PASSWORD},
        headers=WEB_HEADERS,
    )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_FAILED"


def test_login_corrupt_web_auth_file_is_internal_error_not_401(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    web_auth_path = tmp_path / "web-auth.json"
    web_auth_path.chmod(0o644)  # mode changed at runtime -- now insecure
    fresh = _web_client(tmp_path)

    response = fresh.post("/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS)

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "INTERNAL_ERROR"
    assert str(web_auth_path) not in response.text
    assert "web-auth.json" not in response.text


def test_password_change_corrupt_web_auth_file_is_internal_error_not_401(
    tmp_path: Path,
) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    web_auth_path = tmp_path / "web-auth.json"
    web_auth_path.chmod(0o644)

    response = client.put(
        "/v1/setup/password",
        json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
        headers=WEB_HEADERS,
    )

    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "INTERNAL_ERROR"
    assert str(web_auth_path) not in response.text


def test_corrupt_web_auth_file_does_not_feed_rate_limiter(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    web_auth_path = tmp_path / "web-auth.json"
    web_auth_path.chmod(0o644)
    fresh = _web_client(tmp_path)

    for _ in range(6):
        response = fresh.post("/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS)
        assert response.status_code == 500

    web_auth_path.chmod(0o600)
    ok = fresh.post("/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS)
    assert ok.status_code == 201


# ---------------------------------------------------------------------------
# Item 4: `WWW-Authenticate` only on the bearer path (design §2.4/§11 bullet 4).
# ---------------------------------------------------------------------------


def test_password_change_wrong_current_password_via_cookie_has_no_www_authenticate(
    tmp_path: Path,
) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201  # client now carries the session cookie

    response = client.put(
        "/v1/setup/password",
        json={"current_password": "wrong password entirely", "new_password": NEW_PASSWORD},
        headers=WEB_HEADERS,
    )

    assert response.status_code == 401
    assert "WWW-Authenticate" not in response.headers


def test_password_change_wrong_current_password_via_bearer_has_www_authenticate(
    tmp_path: Path,
) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    bearer_only = _web_client(tmp_path)

    response = bearer_only.put(
        "/v1/setup/password",
        json={"current_password": "wrong password entirely", "new_password": NEW_PASSWORD},
        headers={**AUTH, **WEB_HEADERS},
    )

    assert response.status_code == 401
    assert response.headers.get("WWW-Authenticate") == "Bearer"


# ---------------------------------------------------------------------------
# Item 5: full `Set-Cookie` attributes (design §2.2/§11 bullet 2).
# ---------------------------------------------------------------------------


def test_session_cookie_has_full_attributes(tmp_path: Path) -> None:
    client = _web_client(tmp_path)

    response = _setup(client)

    set_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    assert "samesite=strict" in set_cookie.lower()
    assert "Path=/" in set_cookie
    assert "Domain=" not in set_cookie
    assert f"Max-Age={_web_settings(tmp_path).session_max_seconds}" in set_cookie


# ---------------------------------------------------------------------------
# CP4 D1: `x-unflatten-client` must be a known submission header, on both the
# cookie and bearer branches (`docs/runs/web-app-CP4-2026-09-27.md`).
# ---------------------------------------------------------------------------

_SUBMISSION_HEADERS = {
    "Content-Type": "application/pdf",
    "Idempotency-Key": "cp4-d1-request",
    "X-Unflatten-Profile": "word-core-v0",
    "X-Unflatten-Output": "prediction_ir",
}


def test_cookie_submission_with_client_header_returns_202(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    response = client.post(
        "/v1/conversions",
        content=PDF,
        headers={
            **_SUBMISSION_HEADERS,
            "X-Unflatten-Client": "web",
            "Sec-Fetch-Site": "same-origin",
        },
    )

    assert response.status_code == 202
    assert response.json()["state"] == "queued"


def test_bearer_submission_with_client_header_is_not_rejected(tmp_path: Path) -> None:
    client = _web_client(tmp_path)

    response = client.post(
        "/v1/conversions",
        content=PDF,
        headers={**AUTH, **_SUBMISSION_HEADERS, "X-Unflatten-Client": "web"},
    )

    assert response.status_code == 202
    assert response.json()["state"] == "queued"


# ---------------------------------------------------------------------------
# CP4 minor 1: the cookie/login path never sends `WWW-Authenticate` (design
# §2.4) -- `POST /v1/session` never accepts a bearer credential, so a wrong
# password must not hint at one.
# ---------------------------------------------------------------------------


def test_login_wrong_password_401_has_no_www_authenticate(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201
    fresh = _web_client(tmp_path)

    response = fresh.post(
        "/v1/session", json={"password": "not the password at all"}, headers=WEB_HEADERS
    )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_FAILED"
    assert "WWW-Authenticate" not in response.headers


# ---------------------------------------------------------------------------
# RequestValidationError -> 400 problem
# ---------------------------------------------------------------------------


def test_request_validation_error_maps_to_400_problem(tmp_path: Path) -> None:
    client = _web_client(tmp_path)
    assert _setup(client).status_code == 201

    response = client.get(
        "/v1/conversions/job_x/artifacts/prediction_ir/not-an-integer", headers=AUTH
    )

    assert response.status_code == 400
    body = response.json()
    assert body["code"] == "REQUEST_INVALID"
    assert response.headers["content-type"] == "application/problem+json"


# ---------------------------------------------------------------------------
# Tokenless first-run setup (owner 2026-09-29, DEC-052 addendum): a loopback-
# only deployment sets `WebSettings.setup_tokenless`, and `POST /v1/setup`
# then needs no bearer token -- only the request context and a password.
# ---------------------------------------------------------------------------


def _tokenless_client(tmp_path: Path) -> TestClient:
    return _client(
        tmp_path,
        web=WebSettings(web_auth_path=tmp_path / "web-auth.json", setup_tokenless=True),
    )


def _tokenless_setup(client: TestClient, password: str = PASSWORD):
    return client.post("/v1/setup", json={"password": password}, headers=WEB_HEADERS)


def test_session_get_with_tokenless_flag_before_setup_reports_setup_tokenless_true(
    tmp_path: Path,
) -> None:
    client = _tokenless_client(tmp_path)

    body = client.get("/v1/session").json()

    assert body["schema_version"] == "1.1"
    assert body["setup_required"] is True
    assert body["setup_tokenless"] is True


def test_session_get_without_tokenless_flag_reports_setup_tokenless_false(
    tmp_path: Path,
) -> None:
    client = _web_client(tmp_path)

    body = client.get("/v1/session").json()

    assert body["setup_required"] is True
    assert body["setup_tokenless"] is False


def test_setup_tokenless_with_flag_binds_sole_principal_and_returns_session(
    tmp_path: Path,
) -> None:
    client = _tokenless_client(tmp_path)

    response = _tokenless_setup(client)

    assert response.status_code == 201
    body = response.json()
    assert body["authenticated"] is True
    assert body["principal_id"] == "principal_a"
    assert body["auth_kind"] == "session"
    assert body["setup_tokenless"] is False
    assert "unflatten_session" in response.cookies


def test_setup_tokenless_without_flag_returns_401_and_writes_nothing(tmp_path: Path) -> None:
    client = _web_client(tmp_path)

    response = _tokenless_setup(client)

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_FAILED"
    assert not (tmp_path / "web-auth.json").exists()


def test_setup_tokenless_second_attempt_returns_409(tmp_path: Path) -> None:
    client = _tokenless_client(tmp_path)
    assert _tokenless_setup(client).status_code == 201
    stranger = _tokenless_client(tmp_path)

    second = _tokenless_setup(stranger, password="a stranger's own password")
    after = stranger.get("/v1/session").json()

    assert second.status_code == 409
    assert second.json()["code"] == "SETUP_ALREADY_COMPLETE"
    assert after["setup_required"] is False
    assert after["setup_tokenless"] is False


def test_setup_with_tokenless_flag_still_accepts_bearer(tmp_path: Path) -> None:
    client = _tokenless_client(tmp_path)

    response = _setup(client)

    assert response.status_code == 201


def test_setup_with_tokenless_flag_and_wrong_bearer_returns_401(tmp_path: Path) -> None:
    """A present-but-wrong credential is never silently downgraded to the
    tokenless path -- it is the ordinary 401."""
    client = _tokenless_client(tmp_path)

    response = client.post(
        "/v1/setup",
        json={"password": PASSWORD},
        headers={"Authorization": "Bearer " + "b" * 32, **WEB_HEADERS},
    )

    assert response.status_code == 401
    assert not (tmp_path / "web-auth.json").exists()


def test_setup_tokenless_with_two_principals_is_not_offered(tmp_path: Path) -> None:
    """Two principals: tokenless setup cannot tell whose session to bind, so
    it is not offered and `POST /v1/setup` still needs a bearer token."""
    client = _tokenless_client(tmp_path)
    capabilities = frozenset({"read_status"})
    client.app.state.dependencies = dataclasses.replace(
        client.app.state.dependencies,
        principal_store=PrincipalStore(
            [
                (hashlib.sha256(TOKEN.encode()).digest(), Principal("principal_a", capabilities)),
                (hashlib.sha256(b"c" * 32).digest(), Principal("principal_b", capabilities)),
            ]
        ),
    )

    assert client.get("/v1/session").json()["setup_tokenless"] is False
    assert _tokenless_setup(client).status_code == 401
    assert not (tmp_path / "web-auth.json").exists()


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {
            "Content-Type": "text/plain",
            "X-Unflatten-Client": "web",
            "Sec-Fetch-Site": "same-origin",
        },
        {"Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"},
        {"Content-Type": "application/json", "X-Unflatten-Client": "web"},
        {
            "Content-Type": "application/json",
            "X-Unflatten-Client": "web",
            "Sec-Fetch-Site": "same-site",
        },
        {
            "Content-Type": "application/json",
            "X-Unflatten-Client": "web",
            "Sec-Fetch-Site": "cross-site",
        },
    ],
)
def test_setup_tokenless_bad_request_context_returns_403(
    tmp_path: Path, headers: dict[str, str]
) -> None:
    client = _tokenless_client(tmp_path)

    response = client.post("/v1/setup", json={"password": PASSWORD}, headers=headers)

    assert response.status_code == 403
    assert response.json()["code"] == "WEB_REQUEST_REJECTED"
    assert not (tmp_path / "web-auth.json").exists()


def test_setup_tokenless_when_rate_limited_returns_429(tmp_path: Path) -> None:
    """The limiter check still runs before the tokenless branch."""
    client = _tokenless_client(tmp_path)
    limiter = client.app.state.dependencies.login_rate_limiter
    for _ in range(5):
        limiter.record_failure()

    response = _tokenless_setup(client)

    assert response.status_code == 429
    assert response.json()["code"] == "RATE_LIMITED"
    assert not (tmp_path / "web-auth.json").exists()


def test_setup_tokenless_success_does_not_count_as_a_failure(tmp_path: Path) -> None:
    """A tokenless setup leaves the limiter exactly as a bearer setup does:
    four earlier failures plus the tokenless success still allow a login."""
    client = _tokenless_client(tmp_path)
    limiter = client.app.state.dependencies.login_rate_limiter
    for _ in range(4):
        limiter.record_failure()
    assert _tokenless_setup(client).status_code == 201

    login = _web_client(tmp_path).post(
        "/v1/session", json={"password": PASSWORD}, headers=WEB_HEADERS
    )

    assert login.status_code == 201


def test_web_settings_tokenless_with_non_loopback_allowed_host_raises() -> None:
    """A LAN deployment must allow-list its LAN host name -- tokenless setup
    and such an allowlist together are a misconfiguration, refused at start."""
    with pytest.raises(ValueError, match="setup_tokenless"):
        WebSettings(allowed_hosts=("127.0.0.1", "192.168.1.20"), setup_tokenless=True)


def test_web_settings_tokenless_with_default_loopback_hosts_is_accepted() -> None:
    assert WebSettings(setup_tokenless=True).setup_tokenless is True
