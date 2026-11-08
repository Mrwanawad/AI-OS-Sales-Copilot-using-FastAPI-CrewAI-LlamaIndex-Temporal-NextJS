from datetime import timedelta

import jwt
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.crm_copilot.application.access import can_access_lead
from ai_os.shared.auth import Authenticator, Principal, Role, issue_token

from .auth_helpers import AUTH, SECRET, bearer
from .crm_fakes import FakeCrmStore


def client(auth: Authenticator = AUTH) -> TestClient:
    app = FastAPI()

    @app.get("/me")
    def me(user: Principal = Depends(auth)) -> dict[str, str]:  # noqa: B008
        return {"subject": user.subject, "role": user.role.value}

    @app.get("/lead-only")
    def lead_only(
        user: Principal = Depends(auth.require(Role.TEAM_LEAD, Role.MANAGER)),  # noqa: B008
    ) -> dict[str, str]:
        return {"subject": user.subject}

    return TestClient(app)


def with_token(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_valid_token_gives_identity_and_role() -> None:
    response = client().get("/me", headers=bearer("lead-a", Role.TEAM_LEAD))

    assert response.json() == {"subject": "lead-a", "role": "team_lead"}


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer not-a-token"},
        {"Authorization": "Basic abc"},
    ],
)
def test_missing_or_malformed_tokens_are_rejected(headers: dict[str, str]) -> None:
    assert client().get("/me", headers=headers).status_code == 401


def test_expired_token_is_rejected() -> None:
    token = issue_token(SECRET, "rep-a", Role.REP, ttl=timedelta(seconds=-1))

    assert client().get("/me", headers=with_token(token)).status_code == 401


def test_token_signed_with_another_secret_is_rejected() -> None:
    token = issue_token("another-secret-that-is-also-long-enough", "rep-a", Role.MANAGER)

    assert client().get("/me", headers=with_token(token)).status_code == 401


def test_unsigned_token_is_rejected() -> None:
    forged = jwt.encode(
        {"sub": "rep-a", "role": "manager", "exp": 9999999999}, "", algorithm="none"
    )

    assert client().get("/me", headers=with_token(forged)).status_code == 401


@pytest.mark.parametrize(
    "claims",
    [
        {"sub": "rep-a", "role": "rep"},
        {"role": "rep", "exp": 9999999999},
        {"sub": "rep-a", "exp": 9999999999},
        {"sub": "rep-a", "role": "superuser", "exp": 9999999999},
    ],
)
def test_token_with_missing_or_unknown_claims_is_rejected(claims: dict) -> None:
    token = jwt.encode(claims, SECRET, algorithm="HS256")

    assert client().get("/me", headers=with_token(token)).status_code == 401


@pytest.mark.parametrize("secret", [None, "", "too-short"])
def test_api_closes_when_the_secret_is_missing_or_weak(secret: str | None) -> None:
    closed = client(Authenticator(lambda: secret))

    assert closed.get("/me", headers=bearer()).status_code == 503


def test_role_check() -> None:
    test_client = client()

    assert test_client.get("/lead-only", headers=bearer("rep-a", Role.REP)).status_code == 403
    assert test_client.get("/lead-only", headers=bearer("l", Role.TEAM_LEAD)).status_code == 200
    assert test_client.get("/lead-only", headers=bearer("m", Role.MANAGER)).status_code == 200
    assert test_client.get("/lead-only").status_code == 401


async def test_reps_reach_only_their_own_leads() -> None:
    store = FakeCrmStore()

    assert await can_access_lead(store, Principal("rep-a", Role.REP), "L1")
    assert not await can_access_lead(store, Principal("rep-b", Role.REP), "L1")
    assert not await can_access_lead(store, Principal("rep-a", Role.REP), "unknown")
    assert await can_access_lead(store, Principal("lead-a", Role.TEAM_LEAD), "L1")
