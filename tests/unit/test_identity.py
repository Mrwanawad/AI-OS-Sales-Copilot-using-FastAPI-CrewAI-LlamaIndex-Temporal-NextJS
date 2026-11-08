from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.bootstrap.identity import IdentityRevocations, JwtIssuer
from ai_os.modules.identity.adapters.inbound.api import create_router
from ai_os.modules.identity.application.service import Identity
from ai_os.modules.identity.domain.errors import (
    AccountLocked,
    InvalidCredentials,
    InvalidUser,
    UserAlreadyExists,
)
from ai_os.modules.identity.domain.models import MAX_FAILED_LOGINS, User
from ai_os.modules.identity.domain.passwords import (
    WeakPassword,
    hash_password,
    verify_password,
)
from ai_os.shared.auth import Authenticator

from .auth_helpers import SECRET

PASSWORD = "correct horse battery"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 11, 8, 10, 0, 0, 500_000, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class FakeUserStore:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}
        self.revoked: dict[str, datetime] = {}

    async def get(self, subject: str) -> User | None:
        return self.users.get(subject)

    async def add(self, user: User) -> bool:
        if user.subject in self.users:
            return False
        self.users[user.subject] = user
        return True

    async def list(self) -> list[User]:
        return sorted(self.users.values(), key=lambda u: u.subject)

    async def record_failed_login(
        self, subject: str, max_failures: int, locked_until: datetime
    ) -> None:
        user = self.users[subject]
        failures = user.failed_logins + 1
        if failures >= max_failures:
            self.users[subject] = replace(user, failed_logins=0, locked_until=locked_until)
        else:
            self.users[subject] = replace(user, failed_logins=failures)

    async def record_login(self, subject: str) -> None:
        self.users[subject] = replace(self.users[subject], failed_logins=0, locked_until=None)

    async def end_sessions(self, subject: str, at: datetime) -> bool:
        return self._set(subject, sessions_valid_after=at)

    async def set_active(self, subject: str, active: bool, at: datetime) -> bool:
        return self._set(subject, active=active, sessions_valid_after=at)

    async def set_password_hash(self, subject: str, password_hash: str, at: datetime) -> bool:
        return self._set(subject, password_hash=password_hash, sessions_valid_after=at)

    async def revoke_token(self, token_id: str, expires_at: datetime) -> None:
        self.revoked[token_id] = expires_at

    async def is_token_revoked(self, token_id: str) -> bool:
        return token_id in self.revoked

    async def purge_expired_revocations(self, now: datetime) -> int:
        expired = [t for t, at in self.revoked.items() if at < now]
        for token_id in expired:
            del self.revoked[token_id]
        return len(expired)

    def _set(self, subject: str, **values) -> bool:
        if subject not in self.users:
            return False
        self.users[subject] = replace(self.users[subject], **values)
        return True


def build() -> tuple[Identity, FakeUserStore, Clock]:
    store, clock = FakeUserStore(), Clock()
    return Identity(store, JwtIssuer(SECRET, timedelta(hours=8)), clock), store, clock


def test_password_hash_round_trips_and_rejects_wrong_or_short_passwords() -> None:
    stored = hash_password(PASSWORD)

    assert verify_password(PASSWORD, stored)
    assert not verify_password("wrong password!!", stored)
    assert not verify_password(PASSWORD, "not-a-hash")
    with pytest.raises(WeakPassword):
        hash_password("short")


async def test_login_issues_a_token_and_wrong_password_and_unknown_user_look_the_same() -> None:
    identity, _, _ = build()
    await identity.create_user("Rep-A", "Rep A", "rep", PASSWORD)

    issued = await identity.login("rep-a", PASSWORD)

    assert issued.subject == "rep-a" and issued.role == "rep"
    with pytest.raises(InvalidCredentials) as wrong:
        await identity.login("rep-a", "not the password")
    with pytest.raises(InvalidCredentials) as unknown:
        await identity.login("nobody", PASSWORD)
    assert str(wrong.value) == str(unknown.value)


async def test_repeated_failures_lock_the_account_until_the_lock_expires() -> None:
    identity, _, clock = build()
    await identity.create_user("rep-a", "Rep A", "rep", PASSWORD)

    for _ in range(MAX_FAILED_LOGINS):
        with pytest.raises(InvalidCredentials):
            await identity.login("rep-a", "wrong password")

    with pytest.raises(AccountLocked):
        await identity.login("rep-a", PASSWORD)
    clock.now += timedelta(minutes=16)
    assert (await identity.login("rep-a", PASSWORD)).subject == "rep-a"


async def test_disabled_user_cannot_log_in_and_existing_tokens_stop_working() -> None:
    identity, _, clock = build()
    await identity.create_user("rep-a", "Rep A", "rep", PASSWORD)
    issued_at = clock.now.replace(microsecond=0) - timedelta(seconds=1)

    await identity.set_active("rep-a", False)

    with pytest.raises(InvalidCredentials):
        await identity.login("rep-a", PASSWORD)
    assert await identity.is_revoked("rep-a", "t1", issued_at)


async def test_ending_sessions_refuses_older_tokens_but_not_a_new_login() -> None:
    identity, _, clock = build()
    await identity.create_user("rep-a", "Rep A", "rep", PASSWORD)
    old = clock.now - timedelta(minutes=5)

    await identity.end_sessions("rep-a")

    assert await identity.is_revoked("rep-a", "old", old)
    # A login in the same second as the reset carries a whole-second iat.
    assert not await identity.is_revoked("rep-a", "new", clock.now.replace(microsecond=0))


async def test_logout_revokes_only_that_token() -> None:
    identity, store, clock = build()
    await identity.create_user("rep-a", "Rep A", "rep", PASSWORD)

    await identity.logout("t1", clock.now + timedelta(hours=1))

    assert await identity.is_revoked("rep-a", "t1", clock.now)
    assert not await identity.is_revoked("rep-a", "t2", clock.now)
    clock.now += timedelta(hours=2)
    assert await identity.purge_expired_revocations() == 1


async def test_unknown_subject_tokens_are_revoked() -> None:
    identity, _, clock = build()

    assert await identity.is_revoked("ghost", "t1", clock.now)


async def test_create_user_validates_input_and_refuses_duplicates() -> None:
    identity, _, _ = build()
    await identity.create_user("rep-a", "Rep A", "rep", PASSWORD)

    with pytest.raises(UserAlreadyExists):
        await identity.create_user("REP-A", "Again", "rep", PASSWORD)
    with pytest.raises(InvalidUser):
        await identity.create_user("rep a/../", "Bad", "rep", PASSWORD)
    with pytest.raises(InvalidUser):
        await identity.create_user("rep-b", "Rep B", "admin", PASSWORD)


async def test_change_password_needs_the_current_one_and_ends_other_sessions() -> None:
    identity, store, clock = build()
    await identity.create_user("rep-a", "Rep A", "rep", PASSWORD)

    with pytest.raises(InvalidCredentials):
        await identity.change_password("rep-a", "wrong password", "a brand new password")
    await identity.change_password("rep-a", PASSWORD, "a brand new password")

    assert store.users["rep-a"].sessions_valid_after is not None
    assert (await identity.login("rep-a", "a brand new password")).subject == "rep-a"


def api() -> tuple[TestClient, Identity]:
    identity, _, _ = build()
    auth = Authenticator(lambda: SECRET, lambda: IdentityRevocations(identity))
    app = FastAPI()
    app.include_router(create_router(lambda: identity, auth))
    return TestClient(app), identity


def login(client: TestClient, username: str, password: str = PASSWORD) -> dict[str, str]:
    response = client.post("/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_api_login_me_and_logout_revokes_the_token() -> None:
    client, identity = api()
    with client:
        client.portal.call(identity.create_user, "rep-a", "Rep A", "rep", PASSWORD)
        headers = login(client, "rep-a")

        assert client.get("/auth/me", headers=headers).json()["username"] == "rep-a"
        assert client.post("/auth/logout", headers=headers).status_code == 204
        assert client.get("/auth/me", headers=headers).status_code == 401


def test_api_wrong_password_is_401_and_lock_is_429() -> None:
    client, identity = api()
    with client:
        client.portal.call(identity.create_user, "rep-a", "Rep A", "rep", PASSWORD)
        for _ in range(MAX_FAILED_LOGINS):
            bad = {"username": "rep-a", "password": "wrong password"}
            assert client.post("/auth/login", json=bad).status_code == 401
        good = {"username": "rep-a", "password": PASSWORD}
        assert client.post("/auth/login", json=good).status_code == 429


def test_api_only_managers_manage_users_and_disabling_ends_sessions() -> None:
    client, identity = api()
    with client:
        client.portal.call(identity.create_user, "boss", "Boss", "manager", PASSWORD)
        client.portal.call(identity.create_user, "rep-a", "Rep A", "rep", PASSWORD)
        rep = login(client, "rep-a")
        boss = login(client, "boss")
        new_user = {
            "username": "rep-b",
            "display_name": "Rep B",
            "role": "rep",
            "password": PASSWORD,
        }

        assert client.post("/auth/users", json=new_user, headers=rep).status_code == 403
        assert client.post("/auth/users", json=new_user, headers=boss).status_code == 201
        assert client.post("/auth/users", json=new_user, headers=boss).status_code == 409
        assert client.post("/auth/users/boss/disable", headers=boss).status_code == 409
        assert client.post("/auth/users/rep-a/disable", headers=boss).status_code == 204
        assert client.get("/auth/me", headers=rep).status_code == 401
        assert client.post("/auth/users/ghost/disable", headers=boss).status_code == 404
        users = client.get("/auth/users", headers=boss).json()
        assert {u["username"]: u["active"] for u in users} == {
            "boss": True,
            "rep-a": False,
            "rep-b": True,
        }
