import re
from collections.abc import Callable
from datetime import datetime

from ai_os.modules.identity.domain.errors import (
    AccountLocked,
    InvalidCredentials,
    InvalidUser,
    UserAlreadyExists,
    UserNotFound,
)
from ai_os.modules.identity.domain.models import (
    LOCKOUT,
    MAX_FAILED_LOGINS,
    ROLES,
    IssuedToken,
    User,
)
from ai_os.modules.identity.domain.passwords import DUMMY_HASH, hash_password, verify_password
from ai_os.modules.identity.ports.users import TokenIssuer, UserStore

_SUBJECT = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")


class Identity:
    def __init__(self, store: UserStore, issuer: TokenIssuer, now: Callable[[], datetime]) -> None:
        self._store = store
        self._issuer = issuer
        self._now = now

    def _cutoff(self) -> datetime:
        # Token times are whole seconds; flooring keeps a login right after a reset valid.
        return self._now().replace(microsecond=0)

    async def login(self, subject: str, password: str) -> IssuedToken:
        subject = subject.strip().lower()
        user = await self._store.get(subject)
        now = self._now()
        if user is not None and user.is_locked(now):
            raise AccountLocked()
        # Always runs scrypt, so response time doesn't reveal which usernames exist.
        matches = verify_password(password, user.password_hash if user else DUMMY_HASH)
        if user is None or not user.active or not matches:
            if user is not None:
                await self._store.record_failed_login(subject, MAX_FAILED_LOGINS, now + LOCKOUT)
            raise InvalidCredentials()
        await self._store.record_login(subject)
        token, expires_at = self._issuer.issue(user.subject, user.role)
        return IssuedToken(token, user.subject, user.role, user.display_name, expires_at)

    async def logout(self, token_id: str | None, expires_at: datetime | None) -> None:
        if token_id and expires_at:
            await self._store.revoke_token(token_id, expires_at)

    async def is_revoked(
        self, subject: str, token_id: str | None, issued_at: datetime | None
    ) -> bool:
        user = await self._store.get(subject)
        if user is None or not user.active:
            return True
        cutoff = user.sessions_valid_after
        if cutoff is not None and (issued_at is None or issued_at < cutoff):
            return True
        return token_id is not None and await self._store.is_token_revoked(token_id)

    async def create_user(self, subject: str, display_name: str, role: str, password: str) -> User:
        subject = subject.strip().lower()
        if not _SUBJECT.match(subject):
            raise InvalidUser("Usernames are 2-64 lowercase letters, digits, '.', '-' or '_'")
        if role not in ROLES:
            raise InvalidUser(f"Role must be one of {', '.join(sorted(ROLES))}")
        if not display_name.strip():
            raise InvalidUser("A display name is required")
        user = User(subject, display_name.strip(), role, hash_password(password))
        if not await self._store.add(user):
            raise UserAlreadyExists(subject)
        return user

    async def list_users(self) -> list[User]:
        return await self._store.list()

    async def end_sessions(self, subject: str) -> None:
        if not await self._store.end_sessions(subject, self._cutoff()):
            raise UserNotFound(subject)

    async def set_active(self, subject: str, active: bool) -> None:
        # Disabling also ends sessions, so a departing rep loses access at once.
        if not await self._store.set_active(subject, active, self._cutoff()):
            raise UserNotFound(subject)

    async def change_password(self, subject: str, current: str, new: str) -> None:
        user = await self._store.get(subject)
        if user is None or not verify_password(current, user.password_hash):
            raise InvalidCredentials()
        # Other sessions end too: a changed password usually means a leaked one.
        await self._store.set_password_hash(subject, hash_password(new), self._cutoff())

    async def reset_password(self, subject: str, new: str) -> None:
        if not await self._store.set_password_hash(subject, hash_password(new), self._cutoff()):
            raise UserNotFound(subject)

    async def purge_expired_revocations(self) -> int:
        return await self._store.purge_expired_revocations(self._now())
