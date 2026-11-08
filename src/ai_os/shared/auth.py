from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated, Protocol
from uuid import uuid4

import jwt
from fastapi import Depends, Header, HTTPException

_ALGORITHM = "HS256"
_MIN_SECRET_LENGTH = 32


class Role(StrEnum):
    REP = "rep"
    TEAM_LEAD = "team_lead"
    MANAGER = "manager"


@dataclass(frozen=True)
class Principal:
    subject: str
    role: Role
    token_id: str | None = None
    issued_at: datetime | None = None
    expires_at: datetime | None = None

    @property
    def leads_a_team(self) -> bool:
        return self.role is not Role.REP


LeadAccess = Callable[[Principal, str], Awaitable[bool]]


class Revocations(Protocol):
    async def is_revoked(self, principal: Principal) -> bool:
        """True for a logged-out token, or a user whose sessions were ended or disabled."""
        ...


def issue_token(secret: str, subject: str, role: Role, ttl: timedelta = timedelta(hours=8)) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": subject,
        "role": role.value,
        "iat": now,
        "exp": now + ttl,
        # Lets one token be logged out without ending the user's other sessions.
        "jti": uuid4().hex,
    }
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


class Authenticator:
    """Identity comes only from a signed token, never from a request body."""

    def __init__(
        self,
        get_secret: Callable[[], str | None],
        revocations: Callable[[], Revocations | None] | None = None,
        get_previous_secret: Callable[[], str | None] | None = None,
    ) -> None:
        self._get_secret = get_secret
        self._revocations = revocations or (lambda: None)
        # Accepted while a rotated secret's tokens expire, so rotation doesn't log everyone out.
        self._get_previous_secret = get_previous_secret or (lambda: None)

    async def __call__(self, authorization: Annotated[str | None, Header()] = None) -> Principal:
        secret = self._get_secret()
        if not secret or len(secret) < _MIN_SECRET_LENGTH:
            # A missing setting closes the API instead of opening it.
            raise HTTPException(status_code=503, detail="AUTH_SECRET is not configured")
        token = (authorization or "").removeprefix("Bearer ")
        principal = self._verify(token, secret)
        revocations = self._revocations()
        if revocations is not None and await revocations.is_revoked(principal):
            raise HTTPException(status_code=401, detail="Session ended")
        return principal

    def _verify(self, token: str, secret: str) -> Principal:
        secrets = [secret]
        previous = self._get_previous_secret()
        if previous and len(previous) >= _MIN_SECRET_LENGTH:
            secrets.append(previous)
        for candidate in secrets:
            try:
                claims = jwt.decode(
                    token,
                    candidate,
                    # Pinned, so a token can't choose its own verification method.
                    algorithms=[_ALGORITHM],
                    options={"require": ["exp", "sub", "role"]},
                )
                return Principal(
                    claims["sub"],
                    Role(claims["role"]),
                    claims.get("jti"),
                    _time(claims.get("iat")),
                    _time(claims["exp"]),
                )
            except jwt.InvalidSignatureError:
                continue
            except (jwt.InvalidTokenError, ValueError) as error:
                raise HTTPException(status_code=401, detail="Invalid token") from error
        raise HTTPException(status_code=401, detail="Invalid token")

    def require(self, *roles: Role) -> Callable[[Principal], Principal]:
        def check(principal: Annotated[Principal, Depends(self)]) -> Principal:
            if principal.role not in roles:
                raise HTTPException(status_code=403, detail="Your role can't do this")
            return principal

        return check


def _time(value: int | float | None) -> datetime | None:
    return datetime.fromtimestamp(value, UTC) if value is not None else None
