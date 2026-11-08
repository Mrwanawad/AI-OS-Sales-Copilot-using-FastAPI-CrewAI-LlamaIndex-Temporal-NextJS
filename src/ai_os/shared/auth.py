from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated

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

    @property
    def leads_a_team(self) -> bool:
        return self.role is not Role.REP


LeadAccess = Callable[[Principal, str], Awaitable[bool]]


def issue_token(secret: str, subject: str, role: Role, ttl: timedelta = timedelta(hours=8)) -> str:
    now = datetime.now(UTC)
    payload = {"sub": subject, "role": role.value, "iat": now, "exp": now + ttl}
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


class Authenticator:
    """Identity comes only from a signed token, never from a request body."""

    def __init__(self, get_secret: Callable[[], str | None]) -> None:
        self._get_secret = get_secret

    def __call__(self, authorization: Annotated[str | None, Header()] = None) -> Principal:
        secret = self._get_secret()
        if not secret or len(secret) < _MIN_SECRET_LENGTH:
            # A missing setting closes the API instead of opening it.
            raise HTTPException(status_code=503, detail="AUTH_SECRET is not configured")
        token = (authorization or "").removeprefix("Bearer ")
        try:
            claims = jwt.decode(
                token,
                secret,
                # Pinned, so a token can't choose its own verification method.
                algorithms=[_ALGORITHM],
                options={"require": ["exp", "sub", "role"]},
            )
            return Principal(claims["sub"], Role(claims["role"]))
        except (jwt.InvalidTokenError, ValueError) as error:
            raise HTTPException(status_code=401, detail="Invalid token") from error

    def require(self, *roles: Role) -> Callable[[Principal], Principal]:
        def check(principal: Annotated[Principal, Depends(self)]) -> Principal:
            if principal.role not in roles:
                raise HTTPException(status_code=403, detail="Your role can't do this")
            return principal

        return check
