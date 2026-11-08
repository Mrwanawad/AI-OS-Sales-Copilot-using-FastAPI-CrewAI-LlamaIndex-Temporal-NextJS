from datetime import UTC, datetime, timedelta

import jwt
from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.modules.identity.adapters.outbound.postgres_store import PostgresUserStore
from ai_os.modules.identity.application.service import Identity
from ai_os.shared.auth import Principal, Role, issue_token
from ai_os.shared.config import Settings


class JwtIssuer:
    def __init__(self, secret: str, ttl: timedelta) -> None:
        self._secret = secret
        self._ttl = ttl

    def issue(self, subject: str, role: str) -> tuple[str, datetime]:
        token = issue_token(self._secret, subject, Role(role), self._ttl)
        expires = jwt.decode(token, options={"verify_signature": False})["exp"]
        return token, datetime.fromtimestamp(expires, UTC)


class IdentityRevocations:
    def __init__(self, identity: Identity) -> None:
        self._identity = identity

    async def is_revoked(self, principal: Principal) -> bool:
        return await self._identity.is_revoked(
            principal.subject, principal.token_id, principal.issued_at
        )


def build_identity(settings: Settings) -> Identity:
    if not settings.async_database_url:
        raise ValueError("Set ASYNC_DATABASE_URL")
    if settings.auth_secret is None:
        raise ValueError("Set AUTH_SECRET")
    return Identity(
        PostgresUserStore(create_async_engine(settings.async_database_url)),
        JwtIssuer(
            settings.auth_secret.get_secret_value(),
            timedelta(hours=settings.auth_token_hours),
        ),
        lambda: datetime.now(UTC),
    )
