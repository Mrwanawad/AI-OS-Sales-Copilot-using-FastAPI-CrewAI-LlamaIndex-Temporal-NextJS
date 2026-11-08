from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    case,
    delete,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_os.modules.identity.domain.models import User

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("subject", String, primary_key=True),
    Column("display_name", String, nullable=False),
    Column("role", String, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("active", Boolean, nullable=False),
    Column("sessions_valid_after", DateTime(timezone=True)),
    Column("failed_logins", Integer, nullable=False),
    Column("locked_until", DateTime(timezone=True)),
)

revoked_tokens = Table(
    "revoked_tokens",
    metadata,
    Column("token_id", String, primary_key=True),
    # Kept only until the token would have expired anyway.
    Column("expires_at", DateTime(timezone=True), nullable=False, index=True),
)


class PostgresUserStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def get(self, subject: str) -> User | None:
        async with self._engine.connect() as connection:
            row = (
                (await connection.execute(select(users).where(users.c.subject == subject)))
                .mappings()
                .first()
            )
        return User(**row) if row else None

    async def add(self, user: User) -> bool:
        statement = (
            insert(users)
            .values(
                subject=user.subject,
                display_name=user.display_name,
                role=user.role,
                password_hash=user.password_hash,
                active=user.active,
                sessions_valid_after=user.sessions_valid_after,
                failed_logins=user.failed_logins,
                locked_until=user.locked_until,
            )
            .on_conflict_do_nothing(index_elements=[users.c.subject])
        )
        async with self._engine.begin() as connection:
            return (await connection.execute(statement)).rowcount == 1

    async def list(self) -> list[User]:
        async with self._engine.connect() as connection:
            rows = (await connection.execute(select(users).order_by(users.c.subject))).mappings()
            return [User(**row) for row in rows]

    async def record_failed_login(
        self, subject: str, max_failures: int, locked_until: datetime
    ) -> None:
        reached = users.c.failed_logins + 1 >= max_failures
        statement = (
            update(users)
            .where(users.c.subject == subject)
            .values(
                failed_logins=case((reached, 0), else_=users.c.failed_logins + 1),
                locked_until=case((reached, locked_until), else_=users.c.locked_until),
            )
        )
        await self._execute(statement)

    async def record_login(self, subject: str) -> None:
        await self._execute(
            update(users)
            .where(users.c.subject == subject)
            .values(failed_logins=0, locked_until=None)
        )

    async def end_sessions(self, subject: str, at: datetime) -> bool:
        return await self._update_user(subject, sessions_valid_after=at)

    async def set_active(self, subject: str, active: bool, at: datetime) -> bool:
        return await self._update_user(subject, active=active, sessions_valid_after=at)

    async def set_password_hash(self, subject: str, password_hash: str, at: datetime) -> bool:
        return await self._update_user(
            subject, password_hash=password_hash, sessions_valid_after=at, failed_logins=0
        )

    async def revoke_token(self, token_id: str, expires_at: datetime) -> None:
        statement = (
            insert(revoked_tokens)
            .values(token_id=token_id, expires_at=expires_at)
            .on_conflict_do_nothing(index_elements=[revoked_tokens.c.token_id])
        )
        await self._execute(statement)

    async def is_token_revoked(self, token_id: str) -> bool:
        query = select(revoked_tokens.c.token_id).where(revoked_tokens.c.token_id == token_id)
        async with self._engine.connect() as connection:
            return (await connection.execute(query)).first() is not None

    async def purge_expired_revocations(self, now: datetime) -> int:
        statement = delete(revoked_tokens).where(revoked_tokens.c.expires_at < now)
        async with self._engine.begin() as connection:
            return (await connection.execute(statement)).rowcount

    async def _update_user(self, subject: str, **values) -> bool:
        statement = update(users).where(users.c.subject == subject).values(**values)
        async with self._engine.begin() as connection:
            return (await connection.execute(statement)).rowcount == 1

    async def _execute(self, statement) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(statement)
