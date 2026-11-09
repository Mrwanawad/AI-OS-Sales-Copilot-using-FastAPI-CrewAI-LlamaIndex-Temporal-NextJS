from pydantic import TypeAdapter
from sqlalchemy import Column, Integer, MetaData, String, Table, func, select
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_os.modules.coaching.domain.errors import SessionChanged
from ai_os.modules.coaching.domain.models import RolePlaySession

metadata = MetaData()

sessions = Table(
    "role_play_sessions",
    metadata,
    Column("id", String, primary_key=True),
    Column("rep_id", String, nullable=False, index=True),
    Column("payload", JSONB, nullable=False),
)

_SESSION = TypeAdapter(RolePlaySession)


class PostgresSessionStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def save(self, session: RolePlaySession) -> None:
        payload = _SESSION.dump_python(session, mode="json")
        statement = insert(sessions).values(id=session.id, rep_id=session.rep_id, payload=payload)
        stored_version = func.coalesce(sessions.c.payload["version"].astext.cast(Integer), 0)
        # Only the version this one was built from may be replaced, so concurrent saves
        # cannot silently drop each other's turns.
        statement = statement.on_conflict_do_update(
            index_elements=["id"],
            set_={"payload": payload},
            where=stored_version == session.version - 1,
        )
        async with self._engine.begin() as connection:
            result = await connection.execute(statement)
        if result.rowcount != 1:
            raise SessionChanged(session.id)

    async def get(self, session_id: str) -> RolePlaySession | None:
        query = select(sessions.c.payload).where(sessions.c.id == session_id)
        async with self._engine.connect() as connection:
            payload = (await connection.execute(query)).scalar_one_or_none()
        return _SESSION.validate_python(payload) if payload is not None else None
