from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.modules.coaching.adapters.outbound.postgres_store import PostgresSessionStore
from ai_os.modules.coaching.domain.models import (
    CriterionScore,
    RolePlaySession,
    Scorecard,
    SessionStatus,
    Speaker,
    Turn,
)

from .conftest import PostgresUrls

SESSION = RolePlaySession(
    "s1",
    "rep-a",
    "investor",
    datetime(2026, 11, 7, tzinfo=UTC),
    turns=[Turn(Speaker.BUYER, "مرحبا"), Turn(Speaker.REP, "Hello")],
)


@pytest.fixture
async def store(postgres: PostgresUrls) -> AsyncIterator[PostgresSessionStore]:
    schema = f"s_{uuid4().hex}"
    admin = create_async_engine(postgres.asynchronous, poolclass=NullPool)
    async with admin.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
    await admin.dispose()
    engine = create_async_engine(
        postgres.asynchronous,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    sessions = PostgresSessionStore(engine)
    await sessions.create_schema()
    yield sessions
    async with engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    await engine.dispose()


async def test_session_round_trips_and_updates(store: PostgresSessionStore) -> None:
    await store.save(SESSION)
    assert await store.get("s1") == SESSION

    scorecard = Scorecard([CriterionScore("discovery", None, "Unverified")], None)
    finished = replace(SESSION, status=SessionStatus.FINISHED, scorecard=scorecard)
    await store.save(finished)

    assert await store.get("s1") == finished


async def test_missing_session_is_none(store: PostgresSessionStore) -> None:
    assert await store.get("missing") is None
