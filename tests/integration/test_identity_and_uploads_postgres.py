"""User accounts, uploaded files and the document registry on a real Postgres."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.modules.identity.adapters.outbound.postgres_store import PostgresUserStore
from ai_os.modules.identity.domain.models import User
from ai_os.modules.knowledge.adapters.outbound.document_registry import PostgresDocumentRegistry
from ai_os.modules.knowledge.domain.models import DocumentKind
from ai_os.shared.files import PostgresFileStore, StoredFileNotFound

from .conftest import PostgresUrls

NOW = datetime(2026, 11, 8, 10, tzinfo=UTC)


@pytest.fixture
async def engine(postgres: PostgresUrls) -> AsyncIterator[AsyncEngine]:
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
    yield engine
    await engine.dispose()


async def test_stored_files_round_trip_through_a_temporary_copy(engine: AsyncEngine) -> None:
    files = PostgresFileStore(engine)
    await files.create_schema()

    stored = await files.put("../../evil/prices.xlsx", "application/x", b"bytes", "lead-a")

    assert stored.reference.startswith("stored:") and stored.filename == "prices.xlsx"
    async with files.local_copy(stored.reference) as path:
        assert path.name == "prices.xlsx" and path.read_bytes() == b"bytes"
        copied = path
    assert not copied.exists()
    with pytest.raises(StoredFileNotFound):
        async with files.local_copy("stored:missing"):
            pass


async def test_concurrent_uploads_of_one_document_get_different_versions(
    engine: AsyncEngine,
) -> None:
    registry = PostgresDocumentRegistry(engine)
    await registry.create_schema()

    async def upload() -> int:
        return await registry.next_version(
            "bay-faq", "Demo Bay", DocumentKind.FAQ, 1, "stored:x", "lead-a", NOW
        )

    versions = await asyncio.gather(*(upload() for _ in range(8)))

    assert sorted(versions) == list(range(1, 9))
    assert (await registry.get("bay-faq")).latest_version == 8
    jump = await registry.next_version(
        "bay-faq", "Demo Bay", DocumentKind.FAQ, 20, "stored:y", "lead-a", NOW
    )
    assert jump == 20
    assert [d.id for d in await registry.list("Demo Bay", 10)] == ["bay-faq"]
    assert await registry.list("Other", 10) == []


async def test_user_store_counts_failures_locks_and_revokes(engine: AsyncEngine) -> None:
    users = PostgresUserStore(engine)
    await users.create_schema()
    assert await users.add(User("rep-a", "Rep A", "rep", "hash"))
    assert not await users.add(User("rep-a", "Again", "rep", "hash"))

    lock = NOW + timedelta(minutes=15)
    await asyncio.gather(*(users.record_failed_login("rep-a", 5, lock) for _ in range(4)))
    assert (await users.get("rep-a")).failed_logins == 4
    await users.record_failed_login("rep-a", 5, lock)
    locked = await users.get("rep-a")
    assert (locked.failed_logins, locked.locked_until) == (0, lock)
    await users.record_login("rep-a")
    assert (await users.get("rep-a")).locked_until is None

    assert await users.set_active("rep-a", False, NOW)
    disabled = await users.get("rep-a")
    assert (disabled.active, disabled.sessions_valid_after) == (False, NOW)
    assert not await users.end_sessions("ghost", NOW)

    await users.revoke_token("t1", NOW)
    await users.revoke_token("t1", NOW)
    await users.revoke_token("t2", NOW + timedelta(hours=8))
    assert await users.is_token_revoked("t1")
    assert await users.purge_expired_revocations(NOW + timedelta(hours=1)) == 1
    assert not await users.is_token_revoked("t1") and await users.is_token_revoked("t2")
