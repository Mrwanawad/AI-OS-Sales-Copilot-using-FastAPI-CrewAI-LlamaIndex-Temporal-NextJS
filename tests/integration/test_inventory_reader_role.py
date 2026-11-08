from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import make_url, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.modules.inventory.adapters.outbound.postgres_store import PostgresUnitStore

from .conftest import PostgresUrls

ROLE_SQL = Path(__file__).parents[2] / "deploy" / "inventory_reader.sql"


@pytest.fixture
async def reader_store(postgres: PostgresUrls) -> AsyncIterator[PostgresUnitStore]:
    admin = create_async_engine(postgres.asynchronous, poolclass=NullPool)
    bootstrap = PostgresUnitStore(admin, admin)
    await bootstrap.create_schema()
    async with admin.begin() as connection:
        await connection.execute(text("CREATE TABLE leads (phone text)"))
        await connection.execute(
            text(
                "INSERT INTO units VALUES "
                "('A-101', 'P', 'villa', 3, 100, 5000000, 'available', NULL)"
            )
        )
        for statement in ROLE_SQL.read_text().split(";"):
            lines = [li for li in statement.splitlines() if not li.startswith("--")]
            if "".join(lines).strip():
                await connection.execute(text("\n".join(lines)))

    reader_url = make_url(postgres.asynchronous).set(username="inventory_reader")
    reader = create_async_engine(reader_url, poolclass=NullPool)
    yield PostgresUnitStore(reader, reader)
    await reader.dispose()
    async with admin.begin() as connection:
        await connection.execute(text("DROP TABLE units, leads"))
        await connection.execute(text("DROP OWNED BY inventory_reader"))
        await connection.execute(text("DROP ROLE inventory_reader"))
    await admin.dispose()


async def test_reader_can_read_units(reader_store: PostgresUnitStore) -> None:
    rows = await reader_store.run_read_only("SELECT code FROM units")

    assert rows == [{"code": "A-101"}]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT phone FROM leads",
        "SELECT passwd FROM pg_shadow",
        "DELETE FROM units",
        "SELECT pg_read_file('postgresql.conf')",
    ],
)
async def test_reader_is_denied_everything_else(reader_store: PostgresUnitStore, sql: str) -> None:
    with pytest.raises(DBAPIError, match="permission denied|read-only transaction"):
        await reader_store.run_read_only(sql)


async def test_reader_cannot_write_outside_a_read_only_transaction(
    reader_store: PostgresUnitStore,
) -> None:
    with pytest.raises(DBAPIError, match="permission denied"):
        async with reader_store._engine.begin() as connection:
            await connection.execute(text("DELETE FROM units"))
