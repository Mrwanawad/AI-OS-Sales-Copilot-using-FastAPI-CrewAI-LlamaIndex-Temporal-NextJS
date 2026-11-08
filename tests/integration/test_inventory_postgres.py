from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal as D
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.modules.inventory.adapters.outbound.postgres_store import PostgresUnitStore
from ai_os.modules.inventory.adapters.outbound.sql_guard import SqlGlotGuard
from ai_os.modules.inventory.domain.models import Unit, UnitStatus

from .conftest import PostgresUrls


def unit(code: str, price: str = "5000000", status: UnitStatus = UnitStatus.AVAILABLE) -> Unit:
    return Unit(code, "Palm Hills", "apartment", 3, D("150.00"), D(price), status, date(2027, 6, 1))


@pytest.fixture
async def store(postgres: PostgresUrls) -> AsyncIterator[PostgresUnitStore]:
    schema = f"s_{uuid4().hex}"
    args = {"server_settings": {"search_path": schema}}
    admin = create_async_engine(postgres.asynchronous, poolclass=NullPool)
    async with admin.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
    await admin.dispose()

    engine = create_async_engine(postgres.asynchronous, poolclass=NullPool, connect_args=args)
    unit_store = PostgresUnitStore(engine, engine, statement_timeout_ms=200)
    await unit_store.create_schema()
    yield unit_store
    async with engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    await engine.dispose()


async def test_upsert_round_trips_and_updates_price(store: PostgresUnitStore) -> None:
    await store.upsert([unit("A-101"), unit("A-102")])
    await store.upsert([unit("A-101", price="5750000.50", status=UnitStatus.RESERVED)])

    found = await store.get_by_codes(["A-101", "A-102", "missing"])

    assert [u.code for u in found] == ["A-101", "A-102"]
    assert found[0].price_egp == D("5750000.50")
    assert found[0].status is UnitStatus.RESERVED
    assert found[0].delivery_date == date(2027, 6, 1)


async def test_guarded_query_returns_rows(store: PostgresUnitStore) -> None:
    await store.upsert([unit("A-101"), unit("A-102", status=UnitStatus.SOLD)])
    sql = SqlGlotGuard().validate(
        "SELECT code, price_egp FROM units WHERE status = 'available' AND code ILIKE '%101%'"
    )

    rows = await store.run_read_only(sql)

    assert rows == [{"code": "A-101", "price_egp": D("5000000.00")}]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM units",
        "UPDATE units SET price_egp = 1",
        "INSERT INTO units (code) VALUES ('x')",
        "DROP TABLE units",
    ],
)
async def test_database_refuses_writes_even_if_guard_is_bypassed(
    store: PostgresUnitStore, sql: str
) -> None:
    await store.upsert([unit("A-101")])

    with pytest.raises(DBAPIError, match="read-only transaction"):
        await store.run_read_only(sql)

    [found] = await store.get_by_codes(["A-101"])
    assert found.price_egp == D("5000000.00")


async def test_slow_query_is_cancelled(store: PostgresUnitStore) -> None:
    with pytest.raises(DBAPIError, match="statement timeout"):
        await store.run_read_only("SELECT pg_sleep(5)")


async def test_read_only_setting_does_not_leak_to_later_writes(store: PostgresUnitStore) -> None:
    await store.run_read_only("SELECT 1")

    await store.upsert([unit("A-101")])

    assert len(await store.get_by_codes(["A-101"])) == 1
