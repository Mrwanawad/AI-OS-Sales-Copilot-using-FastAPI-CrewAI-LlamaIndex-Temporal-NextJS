from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy import Column, Date, Integer, MetaData, Numeric, String, Table, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_os.modules.inventory.domain.models import Unit, UnitStatus

metadata = MetaData()

units = Table(
    "units",
    metadata,
    Column("code", String, primary_key=True),
    Column("project", String, nullable=False, index=True),
    Column("unit_type", String, nullable=False),
    Column("bedrooms", Integer, nullable=False),
    Column("area_sqm", Numeric(10, 2), nullable=False),
    Column("price_egp", Numeric(14, 2), nullable=False),
    Column("status", String, nullable=False, index=True),
    Column("delivery_date", Date),
)


class PostgresUnitStore:
    def __init__(
        self, engine: AsyncEngine, read_only_engine: AsyncEngine, statement_timeout_ms: int = 3000
    ) -> None:
        self._engine = engine
        self._read_only_engine = read_only_engine
        self._statement_timeout_ms = int(statement_timeout_ms)

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def get_by_codes(self, codes: Sequence[str]) -> list[Unit]:
        if not codes:
            return []
        query = select(units).where(units.c.code.in_(codes)).order_by(units.c.code)
        async with self._read_only_engine.connect() as connection:
            rows = (await connection.execute(query)).mappings().all()
        return [_to_unit(row) for row in rows]

    async def find_available(
        self, max_price: Decimal | None, bedrooms: int | None, project: str | None, limit: int
    ) -> list[Unit]:
        query = select(units).where(units.c.status == UnitStatus.AVAILABLE.value)
        if max_price is not None:
            query = query.where(units.c.price_egp <= max_price)
        if bedrooms is not None:
            query = query.where(units.c.bedrooms == bedrooms)
        if project is not None:
            query = query.where(units.c.project == project)
        query = query.order_by(units.c.price_egp.desc()).limit(limit)
        async with self._read_only_engine.connect() as connection:
            rows = (await connection.execute(query)).mappings().all()
        return [_to_unit(row) for row in rows]

    async def upsert(self, new_units: Sequence[Unit]) -> None:
        if not new_units:
            return
        rows = [_to_row(unit) for unit in new_units]
        statement = insert(units).values(rows)
        updates = {
            column.name: statement.excluded[column.name]
            for column in units.columns
            if column.name != "code"
        }
        async with self._engine.begin() as connection:
            await connection.execute(
                statement.on_conflict_do_update(index_elements=["code"], set_=updates)
            )

    async def run_read_only(self, sql: str) -> list[dict[str, object]]:
        # Second line of defence behind the guard: the database itself refuses writes.
        async with self._read_only_engine.begin() as connection:
            await connection.execute(text("SET TRANSACTION READ ONLY"))
            await connection.execute(
                text(f"SET LOCAL statement_timeout = {self._statement_timeout_ms}")
            )
            result = await connection.exec_driver_sql(sql)
            return [dict(row) for row in result.mappings().all()]


def _to_unit(row: object) -> Unit:
    return Unit(
        code=row["code"],
        project=row["project"],
        unit_type=row["unit_type"],
        bedrooms=row["bedrooms"],
        area_sqm=row["area_sqm"],
        price_egp=row["price_egp"],
        status=UnitStatus(row["status"]),
        delivery_date=row["delivery_date"],
    )


def _to_row(unit: Unit) -> dict[str, object]:
    return {
        "code": unit.code,
        "project": unit.project,
        "unit_type": unit.unit_type,
        "bedrooms": unit.bedrooms,
        "area_sqm": unit.area_sqm,
        "price_egp": unit.price_egp,
        "status": unit.status.value,
        "delivery_date": unit.delivery_date,
    }
