"""The MCP server as a real subprocess over stdio, reading a real database."""

import os
from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal as D
from pathlib import Path
from uuid import uuid4

import pytest
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.modules.crm_copilot.domain.models import Lead, LeadStage
from ai_os.modules.inventory.adapters.outbound.postgres_store import PostgresUnitStore
from ai_os.modules.inventory.domain.models import Unit, UnitStatus

from .conftest import PostgresUrls

SCRIPT = Path(__file__).parents[2] / "scripts" / "mcp_server.py"


@pytest.fixture
async def database(postgres: PostgresUrls) -> AsyncIterator[str]:
    """A fresh database, since a subprocess can't be handed a search_path."""
    name = f"mcp_{uuid4().hex[:12]}"
    admin = create_engine(postgres.sync, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f"CREATE DATABASE {name}"))
    url = make_url(postgres.asynchronous).set(database=name).render_as_string(False)
    engine = create_async_engine(url, poolclass=NullPool)
    units, crm = PostgresUnitStore(engine, engine), PostgresCrmStore(engine)
    await units.create_schema()
    await crm.create_schema()
    await units.upsert(
        [
            Unit(
                "A-101",
                "Palm Hills",
                "villa",
                4,
                D("300"),
                D("9000000"),
                UnitStatus.AVAILABLE,
                date(2027, 6, 1),
            )
        ]
    )
    await crm.add_lead(Lead("L1", "عميل", LeadStage.QUALIFIED, "rep-a", {"budget_egp": "9"}))
    await engine.dispose()
    yield url
    with admin.connect() as connection:
        connection.execute(text(f"DROP DATABASE {name} WITH (FORCE)"))
    admin.dispose()


async def test_tools_answer_over_stdio_from_the_database(database: str) -> None:
    env = {
        **os.environ,
        "ASYNC_DATABASE_URL": database,
        "INVENTORY_READ_ONLY_DATABASE_URL": database,
        "PYTHONUTF8": "1",
    }
    transport = PythonStdioTransport(SCRIPT, env=env, cwd=str(SCRIPT.parents[1]))

    async with Client(transport) as client:
        names = {tool.name for tool in await client.list_tools()}
        units = await client.call_tool("inventory_get_units", {"codes": ["A-101", "nope"]})
        plan = await client.call_tool(
            "inventory_calculate_installments",
            {"price": "5000000", "down_payment_pct": "10", "years": 8},
        )
        lead = await client.call_tool("crm_get_lead_summary", {"lead_id": "L1"})

    assert {
        "inventory_get_units",
        "inventory_ask_inventory",
        "inventory_calculate_installments",
        "inventory_calculate_mortgage",
        "crm_get_context_card",
        "crm_get_lead_summary",
    } <= names
    assert [u["code"] for u in units.data] == ["A-101"]
    assert D(str(units.data[0]["price_egp"])) == D("9000000")
    assert D(str(plan.data["installment"])) == D("46875.00")
    assert lead.data["display_name"] == "عميل"
    assert lead.data["stage"] == "qualified"
