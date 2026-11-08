from collections.abc import Sequence
from decimal import Decimal as D

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from ai_os.mcp_servers.inventory import create_inventory_server
from ai_os.modules.inventory.domain.models import InventoryAnswer, Unit, UnitStatus

UNIT = Unit("A-101", "Palm Hills", "villa", 4, D("300"), D("9000000"), UnitStatus.AVAILABLE)


class FakeStore:
    async def get_by_codes(self, codes: Sequence[str]) -> list[Unit]:
        return [UNIT] if UNIT.code in codes else []


class FakeAsk:
    async def __call__(self, question: str) -> InventoryAnswer:
        return InventoryAnswer(question, "SELECT code FROM units", [{"code": "A-101"}])


def server():
    return create_inventory_server(FakeStore, FakeAsk)


async def test_lists_expected_tools() -> None:
    async with Client(server()) as client:
        names = {tool.name for tool in await client.list_tools()}

    assert names == {
        "get_units",
        "ask_inventory",
        "calculate_installments",
        "calculate_mortgage",
    }


async def test_calculator_tool_returns_exact_amounts() -> None:
    async with Client(server()) as client:
        result = await client.call_tool(
            "calculate_installments",
            {"price": "5000000", "down_payment_pct": "10", "years": 8},
        )

    assert D(str(result.data["installment"])) == D("46875.00")


async def test_get_units_tool_serializes_live_data() -> None:
    async with Client(server()) as client:
        result = await client.call_tool("get_units", {"codes": ["A-101"]})

    assert result.data[0]["code"] == "A-101"
    assert D(str(result.data[0]["price_egp"])) == D("9000000")


async def test_invalid_calculation_surfaces_as_tool_error() -> None:
    async with Client(server()) as client:
        with pytest.raises(ToolError, match="price must be positive"):
            await client.call_tool(
                "calculate_installments",
                {"price": "-5", "down_payment_pct": "10", "years": 8},
            )
