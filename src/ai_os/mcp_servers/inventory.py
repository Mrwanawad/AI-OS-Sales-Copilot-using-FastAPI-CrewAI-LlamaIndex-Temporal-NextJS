from collections.abc import Callable
from dataclasses import asdict
from decimal import Decimal

from fastmcp import FastMCP

from ai_os.modules.inventory.application.ask import AskInventory
from ai_os.modules.inventory.domain.calculator import (
    Frequency,
    installment_plan,
    mortgage_plan,
)
from ai_os.modules.inventory.ports.store import UnitStore


def create_inventory_server(
    get_store: Callable[[], UnitStore], get_ask: Callable[[], AskInventory]
) -> FastMCP:
    server = FastMCP("inventory")

    @server.tool
    async def get_units(codes: list[str]) -> list[dict]:
        """Look up units by code. Returns live prices and availability."""
        return [asdict(unit) for unit in await get_store().get_by_codes(codes)]

    @server.tool
    async def ask_inventory(question: str) -> dict:
        """Answer a question about units with a validated, read-only query."""
        return asdict(await get_ask()(question))

    @server.tool
    def calculate_installments(
        price: str,
        down_payment_pct: str,
        years: int,
        months_between_payments: int = 1,
        maintenance_pct: str = "0",
    ) -> dict:
        """Calculate a developer installment plan. Amounts are decimal strings in EGP."""
        return asdict(
            installment_plan(
                Decimal(price),
                Decimal(down_payment_pct),
                years,
                Frequency(months_between_payments),
                Decimal(maintenance_pct),
            )
        )

    @server.tool
    def calculate_mortgage(principal: str, annual_rate_pct: str, years: int) -> dict:
        """Calculate a mortgage payment. Amounts are decimal strings in EGP."""
        return asdict(mortgage_plan(Decimal(principal), Decimal(annual_rate_pct), years))

    return server
