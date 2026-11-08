from collections.abc import Callable
from dataclasses import asdict

from fastmcp import FastMCP

from ai_os.modules.crm_copilot.ports.crm import CrmStore


def create_crm_server(get_store: Callable[[], CrmStore]) -> FastMCP:
    server = FastMCP("crm")

    @server.tool
    async def get_context_card(lead_id: str) -> dict | None:
        """Get the stored client context card for a lead, or null if none exists."""
        card = await get_store().get_card(lead_id)
        return asdict(card) if card else None

    @server.tool
    async def get_lead_summary(lead_id: str) -> dict | None:
        """Get a lead's stage, assigned rep and CRM fields."""
        lead = await get_store().get_lead(lead_id)
        return asdict(lead) if lead else None

    return server
