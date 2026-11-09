"""Serves the inventory and CRM tools over MCP stdio, for clients such as Claude Code.

The process reads the database with its own credentials, so anyone who can start it sees what
those credentials see, every lead included. Run it only for people allowed that. It is not
served over HTTP: that would need the API's per-rep access checks first.
"""

from functools import cache

from fastmcp import FastMCP

from ai_os.bootstrap.crm import build_crm_store
from ai_os.bootstrap.inventory import build_ask_inventory, build_unit_store
from ai_os.mcp_servers.crm import create_crm_server
from ai_os.mcp_servers.inventory import create_inventory_server
from ai_os.shared.config import Settings


def build_mcp_server(settings: Settings) -> FastMCP:
    # Built on first use, so the calculator tools work before a database or key is set.
    @cache
    def units():
        return build_unit_store(settings)

    @cache
    def ask():
        return build_ask_inventory(settings, units())

    @cache
    def crm():
        return build_crm_store(settings)

    server = FastMCP("ai-os")
    server.mount(create_inventory_server(units, ask), namespace="inventory")
    server.mount(create_crm_server(crm), namespace="crm")
    return server


def main() -> None:
    build_mcp_server(Settings()).run(show_banner=False)


if __name__ == "__main__":
    main()
