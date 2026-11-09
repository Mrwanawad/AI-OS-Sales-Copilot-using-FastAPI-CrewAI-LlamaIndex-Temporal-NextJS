"""MCP stdio entry point. Example client config: command `uv`, args `run python
scripts/mcp_server.py`, with ASYNC_DATABASE_URL and INVENTORY_READ_ONLY_DATABASE_URL set."""

from ai_os.bootstrap.mcp import main

if __name__ == "__main__":
    main()
