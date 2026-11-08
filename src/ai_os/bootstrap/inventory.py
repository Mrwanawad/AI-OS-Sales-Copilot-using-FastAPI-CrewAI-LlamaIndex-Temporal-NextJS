from functools import partial

from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.modules.inventory.adapters.outbound.postgres_store import PostgresUnitStore
from ai_os.modules.inventory.adapters.outbound.sql_guard import SqlGlotGuard
from ai_os.modules.inventory.application.ask import AskInventory
from ai_os.modules.inventory.ports.store import UnitStore
from ai_os.shared.config import Settings
from ai_os.shared.llm.pydantic_ai_adapter import PydanticAIStructuredLLM, build_model


def build_unit_store(settings: Settings) -> PostgresUnitStore:
    if not settings.async_database_url or not settings.inventory_read_only_database_url:
        raise ValueError("Set ASYNC_DATABASE_URL and INVENTORY_READ_ONLY_DATABASE_URL")
    return PostgresUnitStore(
        create_async_engine(settings.async_database_url),
        create_async_engine(settings.inventory_read_only_database_url),
    )


def build_ask_inventory(settings: Settings, store: UnitStore) -> AskInventory:
    settings.require_anthropic_key()
    settings.require_openai_key()
    llm = PydanticAIStructuredLLM(partial(build_model, settings))
    return AskInventory(llm, SqlGlotGuard(), store)
