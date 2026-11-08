from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from qdrant_client import AsyncQdrantClient
from sqlalchemy import create_engine, inspect, make_url, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.bootstrap.demo_data import ingest_documents, load_units, seed_crm, seed_units
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.modules.inventory.adapters.outbound.postgres_store import PostgresUnitStore
from ai_os.modules.knowledge.adapters.outbound.docling_parser import DoclingParser
from ai_os.modules.knowledge.adapters.outbound.llama_index_store import LlamaIndexVectorStore
from ai_os.modules.knowledge.adapters.outbound.price_sheet_parser import PriceSheetParser
from ai_os.modules.knowledge.adapters.outbound.routing_parser import RoutingParser
from ai_os.modules.knowledge.adapters.outbound.stores import qdrant_store
from ai_os.modules.knowledge.application.ingest import IngestDocument

from .conftest import HashingEmbedding, PostgresUrls

ROOT = Path(__file__).parents[2]
TABLES = {
    "units",
    "leads",
    "interactions",
    "context_cards",
    "handovers",
    "follow_ups",
    "role_play_sessions",
    "escalation_tickets",
    "knowledge_candidates",
}


@pytest.fixture
def fresh_database(postgres: PostgresUrls) -> tuple[str, str]:
    name = f"d_{uuid4().hex}"
    admin = create_engine(postgres.sync, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    with admin.connect() as connection:
        connection.execute(text(f"CREATE DATABASE {name}"))
    admin.dispose()
    sync_url = make_url(postgres.sync).set(database=name).render_as_string(False)
    async_url = make_url(postgres.asynchronous).set(database=name).render_as_string(False)
    return sync_url, async_url


def alembic_config(url: str) -> Config:
    config = Config(ROOT / "alembic.ini")
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return config


def tables(url: str) -> set[str]:
    engine = create_engine(url, poolclass=NullPool)
    found = set(inspect(engine).get_table_names())
    engine.dispose()
    return found


def test_migrations_build_the_schema_match_the_code_and_reverse(
    fresh_database: tuple[str, str],
) -> None:
    url = fresh_database[0]
    config = alembic_config(url)

    command.upgrade(config, "head")
    assert TABLES <= tables(url)

    # Fails if a table definition changed without a migration.
    command.check(config)

    command.downgrade(config, "base")
    assert not TABLES & tables(url)


def test_reader_role_script_works_on_migrated_schema(fresh_database: tuple[str, str]) -> None:
    url = fresh_database[0]
    command.upgrade(alembic_config(url), "head")
    engine = create_engine(url, poolclass=NullPool)
    grant = text(
        "DO $$ BEGIN CREATE ROLE migration_test_reader LOGIN; "
        "EXCEPTION WHEN duplicate_object THEN NULL; END $$; "
        "GRANT SELECT ON units TO migration_test_reader"
    )
    with engine.begin() as connection:
        connection.execute(grant)
        connection.execute(text("DROP OWNED BY migration_test_reader"))
    engine.dispose()


@pytest.fixture
async def seeded(
    fresh_database: tuple[str, str],
) -> AsyncIterator[tuple[PostgresUnitStore, PostgresCrmStore]]:
    sync_url, async_url = fresh_database
    command.upgrade(alembic_config(sync_url), "head")
    engine = create_async_engine(async_url, poolclass=NullPool)
    units, crm = PostgresUnitStore(engine, engine), PostgresCrmStore(engine)
    await seed_units(units)
    await seed_crm(crm)
    yield units, crm
    await engine.dispose()


async def test_seed_loads_synthetic_units_and_leads_and_can_rerun(seeded) -> None:
    units, crm = seeded

    await seed_units(units)
    await seed_crm(crm)

    found = await units.get_by_codes([unit.code for unit in load_units()])
    assert len(found) == 8
    assert all(unit.project.startswith("Demo") for unit in found)
    lead = await crm.get_lead("DEMO-LEAD-1")
    assert lead.display_name.startswith("Synthetic")
    assert len(await crm.list_interactions("DEMO-LEAD-1")) == 3


async def test_demo_documents_pass_the_real_ingestion_pipeline() -> None:
    client = AsyncQdrantClient(":memory:")
    store = LlamaIndexVectorStore(qdrant_store(client), HashingEmbedding())
    parser = RoutingParser(PriceSheetParser(), DoclingParser())

    chunks = await ingest_documents(IngestDocument(parser, store))

    assert chunks >= 8 + 5
    [hit] = await store.search("DG-102", 1)
    assert hit.chunk.unit_codes == ("DG-102",)
    assert "5,200,000" in hit.chunk.text
    await client.close()
