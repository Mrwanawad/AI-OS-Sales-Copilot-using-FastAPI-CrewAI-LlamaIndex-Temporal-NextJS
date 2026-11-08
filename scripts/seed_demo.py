"""Loads the synthetic demo data. Run `alembic upgrade head` first. Safe to run twice."""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.bootstrap.demo_data import ingest_documents, seed_crm, seed_units
from ai_os.bootstrap.inventory import build_unit_store
from ai_os.bootstrap.knowledge import build_ingest
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.shared.config import Settings


async def main() -> None:
    settings = Settings()
    if not settings.async_database_url:
        raise SystemExit("Set ASYNC_DATABASE_URL")
    print("units:", await seed_units(build_unit_store(settings)))
    engine = create_async_engine(settings.async_database_url)
    print("leads:", await seed_crm(PostgresCrmStore(engine)))
    await engine.dispose()
    print("chunks:", await ingest_documents(build_ingest(settings)))


if __name__ == "__main__":
    asyncio.run(main())
