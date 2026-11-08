from alembic import context
from sqlalchemy import create_engine, pool

from ai_os.modules.coaching.adapters.outbound.postgres_store import metadata as coaching
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import metadata as crm
from ai_os.modules.escalation.adapters.outbound.postgres_store import metadata as escalation
from ai_os.modules.inventory.adapters.outbound.postgres_store import metadata as inventory
from ai_os.shared.config import Settings

target_metadata = [inventory, crm, coaching, escalation]


def run_migrations() -> None:
    url = context.config.get_main_option("sqlalchemy.url") or Settings().database_url
    if not url:
        raise SystemExit("Set DATABASE_URL")
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        # PGVector's tables are created by LlamaIndex, so migrations leave them alone.
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=_only_our_tables,
        )
        with context.begin_transaction():
            context.run_migrations()


def _only_our_tables(_object, name, type_, reflected, _compare_to) -> bool:
    known = {table for metadata in target_metadata for table in metadata.tables}
    return not (type_ == "table" and reflected and name not in known)


run_migrations()
