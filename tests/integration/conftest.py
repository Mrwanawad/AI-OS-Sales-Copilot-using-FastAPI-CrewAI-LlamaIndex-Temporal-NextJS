import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text

from ai_os.modules.knowledge.adapters.outbound.hashing_embedding import HashingEmbedding

__all__ = ["HashingEmbedding", "PostgresUrls"]


@dataclass(frozen=True)
class PostgresUrls:
    sync: str
    asynchronous: str


def _embedded_postgres(path: Path) -> Iterator[str]:
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(path)
    try:
        yield server.get_uri().replace("postgresql://", "postgresql+psycopg2://")
    finally:
        server.cleanup()


@pytest.fixture(scope="session")
def postgres(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PostgresUrls]:
    """Uses TEST_DATABASE_URL when set, otherwise an embedded Postgres with pgvector."""
    external = os.environ.get("TEST_DATABASE_URL")
    urls = iter([external]) if external else _embedded_postgres(tmp_path_factory.mktemp("pg"))
    sync_url = next(urls)
    engine = create_engine(sync_url)
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    engine.dispose()
    yield PostgresUrls(sync_url, sync_url.replace("postgresql+psycopg2", "postgresql+asyncpg"))
    next(urls, None)
