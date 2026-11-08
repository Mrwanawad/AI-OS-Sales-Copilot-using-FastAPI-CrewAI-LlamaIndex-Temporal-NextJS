"""Runs the whole backend locally without Docker: Postgres, Temporal, the API and the worker.

Usage: python scripts/dev_stack.py [--reset]
Then: cd frontend && npm run dev, and sign in at http://localhost:3000 as rep-a, lead-a or
manager-a with the password printed below.

Data lives in .devdata/ and survives restarts; --reset starts over. Small embedding and
reranker models keep downloads and memory low; set EMBEDDING_MODEL, EMBEDDING_DIM and
RERANKER_MODEL to use the production ones. LLM answers need ANTHROPIC_API_KEY or
OPENAI_API_KEY in .env; without them the agent returns a clean 503.
"""

import argparse
import asyncio
import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pgserver
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from temporalio.testing import WorkflowEnvironment

ROOT = Path(__file__).parents[1]
DATA = ROOT / ".devdata"
PORT = 8000


def secret_file(name: str, length: int = 24) -> str:
    """A random value kept in .devdata, so restarts reuse it."""
    path = DATA / name
    if not path.exists():
        path.write_text(secrets.token_urlsafe(length), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


async def prepare_database(async_url: str) -> None:
    engine = create_async_engine(async_url, poolclass=NullPool)
    async with engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        exists = await connection.scalar(
            text("SELECT 1 FROM pg_roles WHERE rolname = 'inventory_reader'")
        )
        if not exists:
            await connection.execute(text("CREATE ROLE inventory_reader LOGIN"))
    await engine.dispose()


async def grant_reader(async_url: str) -> None:
    engine = create_async_engine(async_url, poolclass=NullPool)
    async with engine.begin() as connection:
        await connection.execute(text("GRANT USAGE ON SCHEMA public TO inventory_reader"))
        await connection.execute(text("GRANT SELECT ON units TO inventory_reader"))
    await engine.dispose()


async def main(reset: bool) -> None:
    if reset and DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(exist_ok=True)
    first_run = not (DATA / "seeded").exists()

    database = pgserver.get_server(DATA / "pg")  # stops on exit, keeps data
    base = database.get_uri()
    sync_url = base.replace("postgresql://", "postgresql+psycopg2://")
    async_url = base.replace("postgresql://", "postgresql+asyncpg://")
    reader_url = make_url(async_url).set(username="inventory_reader").render_as_string(False)
    await prepare_database(async_url)

    temporal = await WorkflowEnvironment.start_local()
    password = secret_file("demo_password", 12)
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "VECTOR_STORE": "pgvector",
        "EMBEDDING_MODEL": os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "EMBEDDING_DIM": os.environ.get("EMBEDDING_DIM", "384"),
        "RERANKER_MODEL": os.environ.get(
            "RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L2-v2"
        ),
        "DATABASE_URL": sync_url,
        "ASYNC_DATABASE_URL": async_url,
        "INVENTORY_READ_ONLY_DATABASE_URL": reader_url,
        "TEMPORAL_ADDRESS": temporal.client.service_client.config.target_host,
        "AUTH_SECRET": secret_file("auth_secret", 48),
        "CRM_WEBHOOK_SECRET": secret_file("webhook_secret"),
        "APP_URL": "http://localhost:3000",
        "DEMO_PASSWORD": password,
        "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", "tiny"),
    }

    def run(*command: str) -> None:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=environment, check=True)

    run("-m", "alembic", "upgrade", "head")
    await grant_reader(async_url)
    if first_run:
        print("Seeding demo data (first run downloads two small models)...")
        run("scripts/seed_demo.py")
        (DATA / "seeded").touch()

    logs = (DATA / "processes.log").open("a", encoding="utf-8")
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ai_os.bootstrap.app:create_app",
                "--factory",
                "--port",
                str(PORT),
            ],
            cwd=ROOT,
            env=environment,
            stdout=logs,
            stderr=subprocess.STDOUT,
        ),
        subprocess.Popen(
            [sys.executable, "-m", "ai_os.bootstrap.worker"],
            cwd=ROOT,
            env=environment,
            stdout=logs,
            stderr=subprocess.STDOUT,
        ),
    ]
    print(f"\nAPI     http://localhost:{PORT}/docs")
    print("Web     cd frontend && npm run dev, then http://localhost:3000")
    print(f"Logins  rep-a, rep-b, lead-a, manager-a / {password}")
    print(f"Logs    {DATA / 'processes.log'}\nCtrl+C stops everything.")
    try:
        while all(process.poll() is None for process in processes):
            await asyncio.sleep(2)
        print("A process exited; see the log.")
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(30)
        logs.close()
        await temporal.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="delete .devdata and start over")
    try:
        asyncio.run(main(parser.parse_args().reset))
    except KeyboardInterrupt:
        pass
