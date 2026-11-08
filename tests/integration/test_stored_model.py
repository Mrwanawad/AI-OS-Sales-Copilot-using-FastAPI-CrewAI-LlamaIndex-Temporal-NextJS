"""The promoted best-time model travels through Postgres and reloads when it changes."""

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.bootstrap.best_time import StoredModel
from ai_os.modules.best_time.domain.features import FEATURE_NAMES
from ai_os.shared.files import PostgresFileStore

from .conftest import PostgresUrls


@pytest.fixture
async def files(postgres: PostgresUrls) -> AsyncIterator[PostgresFileStore]:
    schema = f"s_{uuid4().hex}"
    admin = create_async_engine(postgres.asynchronous, poolclass=NullPool)
    async with admin.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
    await admin.dispose()
    engine = create_async_engine(
        postgres.asynchronous,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    store = PostgresFileStore(engine)
    await store.create_schema()
    yield store
    await engine.dispose()


def booster_bytes(tmp_path: Path, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    rows = rng.random((200, len(FEATURE_NAMES)))
    labels = (rows[:, 0] > 0.5).astype(int)
    data = lgb.Dataset(rows, labels, feature_name=FEATURE_NAMES)
    booster = lgb.train({"objective": "binary", "verbose": -1, "seed": seed}, data, 5)
    path = tmp_path / f"model-{seed}.txt"
    booster.save_model(str(path))
    return path.read_bytes()


async def test_stored_model_loads_and_reloads_after_a_retrain(
    files: PostgresFileStore, tmp_path: Path
) -> None:
    model = StoredModel(files, "stored:best-time-champion", timedelta(0))
    assert await model.get() is None

    await files.put_named("best-time-champion", "m.txt", "x", booster_bytes(tmp_path, 1), "t")
    first = await model.get()
    assert first is not None and len(first.predict([[0.0] * len(FEATURE_NAMES)])) == 1
    assert await model.get() is first

    await files.put_named("best-time-champion", "m.txt", "x", booster_bytes(tmp_path, 2), "t")
    second = await model.get()
    assert second is not None and second is not first
