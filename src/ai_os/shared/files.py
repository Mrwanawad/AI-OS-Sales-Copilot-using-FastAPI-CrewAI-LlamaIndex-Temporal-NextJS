"""Files the API receives and the worker reads, kept where both services can reach them.

Separate Render services share no disk, so uploads go into Postgres and the worker copies
them to a temporary file. A reference is either `stored:<id>` or a plain local path; plain
paths keep working for scripts and tests that run on one machine.
"""

import hashlib
import tempfile
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath
from typing import Protocol
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    LargeBinary,
    MetaData,
    String,
    Table,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

STORED_PREFIX = "stored:"

metadata = MetaData()

stored_files = Table(
    "stored_files",
    metadata,
    Column("id", String, primary_key=True),
    Column("filename", String, nullable=False),
    Column("content_type", String, nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("sha256", String, nullable=False),
    Column("uploaded_by", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("data", LargeBinary, nullable=False),
)


class StoredFileNotFound(LookupError):
    pass


@dataclass(frozen=True)
class StoredFile:
    reference: str
    filename: str
    size_bytes: int
    sha256: str


class FileStore(Protocol):
    async def put(
        self, filename: str, content_type: str, data: bytes, uploaded_by: str
    ) -> StoredFile: ...

    def local_copy(self, reference: str) -> AbstractAsyncContextManager[Path]: ...


def safe_filename(filename: str) -> str:
    """Keeps only the final name, so a client can't pick a path on the worker."""
    name = PurePath(filename.replace("\\", "/")).name.strip()
    return name or "upload"


class PostgresFileStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def put(
        self, filename: str, content_type: str, data: bytes, uploaded_by: str
    ) -> StoredFile:
        file_id = uuid4().hex
        name = safe_filename(filename)
        digest = hashlib.sha256(data).hexdigest()
        async with self._engine.begin() as connection:
            await connection.execute(
                stored_files.insert().values(
                    id=file_id,
                    filename=name,
                    content_type=content_type,
                    size_bytes=len(data),
                    sha256=digest,
                    uploaded_by=uploaded_by,
                    created_at=datetime.now(UTC),
                    data=data,
                )
            )
        return StoredFile(f"{STORED_PREFIX}{file_id}", name, len(data), digest)

    async def put_named(
        self, name: str, filename: str, content_type: str, data: bytes, uploaded_by: str
    ) -> StoredFile:
        """Stores under a fixed reference, `stored:<name>`, replacing what was there."""
        digest = hashlib.sha256(data).hexdigest()
        values = {
            "filename": safe_filename(filename),
            "content_type": content_type,
            "size_bytes": len(data),
            "sha256": digest,
            "uploaded_by": uploaded_by,
            "created_at": datetime.now(UTC),
            "data": data,
        }
        statement = (
            insert(stored_files)
            .values(id=name, **values)
            .on_conflict_do_update(index_elements=[stored_files.c.id], set_=values)
        )
        async with self._engine.begin() as connection:
            await connection.execute(statement)
        return StoredFile(f"{STORED_PREFIX}{name}", values["filename"], len(data), digest)

    async def fingerprint(self, reference: str) -> str | None:
        """The stored file's SHA-256, so readers can tell it changed without downloading it."""
        query = select(stored_files.c.sha256).where(
            stored_files.c.id == reference.removeprefix(STORED_PREFIX)
        )
        async with self._engine.connect() as connection:
            return (await connection.execute(query)).scalar_one_or_none()

    @asynccontextmanager
    async def local_copy(self, reference: str) -> AsyncIterator[Path]:
        if not reference.startswith(STORED_PREFIX):
            yield Path(reference)
            return
        file_id = reference.removeprefix(STORED_PREFIX)
        query = select(stored_files.c.filename, stored_files.c.data).where(
            stored_files.c.id == file_id
        )
        async with self._engine.connect() as connection:
            row = (await connection.execute(query)).first()
        if row is None:
            raise StoredFileNotFound(reference)
        # The original name is kept: parsers choose a format by its extension.
        with tempfile.TemporaryDirectory(prefix="ai-os-") as directory:
            path = Path(directory) / safe_filename(row.filename)
            path.write_bytes(row.data)
            yield path


class LocalPathFiles:
    """For processes without a database: every reference is a local path."""

    @asynccontextmanager
    async def local_copy(self, reference: str) -> AsyncIterator[Path]:
        if reference.startswith(STORED_PREFIX):
            raise StoredFileNotFound(f"{reference} needs a database-backed file store")
        yield Path(reference)
