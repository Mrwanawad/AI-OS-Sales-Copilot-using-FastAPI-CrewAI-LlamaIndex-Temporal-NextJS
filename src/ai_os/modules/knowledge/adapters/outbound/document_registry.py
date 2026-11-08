from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_os.modules.knowledge.domain.models import DocumentKind, RegisteredDocument

metadata = MetaData()

knowledge_documents = Table(
    "knowledge_documents",
    metadata,
    Column("id", String, primary_key=True),
    Column("project", String, nullable=False, index=True),
    Column("kind", String, nullable=False),
    Column("latest_version", Integer, nullable=False),
    Column("source", String, nullable=False),
    Column("uploaded_by", String, nullable=False),
    Column("uploaded_at", DateTime(timezone=True), nullable=False),
)


class PostgresDocumentRegistry:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def next_version(
        self,
        document_id: str,
        project: str,
        kind: DocumentKind,
        at_least: int,
        source: str,
        uploaded_by: str,
        at: datetime,
    ) -> int:
        table = knowledge_documents
        statement = insert(table).values(
            id=document_id,
            project=project,
            kind=kind.value,
            latest_version=at_least,
            source=source,
            uploaded_by=uploaded_by,
            uploaded_at=at,
        )
        # One statement, so concurrent uploads of one document get different versions.
        statement = statement.on_conflict_do_update(
            index_elements=[table.c.id],
            set_={
                "latest_version": func.greatest(table.c.latest_version + 1, at_least),
                "kind": kind.value,
                "source": source,
                "uploaded_by": uploaded_by,
                "uploaded_at": at,
            },
        ).returning(table.c.latest_version)
        async with self._engine.begin() as connection:
            return (await connection.execute(statement)).scalar_one()

    async def get(self, document_id: str) -> RegisteredDocument | None:
        query = select(knowledge_documents).where(knowledge_documents.c.id == document_id)
        async with self._engine.connect() as connection:
            row = (await connection.execute(query)).mappings().first()
        return _document(row) if row else None

    async def list(self, project: str | None, limit: int) -> list[RegisteredDocument]:
        query = select(knowledge_documents).order_by(knowledge_documents.c.uploaded_at.desc())
        if project:
            query = query.where(knowledge_documents.c.project == project)
        async with self._engine.connect() as connection:
            rows = (await connection.execute(query.limit(limit))).mappings().all()
        return [_document(row) for row in rows]


def _document(row) -> RegisteredDocument:
    return RegisteredDocument(
        row["id"],
        row["project"],
        DocumentKind(row["kind"]),
        row["latest_version"],
        row["source"],
        row["uploaded_by"],
        row["uploaded_at"],
    )
