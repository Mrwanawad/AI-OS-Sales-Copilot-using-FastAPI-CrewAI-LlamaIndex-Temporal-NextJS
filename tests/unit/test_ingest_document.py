from collections.abc import Sequence
from pathlib import Path

import pytest

from ai_os.modules.knowledge.application.ingest import IngestDocument
from ai_os.modules.knowledge.domain.errors import EmptyDocument, StaleDocumentVersion
from ai_os.modules.knowledge.domain.models import (
    Chunk,
    DocumentKind,
    ParsedSection,
    ScoredChunk,
    SearchFilters,
    SourceDocument,
)


def document(version: int) -> SourceDocument:
    return SourceDocument("prices", "Palm Hills", DocumentKind.PRICE_SHEET, version)


class FakeParser:
    def __init__(self, sections: list[ParsedSection]) -> None:
        self.sections = sections

    async def parse(self, path: Path, document: SourceDocument) -> list[ParsedSection]:
        return self.sections


class FakeStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.calls: list[str] = []

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        self.calls.append("upsert")
        existing = {c.id: c for c in self.chunks}
        existing.update({c.id: c for c in chunks})
        self.chunks = list(existing.values())

    async def search(
        self, query: str, top_k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        raise NotImplementedError

    async def active_version(self, document_id: str) -> int | None:
        versions = [
            c.citation.document_version
            for c in self.chunks
            if c.citation.document_id == document_id
        ]
        return max(versions, default=None)

    async def delete_older_versions(self, document_id: str, version: int) -> None:
        self.calls.append("delete")
        self.chunks = [
            c
            for c in self.chunks
            if c.citation.document_id != document_id or c.citation.document_version >= version
        ]


async def test_new_version_replaces_old_and_upserts_before_deleting() -> None:
    store = FakeStore()
    ingest = IngestDocument(FakeParser([ParsedSection("Old price")]), store)
    await ingest(Path("v1.xlsx"), document(1))

    new = IngestDocument(FakeParser([ParsedSection("New price")]), store)
    result = await new(Path("v2.xlsx"), document(2))

    assert [c.text for c in store.chunks] == ["New price"]
    assert result.chunks == 1
    assert store.calls[-2:] == ["upsert", "delete"]


async def test_older_version_is_rejected() -> None:
    store = FakeStore()
    await IngestDocument(FakeParser([ParsedSection("Current")]), store)(Path("v"), document(5))

    with pytest.raises(StaleDocumentVersion):
        await IngestDocument(FakeParser([ParsedSection("Old")]), store)(Path("v"), document(4))

    assert [c.text for c in store.chunks] == ["Current"]


async def test_retrying_same_version_is_idempotent() -> None:
    store = FakeStore()
    ingest = IngestDocument(FakeParser([ParsedSection("Price")]), store)

    await ingest(Path("v"), document(1))
    await ingest(Path("v"), document(1))

    assert len(store.chunks) == 1


async def test_empty_parse_keeps_active_version() -> None:
    store = FakeStore()
    await IngestDocument(FakeParser([ParsedSection("Current")]), store)(Path("v"), document(1))

    with pytest.raises(EmptyDocument):
        await IngestDocument(FakeParser([ParsedSection("  ")]), store)(Path("v"), document(2))

    assert [c.text for c in store.chunks] == ["Current"]
