"""One contract, run against each VectorStore backend."""

from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient

from ai_os.modules.knowledge.adapters.outbound.llama_index_store import LlamaIndexVectorStore
from ai_os.modules.knowledge.adapters.outbound.stores import pgvector_store, qdrant_store
from ai_os.modules.knowledge.application.ingest import IngestDocument
from ai_os.modules.knowledge.domain.chunking import chunk_sections
from ai_os.modules.knowledge.domain.models import (
    DocumentKind,
    ParsedSection,
    SearchFilters,
    SourceDocument,
)
from ai_os.modules.knowledge.ports.retrieval import VectorStore

from .conftest import HashingEmbedding, PostgresUrls

BACKENDS = ["qdrant", "pgvector"]


@pytest.fixture(params=BACKENDS)
async def store(request: pytest.FixtureRequest) -> AsyncIterator[VectorStore]:
    if request.param == "qdrant":
        client = AsyncQdrantClient(":memory:")
        yield LlamaIndexVectorStore(qdrant_store(client), HashingEmbedding())
        await client.close()
        return

    urls: PostgresUrls = request.getfixturevalue("postgres")
    backend = pgvector_store(urls.sync, urls.asynchronous, embed_dim=256, table=f"t_{uuid4().hex}")
    yield LlamaIndexVectorStore(backend, HashingEmbedding())
    await backend.aclear()


def source(document_id: str, project: str, kind: DocumentKind, version: int = 1) -> SourceDocument:
    return SourceDocument(document_id, project, kind, version)


async def add(store: VectorStore, document: SourceDocument, *sections: ParsedSection) -> None:
    await store.upsert(chunk_sections(document, sections))


async def test_search_round_trips_citation_and_unit_codes(store: VectorStore) -> None:
    document = source("prices", "Palm Hills", DocumentKind.PRICE_SHEET, 3)
    await add(store, document, ParsedSection("Unit: A-101 | Price: 5,000,000 EGP", 7, ("A-101",)))

    [hit] = await store.search("price of A-101", top_k=5)

    assert hit.chunk.text == "Unit: A-101 | Price: 5,000,000 EGP"
    assert hit.chunk.citation.document_id == "prices"
    assert hit.chunk.citation.document_version == 3
    assert hit.chunk.citation.page == 7
    assert hit.chunk.unit_codes == ("A-101",)
    assert hit.chunk.kind is DocumentKind.PRICE_SHEET


async def test_missing_page_round_trips_as_none(store: VectorStore) -> None:
    await add(store, source("faq", "Palm Hills", DocumentKind.FAQ), ParsedSection("Gym included"))

    [hit] = await store.search("gym", top_k=1)

    assert hit.chunk.citation.page is None


async def test_filters_by_project_and_kind(store: VectorStore) -> None:
    await add(store, source("a", "Palm Hills", DocumentKind.FAQ), ParsedSection("Pool open"))
    await add(store, source("b", "Mountain View", DocumentKind.FAQ), ParsedSection("Pool closed"))
    await add(store, source("c", "Palm Hills", DocumentKind.POLICY), ParsedSection("Pool rules"))

    by_project = await store.search("pool", 10, SearchFilters(project="Mountain View"))
    by_both = await store.search(
        "pool", 10, SearchFilters(project="Palm Hills", kinds=(DocumentKind.POLICY,))
    )

    assert [h.chunk.text for h in by_project] == ["Pool closed"]
    assert [h.chunk.text for h in by_both] == ["Pool rules"]


async def test_active_version_and_delete_older_versions_scope_to_document(
    store: VectorStore,
) -> None:
    assert await store.active_version("prices") is None

    await add(store, source("prices", "P", DocumentKind.PRICE_SHEET, 1), ParsedSection("old"))
    await add(store, source("prices", "P", DocumentKind.PRICE_SHEET, 2), ParsedSection("new"))
    await add(store, source("other", "P", DocumentKind.PRICE_SHEET, 1), ParsedSection("keep"))
    assert await store.active_version("prices") == 2

    await store.delete_older_versions("prices", 2)

    texts = {h.chunk.text for h in await store.search("old new keep", 10)}
    assert texts == {"new", "keep"}
    assert await store.active_version("other") == 1


async def test_active_version_finds_newest_among_many_chunks(store: VectorStore) -> None:
    old = source("prices", "P", DocumentKind.PRICE_SHEET, 1)
    await add(store, old, *[ParsedSection(f"row {n}", atomic=True) for n in range(40)])
    await add(store, source("prices", "P", DocumentKind.PRICE_SHEET, 2), ParsedSection("new"))

    assert await store.active_version("prices") == 2


class StaticParser:
    def __init__(self, *sections: ParsedSection) -> None:
        self.sections = list(sections)

    async def parse(self, path: Path, document: SourceDocument) -> list[ParsedSection]:
        return self.sections


async def test_new_price_sheet_retires_old_price_in_search(store: VectorStore) -> None:
    old = IngestDocument(StaticParser(ParsedSection("Unit A-101 price 5,000,000 EGP")), store)
    new = IngestDocument(StaticParser(ParsedSection("Unit A-101 price 5,750,000 EGP")), store)

    await old(Path("v1.xlsx"), source("prices", "P", DocumentKind.PRICE_SHEET, 1))
    await new(Path("v2.xlsx"), source("prices", "P", DocumentKind.PRICE_SHEET, 2))

    hits = await store.search("A-101 price", 10)
    assert [h.chunk.text for h in hits] == ["Unit A-101 price 5,750,000 EGP"]
    assert hits[0].chunk.citation.document_version == 2


async def test_exact_unit_code_beats_semantic_noise_on_qdrant() -> None:
    client = AsyncQdrantClient(":memory:")
    store = LlamaIndexVectorStore(qdrant_store(client), HashingEmbedding())
    document = source("prices", "P", DocumentKind.PRICE_SHEET)
    noise = [ParsedSection(f"Unit B-{n} villa with pool", atomic=True) for n in range(40)]
    target = ParsedSection("Unit C-777 apartment", atomic=True)
    await add(store, document, *noise, target)

    hits = await store.search("c-777", top_k=3)

    assert hits[0].chunk.text == target.text
    await client.close()


async def test_arabic_spelling_variant_and_digits_match_on_qdrant() -> None:
    client = AsyncQdrantClient(":memory:")
    store = LlamaIndexVectorStore(qdrant_store(client), HashingEmbedding())
    document = source("faq", "P", DocumentKind.FAQ)
    await add(
        store,
        document,
        ParsedSection("الوحدة ١٠١ شقة بإطلالة على البحيرة", atomic=True),
        ParsedSection("فيلا مستقلة بحديقة خاصة", atomic=True),
    )

    by_variant = await store.search("شقه باطلاله", top_k=1)
    by_digits = await store.search("الوحده 101", top_k=1)

    assert by_variant[0].chunk.text.startswith("الوحدة")
    assert by_digits[0].chunk.text.startswith("الوحدة")
    await client.close()
