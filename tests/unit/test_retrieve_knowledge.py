from collections.abc import Sequence

from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.modules.knowledge.domain.models import (
    Chunk,
    Citation,
    DocumentKind,
    ScoredChunk,
    SearchFilters,
)


def scored(chunk_id: str, score: float) -> ScoredChunk:
    citation = Citation(document_id="price-sheet", document_version=1, page=2)
    chunk = Chunk(chunk_id, chunk_id, citation, "Palm Hills", DocumentKind.PRICE_SHEET)
    return ScoredChunk(chunk, score)


class FakeStore:
    def __init__(self, results: list[ScoredChunk]) -> None:
        self.results = results
        self.requested_top_k: int | None = None
        self.requested_filters: SearchFilters | None = None

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        raise NotImplementedError

    async def search(
        self, query: str, top_k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        self.requested_top_k = top_k
        self.requested_filters = filters
        return self.results

    async def active_version(self, document_id: str) -> int | None:
        raise NotImplementedError

    async def delete_older_versions(self, document_id: str, version: int) -> None:
        raise NotImplementedError


class ReversingReranker:
    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(reversed(candidates))[:top_k]


async def test_reranks_candidate_pool_down_to_top_k() -> None:
    store = FakeStore([scored("a", 0.9), scored("b", 0.8), scored("c", 0.7)])
    retrieve = RetrieveKnowledge(store, ReversingReranker(), candidate_pool=30)
    filters = SearchFilters(project="Palm Hills")

    results = await retrieve("3-bedroom units in the compound", top_k=2, filters=filters)

    assert store.requested_top_k == 30
    assert store.requested_filters == filters
    assert [r.chunk.id for r in results] == ["c", "b"]
