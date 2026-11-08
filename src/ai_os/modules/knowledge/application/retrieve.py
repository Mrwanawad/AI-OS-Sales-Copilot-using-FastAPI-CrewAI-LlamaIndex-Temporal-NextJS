from ai_os.modules.knowledge.domain.models import ScoredChunk, SearchFilters
from ai_os.modules.knowledge.ports.retrieval import Reranker, VectorStore


class RetrieveKnowledge:
    def __init__(self, store: VectorStore, reranker: Reranker, candidate_pool: int = 30) -> None:
        self._store = store
        self._reranker = reranker
        self._candidate_pool = candidate_pool

    async def __call__(
        self, query: str, top_k: int = 5, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        candidates = await self._store.search(query, self._candidate_pool, filters)
        return await self._reranker.rerank(query, candidates, top_k)
