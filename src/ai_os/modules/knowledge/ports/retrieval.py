from collections.abc import Sequence
from typing import Protocol

from ai_os.modules.knowledge.domain.models import Chunk, ScoredChunk, SearchFilters


class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    async def search(
        self, query: str, top_k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]: ...

    async def active_version(self, document_id: str) -> int | None: ...

    async def delete_older_versions(self, document_id: str, version: int) -> None: ...


class Reranker(Protocol):
    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...
