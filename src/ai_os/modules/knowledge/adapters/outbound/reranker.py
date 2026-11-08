import asyncio
from collections.abc import Sequence
from typing import Protocol

from ai_os.modules.knowledge.domain.models import ScoredChunk


class CrossEncoderModel(Protocol):
    def predict(self, sentences: list[tuple[str, str]]) -> Sequence[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: CrossEncoderModel) -> None:
        self._model = model

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [ScoredChunk(candidate.chunk, float(score)) for candidate, score in ranked[:top_k]]
