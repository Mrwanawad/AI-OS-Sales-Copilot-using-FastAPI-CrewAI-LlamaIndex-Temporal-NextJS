from ai_os.modules.knowledge.adapters.outbound.reranker import CrossEncoderReranker
from ai_os.modules.knowledge.domain.models import Chunk, Citation, DocumentKind, ScoredChunk


def candidate(text: str) -> ScoredChunk:
    chunk = Chunk(text, text, Citation("d", 1), "P", DocumentKind.FAQ)
    return ScoredChunk(chunk, 0.0)


class LengthModel:
    def predict(self, sentences: list[tuple[str, str]]) -> list[float]:
        return [float(len(text)) for _, text in sentences]


async def test_orders_by_model_score_and_truncates() -> None:
    reranker = CrossEncoderReranker(LengthModel())

    hits = await reranker.rerank("q", [candidate("a"), candidate("ccc"), candidate("bb")], top_k=2)

    assert [h.chunk.text for h in hits] == ["ccc", "bb"]
    assert [h.score for h in hits] == [3.0, 2.0]


async def test_no_candidates_skips_model() -> None:
    assert await CrossEncoderReranker(LengthModel()).rerank("q", [], top_k=3) == []
