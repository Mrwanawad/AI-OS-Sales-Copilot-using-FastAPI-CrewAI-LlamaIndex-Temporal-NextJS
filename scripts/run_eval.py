"""Retrieval release gate: exits non-zero if the gold set falls below evals/thresholds.json.

Usage:
  python scripts/run_eval.py                 # hashing embedder, no downloads, as in CI
  python scripts/run_eval.py --real-models   # the configured embedding and reranker models

The gold set is synthetic. Replace it with questions graded by team leads.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from ai_os.bootstrap.demo_data import ingest_documents
from ai_os.modules.knowledge.adapters.outbound.docling_parser import DoclingParser
from ai_os.modules.knowledge.adapters.outbound.hashing_embedding import HashingEmbedding
from ai_os.modules.knowledge.adapters.outbound.llama_index_store import LlamaIndexVectorStore
from ai_os.modules.knowledge.adapters.outbound.price_sheet_parser import PriceSheetParser
from ai_os.modules.knowledge.adapters.outbound.routing_parser import RoutingParser
from ai_os.modules.knowledge.adapters.outbound.stores import qdrant_store
from ai_os.modules.knowledge.application.ingest import IngestDocument
from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.modules.knowledge.domain.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from ai_os.modules.knowledge.domain.models import ScoredChunk

ROOT = Path(__file__).parents[1]


class KeepOrder:
    """No reranking: the gate then measures hybrid retrieval alone."""

    async def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        return list(candidates)[:top_k]


def _models(real: bool):
    if not real:
        return HashingEmbedding(), KeepOrder()
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from sentence_transformers import CrossEncoder

    from ai_os.modules.knowledge.adapters.outbound.reranker import CrossEncoderReranker
    from ai_os.shared.config import Settings

    settings = Settings()
    return (
        HuggingFaceEmbedding(model_name=settings.embedding_model),
        CrossEncoderReranker(CrossEncoder(settings.reranker_model)),
    )


async def evaluate(real_models: bool) -> dict[str, float]:
    embedder, reranker = _models(real_models)
    client = AsyncQdrantClient(":memory:")
    store = LlamaIndexVectorStore(qdrant_store(client), embedder)
    parser = RoutingParser(PriceSheetParser(), DoclingParser())
    await ingest_documents(IngestDocument(parser, store))
    retrieve = RetrieveKnowledge(store, reranker)

    totals: dict[str, list[float]] = defaultdict(list)
    lines = (ROOT / "evals" / "gold_retrieval.jsonl").read_text(encoding="utf-8").splitlines()
    for case in map(json.loads, lines):
        hits = await retrieve(case["query"], top_k=10)
        ranked = [hit.chunk.citation.document_id for hit in hits]
        recall = recall_at_k(ranked, case["relevant"], 5)
        totals["recall_at_5"].append(recall)
        totals["mrr"].append(reciprocal_rank(ranked, case["relevant"]))
        totals["ndcg_at_10"].append(ndcg_at_k(ranked, case["relevant"], 10))
        if case["tag"] == "exact_code":
            totals["exact_code_recall_at_5"].append(recall)
        if case["tag"] == "arabic":
            totals["arabic_recall_at_5"].append(recall)
    await client.close()
    return {name: round(sum(values) / len(values), 4) for name, values in totals.items()}


def failures(scores: dict[str, float], thresholds: dict[str, float]) -> list[str]:
    return [
        f"{name}: {scores.get(name, 0.0)} is below {minimum}"
        for name, minimum in thresholds.items()
        if scores.get(name, 0.0) < minimum
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-models", action="store_true")
    arguments = parser.parse_args()

    scores = asyncio.run(evaluate(arguments.real_models))
    print(json.dumps(scores, indent=2))
    thresholds = json.loads((ROOT / "evals" / "thresholds.json").read_text(encoding="utf-8"))
    problems = failures(scores, thresholds)
    for problem in problems:
        print("FAILED", problem)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
