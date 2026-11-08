from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import AsyncQdrantClient
from sentence_transformers import CrossEncoder

from ai_os.modules.knowledge.adapters.outbound.docling_parser import DoclingParser
from ai_os.modules.knowledge.adapters.outbound.llama_index_store import LlamaIndexVectorStore
from ai_os.modules.knowledge.adapters.outbound.price_sheet_parser import PriceSheetParser
from ai_os.modules.knowledge.adapters.outbound.reranker import CrossEncoderReranker
from ai_os.modules.knowledge.adapters.outbound.routing_parser import RoutingParser
from ai_os.modules.knowledge.adapters.outbound.stores import pgvector_store, qdrant_store
from ai_os.modules.knowledge.application.ingest import IngestDocument
from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.shared.config import Settings


def build_store(settings: Settings) -> LlamaIndexVectorStore:
    embedder = HuggingFaceEmbedding(model_name=settings.embedding_model)
    if settings.vector_store == "qdrant":
        api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
        client = AsyncQdrantClient(url=settings.qdrant_url, api_key=api_key)
        return LlamaIndexVectorStore(qdrant_store(client), embedder)
    if not settings.database_url or not settings.async_database_url:
        raise ValueError("VECTOR_STORE=pgvector needs DATABASE_URL and ASYNC_DATABASE_URL")
    store = pgvector_store(
        settings.database_url, settings.async_database_url, settings.embedding_dim
    )
    return LlamaIndexVectorStore(store, embedder)


def build_retrieve(settings: Settings) -> RetrieveKnowledge:
    reranker = CrossEncoderReranker(CrossEncoder(settings.reranker_model))
    return RetrieveKnowledge(build_store(settings), reranker)


def build_ingest(settings: Settings) -> IngestDocument:
    parser = RoutingParser(PriceSheetParser(), DoclingParser())
    return IngestDocument(parser, build_store(settings))
