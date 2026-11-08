from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Modifier, SparseVectorParams

from ai_os.modules.knowledge.adapters.outbound.lexical import encode_sparse


def qdrant_store(client: AsyncQdrantClient, collection: str = "knowledge") -> QdrantVectorStore:
    return QdrantVectorStore(
        collection_name=collection,
        aclient=client,
        enable_hybrid=True,
        sparse_doc_fn=encode_sparse,
        sparse_query_fn=encode_sparse,
        sparse_config=SparseVectorParams(modifier=Modifier.IDF),
    )


def pgvector_store(
    connection_string: str,
    async_connection_string: str,
    embed_dim: int,
    table: str = "knowledge",
) -> PGVectorStore:
    return PGVectorStore.from_params(
        connection_string=connection_string,
        async_connection_string=async_connection_string,
        table_name=table,
        embed_dim=embed_dim,
        hybrid_search=True,
        # "simple" skips English stemming, which would mangle Arabic text and unit codes.
        text_search_config="simple",
        use_jsonb=True,
    )
