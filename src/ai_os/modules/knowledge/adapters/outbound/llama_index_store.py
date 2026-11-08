import inspect
import json
from collections.abc import Sequence

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)

from ai_os.modules.knowledge.domain.models import (
    GENERAL_PROJECT,
    Chunk,
    Citation,
    DocumentKind,
    ScoredChunk,
    SearchFilters,
)

# LlamaIndex overwrites a "document_id" payload key with its own reference id.
_DOCUMENT_ID = "source_document_id"
# Qdrant's scroll defaults to 10 points, which could miss the newest version.
_VERSION_SCAN = 10_000


class LlamaIndexVectorStore:
    """Implements the VectorStore port on any LlamaIndex store with hybrid search."""

    def __init__(self, store: BasePydanticVectorStore, embedder: BaseEmbedding) -> None:
        self._store = store
        self._embedder = embedder

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        embeddings = await self._embedder.aget_text_embedding_batch([c.text for c in chunks])
        nodes = [
            _to_node(chunk, embedding) for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        await self._store.async_add(nodes)

    async def search(
        self, query: str, top_k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        result = await self._store.aquery(
            VectorStoreQuery(
                query_embedding=await self._embedder.aget_query_embedding(query),
                query_str=query,
                similarity_top_k=top_k,
                sparse_top_k=top_k,
                hybrid_top_k=top_k,
                mode=VectorStoreQueryMode.HYBRID,
                filters=_to_metadata_filters(filters),
            )
        )
        nodes = result.nodes or []
        scores = result.similarities or [0.0] * len(nodes)
        return [
            ScoredChunk(_to_chunk(node), score) for node, score in zip(nodes, scores, strict=True)
        ]

    async def active_version(self, document_id: str) -> int | None:
        if not await self._store_ready():
            return None
        # Only some stores accept a limit, and Qdrant's default would be too small.
        accepts_limit = "limit" in inspect.signature(self._store.aget_nodes).parameters
        nodes = await self._store.aget_nodes(
            filters=_document_filter(document_id),
            **({"limit": _VERSION_SCAN} if accepts_limit else {}),
        )
        return max((node.metadata["document_version"] for node in nodes), default=None)

    async def delete_older_versions(self, document_id: str, version: int) -> None:
        filters = _document_filter(document_id)
        filters.filters.append(
            MetadataFilter(key="document_version", value=version, operator=FilterOperator.LT)
        )
        await self._store.adelete_nodes(filters=filters)

    async def _store_ready(self) -> bool:
        # Qdrant creates its collection on first write; before that nothing is active.
        exists = getattr(self._store, "_acollection_exists", None)
        collection = getattr(self._store, "collection_name", None)
        return await exists(collection) if exists and collection else True


def _to_node(chunk: Chunk, embedding: list[float]) -> TextNode:
    return TextNode(
        id_=chunk.id,
        text=chunk.text,
        embedding=embedding,
        metadata={
            _DOCUMENT_ID: chunk.citation.document_id,
            "document_version": chunk.citation.document_version,
            "page": chunk.citation.page if chunk.citation.page is not None else -1,
            "project": chunk.project,
            "kind": chunk.kind.value,
            "unit_codes": json.dumps(chunk.unit_codes, ensure_ascii=False),
        },
    )


def _to_chunk(node: BaseNode) -> Chunk:
    metadata = node.metadata
    page = metadata["page"]
    return Chunk(
        id=node.node_id,
        text=node.get_content(),
        citation=Citation(
            document_id=metadata[_DOCUMENT_ID],
            document_version=metadata["document_version"],
            page=None if page == -1 else page,
        ),
        project=metadata["project"],
        kind=DocumentKind(metadata["kind"]),
        unit_codes=tuple(json.loads(metadata["unit_codes"])),
    )


def _document_filter(document_id: str) -> MetadataFilters:
    return MetadataFilters(filters=[MetadataFilter(key=_DOCUMENT_ID, value=document_id)])


def _to_metadata_filters(filters: SearchFilters | None) -> MetadataFilters | None:
    if filters is None:
        return None
    conditions: list[MetadataFilter] = []
    if filters.project is not None:
        conditions.append(
            MetadataFilter(
                key="project",
                value=[filters.project, GENERAL_PROJECT],
                operator=FilterOperator.IN,
            )
        )
    if filters.kinds:
        conditions.append(
            MetadataFilter(
                key="kind", value=[k.value for k in filters.kinds], operator=FilterOperator.IN
            )
        )
    return MetadataFilters(filters=conditions) if conditions else None
