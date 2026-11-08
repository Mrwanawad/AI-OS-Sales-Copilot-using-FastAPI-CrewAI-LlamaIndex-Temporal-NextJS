from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.modules.knowledge.domain.models import DocumentKind, SearchFilters
from ai_os.shared.auth import Authenticator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    project: str | None = None
    kinds: list[DocumentKind] = []


class CitationResponse(BaseModel):
    document_id: str
    document_version: int
    page: int | None


class SearchHit(BaseModel):
    text: str
    score: float
    project: str
    kind: DocumentKind
    unit_codes: list[str]
    citation: CitationResponse


def create_router(get_retrieve: Callable[[], RetrieveKnowledge], auth: Authenticator) -> APIRouter:
    router = APIRouter(prefix="/knowledge", tags=["knowledge"], dependencies=[Depends(auth)])

    @router.post("/search")
    async def search(
        request: SearchRequest,
        retrieve: Annotated[RetrieveKnowledge, Depends(get_retrieve)],
    ) -> list[SearchHit]:
        filters = SearchFilters(project=request.project, kinds=tuple(request.kinds))
        hits = await retrieve(request.query, request.top_k, filters)
        return [
            SearchHit(
                text=hit.chunk.text,
                score=hit.score,
                project=hit.chunk.project,
                kind=hit.chunk.kind,
                unit_codes=list(hit.chunk.unit_codes),
                citation=CitationResponse(
                    document_id=hit.chunk.citation.document_id,
                    document_version=hit.chunk.citation.document_version,
                    page=hit.chunk.citation.page,
                ),
            )
            for hit in hits
        ]

    return router
