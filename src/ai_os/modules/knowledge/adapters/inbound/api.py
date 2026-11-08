from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.modules.knowledge.application.upload import MAX_UPLOAD_BYTES, UploadDocument
from ai_os.modules.knowledge.domain.errors import (
    DocumentProjectMismatch,
    IngestionAlreadyRunning,
    UploadRejected,
)
from ai_os.modules.knowledge.domain.models import (
    DocumentKind,
    IngestionState,
    RegisteredDocument,
    SearchFilters,
)
from ai_os.shared.auth import Authenticator, Principal, Role


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


class UploadAccepted(BaseModel):
    document_id: str
    project: str
    kind: DocumentKind
    version: int
    status_url: str


class DocumentResponse(BaseModel):
    document_id: str
    project: str
    kind: DocumentKind
    latest_version: int
    uploaded_by: str
    uploaded_at: datetime


class IngestionStatusResponse(BaseModel):
    document_id: str
    version: int
    state: IngestionState
    chunks: int | None
    error: str | None


def create_documents_router(
    get_upload: Callable[[], UploadDocument], auth: Authenticator
) -> APIRouter:
    """Team leads and managers add or replace project documents; reps only search them."""
    router = APIRouter(prefix="/knowledge/documents", tags=["knowledge"])
    editor = auth.require(Role.TEAM_LEAD, Role.MANAGER)

    @router.post("", status_code=202)
    async def upload(
        user: Annotated[Principal, Depends(editor)],
        upload_document: Annotated[UploadDocument, Depends(get_upload)],
        file: Annotated[UploadFile, File()],
        document_id: Annotated[str, Form()],
        project: Annotated[str, Form(min_length=1, max_length=120)],
        kind: Annotated[DocumentKind, Form()],
    ) -> UploadAccepted:
        # One byte over the limit is enough to refuse, without reading a huge body.
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            document = await upload_document(
                document_id,
                project,
                kind,
                file.filename or "",
                file.content_type or "application/octet-stream",
                data,
                user.subject,
            )
        except UploadRejected as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (DocumentProjectMismatch, IngestionAlreadyRunning) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return UploadAccepted(
            document_id=document.id,
            project=document.project,
            kind=document.kind,
            version=document.version,
            status_url=f"/knowledge/documents/{document.id}/versions/{document.version}",
        )

    @router.get("")
    async def list_documents(
        _user: Annotated[Principal, Depends(auth)],
        upload_document: Annotated[UploadDocument, Depends(get_upload)],
        project: str | None = None,
    ) -> list[DocumentResponse]:
        return [_document(d) for d in await upload_document.list(project)]

    @router.get("/{document_id}/versions/{version}")
    async def ingestion_status(
        document_id: str,
        version: int,
        _user: Annotated[Principal, Depends(editor)],
        upload_document: Annotated[UploadDocument, Depends(get_upload)],
    ) -> IngestionStatusResponse:
        status = await upload_document.status(document_id, version)
        if status is None:
            raise HTTPException(status_code=404, detail="No such document version")
        return IngestionStatusResponse(
            document_id=document_id,
            version=version,
            state=status.state,
            chunks=status.chunks,
            error=status.error,
        )

    return router


def _document(document: RegisteredDocument) -> DocumentResponse:
    return DocumentResponse(
        document_id=document.id,
        project=document.project,
        kind=document.kind,
        latest_version=document.latest_version,
        uploaded_by=document.uploaded_by,
        uploaded_at=document.uploaded_at,
    )
