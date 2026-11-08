import re
from collections.abc import Callable
from datetime import datetime
from pathlib import PurePath

from ai_os.modules.knowledge.domain.errors import DocumentProjectMismatch, UploadRejected
from ai_os.modules.knowledge.domain.models import (
    DocumentKind,
    IngestionStatus,
    RegisteredDocument,
    SourceDocument,
)
from ai_os.modules.knowledge.ports.retrieval import VectorStore
from ai_os.modules.knowledge.ports.uploads import (
    DocumentRegistry,
    IngestionStarter,
    UploadedFiles,
)

# Formats the Docling and price sheet parsers read.
ALLOWED_SUFFIXES = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".md", ".html", ".htm", ".txt", ".csv"}
)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_DOCUMENT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,79}$")


class UploadDocument:
    def __init__(
        self,
        files: UploadedFiles,
        registry: DocumentRegistry,
        vectors: Callable[[], VectorStore],
        ingestion: IngestionStarter,
        now: Callable[[], datetime],
    ) -> None:
        self._files = files
        self._registry = registry
        # A factory: the vector store loads embedding weights on creation.
        self._vectors = vectors
        self._ingestion = ingestion
        self._now = now

    async def __call__(
        self,
        document_id: str,
        project: str,
        kind: DocumentKind,
        filename: str,
        content_type: str,
        data: bytes,
        uploaded_by: str,
    ) -> SourceDocument:
        _check(document_id, filename, data)
        existing = await self._registry.get(document_id)
        if existing is not None and existing.project != project:
            raise DocumentProjectMismatch(document_id, existing.project)

        source = await self._files.put(filename, content_type, data, uploaded_by)
        # Documents loaded before the registry existed still have an active version.
        active = await self._vectors().active_version(document_id) or 0
        version = await self._registry.next_version(
            document_id, project, kind, active + 1, source, uploaded_by, self._now()
        )
        document = SourceDocument(document_id, project, kind, version)
        await self._ingestion.start(source, document)
        return document

    async def status(self, document_id: str, version: int) -> IngestionStatus | None:
        registered = await self._registry.get(document_id)
        if registered is None or not 1 <= version <= registered.latest_version:
            return None
        document = SourceDocument(document_id, registered.project, registered.kind, version)
        return await self._ingestion.status(document)

    async def list(self, project: str | None, limit: int = 100) -> list[RegisteredDocument]:
        return await self._registry.list(project, limit)


def _check(document_id: str, filename: str, data: bytes) -> None:
    if not _DOCUMENT_ID.match(document_id):
        raise UploadRejected(
            "document_id must be 2-80 lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    suffix = PurePath(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise UploadRejected(f"{suffix or 'Files without an extension'} can't be ingested")
    if not data:
        raise UploadRejected("The file is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(f"Files over {MAX_UPLOAD_BYTES // (1024 * 1024)} MB are refused")
