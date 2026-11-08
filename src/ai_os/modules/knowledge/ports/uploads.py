from datetime import datetime
from typing import Protocol

from ai_os.modules.knowledge.domain.models import (
    DocumentKind,
    IngestionStatus,
    RegisteredDocument,
    SourceDocument,
)


class UploadedFiles(Protocol):
    async def put(self, filename: str, content_type: str, data: bytes, uploaded_by: str) -> str:
        """Stores the bytes and returns a reference the worker can read."""
        ...


class DocumentRegistry(Protocol):
    async def next_version(
        self,
        document_id: str,
        project: str,
        kind: DocumentKind,
        at_least: int,
        source: str,
        uploaded_by: str,
        at: datetime,
    ) -> int:
        """Reserves the next version atomically, so two uploads never share one."""
        ...

    async def get(self, document_id: str) -> RegisteredDocument | None: ...

    async def list(self, project: str | None, limit: int) -> list[RegisteredDocument]: ...


class IngestionStarter(Protocol):
    async def start(self, source: str, document: SourceDocument) -> None: ...

    async def status(self, document: SourceDocument) -> IngestionStatus: ...
