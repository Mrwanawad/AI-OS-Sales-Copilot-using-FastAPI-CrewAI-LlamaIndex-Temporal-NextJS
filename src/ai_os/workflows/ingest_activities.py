from temporalio import activity
from temporalio.exceptions import ApplicationError

from ai_os.modules.knowledge.application.ingest import IngestDocument
from ai_os.modules.knowledge.domain.errors import (
    EmptyDocument,
    InvalidPriceSheet,
    StaleDocumentVersion,
)
from ai_os.modules.knowledge.domain.models import DocumentKind, SourceDocument
from ai_os.shared.files import FileStore, LocalPathFiles, StoredFileNotFound
from ai_os.workflows.ingest_document import IngestRequest, IngestSummary

# Retrying cannot fix these; someone has to supply a corrected file.
_PERMANENT = (EmptyDocument, InvalidPriceSheet, StaleDocumentVersion, StoredFileNotFound)


class IngestActivities:
    def __init__(self, ingest: IngestDocument, files: FileStore | None = None) -> None:
        self._ingest = ingest
        self._files = files or LocalPathFiles()

    @activity.defn(name="ingest_document")
    async def ingest_document(self, request: IngestRequest) -> IngestSummary:
        document = SourceDocument(
            request.document_id, request.project, DocumentKind(request.kind), request.version
        )
        try:
            async with self._files.local_copy(request.source) as path:
                result = await self._ingest(path, document)
        except _PERMANENT as error:
            raise ApplicationError(
                str(error), type=type(error).__name__, non_retryable=True
            ) from error
        return IngestSummary(result.document_id, result.version, result.chunks)
