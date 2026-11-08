from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import create_async_engine
from temporalio.client import Client, WorkflowExecutionStatus, WorkflowFailureError
from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from ai_os.modules.knowledge.adapters.outbound.document_registry import PostgresDocumentRegistry
from ai_os.modules.knowledge.application.upload import UploadDocument
from ai_os.modules.knowledge.domain.errors import IngestionAlreadyRunning
from ai_os.modules.knowledge.domain.models import (
    IngestionState,
    IngestionStatus,
    SourceDocument,
)
from ai_os.modules.knowledge.ports.retrieval import VectorStore
from ai_os.shared.config import Settings
from ai_os.shared.files import PostgresFileStore
from ai_os.workflows.ingest_document import IngestDocumentWorkflow, IngestRequest, IngestSummary

_RUNNING = {WorkflowExecutionStatus.RUNNING, WorkflowExecutionStatus.CONTINUED_AS_NEW}


class StoredUploads:
    def __init__(self, files: PostgresFileStore) -> None:
        self._files = files

    async def put(self, filename: str, content_type: str, data: bytes, uploaded_by: str) -> str:
        return (await self._files.put(filename, content_type, data, uploaded_by)).reference


class TemporalIngestion:
    def __init__(self, connect: Callable[[], Awaitable[Client]], task_queue: str) -> None:
        self._connect = connect
        self._task_queue = task_queue
        self._client: Client | None = None

    async def _get_client(self) -> Client:
        if self._client is None:
            self._client = await self._connect()
        return self._client

    async def start(self, source: str, document: SourceDocument) -> None:
        client = await self._get_client()
        try:
            await client.start_workflow(
                IngestDocumentWorkflow.run,
                IngestRequest(
                    source, document.id, document.project, document.kind.value, document.version
                ),
                id=_workflow_id(document),
                task_queue=self._task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
                id_conflict_policy=WorkflowIDConflictPolicy.FAIL,
            )
        except WorkflowAlreadyStartedError as error:
            raise IngestionAlreadyRunning(document.id, document.version) from error

    async def status(self, document: SourceDocument) -> IngestionStatus:
        handle = (await self._get_client()).get_workflow_handle(
            _workflow_id(document), result_type=IngestSummary
        )
        try:
            description = await handle.describe()
        except RPCError as error:
            if error.status is RPCStatusCode.NOT_FOUND:
                # Older versions loaded by script, or history past retention.
                return IngestionStatus(IngestionState.UNKNOWN)
            raise
        if description.status in _RUNNING:
            return IngestionStatus(IngestionState.RUNNING)
        if description.status is WorkflowExecutionStatus.COMPLETED:
            summary = await handle.result()
            return IngestionStatus(IngestionState.COMPLETED, chunks=summary.chunks)
        try:
            await handle.result()
        except WorkflowFailureError as failure:
            cause = failure.cause
            return IngestionStatus(IngestionState.FAILED, error=str(cause or failure))
        return IngestionStatus(IngestionState.FAILED, error=str(description.status))


def _workflow_id(document: SourceDocument) -> str:
    return f"ingest-{document.id}-v{document.version}"


def build_upload_document(settings: Settings, vectors: Callable[[], VectorStore]) -> UploadDocument:
    if not settings.async_database_url:
        raise ValueError("Set ASYNC_DATABASE_URL")
    engine = create_async_engine(settings.async_database_url)

    async def connect() -> Client:
        return await Client.connect(settings.temporal_address)

    return UploadDocument(
        StoredUploads(PostgresFileStore(engine)),
        PostgresDocumentRegistry(engine),
        vectors,
        TemporalIngestion(connect, settings.temporal_task_queue),
        lambda: datetime.now(UTC),
    )
