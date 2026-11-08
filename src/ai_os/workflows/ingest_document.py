from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy


@dataclass(frozen=True)
class IngestRequest:
    path: str
    document_id: str
    project: str
    kind: str
    version: int


@dataclass(frozen=True)
class IngestSummary:
    document_id: str
    version: int
    chunks: int


@workflow.defn
class IngestDocumentWorkflow:
    @workflow.run
    async def run(self, request: IngestRequest) -> IngestSummary:
        return await workflow.execute_activity(
            "ingest_document",
            request,
            result_type=IngestSummary,
            # Large PDFs run OCR and layout models.
            start_to_close_timeout=timedelta(minutes=30),
            retry_policy=RetryPolicy(maximum_attempts=5),
        )
