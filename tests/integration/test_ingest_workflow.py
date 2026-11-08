from pathlib import Path
from uuid import uuid4

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from ai_os.modules.knowledge.application.ingest import IngestDocument
from ai_os.modules.knowledge.domain.errors import InvalidPriceSheet
from ai_os.modules.knowledge.domain.models import ParsedSection, SourceDocument
from ai_os.workflows.ingest_activities import IngestActivities
from ai_os.workflows.ingest_document import IngestDocumentWorkflow, IngestRequest, IngestSummary

from ..unit.test_ingest_document import FakeStore

REQUEST = IngestRequest("prices.xlsx", "prices", "Palm Hills", "price_sheet", 1)


class ScriptedParser:
    """Raises the scripted errors in order, then succeeds."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def parse(self, path: Path, document: SourceDocument) -> list[ParsedSection]:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return [ParsedSection("Unit A-101 | Price: 5,000,000 EGP")]


async def run_workflow(parser: ScriptedParser) -> IngestSummary:
    queue = f"test-{uuid4()}"
    activities = IngestActivities(IngestDocument(parser, FakeStore()))
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[IngestDocumentWorkflow],
            activities=[activities.ingest_document],
        ):
            return await env.client.execute_workflow(
                IngestDocumentWorkflow.run, REQUEST, id=str(uuid4()), task_queue=queue
            )


async def test_transient_failure_is_retried_until_success() -> None:
    parser = ScriptedParser(ConnectionError("store unavailable"))

    summary = await run_workflow(parser)

    assert summary == IngestSummary("prices", 1, 1)
    assert parser.calls == 2


async def test_invalid_price_sheet_fails_without_retry() -> None:
    parser = ScriptedParser(InvalidPriceSheet(["Phase1: row 4 price failed"]))

    with pytest.raises(WorkflowFailureError) as error:
        await run_workflow(parser)

    cause = error.value.cause.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == "InvalidPriceSheet"
    assert parser.calls == 1
