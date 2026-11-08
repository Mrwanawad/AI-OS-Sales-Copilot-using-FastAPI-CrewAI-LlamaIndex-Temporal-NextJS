import asyncio

from sqlalchemy.ext.asyncio import create_async_engine
from temporalio.client import Client
from temporalio.worker import Worker

from ai_os.bootstrap.crm import build_crm_activities, build_crm_store
from ai_os.bootstrap.inventory import build_unit_store
from ai_os.bootstrap.knowledge import build_ingest
from ai_os.bootstrap.observability import configure_tracing
from ai_os.shared.config import Settings
from ai_os.shared.files import LocalPathFiles, PostgresFileStore
from ai_os.workflows.crm_event import CrmEventWorkflow
from ai_os.workflows.ingest_activities import IngestActivities
from ai_os.workflows.ingest_document import IngestDocumentWorkflow


async def main() -> None:
    settings = Settings()
    configure_tracing(settings)
    client = await Client.connect(settings.temporal_address)
    # Uploads live in Postgres, so the worker needs no disk shared with the API.
    files = (
        PostgresFileStore(create_async_engine(settings.async_database_url))
        if settings.async_database_url
        else LocalPathFiles()
    )
    ingest = IngestActivities(build_ingest(settings), files)
    crm = build_crm_activities(
        settings, build_crm_store(settings), build_unit_store(settings), files
    )
    worker = Worker(
        client,
        task_queue=settings.temporal_task_queue,
        workflows=[IngestDocumentWorkflow, CrmEventWorkflow],
        activities=[
            ingest.ingest_document,
            crm.process_call_recording,
            crm.build_card,
            crm.build_card_if_stale,
            crm.process_post_call,
            crm.build_handover,
        ],
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
