from datetime import UTC, datetime
from functools import partial
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import create_async_engine
from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy
from temporalio.exceptions import WorkflowAlreadyStartedError

from ai_os.bootstrap.best_time import BestTimeCardProvider, build_recommend
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.modules.crm_copilot.adapters.outbound.speech import (
    NoDiarizer,
    PyannoteDiarizer,
    WhisperTranscriber,
)
from ai_os.modules.crm_copilot.application.build_card import BuildContextCard
from ai_os.modules.crm_copilot.application.handover import BuildHandover
from ai_os.modules.crm_copilot.application.post_call import ProcessPostCall
from ai_os.modules.crm_copilot.application.privacy import (
    PersonalDataLifecycle,
    RetentionPeriods,
)
from ai_os.modules.crm_copilot.application.process_call import ProcessCallRecording
from ai_os.modules.crm_copilot.domain.models import MatchedUnit, Preferences
from ai_os.modules.crm_copilot.ports.crm import CrmStore, RecordingFiles
from ai_os.modules.inventory.ports.store import UnitStore
from ai_os.shared.config import Settings
from ai_os.shared.llm.pydantic_ai_adapter import PydanticAIStructuredLLM, build_model
from ai_os.workflows.crm_activities import CrmActivities
from ai_os.workflows.crm_event import CrmEvent, CrmEventWorkflow


def utc_now() -> datetime:
    return datetime.now(UTC)


class InventoryUnitMatcher:
    def __init__(self, units: UnitStore, limit: int = 5) -> None:
        self._units = units
        self._limit = limit

    async def match(self, preferences: Preferences) -> list[MatchedUnit]:
        found = await self._units.find_available(
            preferences.max_budget_egp, preferences.bedrooms, preferences.project, self._limit
        )
        return [MatchedUnit(u.code, u.project, u.price_egp, u.bedrooms) for u in found]


class TemporalEventPublisher:
    def __init__(self, client_factory, task_queue: str) -> None:
        self._client_factory = client_factory
        self._task_queue = task_queue
        self._client: Client | None = None

    async def publish(
        self,
        event_id: str,
        event_type: str,
        lead_id: str,
        interaction_id: str | None,
        from_rep: str | None,
        to_rep: str | None,
    ) -> None:
        if self._client is None:
            self._client = await self._client_factory()
        try:
            await self._client.start_workflow(
                CrmEventWorkflow.run,
                CrmEvent(event_id, event_type, lead_id, interaction_id, from_rep, to_rep),
                # CRMs redeliver webhooks; the event id makes redelivery a no-op.
                id=f"crm-event-{event_id}",
                task_queue=self._task_queue,
                id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
                id_conflict_policy=WorkflowIDConflictPolicy.FAIL,
            )
        except WorkflowAlreadyStartedError:
            return


def build_crm_store(settings: Settings) -> PostgresCrmStore:
    if not settings.async_database_url:
        raise ValueError("Set ASYNC_DATABASE_URL")
    return PostgresCrmStore(create_async_engine(settings.async_database_url))


def build_crm_activities(
    settings: Settings, store: CrmStore, units: UnitStore, files: RecordingFiles | None = None
) -> CrmActivities:
    # Keys are read when a model is first used, so ingestion still runs on a worker without them.
    llm = PydanticAIStructuredLLM(partial(build_model, settings))
    token = settings.hugging_face_token.get_secret_value() if settings.hugging_face_token else None
    diarizer = PyannoteDiarizer(token) if token else NoDiarizer()
    transcriber = WhisperTranscriber(settings.whisper_model)
    best_time = BestTimeCardProvider(
        build_recommend(settings, store), ZoneInfo(settings.business_timezone)
    )
    return CrmActivities(
        ProcessCallRecording(store, transcriber, diarizer, files),
        BuildContextCard(llm, store, InventoryUnitMatcher(units), utc_now, best_time=best_time),
        ProcessPostCall(llm, store, utc_now),
        BuildHandover(llm, store, utc_now),
    )


def build_privacy(settings: Settings) -> PersonalDataLifecycle:
    from ai_os.bootstrap.escalation import build_escalations
    from ai_os.bootstrap.uploads import StoredUploads
    from ai_os.shared.files import PostgresFileStore

    store = build_crm_store(settings)
    periods = RetentionPeriods(
        settings.interaction_content_retention_days,
        settings.dormant_card_retention_days,
        settings.closed_ticket_retention_days,
    )
    files = StoredUploads(PostgresFileStore(create_async_engine(settings.async_database_url)))
    return PersonalDataLifecycle(store, files, [build_escalations(settings)], periods, utc_now)


def build_event_publisher(settings: Settings) -> TemporalEventPublisher:
    async def connect() -> Client:
        return await Client.connect(settings.temporal_address)

    return TemporalEventPublisher(connect, settings.temporal_task_queue)
