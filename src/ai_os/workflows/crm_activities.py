from functools import wraps

from temporalio import activity
from temporalio.exceptions import ApplicationError

from ai_os.modules.crm_copilot.application.build_card import BuildContextCard
from ai_os.modules.crm_copilot.application.handover import BuildHandover
from ai_os.modules.crm_copilot.application.post_call import ProcessPostCall
from ai_os.modules.crm_copilot.application.process_call import ProcessCallRecording
from ai_os.modules.crm_copilot.domain.errors import InteractionNotFound, LeadNotFound
from ai_os.shared.config import MissingApiKey
from ai_os.workflows.crm_event import CrmEvent

# Retrying cannot fix these.
_PERMANENT = (InteractionNotFound, LeadNotFound, MissingApiKey)


def _permanent_errors_stop_retries(step):
    @wraps(step)
    async def wrapper(self, event: CrmEvent) -> None:
        try:
            await step(self, event)
        except _PERMANENT as error:
            raise ApplicationError(
                str(error), type=type(error).__name__, non_retryable=True
            ) from error

    return wrapper


class CrmActivities:
    def __init__(
        self,
        process_call: ProcessCallRecording,
        build_card: BuildContextCard,
        post_call: ProcessPostCall,
        handover: BuildHandover,
    ) -> None:
        self._process_call = process_call
        self._build_card = build_card
        self._post_call = post_call
        self._handover = handover

    @activity.defn(name="process_call_recording")
    @_permanent_errors_stop_retries
    async def process_call_recording(self, event: CrmEvent) -> None:
        await self._process_call(_interaction_id(event))

    @activity.defn(name="build_card")
    @_permanent_errors_stop_retries
    async def build_card(self, event: CrmEvent) -> None:
        await self._build_card(event.lead_id)

    @activity.defn(name="build_card_if_stale")
    @_permanent_errors_stop_retries
    async def build_card_if_stale(self, event: CrmEvent) -> None:
        await self._build_card(event.lead_id, only_if_stale=True)

    @activity.defn(name="process_post_call")
    @_permanent_errors_stop_retries
    async def process_post_call(self, event: CrmEvent) -> None:
        await self._post_call(_interaction_id(event))

    @activity.defn(name="build_handover")
    @_permanent_errors_stop_retries
    async def build_handover(self, event: CrmEvent) -> None:
        await self._handover(event.lead_id, event.from_rep or "", event.to_rep or "")


def _interaction_id(event: CrmEvent) -> str:
    if not event.interaction_id:
        raise ApplicationError(
            f"{event.type} event {event.event_id} has no interaction_id", non_retryable=True
        )
    return event.interaction_id
