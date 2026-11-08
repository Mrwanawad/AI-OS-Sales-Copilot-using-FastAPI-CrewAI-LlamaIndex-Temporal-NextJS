from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

_RETRY = RetryPolicy(maximum_attempts=5)
_LLM_STEP = timedelta(minutes=5)
# Whisper on a long call, on CPU.
_TRANSCRIPTION_STEP = timedelta(hours=2)


@dataclass(frozen=True)
class CrmEvent:
    event_id: str
    type: str
    lead_id: str
    interaction_id: str | None = None
    from_rep: str | None = None
    to_rep: str | None = None


@workflow.defn
class CrmEventWorkflow:
    @workflow.run
    async def run(self, event: CrmEvent) -> None:
        if event.type == "call_ended":
            await self._step("process_call_recording", event, _TRANSCRIPTION_STEP)
            # Post-call first, so the card shows the updated stage.
            await self._step("process_post_call", event, _LLM_STEP)
            await self._step("build_card", event, _LLM_STEP)
        elif event.type == "lead_reassigned":
            await self._step("build_card", event, _LLM_STEP)
            await self._step("build_handover", event, _LLM_STEP)
        elif event.type == "lead_opened":
            await self._step("build_card_if_stale", event, _LLM_STEP)

    async def _step(self, name: str, event: CrmEvent, timeout: timedelta) -> None:
        await workflow.execute_activity(
            name, event, start_to_close_timeout=timeout, retry_policy=_RETRY
        )
