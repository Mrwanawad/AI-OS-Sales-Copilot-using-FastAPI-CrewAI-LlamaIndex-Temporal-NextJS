from dataclasses import dataclass, field
from uuid import uuid4

import pytest
from temporalio.client import WorkflowFailureError
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from ai_os.modules.crm_copilot.domain.errors import LeadNotFound
from ai_os.shared.config import MissingApiKey
from ai_os.workflows.crm_activities import CrmActivities
from ai_os.workflows.crm_event import CrmEvent, CrmEventWorkflow


@dataclass
class Recorder:
    calls: list[str] = field(default_factory=list)
    errors: dict[str, list[Exception]] = field(default_factory=dict)

    def step(self, name: str):
        async def run(*args: object, **kwargs: object) -> None:
            self.calls.append(name)
            pending = self.errors.get(name)
            if pending:
                raise pending.pop(0)

        return run


async def run_event(event: CrmEvent, recorder: Recorder) -> None:
    activities = CrmActivities(
        recorder.step("process_call"),
        recorder.step("build_card"),
        recorder.step("post_call"),
        recorder.step("handover"),
    )
    queue = f"test-{uuid4()}"
    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue=queue,
            workflows=[CrmEventWorkflow],
            activities=[
                activities.process_call_recording,
                activities.build_card,
                activities.build_card_if_stale,
                activities.process_post_call,
                activities.build_handover,
            ],
        ):
            await env.client.execute_workflow(
                CrmEventWorkflow.run, event, id=str(uuid4()), task_queue=queue
            )


async def test_call_ended_runs_recording_then_crm_update_then_card() -> None:
    recorder = Recorder()

    await run_event(CrmEvent("e1", "call_ended", "L1", "c1"), recorder)

    assert recorder.calls == ["process_call", "post_call", "build_card"]


async def test_reassignment_builds_card_then_handover() -> None:
    recorder = Recorder()

    await run_event(CrmEvent("e1", "lead_reassigned", "L1", None, "a", "b"), recorder)

    assert recorder.calls == ["build_card", "handover"]


async def test_lead_opened_only_refreshes_the_card() -> None:
    recorder = Recorder()

    await run_event(CrmEvent("e1", "lead_opened", "L1"), recorder)

    assert recorder.calls == ["build_card"]


async def test_transient_failure_is_retried() -> None:
    recorder = Recorder(errors={"build_card": [ConnectionError("database restarting")]})

    await run_event(CrmEvent("e1", "lead_opened", "L1"), recorder)

    assert recorder.calls == ["build_card", "build_card"]


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (LeadNotFound("L1"), "LeadNotFound"),
        (MissingApiKey("ANTHROPIC_API_KEY is not set"), "MissingApiKey"),
    ],
)
async def test_permanent_failure_stops_without_retry(error: Exception, kind: str) -> None:
    recorder = Recorder(errors={"build_card": [error]})

    with pytest.raises(WorkflowFailureError) as failure:
        await run_event(CrmEvent("e1", "lead_opened", "L1"), recorder)

    cause = failure.value.cause.cause
    assert isinstance(cause, ApplicationError)
    assert cause.type == kind
    assert recorder.calls == ["build_card"]


async def test_call_event_without_interaction_id_fails_without_retry() -> None:
    recorder = Recorder()

    with pytest.raises(WorkflowFailureError):
        await run_event(CrmEvent("e1", "call_ended", "L1"), recorder)

    assert recorder.calls == []
