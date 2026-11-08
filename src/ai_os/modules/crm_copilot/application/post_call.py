from collections.abc import Callable
from datetime import datetime, timedelta

from pydantic import BaseModel

from ai_os.modules.crm_copilot.application.history import history_rule
from ai_os.modules.crm_copilot.domain.errors import InteractionNotFound, LeadNotFound
from ai_os.modules.crm_copilot.domain.grounding import is_supported
from ai_os.modules.crm_copilot.domain.models import (
    HUMAN_ONLY_STAGES,
    UPDATABLE_FIELDS,
    Lead,
    LeadStage,
    PostCallResult,
)
from ai_os.modules.crm_copilot.ports.crm import CrmStore
from ai_os.shared.llm.port import LLMRole, StructuredLLM

_INSTRUCTIONS = f"""\
Summarize this sales call and extract CRM updates. Only extract what was actually said.
Allowed fields: {", ".join(sorted(UPDATABLE_FIELDS))}.
stage: the new pipeline stage only if the call clearly moved the lead forward.
follow_up_at: ISO date and time only if a follow-up was agreed, relative to the call time given.
{history_rule()}
"""

_MAX_FOLLOW_UP = timedelta(days=180)


class PostCallDraft(BaseModel):
    summary: str
    fields: dict[str, str] = {}
    stage: LeadStage | None = None
    follow_up_at: datetime | None = None
    follow_up_reason: str = ""


class ProcessPostCall:
    def __init__(self, llm: StructuredLLM, store: CrmStore, now: Callable[[], datetime]) -> None:
        self._llm = llm
        self._store = store
        self._now = now

    async def __call__(self, interaction_id: str) -> PostCallResult:
        call = await self._store.get_interaction(interaction_id)
        if call is None:
            raise InteractionNotFound(interaction_id)
        lead = await self._store.get_lead(call.lead_id)
        if lead is None:
            raise LeadNotFound(call.lead_id)
        if not call.text.strip():
            # No consent or an unanswered call: nothing was said that could update the CRM.
            return PostCallResult(interaction_id, None)

        draft = await self._llm.extract(
            LLMRole.EXTRACTION,
            _INSTRUCTIONS,
            f"Call at {call.occurred_at:%Y-%m-%d %H:%M}:\n{call.text}",
            PostCallDraft,
        )
        sources = {call.id: call}
        applied: dict[str, str] = {}
        rejected: list[str] = []
        for name, value in draft.fields.items():
            if name in UPDATABLE_FIELDS and is_supported(value, [call.id], sources):
                applied[name] = value
            else:
                rejected.append(name)

        stage = _allowed_stage(lead, draft.stage)
        if applied or stage:
            await self._store.update_lead(lead.id, stage, applied)

        follow_up = self._valid_follow_up(draft.follow_up_at)
        if follow_up:
            await self._store.schedule_follow_up(lead.id, follow_up, draft.follow_up_reason)

        summary = draft.summary if is_supported(draft.summary, [call.id], sources) else None
        return PostCallResult(interaction_id, summary, applied, rejected, stage, follow_up)

    def _valid_follow_up(self, at: datetime | None) -> datetime | None:
        if at is None:
            return None
        now = self._now()
        at = at if at.tzinfo else at.replace(tzinfo=now.tzinfo)
        return at if now < at <= now + _MAX_FOLLOW_UP else None


def _allowed_stage(lead: Lead, proposed: LeadStage | None) -> LeadStage | None:
    """Automation may only move a lead forward, and never into a stage a person must set."""
    if proposed is None or proposed in HUMAN_ONLY_STAGES:
        return None
    if lead.stage in HUMAN_ONLY_STAGES:
        return None
    order = list(LeadStage)
    return proposed if order.index(proposed) > order.index(lead.stage) else None
