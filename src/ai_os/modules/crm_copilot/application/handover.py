from collections.abc import Callable
from datetime import datetime

from pydantic import BaseModel

from ai_os.modules.crm_copilot.application.history import history_rule
from ai_os.modules.crm_copilot.domain.errors import LeadNotFound
from ai_os.modules.crm_copilot.domain.models import ContextCard, HandoverSummary
from ai_os.modules.crm_copilot.ports.crm import CrmStore
from ai_os.shared.llm.port import LLMRole, StructuredLLM
from ai_os.shared.numbers import numbers_in

_INSTRUCTIONS = f"""\
A lead is being reassigned. Write a short handover for the new rep from the client card below:
where the deal stands, what the client wants, what is unresolved, and what was promised.
Use only what the card says. {history_rule()}
"""


class HandoverDraft(BaseModel):
    summary: str


class BuildHandover:
    def __init__(self, llm: StructuredLLM, store: CrmStore, now: Callable[[], datetime]) -> None:
        self._llm = llm
        self._store = store
        self._now = now

    async def __call__(self, lead_id: str, from_rep: str, to_rep: str) -> HandoverSummary:
        # First, so the new rep can open the lead even if the summary fails.
        if to_rep:
            await self._store.assign_rep(lead_id, to_rep)
        card = await self._store.get_card(lead_id)
        if card is None:
            raise LeadNotFound(lead_id)
        card_text = _card_text(card)
        draft = await self._llm.extract(LLMRole.EXTRACTION, _INSTRUCTIONS, card_text, HandoverDraft)
        # A handover quoting an unsupported number falls back to the checked card facts.
        grounded = numbers_in(draft.summary) <= numbers_in(card_text)
        handover = HandoverSummary(
            lead_id, from_rep, to_rep, draft.summary if grounded else card_text, self._now()
        )
        await self._store.save_handover(handover)
        return handover


def _card_text(card: ContextCard) -> str:
    sections = [
        f"Client: {card.display_name}. Stage: {card.stage.value}.",
        _section("Stated needs", [item.text for item in card.stated_needs]),
        _section("Closing drivers", [item.text for item in card.closing_drivers]),
        _section("Open concerns", [item.concern for item in card.open_concerns]),
        _section("Promises", [item.text for item in card.promises]),
        _section("Talking points", [item.text for item in card.talking_points]),
    ]
    return "\n".join(section for section in sections if section)


def _section(title: str, items: list[str]) -> str:
    return f"{title}: " + "; ".join(items) if items else ""
