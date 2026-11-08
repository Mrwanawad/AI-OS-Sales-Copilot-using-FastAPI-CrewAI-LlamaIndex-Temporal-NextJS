from collections.abc import Callable, Sequence
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel

from ai_os.modules.crm_copilot.application.history import format_history, history_rule
from ai_os.modules.crm_copilot.domain.errors import LeadNotFound
from ai_os.modules.crm_copilot.domain.grounding import is_supported, source_text
from ai_os.modules.crm_copilot.domain.models import (
    ContextCard,
    Interaction,
    Lead,
    OpenConcern,
    Preferences,
    SourcedItem,
    TimelineEntry,
)
from ai_os.modules.crm_copilot.ports.crm import BestTimeProvider, CrmStore, UnitMatcher
from ai_os.shared.llm.port import LLMRole, StructuredLLM
from ai_os.shared.numbers import numbers_in

_INSTRUCTIONS = f"""\
Prepare a real estate sales rep to call a client they may never have spoken to.
Use only the interactions given. Do not infer facts about the client beyond what was said.

For every item, list the ids of the interactions that support it in source_ids.
- stated_needs: what the client said they want
- closing_drivers: what would make this client buy
- open_concerns: unresolved objections, with a suggested_answer the rep could give
- promises: commitments made to the client that may still be open
- talking_points: what the rep should raise on the next call
- timeline: one short summary per interaction
- preferences: budget, bedrooms and project only if the client stated them

Write in the language the interactions mostly use. {history_rule()}
"""


class SourcedDraft(BaseModel):
    text: str
    source_ids: list[str] = []


class ConcernDraft(BaseModel):
    concern: str
    suggested_answer: str | None = None
    source_ids: list[str] = []


class TimelineDraft(BaseModel):
    interaction_id: str
    summary: str


class PreferencesDraft(BaseModel):
    max_budget_egp: Decimal | None = None
    bedrooms: int | None = None
    project: str | None = None
    source_ids: list[str] = []


class CardDraft(BaseModel):
    timeline: list[TimelineDraft] = []
    stated_needs: list[SourcedDraft] = []
    closing_drivers: list[SourcedDraft] = []
    open_concerns: list[ConcernDraft] = []
    promises: list[SourcedDraft] = []
    talking_points: list[SourcedDraft] = []
    preferences: PreferencesDraft = PreferencesDraft()


class BuildContextCard:
    def __init__(
        self,
        llm: StructuredLLM,
        store: CrmStore,
        matcher: UnitMatcher,
        now: Callable[[], datetime],
        max_history_chars: int = 40_000,
        best_time: BestTimeProvider | None = None,
    ) -> None:
        self._best_time = best_time
        self._llm = llm
        self._store = store
        self._matcher = matcher
        self._now = now
        self._max_history_chars = max_history_chars

    async def __call__(self, lead_id: str, only_if_stale: bool = False) -> ContextCard:
        lead = await self._store.get_lead(lead_id)
        if lead is None:
            raise LeadNotFound(lead_id)
        interactions = await self._store.list_interactions(lead_id)
        latest = max((i.occurred_at for i in interactions), default=None)

        if only_if_stale:
            existing = await self._store.get_card(lead_id)
            if existing is not None and existing.latest_interaction_at == latest:
                return existing

        history, covered = format_history(interactions, self._max_history_chars)
        if covered:
            draft = await self._llm.extract(LLMRole.EXTRACTION, _INSTRUCTIONS, history, CardDraft)
        else:
            # Nothing to summarize, so the model is not asked to invent anything.
            draft = CardDraft()
        card = await self._assemble(lead, interactions, covered, draft, latest)
        await self._store.save_card(card)
        return card

    async def _assemble(
        self,
        lead: Lead,
        interactions: Sequence[Interaction],
        covered: Sequence[Interaction],
        draft: CardDraft,
        latest: datetime | None,
    ) -> ContextCard:
        by_id = {interaction.id: interaction for interaction in covered}
        dropped = _Dropped()
        preferences = _keep_stated_preferences(draft.preferences, by_id, dropped)
        return ContextCard(
            lead_id=lead.id,
            display_name=lead.display_name,
            stage=lead.stage,
            generated_at=self._now(),
            latest_interaction_at=latest,
            timeline=_timeline(interactions, draft.timeline, by_id, dropped),
            stated_needs=_sourced(draft.stated_needs, by_id, dropped),
            closing_drivers=_sourced(draft.closing_drivers, by_id, dropped),
            open_concerns=_concerns(draft.open_concerns, by_id, dropped),
            promises=_sourced(draft.promises, by_id, dropped),
            talking_points=_sourced(draft.talking_points, by_id, dropped),
            preferences=preferences,
            matched_units=await self._matches(preferences),
            omitted_interactions=sum(1 for i in interactions if i.text.strip()) - len(covered),
            dropped_claims=dropped.count,
            best_time_to_call=(
                await self._best_time.best_time(lead.id) if self._best_time else None
            ),
        )

    async def _matches(self, preferences: Preferences):
        if preferences == Preferences():
            return []
        return await self._matcher.match(preferences)


class _Dropped:
    def __init__(self) -> None:
        self.count = 0


def _sourced(
    drafts: Sequence[SourcedDraft], by_id: dict[str, Interaction], dropped: _Dropped
) -> list[SourcedItem]:
    items: list[SourcedItem] = []
    for draft in drafts:
        if is_supported(draft.text, draft.source_ids, by_id):
            items.append(SourcedItem(draft.text, tuple(draft.source_ids)))
        else:
            dropped.count += 1
    return items


def _concerns(
    drafts: Sequence[ConcernDraft], by_id: dict[str, Interaction], dropped: _Dropped
) -> list[OpenConcern]:
    concerns: list[OpenConcern] = []
    for draft in drafts:
        if not is_supported(draft.concern, draft.source_ids, by_id):
            dropped.count += 1
            continue
        suggestion = draft.suggested_answer
        if suggestion and not is_supported(suggestion, draft.source_ids, by_id):
            # A suggestion quoting an unsupported price is worse than none.
            suggestion = None
            dropped.count += 1
        concerns.append(OpenConcern(draft.concern, tuple(draft.source_ids), suggestion))
    return concerns


def _timeline(
    interactions: Sequence[Interaction],
    drafts: Sequence[TimelineDraft],
    by_id: dict[str, Interaction],
    dropped: _Dropped,
) -> list[TimelineEntry]:
    summaries: dict[str, str] = {}
    for draft in drafts:
        if is_supported(draft.summary, [draft.interaction_id], by_id):
            summaries[draft.interaction_id] = draft.summary
        else:
            dropped.count += 1
    ordered = sorted(interactions, key=lambda item: item.occurred_at)
    return [
        TimelineEntry(item.id, item.occurred_at, item.kind, summaries.get(item.id))
        for item in ordered
    ]


def _keep_stated_preferences(
    draft: PreferencesDraft, by_id: dict[str, Interaction], dropped: _Dropped
) -> Preferences:
    sources = [by_id[i] for i in draft.source_ids if i in by_id]
    stated = " ".join(source_text(source) for source in sources)
    numbers = numbers_in(stated)
    budget = draft.max_budget_egp if draft.max_budget_egp in numbers else None
    bedrooms = draft.bedrooms if draft.bedrooms in numbers else None
    project = draft.project
    if project and project.casefold() not in stated.casefold():
        project = None
    dropped.count += sum(
        kept != proposed
        for kept, proposed in (
            (budget, draft.max_budget_egp),
            (bedrooms, draft.bedrooms),
            (project, draft.project),
        )
    )
    return Preferences(budget, bedrooms, project)
