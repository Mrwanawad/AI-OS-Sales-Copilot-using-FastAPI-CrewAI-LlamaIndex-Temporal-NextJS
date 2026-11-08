from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime

from ai_os.modules.crm_copilot.domain.models import (
    ContextCard,
    HandoverSummary,
    Interaction,
    InteractionKind,
    Lead,
    LeadStage,
    MatchedUnit,
    Preferences,
)
from ai_os.shared.llm.port import LLMRole, OutputT

NOW = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
LEAD = Lead("L1", "Client One", LeadStage.CONTACTED, "rep-a")


def call(
    interaction_id: str = "c1",
    text: str = "Client wants 3 bedrooms, budget 5000000 EGP in Palm Hills.",
    day: int = 1,
    **overrides: object,
) -> Interaction:
    return Interaction(
        interaction_id,
        "L1",
        InteractionKind.CALL,
        datetime(2026, 11, day, 10, 0, tzinfo=UTC),
        text,
        **overrides,
    )


class FakeCrmStore:
    def __init__(self, *interactions: Interaction, lead: Lead | None = LEAD) -> None:
        self.lead = lead
        self.interactions = {item.id: item for item in interactions}
        self.card: ContextCard | None = None
        self.handovers: list[HandoverSummary] = []
        self.stage_updates: list[LeadStage | None] = []
        self.field_updates: dict[str, str] = {}
        self.follow_ups: list[tuple[datetime, str]] = []

    async def get_lead(self, lead_id: str) -> Lead | None:
        return self.lead if self.lead and self.lead.id == lead_id else None

    async def list_leads(self, assigned_rep: str | None, limit: int) -> list[Lead]:
        if self.lead is None or assigned_rep not in (None, self.lead.assigned_rep):
            return []
        return [self.lead][:limit]

    async def get_interaction(self, interaction_id: str) -> Interaction | None:
        return self.interactions.get(interaction_id)

    async def list_interactions(self, lead_id: str) -> list[Interaction]:
        return list(self.interactions.values())

    async def set_recording(self, interaction_id: str, reference: str) -> None:
        current = self.interactions[interaction_id]
        self.interactions[interaction_id] = replace(current, recording_path=reference)

    async def set_interaction_text(self, interaction_id: str, text: str) -> None:
        current = self.interactions[interaction_id]
        self.interactions[interaction_id] = Interaction(
            current.id,
            current.lead_id,
            current.kind,
            current.occurred_at,
            text,
            current.recording_path,
            current.recording_consent,
        )

    async def update_lead(
        self, lead_id: str, stage: LeadStage | None, fields: Mapping[str, str]
    ) -> None:
        self.stage_updates.append(stage)
        self.field_updates.update(fields)

    async def assign_rep(self, lead_id: str, rep: str) -> None:
        self.assigned_to = rep

    async def schedule_follow_up(self, lead_id: str, at: datetime, reason: str) -> None:
        self.follow_ups.append((at, reason))

    async def save_card(self, card: ContextCard) -> None:
        self.card = card

    async def get_card(self, lead_id: str) -> ContextCard | None:
        return self.card

    async def save_handover(self, handover: HandoverSummary) -> None:
        self.handovers.append(handover)


class FakeMatcher:
    def __init__(self, units: list[MatchedUnit] | None = None) -> None:
        self.units = units or []
        self.preferences: list[Preferences] = []

    async def match(self, preferences: Preferences) -> list[MatchedUnit]:
        self.preferences.append(preferences)
        return self.units


class ScriptedLLM:
    def __init__(self, *outputs: object) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    async def extract(
        self, role: LLMRole, instructions: str, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        self.prompts.append(prompt)
        return self.outputs.pop(0)
