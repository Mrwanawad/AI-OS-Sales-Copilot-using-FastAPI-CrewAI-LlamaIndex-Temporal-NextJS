from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ai_os.modules.crm_copilot.domain.models import (
    ContextCard,
    HandoverSummary,
    Interaction,
    Lead,
    LeadStage,
    MatchedUnit,
    Preferences,
    SpeakerTurn,
    SpeechSegment,
)


class CrmStore(Protocol):
    async def get_lead(self, lead_id: str) -> Lead | None: ...

    async def list_leads(self, assigned_rep: str | None, limit: int) -> list[Lead]:
        """All leads when assigned_rep is None."""
        ...

    async def get_interaction(self, interaction_id: str) -> Interaction | None: ...

    async def list_interactions(self, lead_id: str) -> list[Interaction]: ...

    async def set_interaction_text(self, interaction_id: str, text: str) -> None: ...

    async def update_lead(
        self, lead_id: str, stage: LeadStage | None, fields: Mapping[str, str]
    ) -> None: ...

    async def assign_rep(self, lead_id: str, rep: str) -> None: ...

    async def schedule_follow_up(self, lead_id: str, at: datetime, reason: str) -> None: ...

    async def save_card(self, card: ContextCard) -> None: ...

    async def get_card(self, lead_id: str) -> ContextCard | None: ...

    async def save_handover(self, handover: HandoverSummary) -> None: ...


class Transcriber(Protocol):
    async def transcribe(self, path: Path) -> Sequence[SpeechSegment]: ...


class Diarizer(Protocol):
    async def diarize(self, path: Path) -> Sequence[SpeakerTurn]: ...


class RecordingFiles(Protocol):
    def local_copy(self, reference: str) -> AbstractAsyncContextManager[Path]:
        """Yields a readable local file for a stored recording reference."""
        ...


class UnitMatcher(Protocol):
    async def match(self, preferences: Preferences) -> list[MatchedUnit]: ...


class BestTimeProvider(Protocol):
    async def best_time(self, lead_id: str) -> str | None: ...


class EventPublisher(Protocol):
    async def publish(
        self,
        event_id: str,
        event_type: str,
        lead_id: str,
        interaction_id: str | None,
        from_rep: str | None,
        to_rep: str | None,
    ) -> None:
        """Publishing the same event_id twice has no further effect."""
        ...
