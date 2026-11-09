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

    async def set_recording(self, interaction_id: str, reference: str) -> None: ...

    async def update_lead(
        self, lead_id: str, stage: LeadStage | None, fields: Mapping[str, str]
    ) -> None: ...

    async def assign_rep(self, lead_id: str, rep: str) -> None: ...

    async def schedule_follow_up(self, lead_id: str, at: datetime, reason: str) -> None: ...

    async def save_card(self, card: ContextCard) -> None: ...

    async def get_card(self, lead_id: str) -> ContextCard | None: ...

    async def save_handover(self, handover: HandoverSummary) -> None: ...


class PersonalDataStore(Protocol):
    async def erase_lead(self, lead_id: str) -> list[str] | None:
        """Deletes the lead and everything linked to it in one transaction.

        Returns the recording references it held, or None when the lead does not exist.
        """
        ...

    async def purge_interaction_content(self, before: datetime) -> tuple[int, list[str]]:
        """Clears text and recordings of interactions before the cutoff, keeping the time and
        outcome the best-time model trains on.

        Returns how many interactions were cleared and the recording references they held.
        """
        ...

    async def purge_dormant_cards(self, before: datetime) -> int:
        """Deletes cards and handovers of leads with no interaction since the cutoff."""
        ...


class RecordingEraser(Protocol):
    async def delete(self, reference: str) -> None:
        """Deletes a stored recording. References it does not hold are ignored."""
        ...


class LinkedPersonalData(Protocol):
    """Personal data about a lead that another module holds, such as escalation tickets."""

    async def erase_lead(self, lead_id: str) -> int: ...

    async def purge(self, before: datetime) -> int: ...


class Transcriber(Protocol):
    async def transcribe(self, path: Path) -> Sequence[SpeechSegment]: ...


class Diarizer(Protocol):
    async def diarize(self, path: Path) -> Sequence[SpeakerTurn]: ...


class RecordingUploads(Protocol):
    async def put(self, filename: str, content_type: str, data: bytes, uploaded_by: str) -> str:
        """Stores the bytes and returns a reference the worker can read."""
        ...


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
