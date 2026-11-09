from datetime import datetime
from typing import Protocol

from ai_os.modules.escalation.domain.models import (
    CandidateStatus,
    KnowledgeCandidate,
    Ticket,
    TicketStatus,
)


class EscalationStore(Protocol):
    async def add_ticket(self, ticket: Ticket) -> None: ...

    async def get_ticket(self, ticket_id: str) -> Ticket | None: ...

    async def list_tickets(self, status: TicketStatus, limit: int) -> list[Ticket]: ...

    async def list_tickets_asked_by(self, asked_by: str, limit: int) -> list[Ticket]: ...

    async def update_ticket(self, ticket: Ticket, expected: TicketStatus) -> bool:
        """Returns False if the stored status no longer matches, so nothing changed."""
        ...

    async def delete_tickets_for_lead(self, lead_id: str) -> int: ...

    async def delete_closed_tickets(self, before: datetime) -> int:
        """Deletes answered and dismissed tickets closed before the cutoff."""
        ...

    async def add_candidate(self, candidate: KnowledgeCandidate) -> None: ...

    async def get_candidate(self, candidate_id: str) -> KnowledgeCandidate | None: ...

    async def list_candidates(
        self, status: CandidateStatus, limit: int
    ) -> list[KnowledgeCandidate]: ...

    async def update_candidate(
        self, candidate: KnowledgeCandidate, expected: CandidateStatus
    ) -> bool:
        """Returns False if the stored status no longer matches, so nothing changed."""
        ...


class KnowledgePublisher(Protocol):
    async def publish(self, candidate: KnowledgeCandidate) -> None: ...

    async def remove(self, candidate: KnowledgeCandidate) -> None: ...


class TicketNotifier(Protocol):
    async def ticket_opened(self, ticket: Ticket) -> None: ...

    async def ticket_answered(self, ticket: Ticket) -> None: ...
