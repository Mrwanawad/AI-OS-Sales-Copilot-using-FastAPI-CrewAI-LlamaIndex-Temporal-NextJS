from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from uuid import uuid4

from ai_os.modules.escalation.domain.errors import (
    CandidateNotFound,
    ContainsPersonalData,
    InvalidTransition,
    NotShareable,
    SameReviewer,
    TicketNotFound,
)
from ai_os.modules.escalation.domain.models import (
    CandidateStatus,
    KnowledgeCandidate,
    Ticket,
    TicketStatus,
)
from ai_os.modules.escalation.domain.personal_data import find_personal_data
from ai_os.modules.escalation.ports.escalation import (
    EscalationStore,
    KnowledgePublisher,
    TicketNotifier,
)


class Escalations:
    def __init__(
        self,
        store: EscalationStore,
        publisher: KnowledgePublisher,
        notifier: TicketNotifier,
        now: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._publisher = publisher
        self._notifier = notifier
        self._now = now

    async def open_ticket(
        self,
        question: str,
        reason: str,
        project: str | None = None,
        lead_id: str | None = None,
        context: str = "",
        asked_by: str | None = None,
    ) -> Ticket:
        ticket = Ticket(
            uuid4().hex[:12],
            question,
            reason,
            self._now(),
            project=project,
            lead_id=lead_id,
            asked_by=asked_by,
            context=context,
        )
        await self._store.add_ticket(ticket)
        await self._notifier.ticket_opened(ticket)
        return ticket

    async def get_ticket(self, ticket_id: str) -> Ticket:
        ticket = await self._store.get_ticket(ticket_id)
        if ticket is None:
            raise TicketNotFound(ticket_id)
        return ticket

    async def list_tickets(self, status: TicketStatus, limit: int = 100) -> list[Ticket]:
        return await self._store.list_tickets(status, limit)

    async def tickets_asked_by(self, asked_by: str, limit: int = 100) -> list[Ticket]:
        return await self._store.list_tickets_asked_by(asked_by, limit)

    async def answer_ticket(
        self,
        ticket_id: str,
        answer: str,
        answered_by: str,
        share_with_project: str | None = None,
    ) -> tuple[Ticket, KnowledgeCandidate | None]:
        """Answers the rep now. share_with_project also proposes the answer for shared knowledge."""
        ticket = await self.get_ticket(ticket_id)
        candidate_text = None
        if share_with_project:
            # Checked first, so a refused proposal leaves the ticket open to be answered again.
            if ticket.lead_id:
                raise NotShareable("This answer is about a specific client, so it can't be shared")
            candidate_text = f"{ticket.question}\n{answer}"
            _refuse_personal_data(candidate_text)

        answered = replace(
            ticket,
            status=TicketStatus.ANSWERED,
            answer=answer,
            answered_by=answered_by,
            closed_at=self._now(),
        )
        await self._move_ticket(answered, TicketStatus.OPEN)
        if not share_with_project:
            return answered, None
        candidate = KnowledgeCandidate(
            uuid4().hex[:12],
            ticket.id,
            ticket.question,
            answer,
            share_with_project,
            answered_by,
            self._now(),
        )
        await self._store.add_candidate(candidate)
        return answered, candidate

    async def dismiss_ticket(self, ticket_id: str, dismissed_by: str, note: str) -> Ticket:
        ticket = await self.get_ticket(ticket_id)
        dismissed = replace(
            ticket,
            status=TicketStatus.DISMISSED,
            answer=note,
            answered_by=dismissed_by,
            closed_at=self._now(),
        )
        await self._move_ticket(dismissed, TicketStatus.OPEN)
        return dismissed

    async def list_candidates(
        self, status: CandidateStatus, limit: int = 100
    ) -> list[KnowledgeCandidate]:
        return await self._store.list_candidates(status, limit)

    async def approve(self, candidate_id: str, reviewed_by: str) -> KnowledgeCandidate:
        candidate = await self._get_candidate(candidate_id)
        if reviewed_by == candidate.proposed_by:
            raise SameReviewer()
        _refuse_personal_data(f"{candidate.question}\n{candidate.answer}")
        approved = self._reviewed(candidate, CandidateStatus.APPROVED, reviewed_by)
        # Claimed before publishing, so a simultaneous rejection can't leave it live and rejected.
        await self._move_candidate(approved, CandidateStatus.PENDING)
        try:
            await self._publisher.publish(approved)
        except Exception:
            await self._store.update_candidate(candidate, CandidateStatus.APPROVED)
            raise
        return approved

    async def reject(self, candidate_id: str, reviewed_by: str, note: str) -> KnowledgeCandidate:
        candidate = await self._get_candidate(candidate_id)
        rejected = self._reviewed(candidate, CandidateStatus.REJECTED, reviewed_by, note)
        await self._move_candidate(rejected, CandidateStatus.PENDING)
        return rejected

    async def revoke(self, candidate_id: str, revoked_by: str, note: str) -> KnowledgeCandidate:
        candidate = await self._get_candidate(candidate_id)
        revoked = self._reviewed(candidate, CandidateStatus.REVOKED, revoked_by, note)
        await self._move_candidate(revoked, CandidateStatus.APPROVED)
        try:
            await self._publisher.remove(revoked)
        except Exception:
            await self._store.update_candidate(candidate, CandidateStatus.REVOKED)
            raise
        return revoked

    def _reviewed(
        self,
        candidate: KnowledgeCandidate,
        status: CandidateStatus,
        reviewed_by: str,
        note: str = "",
    ) -> KnowledgeCandidate:
        return replace(
            candidate, status=status, reviewed_by=reviewed_by, reviewed_at=self._now(), note=note
        )

    async def _get_candidate(self, candidate_id: str) -> KnowledgeCandidate:
        candidate = await self._store.get_candidate(candidate_id)
        if candidate is None:
            raise CandidateNotFound(candidate_id)
        return candidate

    async def _move_ticket(self, ticket: Ticket, expected: TicketStatus) -> None:
        if not await self._store.update_ticket(ticket, expected):
            raise InvalidTransition(f"Ticket {ticket.id} is no longer {expected.value}")

    async def _move_candidate(
        self, candidate: KnowledgeCandidate, expected: CandidateStatus
    ) -> None:
        if not await self._store.update_candidate(candidate, expected):
            raise InvalidTransition(f"Candidate {candidate.id} is no longer {expected.value}")


def _refuse_personal_data(text: str) -> None:
    kinds = find_personal_data(text)
    if kinds:
        raise ContainsPersonalData(kinds)
