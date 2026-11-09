from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TicketStatus(StrEnum):
    OPEN = "open"
    ANSWERED = "answered"
    DISMISSED = "dismissed"


class CandidateStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"
    # Taken out of search when its review period ran out; approve a fresh answer to restore it.
    EXPIRED = "expired"


@dataclass(frozen=True)
class Ticket:
    id: str
    question: str
    reason: str
    created_at: datetime
    status: TicketStatus = TicketStatus.OPEN
    project: str | None = None
    lead_id: str | None = None
    asked_by: str | None = None
    # Background for the person answering, such as the client's context card.
    context: str = ""
    answer: str | None = None
    answered_by: str | None = None
    closed_at: datetime | None = None


@dataclass(frozen=True)
class KnowledgeCandidate:
    id: str
    ticket_id: str
    question: str
    answer: str
    project: str
    proposed_by: str
    created_at: datetime
    status: CandidateStatus = CandidateStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    note: str = ""
    # Approved answers can quote prices that change, so each one leaves search at this time.
    expires_at: datetime | None = None
