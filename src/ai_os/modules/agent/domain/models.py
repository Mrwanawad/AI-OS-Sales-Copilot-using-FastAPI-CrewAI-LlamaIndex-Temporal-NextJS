from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum


class Intent(StrEnum):
    KNOWLEDGE = "knowledge"
    INVENTORY = "inventory"
    CALCULATION = "calculation"
    LEAD_COACHING = "lead_coaching"
    ROLE_PLAY = "role_play"
    CONTEXT_CARD = "context_card"
    ESCALATE = "escalate"


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ESCALATED = "escalated"
    NEEDS_CLARIFICATION = "needs_clarification"
    USE_ROLE_PLAY = "use_role_play"


class SourceKind(StrEnum):
    DOCUMENT = "document"
    INVENTORY = "inventory"
    CALCULATOR = "calculator"
    CRM = "crm"


@dataclass(frozen=True)
class EvidenceItem:
    text: str
    kind: SourceKind
    reference: str


@dataclass(frozen=True)
class Evidence:
    id: str
    item: EvidenceItem


@dataclass(frozen=True)
class AnswerCitation:
    kind: SourceKind
    reference: str


@dataclass(frozen=True)
class AgentAnswer:
    text: str
    status: AnswerStatus
    intent: Intent
    citations: list[AnswerCitation] = field(default_factory=list)
    confidence: float = 0.0
    ticket_id: str | None = None


@dataclass(frozen=True)
class InstallmentQuery:
    down_payment_pct: Decimal
    years: int
    months_between_payments: int = 1
    maintenance_pct: Decimal = Decimal(0)
    price: Decimal | None = None
    unit_code: str | None = None


@dataclass(frozen=True)
class MortgageQuery:
    annual_rate_pct: Decimal
    years: int
    down_payment_pct: Decimal = Decimal(0)
    price: Decimal | None = None
    unit_code: str | None = None
