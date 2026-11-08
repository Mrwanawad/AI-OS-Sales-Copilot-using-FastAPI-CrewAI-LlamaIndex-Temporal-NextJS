from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class InteractionKind(StrEnum):
    CALL = "call"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    MEETING = "meeting"
    NOTE = "note"


class LeadStage(StrEnum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    SITE_VISIT = "site_visit"
    NEGOTIATION = "negotiation"
    RESERVED = "reserved"
    WON = "won"
    LOST = "lost"


# Declaration order is pipeline order.
# These stages change money or end a lead, so a person sets them.
HUMAN_ONLY_STAGES = frozenset({LeadStage.RESERVED, LeadStage.WON, LeadStage.LOST})
UPDATABLE_FIELDS = frozenset(
    {
        "budget_egp",
        "preferred_project",
        "preferred_bedrooms",
        "preferred_delivery",
        "decision_maker",
    }
)


@dataclass(frozen=True)
class Lead:
    id: str
    display_name: str
    stage: LeadStage
    assigned_rep: str
    fields: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Interaction:
    id: str
    lead_id: str
    kind: InteractionKind
    occurred_at: datetime
    text: str = ""
    recording_path: str | None = None
    recording_consent: bool = False
    answered: bool | None = None


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class SourcedItem:
    text: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class OpenConcern:
    concern: str
    source_ids: tuple[str, ...]
    suggested_answer: str | None = None


@dataclass(frozen=True)
class TimelineEntry:
    interaction_id: str
    occurred_at: datetime
    kind: InteractionKind
    summary: str | None = None


@dataclass(frozen=True)
class Preferences:
    max_budget_egp: Decimal | None = None
    bedrooms: int | None = None
    project: str | None = None


@dataclass(frozen=True)
class MatchedUnit:
    code: str
    project: str
    price_egp: Decimal
    bedrooms: int


@dataclass(frozen=True)
class ContextCard:
    lead_id: str
    display_name: str
    stage: LeadStage
    generated_at: datetime
    latest_interaction_at: datetime | None
    timeline: list[TimelineEntry] = field(default_factory=list)
    stated_needs: list[SourcedItem] = field(default_factory=list)
    closing_drivers: list[SourcedItem] = field(default_factory=list)
    open_concerns: list[OpenConcern] = field(default_factory=list)
    promises: list[SourcedItem] = field(default_factory=list)
    talking_points: list[SourcedItem] = field(default_factory=list)
    preferences: Preferences = field(default_factory=Preferences)
    matched_units: list[MatchedUnit] = field(default_factory=list)
    # Older interactions left out to fit the prompt budget.
    omitted_interactions: int = 0
    # Claims removed because their sources did not support them.
    dropped_claims: int = 0
    # Filled by the best-time model (build step 7).
    best_time_to_call: str | None = None


@dataclass(frozen=True)
class HandoverSummary:
    lead_id: str
    from_rep: str
    to_rep: str
    summary: str
    created_at: datetime


@dataclass(frozen=True)
class PostCallResult:
    interaction_id: str
    summary: str | None
    applied_fields: dict[str, str] = field(default_factory=dict)
    rejected_fields: list[str] = field(default_factory=list)
    new_stage: LeadStage | None = None
    follow_up_at: datetime | None = None
