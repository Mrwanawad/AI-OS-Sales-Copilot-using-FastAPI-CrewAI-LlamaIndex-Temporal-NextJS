from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CallRecord:
    lead_id: str
    at: datetime
    answered: bool


@dataclass(frozen=True)
class SlotScore:
    at: datetime
    probability: float


@dataclass(frozen=True)
class BestTimeRecommendation:
    lead_id: str
    slots: list[SlotScore]
    calls_seen: int
    # Too few past calls with this client: the estimate mostly reflects clients in general.
    low_data: bool
