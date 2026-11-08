from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class CallingWindow:
    """When reps may call, in business local time. Set from the brokerage's policy."""

    first_hour: int = 9
    last_hour: int = 21
    # Python weekday numbers: Monday is 0, Friday is 4.
    days: frozenset[int] = field(default_factory=lambda: frozenset(range(7)))

    def __post_init__(self) -> None:
        if not 0 <= self.first_hour <= self.last_hour <= 23:
            raise ValueError("Calling hours must satisfy 0 <= first_hour <= last_hour <= 23")
        if not self.days or not self.days <= frozenset(range(7)):
            raise ValueError("Calling days must be weekday numbers 0-6")
