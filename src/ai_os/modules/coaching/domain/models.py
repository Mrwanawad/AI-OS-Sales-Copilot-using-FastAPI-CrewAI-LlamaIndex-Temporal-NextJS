from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Speaker(StrEnum):
    REP = "rep"
    BUYER = "buyer"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    FINISHED = "finished"


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    description: str
    objections: tuple[str, ...]
    opening_line: str


@dataclass(frozen=True)
class Turn:
    speaker: Speaker
    text: str


@dataclass(frozen=True)
class CriterionScore:
    criterion: str
    # None means the score could not be verified, so it does not count.
    score: int | None
    feedback: str = ""
    quote: str = ""


@dataclass(frozen=True)
class Scorecard:
    criteria: list[CriterionScore]
    # 0 to 100, or None when too little of the rubric could be verified.
    overall: int | None


@dataclass(frozen=True)
class RolePlaySession:
    id: str
    rep_id: str
    persona_id: str
    started_at: datetime
    status: SessionStatus = SessionStatus.ACTIVE
    turns: list[Turn] = field(default_factory=list)
    scorecard: Scorecard | None = None
