from collections.abc import Sequence
from dataclasses import dataclass

from ai_os.modules.coaching.domain.models import CriterionScore, Scorecard, Speaker, Turn

MAX_SCORE = 5
# Below this share of the rubric, an overall number would mislead.
_MIN_ASSESSED_WEIGHT = 0.5

RUBRIC = {
    "discovery": ("Asks questions that uncover the buyer's needs and budget", 0.25),
    "objection_handling": ("Acknowledges objections and addresses them directly", 0.30),
    "value_communication": ("Links the offer to what this buyer said they care about", 0.20),
    "next_step": ("Secures a concrete next step such as a visit or a call", 0.15),
    "rapport": ("Listens, stays respectful, and builds trust", 0.10),
}


@dataclass(frozen=True)
class ProposedScore:
    criterion: str
    score: int
    quote: str = ""
    feedback: str = ""


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def build_scorecard(proposed: Sequence[ProposedScore], turns: Sequence[Turn]) -> Scorecard:
    """A score above zero must quote the rep's own words, or it cannot be trusted."""
    rep_words = "\n".join(_normalized(t.text) for t in turns if t.speaker is Speaker.REP)
    by_criterion = {item.criterion: item for item in proposed}
    criteria = [_verified(name, by_criterion.get(name), rep_words) for name in RUBRIC]
    return Scorecard(criteria, _overall(criteria))


def _verified(name: str, item: ProposedScore | None, rep_words: str) -> CriterionScore:
    if item is None or not 0 <= item.score <= MAX_SCORE:
        return CriterionScore(name, None)
    # Zero means the behaviour never happened, so there is nothing to quote.
    quote = _normalized(item.quote)
    if item.score > 0 and (not quote or quote not in rep_words):
        return CriterionScore(name, None, item.feedback)
    return CriterionScore(name, item.score, item.feedback, item.quote)


def _overall(criteria: Sequence[CriterionScore]) -> int | None:
    assessed = [c for c in criteria if c.score is not None]
    weight = sum(RUBRIC[c.criterion][1] for c in assessed)
    if weight < _MIN_ASSESSED_WEIGHT:
        return None
    points = sum(RUBRIC[c.criterion][1] * c.score / MAX_SCORE for c in assessed)
    return round(100 * points / weight)
