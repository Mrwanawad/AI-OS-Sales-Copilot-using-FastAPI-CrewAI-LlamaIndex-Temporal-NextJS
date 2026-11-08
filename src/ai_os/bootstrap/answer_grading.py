from dataclasses import dataclass, field

from ai_os.modules.agent.domain.models import AgentAnswer
from ai_os.shared.numbers import numbers_in


@dataclass(frozen=True)
class AnswerCase:
    question: str
    status: str
    intent: str | None = None
    project: str | None = None
    must_contain: list[str] = field(default_factory=list)
    must_contain_numbers: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    citation_kind: str | None = None


def grade(case: AnswerCase, answer: AgentAnswer) -> list[str]:
    """Checks a real model's answer against what the gold case demands. Empty means it passed."""
    problems: list[str] = []
    if answer.status.value != case.status:
        problems.append(f"status {answer.status.value}, expected {case.status}")
    if case.intent and answer.intent.value != case.intent:
        problems.append(f"intent {answer.intent.value}, expected {case.intent}")
    text = answer.text.casefold()
    for phrase in case.must_contain:
        if phrase.casefold() not in text:
            problems.append(f"missing {phrase!r}")
    for phrase in case.must_not_contain:
        if phrase.casefold() in text:
            problems.append(f"should not say {phrase!r}")
    found = numbers_in(answer.text)
    for number in case.must_contain_numbers:
        if not numbers_in(number) <= found:
            problems.append(f"missing number {number}")
    if case.citation_kind and case.citation_kind not in {c.kind.value for c in answer.citations}:
        problems.append(f"no {case.citation_kind} citation")
    return problems
