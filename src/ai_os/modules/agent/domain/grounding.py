from collections.abc import Sequence

from ai_os.modules.agent.domain.models import Evidence
from ai_os.shared.numbers import numbers_in

__all__ = ["find_grounding_problems", "numbers_in"]


def find_grounding_problems(
    answer: str, cited_ids: Sequence[str], evidence: Sequence[Evidence], question: str
) -> list[str]:
    """Numbers are the claims reps act on, so each must trace to cited evidence or the question."""
    by_id = {item.id: item for item in evidence}
    problems: list[str] = []
    if not cited_ids:
        problems.append("the answer cites no evidence")
    unknown = [cited for cited in cited_ids if cited not in by_id]
    if unknown:
        problems.append(f"unknown evidence ids: {', '.join(unknown)}")

    allowed = numbers_in(question)
    for cited in cited_ids:
        if cited in by_id:
            allowed |= numbers_in(by_id[cited].item.text)
    invented = sorted(numbers_in(answer) - allowed)
    if invented:
        listed = ", ".join(str(number) for number in invented)
        problems.append(f"numbers not found in the cited evidence: {listed}")
    return problems
