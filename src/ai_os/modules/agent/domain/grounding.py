from collections.abc import Sequence

from ai_os.modules.agent.domain.models import Evidence
from ai_os.shared.numbers import numbers_in, percentages_in

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

    sources = [question] + [by_id[c].item.text for c in cited_ids if c in by_id]
    allowed = set().union(*(numbers_in(text) for text in sources))
    invented = sorted(numbers_in(answer) - allowed)
    if invented:
        listed = ", ".join(str(number) for number in invented)
        problems.append(f"numbers not found in the cited evidence: {listed}")
    # "8%" is not supported by "8 years": a percentage must appear as a percentage.
    allowed_percentages = set().union(*(percentages_in(text) for text in sources))
    wrong_unit = sorted(percentages_in(answer) - allowed_percentages - set(invented))
    if wrong_unit:
        listed = ", ".join(f"{number}%" for number in wrong_unit)
        problems.append(f"percentages not found as percentages in the cited evidence: {listed}")
    return problems
