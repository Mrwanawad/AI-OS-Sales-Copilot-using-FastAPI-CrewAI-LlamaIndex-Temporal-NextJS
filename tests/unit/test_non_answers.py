import json
from pathlib import Path

import pytest

from ai_os.modules.agent.domain.non_answers import says_evidence_is_silent

CASES = [
    json.loads(line)
    for line in (Path(__file__).parents[2] / "evals" / "non_answer_cases.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["text"][:50])
def test_replies_that_only_say_the_evidence_is_silent_are_caught(case: dict) -> None:
    assert says_evidence_is_silent(case["text"]) is case["non_answer"]
