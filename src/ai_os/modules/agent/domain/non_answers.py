"""Replies that only say the evidence is silent. They must reach a person, not the rep.

The composer also reports this itself (answered_from_evidence), but models don't always set
it: the Gemini eval run marked "the documents do not mention a helipad" as answered. This
check is the backstop, tuned on evals/non_answer_cases.jsonl.
"""

import re

_PATTERNS = [
    # "The documents do not mention / contain / say ..."
    r"\b(?:does|do|did)\s*(?:not|n't)\s+(?:mention|contain|include|specify|say|state|cover"
    r"|provide|address|indicate)\b",
    r"\b(?:is|are|was|were)\s*(?:not|n't)\s+(?:mentioned|specified|covered|stated|included"
    r"|addressed|provided)\b",
    r"\bno\s+(?:\w+\s+){0,4}?(?:information|details|data|mention)\b",
    r"\bnot\s+(?:enough|any)\s+(?:information|details|data)\b",
    r"\b(?:could|can)\s*(?:not|n't)\s+find\b",
    r"\bcannot\s+be\s+determined\b",
    r"لا\s+(?:تذكر|يذكر|تحتوي|يحتوي|تتضمن|يتضمن)",
    r"لا\s+(?:توجد|يوجد|تتوفر|يتوفر)\s+(?:أي\s+)?معلومات",
    r"لم\s+(?:يتم\s+)?(?:ذكر|يذكر|تذكر)",
]
_NON_ANSWER = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def says_evidence_is_silent(text: str) -> bool:
    return bool(_NON_ANSWER.search(text))
