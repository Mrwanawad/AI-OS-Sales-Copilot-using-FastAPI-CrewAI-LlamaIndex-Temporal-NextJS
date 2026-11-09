"""Questions a person must decide, however well the documents seem to cover them.

Policy text answers "what is the rule"; it can't decide whether to match a competitor, grant an
exception or promise something off-contract. Models tend to answer these from policy anyway:
the Gemini eval run answered "a competitor offered 20% off, what should I do?" from the
discount policy. This check sends them to a person before the model is asked. Tuned on
evals/judgment_cases.jsonl, which also holds the policy questions it must leave alone.
"""

import re

_MOVE = r"(?:rise|go\s+up|increase|drop|fall|go\s+down|double)\b"
_PATTERNS = {
    "a competitor's offer": [
        r"\bcompetit(?:or|ors|ion|ing)\b",
        r"\b(?:another|other|different)\s+(?:developer|broker|brokerage|agency|company)\b"
        r"\s+(?:\w+\s+){0,3}?(?:offer|give|quote)",
        r"\b(?:match|beat|counter)\s+(?:\w+\s+){0,3}?(?:offer|price|discount|deal)\b",
        r"\b(?:can|could|should)\s+(?:i|we)\s+(?:match|beat)\b",
        r"منافس",
        r"(?:شركة|مطور|بروكر)\s+(?:تانية|تاني|ثانية|آخر|اخرى|أخرى)",
    ],
    "a discount or exception to approve": [
        r"\b(?:can|could|may|should)\s+(?:i|we)\s+(?:give|offer|grant|approve|allow)\s+"
        r"(?:\S+\s+){0,4}?(?:discount|exception|waiver|extension|free)\b",
        r"\b(?:can|could|may|should)\s+(?:i|we)\s+(?:waive|skip|extend|override)\b",
        r"\bspecial\s+approval\b",
        r"\b(?:need|needs|want)\s+(?:\w+\s+){0,3}?to\s+approve\b",
        r"(?:ممكن|ينفع|أقدر|اقدر|نقدر)\s+(?:\S+\s+){0,3}?(?:خصم|استثناء)",
    ],
    "a promise beyond the contract": [
        r"\b(?:can|could|may|should)\s+(?:i|we)\s+(?:promise|guarantee|commit)\b",
        r"\b(?:promise|guarantee)\s+(?:the\s+)?(?:client|buyer|customer)\b",
        r"(?:أوعد|اوعد|أضمن|اضمن)",
    ],
    "a forecast": [
        r"\bwill\s+(?:\w+\s+){0,4}?(?:prices?|values?)\s+(?:\w+\s+){0,4}?" + _MOVE,
        r"\b(?:prices?|values?)\s+(?:\w+\s+){0,3}?(?:will|going\s+to)\s+(?:\w+\s+){0,2}?" + _MOVE,
        r"\b(?:prices?|values?)\b(?:\s+\S+){0,8}?\s+(?:in|by)\s+20[3-9]\d\b",
        r"(?:هتزيد|هتقل|هيزيد|هيقل)\s+(?:\S+\s+){0,2}?(?:الأسعار|الاسعار|السعر)",
        r"(?:الأسعار|الاسعار|السعر)\s+(?:\S+\s+){0,2}?(?:هتزيد|هتقل|هيزيد|هيقل|هتغلى|هيغلى)",
    ],
}
_COMPILED = {
    reason: re.compile("|".join(patterns), re.IGNORECASE) for reason, patterns in _PATTERNS.items()
}


def needs_judgment(question: str) -> str | None:
    """The kind of decision the question asks a person to make, or None."""
    for reason, pattern in _COMPILED.items():
        if pattern.search(question):
            return reason
    return None
