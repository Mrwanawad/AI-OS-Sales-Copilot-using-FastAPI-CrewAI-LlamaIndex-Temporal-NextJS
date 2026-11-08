import re
from decimal import Decimal, InvalidOperation

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_EVIDENCE_REFERENCE = re.compile(r"\[?\bE\d+\b\]?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
# A number, then an optional scale word ("5.2M", "3.9 million", "٣٫٩ مليون") or percent sign.
_NUMBER = re.compile(
    r"(?P<value>\d+(?:[.,٬]\d+)*)"
    # Abbreviations count only when attached ("5.2M"), so "70 m2" stays seventy.
    r"(?:\s*(?P<word>billion|million|thousand|مليار|مليون|ألف|الف)(?![a-z])"
    r"|(?P<abbreviation>bn|mn|m|k)(?![\w²]))?"
    r"(?P<percent>\s*(?:%|٪|percent\b|per cent\b|في المائة|في المية|بالمائة))?",
    re.IGNORECASE,
)
_SCALES = {
    "billion": 10**9,
    "bn": 10**9,
    "مليار": 10**9,
    "million": 10**6,
    "mn": 10**6,
    "m": 10**6,
    "مليون": 10**6,
    "thousand": 10**3,
    "k": 10**3,
    "ألف": 10**3,
    "الف": 10**3,
}


def _matches(text: str):
    cleaned = _LIST_MARKER.sub("", _EVIDENCE_REFERENCE.sub("", text))
    for match in _NUMBER.finditer(cleaned.translate(_ARABIC_DIGITS)):
        try:
            value = Decimal(match["value"].replace(",", "").replace("٬", ""))
        except InvalidOperation:
            continue
        scale = match["word"] or match["abbreviation"]
        if scale:
            value *= _SCALES[scale.lower()]
        yield value, bool(match["percent"])


def numbers_in(text: str) -> set[Decimal]:
    """Every number in the text, with "5.2M" and "3.9 million" written out in full."""
    return {_plain(value) for value, _percent in _matches(text)}


def percentages_in(text: str) -> set[Decimal]:
    """Numbers written as percentages: "10%", "10 percent", "١٠٪"."""
    return {_plain(value) for value, percent in _matches(text) if percent}


def _plain(value: Decimal) -> Decimal:
    # Decimal("5.2E+6") and Decimal("5200000") are equal; keep one plain form for messages.
    return value.quantize(Decimal(1)) if value == value.to_integral() else value
