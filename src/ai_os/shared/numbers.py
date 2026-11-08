import re
from decimal import Decimal, InvalidOperation

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹٫", "01234567890123456789.")
_NUMBER = re.compile(r"\d+(?:[.,٬]\d+)*")
_EVIDENCE_REFERENCE = re.compile(r"\[?\bE\d+\b\]?")
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)


def numbers_in(text: str) -> set[Decimal]:
    cleaned = _LIST_MARKER.sub("", _EVIDENCE_REFERENCE.sub("", text))
    numbers: set[Decimal] = set()
    for match in _NUMBER.findall(cleaned.translate(_ARABIC_DIGITS)):
        try:
            numbers.add(Decimal(match.replace(",", "").replace("٬", "")))
        except InvalidOperation:
            continue
    return numbers
