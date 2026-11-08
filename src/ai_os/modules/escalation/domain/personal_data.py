import re

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIGIT_SEPARATOR = re.compile(r"(?<=\d)[\s-](?=\d)")
_PATTERNS = {
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone number": re.compile(r"(?<!\d)(?:(?:\+|00)?20)?0?1[0125]\d{8}(?!\d)"),
    "national ID": re.compile(r"(?<!\d)[23]\d{13}(?!\d)"),
}


def find_personal_data(text: str) -> list[str]:
    """Catches the formats a rule can catch. Names still depend on the human reviewer."""
    normalized = _DIGIT_SEPARATOR.sub("", text.translate(_ARABIC_DIGITS))
    return [kind for kind, pattern in _PATTERNS.items() if pattern.search(normalized)]
