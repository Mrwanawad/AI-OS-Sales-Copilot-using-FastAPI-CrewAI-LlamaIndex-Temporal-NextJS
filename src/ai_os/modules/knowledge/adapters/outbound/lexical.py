import hashlib
import re
from collections import Counter

from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import (
    normalize_alef_ar,
    normalize_alef_maksura_ar,
    normalize_teh_marbuta_ar,
)

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_TOKEN = re.compile(r"\w+(?:[-/]\w+)*")
_HASH_SPACE = 2**31 - 1
# BM25 term-frequency saturation. Qdrant's IDF modifier supplies the rest of the formula.
_K1 = 1.2


def normalize(text: str) -> str:
    text = dediac_ar(text).replace("ـ", "")
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text.translate(_ARABIC_DIGITS).casefold()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _TOKEN.findall(normalize(text)):
        tokens.append(token)
        # A unit code like A-101 must match both whole and as "101".
        parts = re.split(r"[-/]", token)
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def _token_id(token: str) -> int:
    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
    return int.from_bytes(digest) % _HASH_SPACE


def _encode_one(text: str) -> tuple[list[int], list[float]]:
    weights: dict[int, float] = {}
    for token, count in Counter(tokenize(text)).items():
        weights[_token_id(token)] = count * (_K1 + 1) / (count + _K1)
    return list(weights), list(weights.values())


def encode_sparse(texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
    encoded = [_encode_one(text) for text in texts]
    return [indices for indices, _ in encoded], [values for _, values in encoded]
