import hashlib

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr


class HashingEmbedding(BaseEmbedding):
    """Deterministic bag-of-words vectors, for tests and CI where no model can be downloaded."""

    _dim: int = PrivateAttr(default=256)

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for word in text.casefold().split():
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            vector[int.from_bytes(digest) % self._dim] += 1.0
        return vector

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vector(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vector(query)
