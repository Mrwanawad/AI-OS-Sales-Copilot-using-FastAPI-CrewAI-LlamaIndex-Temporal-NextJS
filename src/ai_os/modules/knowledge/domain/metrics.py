import math
from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    # One document can return many chunks, and only its first appearance earns credit.
    gains: list[float] = []
    seen: set[str] = set()
    for item in ranked[:k]:
        gains.append(1.0 if item in relevant and item not in seen else 0.0)
        seen.add(item)
    dcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
    ideal = sum(1 / math.log2(position + 2) for position in range(min(len(set(relevant)), k)))
    return dcg / ideal if ideal else 0.0
