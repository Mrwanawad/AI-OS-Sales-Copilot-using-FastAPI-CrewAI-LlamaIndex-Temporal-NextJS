import pytest

from ai_os.modules.knowledge.domain.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_relevant_documents_found_within_k() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, 3) == 1
    assert recall_at_k(["a", "a"], {"a", "b"}, 2) == 0.5
    assert recall_at_k(["a"], set(), 5) == 0


def test_reciprocal_rank_uses_first_relevant_position() -> None:
    assert reciprocal_rank(["x", "a", "b"], {"a", "b"}) == 0.5
    assert reciprocal_rank(["x"], {"a"}) == 0


def test_ndcg_is_one_for_perfect_order_and_lower_when_later() -> None:
    assert ndcg_at_k(["a", "x"], {"a"}, 10) == 1
    assert ndcg_at_k(["x", "a"], {"a"}, 10) == pytest.approx(0.6309, abs=1e-3)
    assert ndcg_at_k(["x"], {"a"}, 10) == 0


def test_ndcg_gives_no_extra_credit_for_repeated_chunks_of_one_document() -> None:
    assert ndcg_at_k(["a", "a", "a"], {"a"}, 10) == 1
