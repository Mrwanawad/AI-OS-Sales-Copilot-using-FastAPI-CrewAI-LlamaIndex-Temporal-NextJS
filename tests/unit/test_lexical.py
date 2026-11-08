from ai_os.modules.knowledge.adapters.outbound.lexical import encode_sparse, normalize, tokenize


def test_normalizes_arabic_spelling_variants_and_digits() -> None:
    assert normalize("أَحْمَد") == normalize("احمد")
    assert normalize("شقة") == normalize("شقه")
    assert normalize("٣ غرف") == "3 غرف"
    assert normalize("A-101") == normalize("a-101")


def test_unit_code_matches_whole_and_by_part() -> None:
    assert {"a-101", "a", "101"} <= set(tokenize("Unit A-101"))
    assert "١٠١" not in tokenize("وحدة ١٠١") and "101" in tokenize("وحدة ١٠١")


def test_sparse_encoding_is_stable_and_saturates_repeats() -> None:
    (once_ids,), (once_values,) = encode_sparse(["villa"])
    (again_ids,), _ = encode_sparse(["villa"])
    _, (twice_values,) = encode_sparse(["villa villa"])

    assert once_ids == again_ids
    assert once_values[0] == 1.0
    assert 1.0 < twice_values[0] < 2.0
