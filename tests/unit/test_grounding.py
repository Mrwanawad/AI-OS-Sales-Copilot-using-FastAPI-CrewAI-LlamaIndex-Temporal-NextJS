from decimal import Decimal as D

from ai_os.modules.agent.domain.grounding import find_grounding_problems, numbers_in
from ai_os.modules.agent.domain.models import Evidence, EvidenceItem, SourceKind

EVIDENCE = [
    Evidence("E1", EvidenceItem("Unit A-101 price 5,000,000 EGP", SourceKind.INVENTORY, "unit")),
    Evidence("E2", EvidenceItem("Delivery in 2027", SourceKind.DOCUMENT, "brochure")),
]


def problems(answer: str, cited: list[str], question: str = "What is the price?") -> list[str]:
    return find_grounding_problems(answer, cited, EVIDENCE, question)


def test_numbers_normalize_commas_decimals_and_arabic_digits() -> None:
    assert numbers_in("5,000,000 and 46875.00 and ٣") == {D("5000000"), D("46875"), D("3")}
    assert numbers_in("٥٬٠٠٠٬٠٠٠") == {D("5000000")}
    assert numbers_in("١٢٫٥") == {D("12.5")}


def test_numbers_ignore_evidence_references_and_list_markers() -> None:
    assert numbers_in("See [E7] and E12") == set()
    assert numbers_in("1. first\n2) second") == set()
    assert numbers_in("Ends with 500.") == {D(500)}


def test_grounded_answer_has_no_problems() -> None:
    assert problems("Unit A-101 costs 5,000,000 EGP [E1].", ["E1"]) == []


def test_same_value_in_other_format_is_grounded() -> None:
    assert problems("It costs ٥٬٠٠٠٬٠٠٠ EGP.", ["E1"]) == []
    assert problems("It costs 5000000.00 EGP.", ["E1"]) == []


def test_invented_price_is_flagged() -> None:
    assert "4900000" in problems("It costs 4,900,000 EGP.", ["E1"])[0]


def test_number_from_uncited_evidence_is_flagged() -> None:
    assert "2027" in problems("Delivery is 2027.", ["E1"])[0]


def test_number_from_question_is_allowed() -> None:
    assert problems("Yes, a 10% down payment works.", ["E1"], "Can I pay 10% down?") == []


def test_missing_and_unknown_citations_are_flagged() -> None:
    assert problems("No numbers here.", []) == ["the answer cites no evidence"]
    assert "E9" in problems("No numbers here.", ["E9"])[0]


def test_shorthand_amounts_match_their_full_form() -> None:
    assert numbers_in("EGP 5.2M, 520K, 3.9 million") == {D("5200000"), D("520000"), D("3900000")}
    assert numbers_in("٣٫٩ مليون جنيه") == {D("3900000")}
    assert numbers_in("150 m² and 12 km") == {D("150"), D("12")}
    assert problems("It costs 5M EGP [E1].", ["E1"]) == []


def test_a_percentage_needs_a_percentage_in_the_evidence() -> None:
    plan = [
        Evidence("P", EvidenceItem("10 percent down over 8 years", SourceKind.DOCUMENT, "plan"))
    ]

    assert find_grounding_problems("10% down over 8 years.", ["P"], plan, "Plan?") == []
    flagged = find_grounding_problems("Maintenance is 8%.", ["P"], plan, "Maintenance?")
    assert "8%" in flagged[0]
    assert find_grounding_problems("Yes, 8% works.", ["P"], plan, "Is 8% ok?") == []
