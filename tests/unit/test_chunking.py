from ai_os.modules.knowledge.domain.chunking import chunk_sections
from ai_os.modules.knowledge.domain.models import DocumentKind, ParsedSection, SourceDocument

DOCUMENT = SourceDocument("brochure-1", "Palm Hills", DocumentKind.BROCHURE, 2)


def test_short_section_stays_one_chunk_with_citation() -> None:
    chunks = chunk_sections(DOCUMENT, [ParsedSection("Delivery in 2027.", page=4)])

    assert [c.text for c in chunks] == ["Delivery in 2027."]
    assert chunks[0].citation.page == 4
    assert chunks[0].citation.document_version == 2


def test_long_text_splits_without_exceeding_limit_or_losing_words() -> None:
    words = [f"word{i}" for i in range(400)]

    chunks = chunk_sections(DOCUMENT, [ParsedSection(" ".join(words))], max_chars=200, overlap=0)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert " ".join(c.text for c in chunks).split() == words


def test_overlap_repeats_previous_tail_within_a_section_only() -> None:
    long_text = " ".join(f"word{i}" for i in range(100))
    sections = [ParsedSection(long_text, page=1), ParsedSection("Next page.", page=2)]

    without = chunk_sections(DOCUMENT, sections, max_chars=200, overlap=0)
    chunks = chunk_sections(DOCUMENT, sections, max_chars=200, overlap=40)

    tail = chunks[1].text.split("\n")[0]
    assert tail and without[0].text.endswith(tail)
    assert chunks[1].text.endswith(without[1].text)
    assert chunks[-1].text == "Next page."


def test_atomic_price_row_is_never_split_or_overlapped() -> None:
    row = "Unit A-101 | 3 bedrooms | " + "x" * 300
    sections = [
        ParsedSection(row, page=1, unit_codes=("A-101",), atomic=True),
        ParsedSection(row.replace("A-101", "A-102"), page=1, unit_codes=("A-102",), atomic=True),
    ]

    chunks = chunk_sections(DOCUMENT, sections, max_chars=100, overlap=40)

    assert [c.text for c in chunks] == [s.text for s in sections]
    assert [c.unit_codes for c in chunks] == [("A-101",), ("A-102",)]


def test_chunk_ids_are_stable_across_retries_and_change_per_version() -> None:
    section = [ParsedSection("Same text")]
    next_version = SourceDocument("brochure-1", "Palm Hills", DocumentKind.BROCHURE, 3)

    first = chunk_sections(DOCUMENT, section)[0].id

    assert first == chunk_sections(DOCUMENT, section)[0].id
    assert first != chunk_sections(next_version, section)[0].id
