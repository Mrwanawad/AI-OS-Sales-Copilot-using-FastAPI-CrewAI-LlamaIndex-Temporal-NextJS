from pathlib import Path

from ai_os.modules.knowledge.adapters.outbound.docling_parser import DoclingParser
from ai_os.modules.knowledge.domain.models import DocumentKind, SourceDocument

DOCUMENT = SourceDocument("brochure", "Palm Hills", DocumentKind.BROCHURE, 1)


async def test_parses_text_and_keeps_tables_atomic(tmp_path: Path) -> None:
    path = tmp_path / "brochure.md"
    path.write_text(
        "# Palm Hills\n\nDelivery in 2027.\n\n"
        "| Unit | Price |\n|---|---|\n| A-101 | 5000000 |\n\n"
        "Clubhouse and gym included.\n",
        encoding="utf-8",
    )

    sections = await DoclingParser().parse(path, DOCUMENT)

    tables = [s for s in sections if s.atomic]
    text = "\n".join(s.text for s in sections if not s.atomic)
    assert len(tables) == 1
    assert "A-101" in tables[0].text and "5000000" in tables[0].text
    assert "Delivery in 2027." in text
    assert "Clubhouse and gym included." in text
