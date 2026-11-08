import asyncio
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItem, DoclingDocument, TableItem

from ai_os.modules.knowledge.domain.models import ParsedSection, SourceDocument


class DoclingParser:
    def __init__(self, converter: DocumentConverter | None = None) -> None:
        self._converter = converter

    async def parse(self, path: Path, document: SourceDocument) -> list[ParsedSection]:
        return await asyncio.to_thread(self._parse, path)

    def _parse(self, path: Path) -> list[ParsedSection]:
        # Built on first use because it loads layout models.
        if self._converter is None:
            self._converter = DocumentConverter()
        return sections_from(self._converter.convert(path).document)


def sections_from(document: DoclingDocument) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    paragraphs: list[str] = []
    current_page: int | None = None

    def flush() -> None:
        if paragraphs:
            sections.append(ParsedSection("\n\n".join(paragraphs), page=current_page))
            paragraphs.clear()

    for item, _level in document.iterate_items():
        if not isinstance(item, DocItem):
            continue
        page = item.prov[0].page_no if item.prov else current_page
        if page != current_page:
            flush()
            current_page = page
        if isinstance(item, TableItem):
            flush()
            table = item.export_to_markdown(doc=document)
            if table.strip():
                sections.append(ParsedSection(table, page=page, atomic=True))
        elif text := getattr(item, "text", "").strip():
            paragraphs.append(text)
    flush()
    return sections
