from pathlib import Path

from ai_os.modules.knowledge.domain.models import DocumentKind, ParsedSection, SourceDocument
from ai_os.modules.knowledge.ports.parsing import DocumentParser

_SPREADSHEETS = {".xlsx", ".xls"}


class RoutingParser:
    def __init__(self, price_sheets: DocumentParser, documents: DocumentParser) -> None:
        self._price_sheets = price_sheets
        self._documents = documents

    async def parse(self, path: Path, document: SourceDocument) -> list[ParsedSection]:
        is_price_sheet = (
            document.kind is DocumentKind.PRICE_SHEET and path.suffix.lower() in _SPREADSHEETS
        )
        parser = self._price_sheets if is_price_sheet else self._documents
        return await parser.parse(path, document)
