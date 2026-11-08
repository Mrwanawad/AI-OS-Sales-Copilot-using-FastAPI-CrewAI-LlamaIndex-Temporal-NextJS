from pathlib import Path
from typing import Protocol

from ai_os.modules.knowledge.domain.models import ParsedSection, SourceDocument


class DocumentParser(Protocol):
    async def parse(self, path: Path, document: SourceDocument) -> list[ParsedSection]: ...
