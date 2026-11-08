from pathlib import Path

from ai_os.modules.knowledge.domain.chunking import chunk_sections
from ai_os.modules.knowledge.domain.errors import EmptyDocument, StaleDocumentVersion
from ai_os.modules.knowledge.domain.models import IngestResult, SourceDocument
from ai_os.modules.knowledge.ports.parsing import DocumentParser
from ai_os.modules.knowledge.ports.retrieval import VectorStore


class IngestDocument:
    def __init__(self, parser: DocumentParser, store: VectorStore) -> None:
        self._parser = parser
        self._store = store

    async def __call__(self, path: Path, document: SourceDocument) -> IngestResult:
        active = await self._store.active_version(document.id)
        if active is not None and document.version < active:
            raise StaleDocumentVersion(document.id, document.version, active)

        chunks = chunk_sections(document, await self._parser.parse(path, document))
        if not chunks:
            raise EmptyDocument(document.id)

        # New version goes in before the old one leaves, so reps never see a gap.
        await self._store.upsert(chunks)
        await self._store.delete_older_versions(document.id, document.version)
        return IngestResult(document.id, document.version, len(chunks))
