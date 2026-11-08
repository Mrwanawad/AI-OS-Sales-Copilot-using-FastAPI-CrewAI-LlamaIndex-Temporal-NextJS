from ai_os.modules.knowledge.domain.chunking import chunk_sections
from ai_os.modules.knowledge.domain.models import DocumentKind, ParsedSection, SourceDocument
from ai_os.modules.knowledge.ports.retrieval import VectorStore

_VERSION = 1


class ApprovedAnswers:
    """Answers people reviewed become searchable knowledge, and can be taken back out."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    async def publish(self, document_id: str, project: str, question: str, answer: str) -> None:
        document = SourceDocument(document_id, project, DocumentKind.FAQ, _VERSION)
        section = ParsedSection(f"Question: {question}\nAnswer: {answer}", atomic=True)
        await self._store.upsert(chunk_sections(document, [section]))

    async def remove(self, document_id: str) -> None:
        await self._store.delete_older_versions(document_id, _VERSION + 1)
