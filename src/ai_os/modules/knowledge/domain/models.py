from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

# Knowledge under this project applies to every project.
GENERAL_PROJECT = "general"


class DocumentKind(StrEnum):
    BROCHURE = "brochure"
    PRICE_SHEET = "price_sheet"
    PAYMENT_PLAN = "payment_plan"
    MASTER_PLAN = "master_plan"
    FAQ = "faq"
    POLICY = "policy"
    PLAYBOOK = "playbook"


@dataclass(frozen=True)
class SourceDocument:
    id: str
    project: str
    kind: DocumentKind
    version: int


@dataclass(frozen=True)
class ParsedSection:
    text: str
    page: int | None = None
    unit_codes: tuple[str, ...] = ()
    atomic: bool = False


@dataclass(frozen=True)
class Citation:
    document_id: str
    document_version: int
    page: int | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    citation: Citation
    project: str
    kind: DocumentKind
    unit_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class SearchFilters:
    project: str | None = None
    kinds: tuple[DocumentKind, ...] = ()


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    version: int
    chunks: int


class IngestionState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class IngestionStatus:
    state: IngestionState
    chunks: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class RegisteredDocument:
    id: str
    project: str
    kind: DocumentKind
    latest_version: int
    source: str
    uploaded_by: str
    uploaded_at: datetime
