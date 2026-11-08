from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.knowledge.adapters.inbound.api import create_documents_router
from ai_os.modules.knowledge.application.upload import UploadDocument
from ai_os.modules.knowledge.domain.errors import (
    DocumentProjectMismatch,
    IngestionAlreadyRunning,
    UploadRejected,
)
from ai_os.modules.knowledge.domain.models import (
    DocumentKind,
    IngestionState,
    IngestionStatus,
    RegisteredDocument,
    SourceDocument,
)
from ai_os.shared.auth import Role
from ai_os.shared.files import safe_filename

from .auth_helpers import AUTH, bearer

NOW = datetime(2026, 11, 8, tzinfo=UTC)


class FakeFiles:
    def __init__(self) -> None:
        self.saved: list[tuple[str, bytes, str]] = []

    async def put(self, filename: str, content_type: str, data: bytes, uploaded_by: str) -> str:
        self.saved.append((filename, data, uploaded_by))
        return f"stored:{len(self.saved)}"


class FakeRegistry:
    def __init__(self) -> None:
        self.documents: dict[str, RegisteredDocument] = {}

    async def next_version(
        self, document_id, project, kind, at_least, source, uploaded_by, at
    ) -> int:
        current = self.documents.get(document_id)
        version = max(current.latest_version + 1, at_least) if current else at_least
        self.documents[document_id] = RegisteredDocument(
            document_id, project, kind, version, source, uploaded_by, at
        )
        return version

    async def get(self, document_id: str) -> RegisteredDocument | None:
        return self.documents.get(document_id)

    async def list(self, project: str | None, limit: int) -> list[RegisteredDocument]:
        return [d for d in self.documents.values() if project in (None, d.project)][:limit]


class FakeVectors:
    def __init__(self, active: dict[str, int] | None = None) -> None:
        self.active = active or {}

    async def active_version(self, document_id: str) -> int | None:
        return self.active.get(document_id)


class FakeIngestion:
    def __init__(self, busy: bool = False) -> None:
        self.started: list[tuple[str, SourceDocument]] = []
        self.busy = busy

    async def start(self, source: str, document: SourceDocument) -> None:
        if self.busy:
            raise IngestionAlreadyRunning(document.id, document.version)
        self.started.append((source, document))

    async def status(self, document: SourceDocument) -> IngestionStatus:
        return IngestionStatus(IngestionState.COMPLETED, chunks=3)


def build(
    vectors: FakeVectors | None = None, ingestion: FakeIngestion | None = None
) -> tuple[UploadDocument, FakeFiles, FakeIngestion]:
    files, started = FakeFiles(), ingestion or FakeIngestion()
    use_case = UploadDocument(
        files, FakeRegistry(), lambda: vectors or FakeVectors(), started, lambda: NOW
    )
    return use_case, files, started


async def test_upload_stores_the_file_and_starts_ingestion_at_the_next_version() -> None:
    upload, files, ingestion = build(FakeVectors({"bay-brochure": 3}))

    first = await upload(
        "bay-brochure", "Demo Bay", DocumentKind.BROCHURE, "b.pdf", "application/pdf", b"%PDF",
        "lead-a",
    )
    second = await upload(
        "bay-brochure", "Demo Bay", DocumentKind.BROCHURE, "b.pdf", "application/pdf", b"%PDF2",
        "lead-a",
    )

    # Version 3 was loaded by script before the registry existed.
    assert (first.version, second.version) == (4, 5)
    assert [source for source, _ in ingestion.started] == ["stored:1", "stored:2"]
    assert files.saved[0][2] == "lead-a"


@pytest.mark.parametrize(
    ("document_id", "filename", "data"),
    [
        ("Bad Id!", "a.pdf", b"x"),
        ("ok-id", "a.exe", b"x"),
        ("ok-id", "noextension", b"x"),
        ("ok-id", "a.pdf", b""),
    ],
)
async def test_upload_rejects_bad_ids_formats_and_empty_files(document_id, filename, data) -> None:
    upload, files, ingestion = build()

    with pytest.raises(UploadRejected):
        await upload(document_id, "P", DocumentKind.FAQ, filename, "x", data, "lead-a")
    assert files.saved == [] and ingestion.started == []


async def test_upload_refuses_moving_a_document_to_another_project() -> None:
    upload, _, _ = build()
    await upload("faq", "P1", DocumentKind.FAQ, "f.md", "text/markdown", b"# Q", "lead-a")

    with pytest.raises(DocumentProjectMismatch):
        await upload("faq", "P2", DocumentKind.FAQ, "f.md", "text/markdown", b"# Q", "lead-a")


async def test_status_is_none_for_versions_never_uploaded() -> None:
    upload, _, _ = build()
    await upload("faq", "P1", DocumentKind.FAQ, "f.md", "text/markdown", b"# Q", "lead-a")

    assert (await upload.status("faq", 1)).chunks == 3
    assert await upload.status("faq", 2) is None
    assert await upload.status("missing", 1) is None


def test_filenames_lose_any_directory_part() -> None:
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\temp\\prices.xlsx") == "prices.xlsx"
    assert safe_filename("  ") == "upload"


def client_for(upload: UploadDocument) -> TestClient:
    app = FastAPI()
    app.include_router(create_documents_router(lambda: upload, AUTH))
    return TestClient(app)


FORM = {"document_id": "bay-faq", "project": "Demo Bay", "kind": "faq"}


def test_api_upload_is_for_team_leads_and_returns_a_status_url() -> None:
    upload, _, _ = build()
    client = client_for(upload)
    file = {"file": ("faq.md", b"# Q\nA", "text/markdown")}

    refused = client.post("/knowledge/documents", data=FORM, files=file, headers=bearer())
    accepted = client.post(
        "/knowledge/documents",
        data=FORM,
        files=file,
        headers=bearer("lead-a", Role.TEAM_LEAD),
    )

    assert refused.status_code == 403
    assert accepted.status_code == 202
    assert accepted.json()["status_url"] == "/knowledge/documents/bay-faq/versions/1"
    status = client.get(accepted.json()["status_url"], headers=bearer("lead-a", Role.TEAM_LEAD))
    assert status.json()["state"] == "completed"
    listed = client.get("/knowledge/documents", headers=bearer())
    assert [d["document_id"] for d in listed.json()] == ["bay-faq"]


def test_api_maps_rejections_and_conflicts() -> None:
    upload, _, _ = build(ingestion=FakeIngestion(busy=True))
    client = client_for(upload)
    lead = bearer("lead-a", Role.TEAM_LEAD)

    bad = client.post(
        "/knowledge/documents",
        data=FORM,
        files={"file": ("tool.exe", b"MZ", "application/octet-stream")},
        headers=lead,
    )
    busy = client.post(
        "/knowledge/documents",
        data=FORM,
        files={"file": ("faq.md", b"# Q", "text/markdown")},
        headers=lead,
    )

    assert bad.status_code == 422
    assert busy.status_code == 409
