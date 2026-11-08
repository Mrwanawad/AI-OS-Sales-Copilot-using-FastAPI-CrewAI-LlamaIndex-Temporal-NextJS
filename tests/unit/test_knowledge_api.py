from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.knowledge.adapters.inbound.api import create_router
from ai_os.modules.knowledge.domain.models import (
    Chunk,
    Citation,
    DocumentKind,
    ScoredChunk,
    SearchFilters,
)

from .auth_helpers import AUTH, bearer


class FakeRetrieve:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, SearchFilters | None]] = []

    async def __call__(
        self, query: str, top_k: int = 5, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        self.calls.append((query, top_k, filters))
        citation = Citation("prices", 2, 7)
        chunk = Chunk("1", "Unit A-101", citation, "P", DocumentKind.PRICE_SHEET, ("A-101",))
        return [ScoredChunk(chunk, 0.9)]


def client_for(retrieve: FakeRetrieve) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(lambda: retrieve, AUTH))
    return TestClient(app, headers=bearer())


def test_search_returns_citations_and_passes_filters() -> None:
    retrieve = FakeRetrieve()

    response = client_for(retrieve).post(
        "/knowledge/search",
        json={"query": "A-101", "top_k": 3, "project": "P", "kinds": ["price_sheet"]},
    )

    assert response.status_code == 200
    assert response.json()[0]["citation"] == {
        "document_id": "prices",
        "document_version": 2,
        "page": 7,
    }
    assert response.json()[0]["unit_codes"] == ["A-101"]
    assert retrieve.calls == [
        ("A-101", 3, SearchFilters(project="P", kinds=(DocumentKind.PRICE_SHEET,)))
    ]


def test_search_needs_a_token() -> None:
    response = client_for(FakeRetrieve()).post(
        "/knowledge/search", json={"query": "x"}, headers={"Authorization": ""}
    )

    assert response.status_code == 401


def test_rejects_empty_query_and_oversized_top_k() -> None:
    client = client_for(FakeRetrieve())

    assert client.post("/knowledge/search", json={"query": ""}).status_code == 422
    assert client.post("/knowledge/search", json={"query": "x", "top_k": 500}).status_code == 422
