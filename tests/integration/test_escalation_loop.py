"""The memory loop against a real database and a real vector store."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.bootstrap.escalation import KnowledgePublisherAdapter, QueueOnlyNotifier
from ai_os.modules.escalation.adapters.outbound.postgres_store import PostgresEscalationStore
from ai_os.modules.escalation.application.service import Escalations
from ai_os.modules.escalation.domain.errors import InvalidTransition
from ai_os.modules.escalation.domain.models import CandidateStatus, TicketStatus
from ai_os.modules.knowledge.adapters.outbound.llama_index_store import LlamaIndexVectorStore
from ai_os.modules.knowledge.adapters.outbound.stores import qdrant_store
from ai_os.modules.knowledge.application.approved_answers import ApprovedAnswers
from ai_os.modules.knowledge.domain.chunking import chunk_sections
from ai_os.modules.knowledge.domain.models import (
    GENERAL_PROJECT,
    DocumentKind,
    ParsedSection,
    SearchFilters,
    SourceDocument,
)

from .conftest import HashingEmbedding, PostgresUrls


@dataclass
class Loop:
    service: Escalations
    store: PostgresEscalationStore
    knowledge: LlamaIndexVectorStore


@pytest.fixture
async def loop(postgres: PostgresUrls) -> AsyncIterator[Loop]:
    schema = f"s_{uuid4().hex}"
    admin = create_async_engine(postgres.asynchronous, poolclass=NullPool)
    async with admin.begin() as connection:
        await connection.execute(text(f"CREATE SCHEMA {schema}"))
    await admin.dispose()
    engine = create_async_engine(
        postgres.asynchronous,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    store = PostgresEscalationStore(engine)
    await store.create_schema()
    client = AsyncQdrantClient(":memory:")
    knowledge = LlamaIndexVectorStore(qdrant_store(client), HashingEmbedding())
    answers = ApprovedAnswers(knowledge)
    service = Escalations(
        store,
        KnowledgePublisherAdapter(lambda: answers),
        QueueOnlyNotifier(),
        lambda: datetime.now(UTC),
    )
    yield Loop(service, store, knowledge)
    await client.close()
    async with engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    await engine.dispose()


async def share(loop: Loop, project: str = "PH") -> str:
    ticket = await loop.service.open_ticket("Does the clubhouse have a gym?", "no evidence found")
    _ticket, candidate = await loop.service.answer_ticket(
        ticket.id, "Yes, the clubhouse has a gym.", "lead-a", project
    )
    return candidate.id


async def test_ticket_and_candidate_round_trip_through_postgres(loop: Loop) -> None:
    ticket = await loop.service.open_ticket("سؤال عن النادي", "low confidence", "PH", "L1", "Card")

    assert await loop.service.get_ticket(ticket.id) == ticket
    assert await loop.service.list_tickets(TicketStatus.OPEN) == [ticket]
    assert await loop.service.list_tickets(TicketStatus.ANSWERED) == []


async def test_rep_finds_their_answered_ticket(loop: Loop) -> None:
    mine = await loop.service.open_ticket("Q?", "reason", asked_by="rep-a")
    await loop.service.open_ticket("Other?", "reason", asked_by="rep-b")
    await loop.service.answer_ticket(mine.id, "The answer", "lead-a")

    tickets = await loop.service.tickets_asked_by("rep-a")

    assert [ticket.answer for ticket in tickets] == ["The answer"]


async def test_approved_answer_becomes_searchable_with_a_citation(loop: Loop) -> None:
    candidate_id = await share(loop)
    await loop.knowledge.upsert(
        chunk_sections(
            SourceDocument("other", "PH", DocumentKind.FAQ, 1), [ParsedSection("Parking info")]
        )
    )
    assert not [
        hit for hit in await loop.knowledge.search("gym clubhouse", 5) if "gym" in hit.chunk.text
    ]

    await loop.service.approve(candidate_id, "manager-b")

    hits = await loop.knowledge.search("gym clubhouse", 1, SearchFilters(project="PH"))
    assert "Yes, the clubhouse has a gym." in hits[0].chunk.text
    assert hits[0].chunk.citation.document_id == f"approved-answer-{candidate_id}"
    assert hits[0].chunk.kind is DocumentKind.FAQ


async def test_revoked_answer_leaves_search_and_other_knowledge_stays(loop: Loop) -> None:
    candidate_id = await share(loop)
    await loop.service.approve(candidate_id, "manager-b")
    await loop.knowledge.upsert(
        chunk_sections(
            SourceDocument("other", "PH", DocumentKind.FAQ, 1),
            [ParsedSection("The gym policy is separate")],
        )
    )

    await loop.service.revoke(candidate_id, "manager-b", "No longer true")

    texts = [hit.chunk.text for hit in await loop.knowledge.search("gym", 10)]
    assert texts == ["The gym policy is separate"]


async def test_general_answers_reach_every_project(loop: Loop) -> None:
    candidate_id = await share(loop, project=GENERAL_PROJECT)
    await loop.service.approve(candidate_id, "manager-b")

    hits = await loop.knowledge.search("gym", 5, SearchFilters(project="Any Project"))

    assert len(hits) == 1


async def test_two_simultaneous_reviews_only_one_lands(loop: Loop) -> None:
    candidate_id = await share(loop)

    results = await asyncio.gather(
        loop.service.approve(candidate_id, "manager-b"),
        loop.service.reject(candidate_id, "manager-c", "Wrong"),
        return_exceptions=True,
    )

    assert sum(isinstance(result, InvalidTransition) for result in results) == 1
    stored = await loop.store.get_candidate(candidate_id)
    live = (
        await loop.knowledge.search("gym", 5) if stored.status is CandidateStatus.APPROVED else []
    )
    assert bool(live) is (stored.status is CandidateStatus.APPROVED)
