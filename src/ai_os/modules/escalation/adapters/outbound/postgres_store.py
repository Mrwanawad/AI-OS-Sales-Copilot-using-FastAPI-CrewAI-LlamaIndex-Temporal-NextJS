from typing import TypeVar

from pydantic import TypeAdapter
from sqlalchemy import Column, DateTime, Index, MetaData, String, Table, select, update
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_os.modules.escalation.domain.models import (
    CandidateStatus,
    KnowledgeCandidate,
    Ticket,
    TicketStatus,
)

metadata = MetaData()


def _table(name: str) -> Table:
    return Table(
        name,
        metadata,
        Column("id", String, primary_key=True),
        Column("status", String, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("payload", JSONB, nullable=False),
        Index(f"{name}_status_time", "status", "created_at"),
    )


tickets = _table("escalation_tickets")
candidates = _table("knowledge_candidates")

_TICKET = TypeAdapter(Ticket)
_CANDIDATE = TypeAdapter(KnowledgeCandidate)
ItemT = TypeVar("ItemT")


class PostgresEscalationStore:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def add_ticket(self, ticket: Ticket) -> None:
        await self._add(tickets, _TICKET, ticket)

    async def get_ticket(self, ticket_id: str) -> Ticket | None:
        return await self._get(tickets, _TICKET, ticket_id)

    async def list_tickets(self, status: TicketStatus, limit: int) -> list[Ticket]:
        return await self._list(tickets, _TICKET, status.value, limit)

    async def list_tickets_asked_by(self, asked_by: str, limit: int) -> list[Ticket]:
        query = (
            select(tickets.c.payload)
            .where(tickets.c.payload["asked_by"].astext == asked_by)
            .order_by(tickets.c.created_at.desc())
            .limit(limit)
        )
        async with self._engine.connect() as connection:
            payloads = (await connection.execute(query)).scalars().all()
        return [_TICKET.validate_python(payload) for payload in payloads]

    async def update_ticket(self, ticket: Ticket, expected: TicketStatus) -> bool:
        return await self._update(tickets, _TICKET, ticket, expected.value)

    async def add_candidate(self, candidate: KnowledgeCandidate) -> None:
        await self._add(candidates, _CANDIDATE, candidate)

    async def get_candidate(self, candidate_id: str) -> KnowledgeCandidate | None:
        return await self._get(candidates, _CANDIDATE, candidate_id)

    async def list_candidates(
        self, status: CandidateStatus, limit: int
    ) -> list[KnowledgeCandidate]:
        return await self._list(candidates, _CANDIDATE, status.value, limit)

    async def update_candidate(
        self, candidate: KnowledgeCandidate, expected: CandidateStatus
    ) -> bool:
        return await self._update(candidates, _CANDIDATE, candidate, expected.value)

    async def _add(self, table: Table, adapter: TypeAdapter, item: object) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(table).values(
                    id=item.id,
                    status=item.status.value,
                    created_at=item.created_at,
                    payload=adapter.dump_python(item, mode="json"),
                )
            )

    async def _get(self, table: Table, adapter: TypeAdapter[ItemT], item_id: str) -> ItemT | None:
        query = select(table.c.payload).where(table.c.id == item_id)
        async with self._engine.connect() as connection:
            payload = (await connection.execute(query)).scalar_one_or_none()
        return adapter.validate_python(payload) if payload is not None else None

    async def _list(
        self, table: Table, adapter: TypeAdapter[ItemT], status: str, limit: int
    ) -> list[ItemT]:
        query = (
            select(table.c.payload)
            .where(table.c.status == status)
            .order_by(table.c.created_at)
            .limit(limit)
        )
        async with self._engine.connect() as connection:
            payloads = (await connection.execute(query)).scalars().all()
        return [adapter.validate_python(payload) for payload in payloads]

    async def _update(
        self, table: Table, adapter: TypeAdapter, item: object, expected: str
    ) -> bool:
        # The status condition makes two people acting at once safe: only one update lands.
        statement = (
            update(table)
            .where(table.c.id == item.id, table.c.status == expected)
            .values(status=item.status.value, payload=adapter.dump_python(item, mode="json"))
        )
        async with self._engine.begin() as connection:
            result = await connection.execute(statement)
        return result.rowcount == 1
