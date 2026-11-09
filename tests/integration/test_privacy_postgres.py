"""PDPL erasure and retention against a real database."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.bootstrap.escalation import QueueOnlyNotifier
from ai_os.bootstrap.uploads import StoredUploads
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.modules.crm_copilot.application.privacy import (
    PersonalDataLifecycle,
    RetentionPeriods,
)
from ai_os.modules.crm_copilot.domain.errors import LeadNotFound
from ai_os.modules.crm_copilot.domain.models import (
    ContextCard,
    HandoverSummary,
    Interaction,
    InteractionKind,
    Lead,
    LeadStage,
)
from ai_os.modules.escalation.adapters.outbound.postgres_store import PostgresEscalationStore
from ai_os.modules.escalation.application.service import Escalations
from ai_os.modules.escalation.domain.models import TicketStatus
from ai_os.shared.files import PostgresFileStore, StoredFileNotFound

from .conftest import PostgresUrls

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
PERIODS = RetentionPeriods(interaction_content_days=30, dormant_card_days=60, closed_ticket_days=90)


@dataclass
class Setup:
    crm: PostgresCrmStore
    files: PostgresFileStore
    escalations: Escalations
    tickets: PostgresEscalationStore
    lifecycle: PersonalDataLifecycle
    clock: list[datetime]


@pytest.fixture
async def setup(postgres: PostgresUrls) -> AsyncIterator[Setup]:
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
    crm, files, tickets = (
        PostgresCrmStore(engine),
        PostgresFileStore(engine),
        PostgresEscalationStore(engine),
    )
    for store in (crm, files, tickets):
        await store.create_schema()
    clock = [NOW]
    escalations = Escalations(tickets, None, QueueOnlyNotifier(), lambda: clock[0])
    lifecycle = PersonalDataLifecycle(
        crm, StoredUploads(files), [escalations], PERIODS, lambda: clock[0]
    )
    yield Setup(crm, files, escalations, tickets, lifecycle, clock)
    async with engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    await engine.dispose()


async def add_lead(setup: Setup, lead_id: str, days_ago: int) -> str:
    """A lead with one recorded call `days_ago`, a card, a handover and a follow-up."""
    when = NOW - timedelta(days=days_ago)
    recording = (await setup.files.put("call.wav", "audio/wav", b"RIFF", "dialer")).reference
    await setup.crm.add_lead(Lead(lead_id, "عميل", LeadStage.CONTACTED, "rep-a"))
    await setup.crm.add_interaction(
        Interaction(
            f"{lead_id}-c1",
            lead_id,
            InteractionKind.CALL,
            when,
            "Client said budget 5M, phone 01012345678",
            recording,
            True,
            True,
        )
    )
    await setup.crm.save_card(ContextCard(lead_id, "عميل", LeadStage.CONTACTED, when, when))
    await setup.crm.save_handover(HandoverSummary(lead_id, "rep-a", "rep-b", "Summary", when))
    await setup.crm.schedule_follow_up(lead_id, when, "Call back")
    return recording


async def assert_file_gone(setup: Setup, reference: str) -> None:
    with pytest.raises(StoredFileNotFound):
        async with setup.files.local_copy(reference):
            pass


async def test_erasure_removes_the_lead_everywhere_and_leaves_others(setup: Setup) -> None:
    recording = await add_lead(setup, "L1", days_ago=1)
    other = await add_lead(setup, "L2", days_ago=1)
    await setup.escalations.open_ticket("Q about L1", "reason", "PH", "L1", "Card text")
    kept = await setup.escalations.open_ticket("Q about L2", "reason", "PH", "L2", "Card")

    report = await setup.lifecycle.erase_lead("L1", "manager-a")

    assert (report.recordings_deleted, report.linked_records_deleted) == (1, 1)
    assert await setup.crm.get_lead("L1") is None
    assert await setup.crm.list_interactions("L1") == []
    assert await setup.crm.get_card("L1") is None
    assert await setup.crm.list_follow_ups("L1") == []
    await assert_file_gone(setup, recording)
    assert [t.id for t in await setup.escalations.list_tickets(TicketStatus.OPEN)] == [kept.id]

    assert await setup.crm.get_card("L2") is not None
    async with setup.files.local_copy(other) as path:
        assert path.read_bytes() == b"RIFF"


async def test_erasing_an_unknown_lead_raises(setup: Setup) -> None:
    with pytest.raises(LeadNotFound):
        await setup.lifecycle.erase_lead("missing", "manager-a")


async def test_retention_clears_old_content_but_keeps_call_outcomes(setup: Setup) -> None:
    old_recording = await add_lead(setup, "OLD", days_ago=90)
    new_recording = await add_lead(setup, "NEW", days_ago=5)

    report = await setup.lifecycle.apply_retention()

    assert (report.interactions_cleared, report.recordings_deleted) == (1, 1)
    [old_call] = await setup.crm.list_interactions("OLD")
    assert (old_call.text, old_call.recording_path, old_call.answered) == ("", None, True)
    await assert_file_gone(setup, old_recording)
    # Dormant for 90 days, past the 60-day card period.
    assert report.cards_deleted == 1
    assert await setup.crm.get_card("OLD") is None

    [new_call] = await setup.crm.list_interactions("NEW")
    assert new_call.text.startswith("Client said")
    assert await setup.crm.get_card("NEW") is not None
    async with setup.files.local_copy(new_recording):
        pass

    # Nothing left to clear, so a second run changes nothing.
    again = await setup.lifecycle.apply_retention()
    assert (again.interactions_cleared, again.cards_deleted) == (0, 0)


async def test_retention_deletes_only_old_closed_tickets(setup: Setup) -> None:
    setup.clock[0] = NOW - timedelta(days=120)
    old = await setup.escalations.open_ticket("Old", "reason")
    await setup.escalations.answer_ticket(old.id, "Answer", "lead-a")
    still_open = await setup.escalations.open_ticket("Still open", "reason")
    setup.clock[0] = NOW - timedelta(days=10)
    recent = await setup.escalations.open_ticket("Recent", "reason")
    await setup.escalations.dismiss_ticket(recent.id, "lead-a", "Duplicate")
    setup.clock[0] = NOW

    report = await setup.lifecycle.apply_retention()

    assert report.tickets_deleted == 1
    assert await setup.tickets.get_ticket(old.id) is None
    assert await setup.tickets.get_ticket(still_open.id) is not None
    assert await setup.tickets.get_ticket(recent.id) is not None
