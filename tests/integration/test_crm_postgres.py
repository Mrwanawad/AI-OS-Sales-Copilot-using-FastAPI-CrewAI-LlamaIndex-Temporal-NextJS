from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal as D
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.modules.crm_copilot.domain.models import (
    ContextCard,
    HandoverSummary,
    Interaction,
    InteractionKind,
    Lead,
    LeadStage,
    MatchedUnit,
    OpenConcern,
    Preferences,
    SourcedItem,
    TimelineEntry,
)

from .conftest import PostgresUrls

WHEN = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
LEAD = Lead("L1", "عميل تجريبي", LeadStage.CONTACTED, "rep-a", {"budget_egp": "1"})


@pytest.fixture
async def store(postgres: PostgresUrls) -> AsyncIterator[PostgresCrmStore]:
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
    crm = PostgresCrmStore(engine)
    await crm.create_schema()
    await crm.add_lead(LEAD)
    yield crm
    async with engine.begin() as connection:
        await connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    await engine.dispose()


def interaction(interaction_id: str, day: int = 1, text_value: str = "Hello") -> Interaction:
    return Interaction(
        interaction_id,
        "L1",
        InteractionKind.CALL,
        WHEN.replace(day=day),
        text_value,
        "a.wav",
        True,
        True,
    )


async def test_lead_round_trips_with_arabic_name(store: PostgresCrmStore) -> None:
    assert await store.get_lead("L1") == LEAD
    assert await store.get_lead("missing") is None


async def test_interactions_round_trip_in_time_order(store: PostgresCrmStore) -> None:
    await store.add_interaction(interaction("new", day=2))
    await store.add_interaction(interaction("old", day=1))

    found = await store.list_interactions("L1")

    assert [item.id for item in found] == ["old", "new"]
    assert await store.get_interaction("old") == interaction("old", day=1)


async def test_answer_outcomes_list_only_calls_with_known_results(
    store: PostgresCrmStore,
) -> None:
    await store.add_interaction(interaction("answered"))
    await store.add_interaction(
        Interaction("unknown", "L1", InteractionKind.CALL, WHEN.replace(day=2))
    )
    await store.add_interaction(
        Interaction("message", "L1", InteractionKind.WHATSAPP, WHEN.replace(day=3), answered=True)
    )

    assert await store.list_answer_outcomes() == [("L1", WHEN, True)]


async def test_set_interaction_text(store: PostgresCrmStore) -> None:
    await store.add_interaction(interaction("c1", text_value=""))

    await store.set_interaction_text("c1", "Transcript")

    assert (await store.get_interaction("c1")).text == "Transcript"


async def test_update_lead_merges_fields_and_keeps_stage_when_none(
    store: PostgresCrmStore,
) -> None:
    await store.update_lead("L1", None, {"preferred_bedrooms": "3"})
    lead = await store.get_lead("L1")
    assert lead.fields == {"budget_egp": "1", "preferred_bedrooms": "3"}
    assert lead.stage is LeadStage.CONTACTED

    await store.update_lead("L1", LeadStage.QUALIFIED, {})
    assert (await store.get_lead("L1")).stage is LeadStage.QUALIFIED


async def test_assign_rep(store: PostgresCrmStore) -> None:
    await store.assign_rep("L1", "rep-b")

    assert (await store.get_lead("L1")).assigned_rep == "rep-b"


async def test_follow_up_retry_does_not_book_twice(store: PostgresCrmStore) -> None:
    await store.schedule_follow_up("L1", WHEN, "Call back")
    await store.schedule_follow_up("L1", WHEN, "Call back")

    assert await store.list_follow_ups("L1") == [(WHEN, "Call back")]


async def test_full_card_round_trips_exact_decimals(store: PostgresCrmStore) -> None:
    card = ContextCard(
        lead_id="L1",
        display_name="عميل تجريبي",
        stage=LeadStage.QUALIFIED,
        generated_at=WHEN,
        latest_interaction_at=WHEN,
        timeline=[TimelineEntry("c1", WHEN, InteractionKind.CALL, "Summary")],
        stated_needs=[SourcedItem("Needs", ("c1",))],
        open_concerns=[OpenConcern("Price", ("c1",), "Answer")],
        preferences=Preferences(D("5000000.50"), 3, "Palm Hills"),
        matched_units=[MatchedUnit("A-101", "Palm Hills", D("5000000.50"), 3)],
        omitted_interactions=2,
        dropped_claims=1,
    )

    await store.save_card(card)
    assert await store.get_card("L1") == card

    await store.save_card(ContextCard("L1", "x", LeadStage.NEW, WHEN, None))
    assert (await store.get_card("L1")).display_name == "x"


async def test_missing_card_is_none_and_handover_saves(store: PostgresCrmStore) -> None:
    assert await store.get_card("L1") is None

    await store.save_handover(HandoverSummary("L1", "a", "b", "Summary", WHEN))
    await store.save_handover(HandoverSummary("L1", "a", "b", "Summary", WHEN))
