from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime

from pydantic import TypeAdapter
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    Table,
    Text,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.ext.asyncio import AsyncEngine

from ai_os.modules.crm_copilot.domain.models import (
    ContextCard,
    HandoverSummary,
    Interaction,
    InteractionKind,
    Lead,
    LeadStage,
)

metadata = MetaData()

leads = Table(
    "leads",
    metadata,
    Column("id", String, primary_key=True),
    Column("display_name", String, nullable=False),
    Column("stage", String, nullable=False),
    Column("assigned_rep", String, nullable=False),
    Column("fields", JSONB, nullable=False, default=dict),
)
interactions = Table(
    "interactions",
    metadata,
    Column("id", String, primary_key=True),
    Column("lead_id", ForeignKey("leads.id"), nullable=False),
    Column("kind", String, nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("text", Text, nullable=False, default=""),
    Column("recording_path", String),
    Column("recording_consent", Boolean, nullable=False, default=False),
    Column("answered", Boolean),
    Index("interactions_lead_time", "lead_id", "occurred_at"),
)
context_cards = Table(
    "context_cards",
    metadata,
    Column("lead_id", ForeignKey("leads.id"), primary_key=True),
    Column("card", JSONB, nullable=False),
)
handovers = Table(
    "handovers",
    metadata,
    Column("id", String, primary_key=True),
    Column("lead_id", ForeignKey("leads.id"), nullable=False),
    Column("payload", JSONB, nullable=False),
)
follow_ups = Table(
    "follow_ups",
    metadata,
    Column("id", String, primary_key=True),
    Column("lead_id", ForeignKey("leads.id"), nullable=False),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("reason", Text, nullable=False),
)

_CARD = TypeAdapter(ContextCard)
_HANDOVER = TypeAdapter(HandoverSummary)


class PostgresCrmStore:
    """The demo's in-house CRM. A Salesforce, HubSpot or Zoho adapter would replace it."""

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_schema(self) -> None:
        async with self._engine.begin() as connection:
            await connection.run_sync(metadata.create_all)

    async def add_lead(self, lead: Lead) -> None:
        values = {**asdict(lead), "stage": lead.stage.value}
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(leads)
                .values(values)
                .on_conflict_do_update(index_elements=["id"], set_=values)
            )

    async def add_interaction(self, interaction: Interaction) -> None:
        values = {**asdict(interaction), "kind": interaction.kind.value}
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(interactions)
                .values(values)
                .on_conflict_do_update(index_elements=["id"], set_=values)
            )

    async def get_lead(self, lead_id: str) -> Lead | None:
        async with self._engine.connect() as connection:
            row = (
                (await connection.execute(select(leads).where(leads.c.id == lead_id)))
                .mappings()
                .first()
            )
        return _to_lead(row) if row else None

    async def list_leads(self, assigned_rep: str | None, limit: int) -> list[Lead]:
        query = select(leads).order_by(leads.c.display_name).limit(limit)
        if assigned_rep is not None:
            query = query.where(leads.c.assigned_rep == assigned_rep)
        async with self._engine.connect() as connection:
            rows = (await connection.execute(query)).mappings().all()
        return [_to_lead(row) for row in rows]

    async def get_interaction(self, interaction_id: str) -> Interaction | None:
        query = select(interactions).where(interactions.c.id == interaction_id)
        async with self._engine.connect() as connection:
            row = (await connection.execute(query)).mappings().first()
        return _to_interaction(row) if row else None

    async def list_interactions(self, lead_id: str) -> list[Interaction]:
        query = (
            select(interactions)
            .where(interactions.c.lead_id == lead_id)
            .order_by(interactions.c.occurred_at)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(query)).mappings().all()
        return [_to_interaction(row) for row in rows]

    async def list_answer_outcomes(self) -> list[tuple[str, datetime, bool]]:
        """Every call with a known outcome, for training the best-time model."""
        query = select(
            interactions.c.lead_id, interactions.c.occurred_at, interactions.c.answered
        ).where(
            interactions.c.kind == InteractionKind.CALL.value, interactions.c.answered.is_not(None)
        )
        async with self._engine.connect() as connection:
            rows = (await connection.execute(query)).all()
        return [(row[0], row[1], row[2]) for row in rows]

    async def set_interaction_text(self, interaction_id: str, text: str) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                update(interactions).where(interactions.c.id == interaction_id).values(text=text)
            )

    async def set_recording(self, interaction_id: str, reference: str) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                update(interactions)
                .where(interactions.c.id == interaction_id)
                .values(recording_path=reference)
            )

    async def update_lead(
        self, lead_id: str, stage: LeadStage | None, fields: Mapping[str, str]
    ) -> None:
        async with self._engine.begin() as connection:
            current = (
                await connection.execute(
                    select(leads.c.fields).where(leads.c.id == lead_id).with_for_update()
                )
            ).scalar_one()
            values: dict[str, object] = {"fields": {**current, **fields}}
            if stage is not None:
                values["stage"] = stage.value
            await connection.execute(update(leads).where(leads.c.id == lead_id).values(values))

    async def assign_rep(self, lead_id: str, rep: str) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(
                update(leads).where(leads.c.id == lead_id).values(assigned_rep=rep)
            )

    async def schedule_follow_up(self, lead_id: str, at: datetime, reason: str) -> None:
        # Keyed on lead and time, so an activity retry does not book a second follow-up.
        follow_up_id = f"{lead_id}:{at.isoformat()}"
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(follow_ups)
                .values(id=follow_up_id, lead_id=lead_id, at=at, reason=reason)
                .on_conflict_do_nothing(index_elements=["id"])
            )

    async def list_follow_ups(self, lead_id: str) -> list[tuple[datetime, str]]:
        query = select(follow_ups).where(follow_ups.c.lead_id == lead_id)
        async with self._engine.connect() as connection:
            rows = (await connection.execute(query)).mappings().all()
        return [(row["at"], row["reason"]) for row in rows]

    async def save_card(self, card: ContextCard) -> None:
        payload = _CARD.dump_python(card, mode="json")
        statement = insert(context_cards).values(lead_id=card.lead_id, card=payload)
        async with self._engine.begin() as connection:
            await connection.execute(
                statement.on_conflict_do_update(index_elements=["lead_id"], set_={"card": payload})
            )

    async def get_card(self, lead_id: str) -> ContextCard | None:
        query = select(context_cards.c.card).where(context_cards.c.lead_id == lead_id)
        async with self._engine.connect() as connection:
            payload = (await connection.execute(query)).scalar_one_or_none()
        return _CARD.validate_python(payload) if payload is not None else None

    async def save_handover(self, handover: HandoverSummary) -> None:
        handover_id = f"{handover.lead_id}:{handover.created_at.isoformat()}"
        async with self._engine.begin() as connection:
            await connection.execute(
                insert(handovers)
                .values(
                    id=handover_id,
                    lead_id=handover.lead_id,
                    payload=_HANDOVER.dump_python(handover, mode="json"),
                )
                .on_conflict_do_nothing(index_elements=["id"])
            )


def _to_interaction(row: Mapping) -> Interaction:
    return Interaction(
        id=row["id"],
        lead_id=row["lead_id"],
        kind=InteractionKind(row["kind"]),
        occurred_at=row["occurred_at"],
        text=row["text"],
        recording_path=row["recording_path"],
        recording_consent=row["recording_consent"],
        answered=row["answered"],
    )


def _to_lead(row) -> Lead:
    return Lead(
        row["id"],
        row["display_name"],
        LeadStage(row["stage"]),
        row["assigned_rep"],
        dict(row["fields"]),
    )
