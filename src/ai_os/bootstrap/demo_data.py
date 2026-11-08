"""Loads the invented projects, units and clients in demo_data/. Nothing here is real."""

import csv
import json
import tempfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd

from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.modules.crm_copilot.domain.models import (
    Interaction,
    InteractionKind,
    Lead,
    LeadStage,
)
from ai_os.modules.identity.application.service import Identity
from ai_os.modules.identity.domain.errors import UserAlreadyExists
from ai_os.modules.inventory.domain.models import Unit, UnitStatus
from ai_os.modules.inventory.ports.store import UnitStore
from ai_os.modules.knowledge.application.ingest import IngestDocument
from ai_os.modules.knowledge.domain.models import DocumentKind, SourceDocument

DEMO_DATA = Path(__file__).parents[3] / "demo_data"
_PRICE_SHEETS = {"Demo Gardens": "demo-gardens-prices", "Demo Bay": "demo-bay-prices"}
# The reps match the assigned_rep values in leads.json.
DEMO_USERS = (
    ("rep-a", "Rep A", "rep"),
    ("rep-b", "Rep B", "rep"),
    ("lead-a", "Team Lead A", "team_lead"),
    ("manager-a", "Manager A", "manager"),
)


async def seed_users(identity: Identity, password: str) -> int:
    """Creates the demo logins with one shared password. Returns how many were new."""
    created = 0
    for subject, name, role in DEMO_USERS:
        try:
            await identity.create_user(subject, name, role, password)
            created += 1
        except UserAlreadyExists:
            continue
    return created


def load_units(directory: Path = DEMO_DATA) -> list[Unit]:
    with (directory / "units.csv").open(encoding="utf-8") as file:
        return [
            Unit(
                code=row["code"],
                project=row["project"],
                unit_type=row["unit_type"],
                bedrooms=int(row["bedrooms"]),
                area_sqm=Decimal(row["area_sqm"]),
                price_egp=Decimal(row["price_egp"]),
                status=UnitStatus(row["status"]),
                delivery_date=date.fromisoformat(row["delivery_date"]),
            )
            for row in csv.DictReader(file)
        ]


async def ingest_documents(ingest: IngestDocument, directory: Path = DEMO_DATA) -> int:
    """Runs the demo documents through the real ingestion pipeline. Returns the chunk count."""
    chunks = 0
    for entry in json.loads((directory / "documents.json").read_text(encoding="utf-8")):
        document = SourceDocument(entry["id"], entry["project"], DocumentKind(entry["kind"]), 1)
        chunks += (await ingest(directory / entry["file"], document)).chunks
    units = load_units(directory)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        for project, document_id in _PRICE_SHEETS.items():
            rows = [
                {
                    "Unit Code": unit.code,
                    "Price": float(unit.price_egp),
                    "Bedrooms": unit.bedrooms,
                    "Area": float(unit.area_sqm),
                    "Delivery": unit.delivery_date.isoformat(),
                }
                for unit in units
                if unit.project == project
            ]
            path = Path(temporary) / f"{document_id}.xlsx"
            pd.DataFrame(rows).to_excel(path, index=False)
            document = SourceDocument(document_id, project, DocumentKind.PRICE_SHEET, 1)
            chunks += (await ingest(path, document)).chunks
    return chunks


async def seed_crm(crm: PostgresCrmStore, directory: Path = DEMO_DATA) -> int:
    leads = json.loads((directory / "leads.json").read_text(encoding="utf-8"))
    for lead in leads:
        await crm.add_lead(
            Lead(lead["id"], lead["display_name"], LeadStage(lead["stage"]), lead["assigned_rep"])
        )
        for item in lead["interactions"]:
            await crm.add_interaction(
                Interaction(
                    item["id"],
                    lead["id"],
                    InteractionKind(item["kind"]),
                    datetime.fromisoformat(item["occurred_at"]),
                    item["text"],
                    answered=item.get("answered"),
                )
            )
    return len(leads)


async def seed_units(store: UnitStore, directory: Path = DEMO_DATA) -> int:
    units = load_units(directory)
    await store.upsert(units)
    return len(units)
