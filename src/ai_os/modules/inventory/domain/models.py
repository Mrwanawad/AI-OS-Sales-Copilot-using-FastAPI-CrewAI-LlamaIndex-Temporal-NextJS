from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum


class UnitStatus(StrEnum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    SOLD = "sold"


@dataclass(frozen=True)
class Unit:
    code: str
    project: str
    unit_type: str
    bedrooms: int
    area_sqm: Decimal
    price_egp: Decimal
    status: UnitStatus
    delivery_date: date | None = None


@dataclass(frozen=True)
class InventoryAnswer:
    question: str
    sql: str
    rows: list[dict[str, object]]
