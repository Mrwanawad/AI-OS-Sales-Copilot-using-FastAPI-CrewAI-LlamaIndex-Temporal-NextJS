from collections.abc import Sequence
from decimal import Decimal
from typing import Protocol

from ai_os.modules.inventory.domain.models import Unit


class UnitStore(Protocol):
    async def get_by_codes(self, codes: Sequence[str]) -> list[Unit]: ...

    async def find_available(
        self, max_price: Decimal | None, bedrooms: int | None, project: str | None, limit: int
    ) -> list[Unit]: ...

    async def upsert(self, units: Sequence[Unit]) -> None: ...

    async def run_read_only(self, sql: str) -> list[dict[str, object]]: ...


class SqlGuard(Protocol):
    def validate(self, sql: str) -> str:
        """Returns safe SQL to run, or raises UnsafeSql."""
        ...
