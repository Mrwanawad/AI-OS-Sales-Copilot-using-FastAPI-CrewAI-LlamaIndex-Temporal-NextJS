from collections.abc import Sequence
from typing import Protocol

from ai_os.modules.best_time.domain.models import CallRecord


class AnswerModel(Protocol):
    def predict(self, rows: Sequence[Sequence[float]]) -> Sequence[float]: ...


class CallHistory(Protocol):
    async def calls_for(self, lead_id: str) -> list[CallRecord]: ...
