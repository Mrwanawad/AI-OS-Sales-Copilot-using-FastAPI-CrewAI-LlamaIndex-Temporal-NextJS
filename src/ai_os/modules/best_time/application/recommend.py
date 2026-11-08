import inspect
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ai_os.modules.best_time.domain.features import features
from ai_os.modules.best_time.domain.models import (
    BestTimeRecommendation,
    CallRecord,
    SlotScore,
)
from ai_os.modules.best_time.ports.model import AnswerModel, CallHistory

# Placeholder calling window. Replace with the brokerage's real policy.
CALL_HOURS = range(9, 22)
_DAYS_AHEAD = 7
_MIN_CALLS = 3


def candidate_slots(now: datetime, tz: ZoneInfo) -> list[datetime]:
    """Every allowed hour in the next week, in local time, skipping the current hour."""
    local = now.astimezone(tz).replace(minute=0, second=0, microsecond=0)
    slots: list[datetime] = []
    for offset in range(_DAYS_AHEAD + 1):
        day = local + timedelta(days=offset)
        for hour in CALL_HOURS:
            slot = day.replace(hour=hour)
            if now < slot <= now + timedelta(days=_DAYS_AHEAD):
                slots.append(slot)
    return slots


def rank_slots(
    model: AnswerModel, prior: Sequence[CallRecord], now: datetime, tz: ZoneInfo, top: int
) -> list[SlotScore]:
    slots = candidate_slots(now, tz)
    scores = model.predict([features(prior, slot, tz) for slot in slots])
    ranked = sorted(zip(slots, scores, strict=True), key=lambda pair: pair[1], reverse=True)
    return [SlotScore(slot, float(score)) for slot, score in ranked[:top]]


class RecommendBestTime:
    def __init__(
        self,
        # May be async, so a model stored in the database can be reloaded when it changes.
        get_model: Callable[[], AnswerModel | None | Awaitable[AnswerModel | None]],
        history: CallHistory,
        tz: ZoneInfo,
        now: Callable[[], datetime],
        top: int = 3,
    ) -> None:
        self._get_model = get_model
        self._history = history
        self._tz = tz
        self._now = now
        self._top = top

    async def __call__(self, lead_id: str) -> BestTimeRecommendation | None:
        """None when no model has passed the promotion gate yet."""
        model = self._get_model()
        if inspect.isawaitable(model):
            model = await model
        if model is None:
            return None
        calls = sorted(await self._history.calls_for(lead_id), key=lambda call: call.at)
        return BestTimeRecommendation(
            lead_id,
            rank_slots(model, calls, self._now(), self._tz, self._top),
            len(calls),
            len(calls) < _MIN_CALLS,
        )
