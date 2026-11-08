import inspect
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ai_os.modules.best_time.domain.features import features
from ai_os.modules.best_time.domain.models import (
    BestTimeRecommendation,
    CallingWindow,
    CallRecord,
    SlotScore,
)
from ai_os.modules.best_time.ports.model import AnswerModel, CallHistory

_DAYS_AHEAD = 7
_MIN_CALLS = 3


def candidate_slots(
    now: datetime, tz: ZoneInfo, window: CallingWindow | None = None
) -> list[datetime]:
    """Every allowed hour in the next week, in local time, skipping the current hour."""
    window = window or CallingWindow()
    local = now.astimezone(tz).replace(minute=0, second=0, microsecond=0)
    slots: list[datetime] = []
    for offset in range(_DAYS_AHEAD + 1):
        day = local + timedelta(days=offset)
        if day.weekday() not in window.days:
            continue
        for hour in range(window.first_hour, window.last_hour + 1):
            slot = day.replace(hour=hour)
            if now < slot <= now + timedelta(days=_DAYS_AHEAD):
                slots.append(slot)
    return slots


def rank_slots(
    model: AnswerModel,
    prior: Sequence[CallRecord],
    now: datetime,
    tz: ZoneInfo,
    top: int,
    window: CallingWindow | None = None,
) -> list[SlotScore]:
    slots = candidate_slots(now, tz, window)
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
        window: CallingWindow | None = None,
    ) -> None:
        self._get_model = get_model
        self._history = history
        self._tz = tz
        self._now = now
        self._top = top
        self._window = window or CallingWindow()

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
            rank_slots(model, calls, self._now(), self._tz, self._top, self._window),
            len(calls),
            len(calls) < _MIN_CALLS,
        )
