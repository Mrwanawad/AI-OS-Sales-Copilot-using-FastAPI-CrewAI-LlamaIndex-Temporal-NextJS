from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ai_os.modules.best_time.domain.models import CallRecord

FEATURE_NAMES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "attempts_so_far",
    "answer_rate",
    "answer_rate_same_daypart",
    "attempts_same_daypart",
    "answer_rate_same_day_type",
    "hours_since_last_call",
]

_EGYPT_WEEKEND = {4, 5}  # Friday and Saturday
_SMOOTHING_CALLS = 2
_SMOOTHING_RATE = 0.5
_MAX_HOURS_SINCE = 24 * 60
_NO_PREVIOUS_CALL = -1.0
# Keeps old behaviour from outweighing recent behaviour, and bounds the work per slot.
_HISTORY_WINDOW = 100


def _daypart(hour: int) -> int:
    if hour < 12:
        return 0
    return 1 if hour < 17 else 2


def _rate(answered: int, attempts: int) -> float:
    return (answered + _SMOOTHING_CALLS * _SMOOTHING_RATE) / (attempts + _SMOOTHING_CALLS)


def features(prior: Sequence[CallRecord], at: datetime, tz: ZoneInfo) -> list[float]:
    """Describes a call at `at` using only earlier calls, so training cannot see the future."""
    recent = prior[-_HISTORY_WINDOW:]
    local = at.astimezone(tz)
    weekend = local.weekday() in _EGYPT_WEEKEND
    daypart = _daypart(local.hour)

    same_daypart = same_day_type = same_daypart_answered = same_day_type_answered = answered = 0
    for call in recent:
        call_local = call.at.astimezone(tz)
        answered += call.answered
        if _daypart(call_local.hour) == daypart:
            same_daypart += 1
            same_daypart_answered += call.answered
        if (call_local.weekday() in _EGYPT_WEEKEND) == weekend:
            same_day_type += 1
            same_day_type_answered += call.answered

    if recent:
        hours = (at - recent[-1].at) / timedelta(hours=1)
        since_last = min(hours, _MAX_HOURS_SINCE)
    else:
        since_last = _NO_PREVIOUS_CALL
    return [
        float(local.hour),
        float(local.weekday()),
        float(weekend),
        float(len(recent)),
        _rate(answered, len(recent)),
        _rate(same_daypart_answered, same_daypart),
        float(same_daypart),
        _rate(same_day_type_answered, same_day_type),
        since_last,
    ]


def training_rows(
    calls: Iterable[CallRecord], tz: ZoneInfo
) -> list[tuple[datetime, list[float], int]]:
    """One (time, features, answered) row per call, oldest first."""
    by_lead: dict[str, list[CallRecord]] = defaultdict(list)
    for call in calls:
        by_lead[call.lead_id].append(call)
    rows: list[tuple[datetime, list[float], int]] = []
    for history in by_lead.values():
        history.sort(key=lambda call: call.at)
        for index, call in enumerate(history):
            rows.append((call.at, features(history[:index], call.at, tz), int(call.answered)))
    rows.sort(key=lambda row: row[0])
    return rows
