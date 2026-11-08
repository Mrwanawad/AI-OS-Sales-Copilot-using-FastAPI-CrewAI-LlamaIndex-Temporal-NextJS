"""Invented call logs with a known answer pattern. Results on this data prove the pipeline only."""

import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ai_os.modules.best_time.domain.models import CallRecord

_PROFILES = {
    "office_worker": {"morning": 0.15, "afternoon": 0.2, "evening": 0.7, "weekend": 0.65},
    "business_owner": {"morning": 0.3, "afternoon": 0.65, "evening": 0.3, "weekend": 0.25},
    "retiree": {"morning": 0.7, "afternoon": 0.45, "evening": 0.2, "weekend": 0.5},
}
# Reps mostly call during their own working hours, so evening slots are under-sampled, as in life.
_HOUR_WEIGHTS = {hour: (3 if 10 <= hour <= 17 else 1) for hour in range(9, 22)}


def answer_probability(profile: str, weekday: int, hour: int) -> float:
    rates = _PROFILES[profile]
    if weekday in {4, 5}:
        return rates["weekend"]
    if hour < 12:
        return rates["morning"]
    return rates["afternoon"] if hour < 17 else rates["evening"]


def generate_calls(
    leads: int, seed: int, tz: ZoneInfo, start: datetime
) -> tuple[list[CallRecord], dict[str, str]]:
    rng = random.Random(seed)
    hours = list(_HOUR_WEIGHTS)
    weights = list(_HOUR_WEIGHTS.values())
    calls: list[CallRecord] = []
    profiles: dict[str, str] = {}
    for number in range(leads):
        lead_id = f"synthetic-{number}"
        profile = rng.choice(list(_PROFILES))
        profiles[lead_id] = profile
        first_day = rng.randrange(150)
        for _ in range(rng.randint(5, 25)):
            day = first_day + rng.randrange(30)
            hour = rng.choices(hours, weights)[0]
            local = (start + timedelta(days=day)).astimezone(tz)
            at = local.replace(hour=hour, minute=rng.randrange(60), second=0, microsecond=0)
            answered = rng.random() < answer_probability(profile, at.weekday(), hour)
            calls.append(CallRecord(lead_id, at, answered))
    return calls, profiles
