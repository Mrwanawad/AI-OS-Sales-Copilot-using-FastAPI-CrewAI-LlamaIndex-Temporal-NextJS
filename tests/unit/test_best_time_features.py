from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from ai_os.modules.best_time.application.recommend import (
    CALL_HOURS,
    RecommendBestTime,
    candidate_slots,
)
from ai_os.modules.best_time.domain.features import FEATURE_NAMES, features, training_rows
from ai_os.modules.best_time.domain.models import CallRecord

CAIRO = ZoneInfo("Africa/Cairo")
# A Monday.
NOW = datetime(2026, 11, 2, 6, 30, tzinfo=UTC)


def feature(row: list[float], name: str) -> float:
    return row[FEATURE_NAMES.index(name)]


def call(hours_ago: float, answered: bool, lead: str = "L1") -> CallRecord:
    return CallRecord(lead, NOW - timedelta(hours=hours_ago), answered)


def test_hour_and_day_use_business_timezone_not_utc() -> None:
    row = features([], datetime(2026, 11, 5, 22, 30, tzinfo=UTC), CAIRO)

    assert feature(row, "hour") == 0
    assert feature(row, "day_of_week") == 4
    assert feature(row, "is_weekend") == 1


def test_no_history_gives_neutral_rates() -> None:
    row = features([], NOW, CAIRO)

    assert feature(row, "answer_rate") == 0.5
    assert feature(row, "attempts_so_far") == 0
    assert feature(row, "hours_since_last_call") == -1


def test_history_raises_rate_only_for_the_matching_daypart() -> None:
    # 72 hours before an 08:30 Cairo Monday morning is a morning too.
    prior = [call(72, True), call(72 - 24, True)]
    morning = features(prior, NOW, CAIRO)
    evening = features(prior, NOW + timedelta(hours=10), CAIRO)

    assert feature(morning, "answer_rate_same_daypart") > 0.5
    assert feature(evening, "answer_rate_same_daypart") == 0.5
    assert feature(morning, "hours_since_last_call") == 48


def test_training_rows_never_use_the_call_being_predicted_or_later_ones() -> None:
    calls = [call(30, True), call(20, False), call(10, True)]
    changed_future = [call(30, True), call(20, False), call(10, False)]

    rows = training_rows(calls, CAIRO)
    other = training_rows(changed_future, CAIRO)

    assert [row[1] for row in rows] == [row[1] for row in other]
    assert [row[2] for row in rows] == [1, 0, 1]


def test_training_rows_keep_leads_separate_and_sort_by_time() -> None:
    rows = training_rows([call(5, True, "A"), call(9, False, "B"), call(1, True, "B")], CAIRO)

    assert [row[0] for row in rows] == sorted(row[0] for row in rows)
    assert feature(rows[1][1], "attempts_so_far") == 0
    assert feature(rows[2][1], "attempts_so_far") == 1


def test_training_and_serving_share_identical_features() -> None:
    history = [call(30, True), call(20, False)]
    target = call(10, True)

    served = features(history, target.at, CAIRO)
    trained = training_rows([*history, target], CAIRO)[-1][1]

    assert served == trained


def test_slots_stay_in_calling_hours_and_the_next_week() -> None:
    slots = candidate_slots(NOW, CAIRO)

    assert all(slot.hour in CALL_HOURS for slot in slots)
    assert all(NOW < slot <= NOW + timedelta(days=7) for slot in slots)
    assert len(slots) == 7 * len(CALL_HOURS)


class PrefersEvening:
    def predict(self, rows: list[list[float]]) -> list[float]:
        return [row[0] / 24 for row in rows]


class History:
    def __init__(self, calls: list[CallRecord]) -> None:
        self.calls = calls

    async def calls_for(self, lead_id: str) -> list[CallRecord]:
        return self.calls


async def test_recommends_highest_scored_slots() -> None:
    recommend = RecommendBestTime(
        PrefersEvening, History([call(5, True)] * 4), CAIRO, lambda: NOW, top=2
    )

    result = await recommend("L1")

    assert len(result.slots) == 2
    assert all(slot.at.astimezone(CAIRO).hour == 21 for slot in result.slots)
    assert result.slots[0].probability >= result.slots[1].probability
    assert not result.low_data


@pytest.mark.parametrize(("calls", "low"), [(0, True), (2, True), (3, False)])
async def test_flags_low_data(calls: int, low: bool) -> None:
    history = History([call(5 + i, True) for i in range(calls)])

    result = await RecommendBestTime(PrefersEvening, history, CAIRO, lambda: NOW)("L1")

    assert result.low_data is low
    assert result.calls_seen == calls


async def test_no_model_means_no_recommendation() -> None:
    recommend = RecommendBestTime(lambda: None, History([]), CAIRO, lambda: NOW)

    assert await recommend("L1") is None
