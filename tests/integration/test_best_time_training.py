from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import mlflow
import pytest

from ai_os.modules.best_time.adapters.outbound.lightgbm_model import LightGBMAnswerModel
from ai_os.modules.best_time.adapters.outbound.synthetic import (
    answer_probability,
    generate_calls,
)
from ai_os.modules.best_time.adapters.outbound.trainer import (
    REGISTERED_MODEL,
    TrainingReport,
    answered_by_attempt,
    train_best_time,
)
from ai_os.modules.best_time.application.recommend import candidate_slots, rank_slots
from ai_os.modules.best_time.domain.models import CallRecord

CAIRO = ZoneInfo("Africa/Cairo")
START = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    directory = tmp_path_factory.mktemp("best-time")
    calls, profiles = generate_calls(600, 7, CAIRO, START)
    output = directory / "model.txt"
    report = train_best_time(
        calls, CAIRO, output, f"sqlite:///{(directory / 'mlflow.db').as_posix()}", trials=4
    )
    return report, output, calls, profiles, directory


def test_model_beats_baseline_and_is_promoted(trained) -> None:
    report: TrainingReport = trained[0]

    assert report.promoted
    assert report.model_auc > report.baseline_auc
    assert trained[1].exists()
    assert report.top_decile_lift > 1


def test_model_logged_and_registered_in_mlflow(trained) -> None:
    report, _output, _calls, _profiles, directory = trained
    client = mlflow.MlflowClient(f"sqlite:///{(directory / 'mlflow.db').as_posix()}")

    version = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
    run = client.get_run(version.run_id)

    assert run.data.metrics["model_auc"] == report.model_auc
    assert run.data.tags["promoted"] == "True"


def test_recommendations_find_better_times_than_random_on_known_truth(trained) -> None:
    _report, output, calls, profiles, _directory = trained
    model = LightGBMAnswerModel.load(output)
    now = START + timedelta(days=190)
    chosen: list[float] = []
    average: list[float] = []
    for lead_id, profile in list(profiles.items())[:100]:
        prior = sorted((c for c in calls if c.lead_id == lead_id), key=lambda c: c.at)
        best = rank_slots(model, prior, now, CAIRO, top=1)[0].at.astimezone(CAIRO)
        chosen.append(answer_probability(profile, best.weekday(), best.hour))
        slots = candidate_slots(now, CAIRO)
        truths = [answer_probability(profile, s.weekday(), s.hour) for s in slots]
        average.append(sum(truths) / len(truths))

    assert sum(chosen) / len(chosen) > 1.2 * sum(average) / len(average)


def test_too_few_calls_refuses_to_train(tmp_path: Path) -> None:
    calls, _profiles = generate_calls(5, 7, CAIRO, START)
    output = tmp_path / "model.txt"

    report = train_best_time(calls, CAIRO, output, f"sqlite:///{tmp_path.as_posix()}/m.db")

    assert not report.promoted
    assert "need" in report.reason
    assert not output.exists()


def test_pure_noise_is_not_promoted(tmp_path: Path) -> None:
    """With nothing to learn, the gate must keep the model out."""
    import random

    rng = random.Random(3)
    calls = [
        CallRecord(
            f"L{rng.randrange(300)}",
            START + timedelta(hours=rng.randrange(24 * 150)),
            rng.random() < 0.4,
        )
        for _ in range(4000)
    ]
    output = tmp_path / "model.txt"

    report = train_best_time(
        calls, CAIRO, output, f"sqlite:///{tmp_path.as_posix()}/m.db", trials=3
    )

    assert not report.promoted
    assert not output.exists()


def test_survival_curve_on_a_known_case() -> None:
    def history(lead: str, outcomes: list[bool]) -> list[CallRecord]:
        return [
            CallRecord(lead, START + timedelta(hours=i), outcome)
            for i, outcome in enumerate(outcomes)
        ]

    calls = [
        *history("first-try", [True]),
        *history("second-try", [False, True]),
        *history("third-try", [False, False, True]),
        *history("fourth-try", [False, False, False, True]),
    ]

    curve = answered_by_attempt(calls, attempts=(1, 2, 4))

    assert curve == {1: 0.25, 2: 0.5, 4: 1.0}


def test_survival_curve_counts_unreached_leads_as_censored() -> None:
    calls = [
        CallRecord("reached", START, True),
        CallRecord("unreached", START, False),
    ]

    assert answered_by_attempt(calls, attempts=(1,)) == {1: 0.5}


def test_saved_model_with_different_features_is_refused(tmp_path: Path) -> None:
    import lightgbm as lgb
    import numpy as np

    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "min_data_in_leaf": 1},
        lgb.Dataset(np.array([[0.0], [1.0]]), np.array([0, 1]), feature_name=["other"]),
        num_boost_round=1,
    )
    path = tmp_path / "old.txt"
    booster.save_model(str(path))

    with pytest.raises(ValueError, match="different features"):
        LightGBMAnswerModel.load(path)
