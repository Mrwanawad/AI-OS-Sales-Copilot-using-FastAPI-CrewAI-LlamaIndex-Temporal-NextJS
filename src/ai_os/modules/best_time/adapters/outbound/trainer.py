import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

# MLflow reads this at import. Training metadata stays on our infrastructure.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import lightgbm as lgb  # noqa: E402
import mlflow  # noqa: E402
import numpy as np
import optuna
from lifelines import KaplanMeierFitter
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ai_os.modules.best_time.adapters.outbound.lightgbm_model import LightGBMAnswerModel
from ai_os.modules.best_time.domain.features import FEATURE_NAMES, training_rows
from ai_os.modules.best_time.domain.models import CallRecord

logger = logging.getLogger(__name__)

REGISTERED_MODEL = "best-time-to-call"
_MIN_AUC_GAIN = 0.01
_TRAIN_SHARE = 0.7
_VALID_SHARE = 0.15
_LOGGED_METRICS = {"rows", "model_auc", "baseline_auc", "log_loss", "brier", "top_decile_lift"}
_HOUR = 0
_DAY = 1


@dataclass(frozen=True)
class TrainingReport:
    rows: int
    promoted: bool
    reason: str
    model_auc: float | None = None
    baseline_auc: float | None = None
    log_loss: float | None = None
    brier: float | None = None
    # Answer rate of the model's top-scored tenth of calls over the overall rate. Observational:
    # reps chose when to call, so this is not the effect of following the recommendation.
    top_decile_lift: float | None = None
    best_params: dict[str, float] = field(default_factory=dict)
    answered_by_attempt: dict[int, float] = field(default_factory=dict)


def answered_by_attempt(
    calls: Sequence[CallRecord], attempts: Sequence[int] = (1, 2, 3, 5, 10)
) -> dict[int, float]:
    """Survival curve of attempts until first answer. Leads never reached count as censored."""
    by_lead: dict[str, list[CallRecord]] = defaultdict(list)
    for call in calls:
        by_lead[call.lead_id].append(call)
    durations: list[int] = []
    reached: list[bool] = []
    for history in by_lead.values():
        history.sort(key=lambda call: call.at)
        first = next((i for i, call in enumerate(history, start=1) if call.answered), None)
        durations.append(first or len(history))
        reached.append(first is not None)
    fitter = KaplanMeierFitter().fit(durations, event_observed=reached)
    return {k: round(1 - float(fitter.predict(k)), 4) for k in attempts}


def _baseline_scores(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    """The obvious rule to beat: the historical answer rate of each hour and weekday."""
    overall = float(train_y.mean())
    totals: dict[tuple[float, float], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row, label in zip(train_x, train_y, strict=True):
        cell = totals[(row[_HOUR], row[_DAY])]
        cell[0] += label
        cell[1] += 1
    smoothing = 5
    return np.array(
        [
            (totals[(r[_HOUR], r[_DAY])][0] + smoothing * overall)
            / (totals[(r[_HOUR], r[_DAY])][1] + smoothing)
            for r in test_x
        ]
    )


def _params(trial: optuna.Trial) -> dict[str, float]:
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
    }


def _fit(
    params: dict[str, float],
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    rounds: int,
) -> lgb.Booster:
    full = {
        **params,
        "objective": "binary",
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbosity": -1,
    }
    return lgb.train(full, lgb.Dataset(x, y, feature_name=FEATURE_NAMES), num_boost_round=rounds)


def _tune(
    train: tuple[np.ndarray, np.ndarray],
    valid: tuple[np.ndarray, np.ndarray],
    trials: int,
    seed: int,
) -> tuple[dict[str, float], int]:
    rounds_seen: dict[int, int] = {}

    def objective(trial: optuna.Trial) -> float:
        booster = lgb.train(
            {
                **_params(trial),
                "objective": "binary",
                "metric": "binary_logloss",
                "seed": seed,
                "deterministic": True,
                "force_row_wise": True,
                "num_threads": 1,
                "verbosity": -1,
            },
            lgb.Dataset(train[0], train[1], feature_name=FEATURE_NAMES),
            num_boost_round=500,
            valid_sets=[lgb.Dataset(valid[0], valid[1], feature_name=FEATURE_NAMES)],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        rounds_seen[trial.number] = booster.best_iteration
        return float(log_loss(valid[1], booster.predict(valid[0]), labels=[0, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    return study.best_params, rounds_seen[study.best_trial.number]


def train_best_time(
    calls: Sequence[CallRecord],
    tz: ZoneInfo,
    output_path: Path,
    tracking_uri: str,
    trials: int = 20,
    min_rows: int = 2000,
    seed: int = 7,
) -> TrainingReport:
    """Writes the model to output_path only if it beats the baseline on the newest calls."""
    rows = training_rows(calls, tz)
    if len(rows) < min_rows:
        return TrainingReport(len(rows), False, f"only {len(rows)} calls, need {min_rows}")

    x = np.array([row[1] for row in rows], dtype=float)
    y = np.array([row[2] for row in rows], dtype=float)
    # Split by time, as in production: the model always predicts calls after those it saw.
    train_end = int(len(rows) * _TRAIN_SHARE)
    valid_end = int(len(rows) * (_TRAIN_SHARE + _VALID_SHARE))
    train, valid = (x[:train_end], y[:train_end]), (x[train_end:valid_end], y[train_end:valid_end])
    test_x, test_y = x[valid_end:], y[valid_end:]
    if len(set(y[:valid_end])) < 2 or len(set(test_y)) < 2:
        return TrainingReport(len(rows), False, "needs both answered and missed calls")

    best_params, rounds = _tune(train, valid, trials, seed)
    booster = _fit(best_params, x[:valid_end], y[:valid_end], seed, rounds)
    scores = booster.predict(test_x)
    baseline = _baseline_scores(x[:valid_end], y[:valid_end], test_x)
    model_auc = float(roc_auc_score(test_y, scores))
    baseline_auc = float(roc_auc_score(test_y, baseline))
    top = np.argsort(scores)[::-1][: max(1, len(scores) // 10)]
    promoted = model_auc >= baseline_auc + _MIN_AUC_GAIN
    report = TrainingReport(
        rows=len(rows),
        promoted=promoted,
        reason=(
            "beat the baseline"
            if promoted
            else f"AUC {model_auc:.3f} did not beat baseline {baseline_auc:.3f} by {_MIN_AUC_GAIN}"
        ),
        model_auc=round(model_auc, 4),
        baseline_auc=round(baseline_auc, 4),
        log_loss=round(float(log_loss(test_y, scores, labels=[0, 1])), 4),
        brier=round(float(brier_score_loss(test_y, scores)), 4),
        top_decile_lift=round(float(test_y[top].mean() / test_y.mean()), 3),
        best_params=best_params,
        answered_by_attempt=answered_by_attempt(calls),
    )
    if promoted:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(output_path))
        LightGBMAnswerModel.load(output_path)
    _log_run(report, booster, tracking_uri)
    return report


def _log_run(report: TrainingReport, booster: lgb.Booster, tracking_uri: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(REGISTERED_MODEL)
    with mlflow.start_run():
        mlflow.log_params(report.best_params)
        mlflow.set_tag("promoted", str(report.promoted))
        mlflow.set_tag("reason", report.reason)
        mlflow.log_metrics(
            {
                key: value
                for key, value in asdict(report).items()
                if key
                in {"rows", "model_auc", "baseline_auc", "log_loss", "brier", "top_decile_lift"}
            }
        )
        mlflow.log_metrics(
            {f"answered_by_attempt_{k}": v for k, v in report.answered_by_attempt.items()}
        )
        if report.promoted:
            logged = mlflow.lightgbm.log_model(
                booster, name="model", registered_model_name=REGISTERED_MODEL
            )
            mlflow.MlflowClient().set_registered_model_alias(
                REGISTERED_MODEL, "champion", logged.registered_model_version
            )
