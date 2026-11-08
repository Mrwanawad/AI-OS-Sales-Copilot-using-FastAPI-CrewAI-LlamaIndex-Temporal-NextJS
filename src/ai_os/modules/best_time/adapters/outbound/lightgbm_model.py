from collections.abc import Sequence
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ai_os.modules.best_time.domain.features import FEATURE_NAMES


class LightGBMAnswerModel:
    def __init__(self, booster: lgb.Booster) -> None:
        if booster.feature_name() != FEATURE_NAMES:
            raise ValueError("Saved model was trained on different features; retrain it")
        self._booster = booster

    @classmethod
    def load(cls, path: Path) -> "LightGBMAnswerModel":
        return cls(lgb.Booster(model_file=str(path)))

    def predict(self, rows: Sequence[Sequence[float]]) -> Sequence[float]:
        return self._booster.predict(np.asarray(rows, dtype=float)).tolist()
