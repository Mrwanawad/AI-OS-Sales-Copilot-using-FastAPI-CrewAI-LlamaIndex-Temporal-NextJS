import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from ai_os.modules.best_time.application.recommend import RecommendBestTime
from ai_os.modules.best_time.domain.models import CallRecord
from ai_os.modules.best_time.ports.model import AnswerModel
from ai_os.modules.crm_copilot.domain.models import InteractionKind
from ai_os.modules.crm_copilot.ports.crm import CrmStore
from ai_os.shared.config import Settings

logger = logging.getLogger(__name__)


class CrmCallHistory:
    def __init__(self, store: CrmStore) -> None:
        self._store = store

    async def calls_for(self, lead_id: str) -> list[CallRecord]:
        return [
            CallRecord(lead_id, item.occurred_at, item.answered)
            for item in await self._store.list_interactions(lead_id)
            if item.kind is InteractionKind.CALL and item.answered is not None
        ]


@lru_cache
def _load_model(path: str) -> AnswerModel | None:
    # Imported here so the app runs without the ml extra until a model is configured.
    from ai_os.modules.best_time.adapters.outbound.lightgbm_model import LightGBMAnswerModel

    if not Path(path).exists():
        logger.warning("No best-time model at %s", path)
        return None
    return LightGBMAnswerModel.load(Path(path))


def build_recommend(settings: Settings, store: CrmStore) -> RecommendBestTime:
    def get_model() -> AnswerModel | None:
        path = settings.best_time_model_path
        return _load_model(path) if path else None

    return RecommendBestTime(
        get_model,
        CrmCallHistory(store),
        ZoneInfo(settings.business_timezone),
        lambda: datetime.now(UTC),
    )


class BestTimeCardProvider:
    """A context card is still useful without this estimate, so a failure here must not block it."""

    def __init__(self, recommend: RecommendBestTime, tz: ZoneInfo) -> None:
        self._recommend = recommend
        self._tz = tz

    async def best_time(self, lead_id: str) -> str | None:
        try:
            recommendation = await self._recommend(lead_id)
        except Exception:
            logger.exception("Best-time estimate failed for lead %s", lead_id)
            return None
        if recommendation is None or not recommendation.slots:
            return None
        slot = recommendation.slots[0]
        local = slot.at.astimezone(self._tz)
        basis = "general patterns" if recommendation.low_data else "this client's past calls"
        return f"{local:%A} around {local:%H:%M} (estimated from {basis})"
