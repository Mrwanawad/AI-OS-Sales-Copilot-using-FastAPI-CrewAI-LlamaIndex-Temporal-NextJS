import asyncio
import logging
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.modules.best_time.application.recommend import RecommendBestTime
from ai_os.modules.best_time.domain.models import CallingWindow, CallRecord
from ai_os.modules.best_time.ports.model import AnswerModel
from ai_os.modules.crm_copilot.domain.models import InteractionKind
from ai_os.modules.crm_copilot.ports.crm import CrmStore
from ai_os.shared.config import Settings
from ai_os.shared.files import STORED_PREFIX, PostgresFileStore

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


class StoredModel:
    """A model kept in Postgres (`stored:<name>`), reloaded when a retrain replaces it.

    The API and worker are separate services with no shared disk, so both read the
    promoted model from the database instead of a file path.
    """

    def __init__(self, files: PostgresFileStore, reference: str, check_every: timedelta) -> None:
        self._files = files
        self._reference = reference
        self._check_every = check_every
        self._model: AnswerModel | None = None
        self._fingerprint: str | None = None
        self._checked_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def get(self) -> AnswerModel | None:
        if self._fresh():
            return self._model
        async with self._lock:
            if not self._fresh():
                try:
                    await self._refresh()
                except Exception:
                    # Keep the last good model; cards work without an estimate anyway.
                    logger.exception("Could not refresh the best-time model")
                self._checked_at = datetime.now(UTC)
        return self._model

    def _fresh(self) -> bool:
        checked = self._checked_at
        return checked is not None and datetime.now(UTC) - checked < self._check_every

    async def _refresh(self) -> None:
        from ai_os.modules.best_time.adapters.outbound.lightgbm_model import LightGBMAnswerModel

        fingerprint = await self._files.fingerprint(self._reference)
        if fingerprint is None:
            self._model, self._fingerprint = None, None
            return
        if fingerprint == self._fingerprint:
            return
        async with self._files.local_copy(self._reference) as path:
            self._model = LightGBMAnswerModel.load(path)
        self._fingerprint = fingerprint
        logger.info("Loaded best-time model %s", fingerprint[:12])


@lru_cache
def _stored_model(database_url: str, reference: str) -> StoredModel:
    files = PostgresFileStore(create_async_engine(database_url))
    return StoredModel(files, reference, timedelta(minutes=10))


def build_recommend(settings: Settings, store: CrmStore) -> RecommendBestTime:
    path = settings.best_time_model_path
    if path and path.startswith(STORED_PREFIX) and settings.async_database_url:
        get_model = _stored_model(settings.async_database_url, path).get
    else:

        def get_model() -> AnswerModel | None:
            return _load_model(path) if path else None

    return RecommendBestTime(
        get_model,
        CrmCallHistory(store),
        ZoneInfo(settings.business_timezone),
        lambda: datetime.now(UTC),
        window=CallingWindow(
            settings.call_window_first_hour,
            settings.call_window_last_hour,
            frozenset(int(day) for day in settings.call_window_days.split(",") if day.strip()),
        ),
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
