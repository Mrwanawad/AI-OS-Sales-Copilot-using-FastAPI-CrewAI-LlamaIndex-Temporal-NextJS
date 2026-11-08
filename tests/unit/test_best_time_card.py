from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.bootstrap.best_time import BestTimeCardProvider, CrmCallHistory
from ai_os.modules.best_time.adapters.inbound.api import create_router
from ai_os.modules.best_time.domain.models import BestTimeRecommendation, SlotScore
from ai_os.modules.crm_copilot.domain.models import InteractionKind

from .auth_helpers import AUTH, allow_all, bearer, deny_all
from .crm_fakes import FakeCrmStore, call

CAIRO = ZoneInfo("Africa/Cairo")
SLOT = SlotScore(datetime(2026, 11, 3, 17, 0, tzinfo=UTC), 0.6)


def recommend_with(result: object):
    async def recommend(lead_id: str) -> object:
        if isinstance(result, Exception):
            raise result
        return result

    return recommend


async def test_card_text_uses_business_timezone_and_states_its_basis() -> None:
    result = BestTimeRecommendation("L1", [SLOT], 5, low_data=False)

    text = await BestTimeCardProvider(recommend_with(result), CAIRO).best_time("L1")

    assert text == "Tuesday around 19:00 (estimated from this client's past calls)"


async def test_card_text_admits_when_it_rests_on_general_patterns() -> None:
    result = BestTimeRecommendation("L1", [SLOT], 1, low_data=True)

    text = await BestTimeCardProvider(recommend_with(result), CAIRO).best_time("L1")

    assert "general patterns" in text


async def test_card_survives_no_model_and_model_failure() -> None:
    no_model = BestTimeCardProvider(recommend_with(None), CAIRO)
    broken = BestTimeCardProvider(recommend_with(RuntimeError("bad model file")), CAIRO)

    assert await no_model.best_time("L1") is None
    assert await broken.best_time("L1") is None


async def test_history_keeps_only_calls_with_known_outcomes() -> None:
    message = call("m1", answered=True)
    store = FakeCrmStore(
        call("a", answered=True),
        call("b", answered=False),
        call("unknown"),
        message.__class__("w1", "L1", InteractionKind.WHATSAPP, message.occurred_at, answered=True),
    )

    records = await CrmCallHistory(store).calls_for("L1")

    assert [record.answered for record in records] == [True, False]


def api(result: object, lead_access=allow_all) -> TestClient:
    app = FastAPI()
    app.include_router(create_router(lambda: recommend_with(result), AUTH, lead_access))
    return TestClient(app)


def test_route_needs_token_and_reports_missing_model() -> None:
    auth = bearer()
    url = "/best-time/leads/L1"

    assert api(None).get(url).status_code == 401
    assert api(None).get(url, headers=auth).status_code == 404
    found = api(BestTimeRecommendation("L1", [SLOT], 5, False)).get(url, headers=auth)
    assert found.json()["slots"][0]["probability"] == 0.6
    assert api(None, deny_all).get(url, headers=auth).status_code == 403
