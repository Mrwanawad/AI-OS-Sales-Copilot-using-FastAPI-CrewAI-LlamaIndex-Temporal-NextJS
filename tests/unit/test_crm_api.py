import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.crm_copilot.adapters.inbound.api import create_router, sign
from ai_os.modules.crm_copilot.domain.models import ContextCard
from ai_os.shared.auth import Role

from .auth_helpers import AUTH, allow_all, bearer, deny_all
from .crm_fakes import LEAD, NOW, FakeCrmStore, call

SECRET = "webhook-secret"
EVENT = {"event_id": "e1", "type": "call_ended", "lead_id": "L1", "interaction_id": "c1"}


class FakePublisher:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def publish(self, *event: object) -> None:
        self.events.append(event)


def client(
    store: FakeCrmStore | None = None,
    publisher: FakePublisher | None = None,
    secret: str | None = SECRET,
    lead_access=allow_all,
) -> TestClient:
    app = FastAPI()
    app.include_router(
        create_router(
            lambda: store or FakeCrmStore(),
            lambda: publisher or FakePublisher(),
            lambda: secret,
            AUTH,
            lead_access,
        )
    )
    return TestClient(app)


def signed(body: bytes, secret: str = SECRET, timestamp: str | None = None) -> dict[str, str]:
    stamp = timestamp or str(time.time())
    return {"X-Timestamp": stamp, "X-Signature": sign(secret, stamp, body)}


def post(test_client: TestClient, body: bytes, headers: dict[str, str]):
    return test_client.post("/crm/webhooks/events", content=body, headers=headers)


def test_signed_event_is_published() -> None:
    publisher = FakePublisher()
    body = json.dumps(EVENT).encode()

    response = post(client(publisher=publisher), body, signed(body))

    assert response.status_code == 202
    assert publisher.events == [("e1", "call_ended", "L1", "c1", None, None)]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-Timestamp": str(time.time())},
        {"X-Signature": "abc"},
        {"X-Timestamp": "not-a-time", "X-Signature": "abc"},
    ],
)
def test_missing_or_malformed_signature_is_rejected(headers: dict[str, str]) -> None:
    publisher = FakePublisher()

    response = post(client(publisher=publisher), json.dumps(EVENT).encode(), headers)

    assert response.status_code == 401
    assert publisher.events == []


def test_wrong_secret_tampered_body_and_replay_are_rejected() -> None:
    publisher = FakePublisher()
    test_client = client(publisher=publisher)
    body = json.dumps(EVENT).encode()

    assert post(test_client, body, signed(body, "wrong")).status_code == 401
    assert post(test_client, body + b" ", signed(body)).status_code == 401
    old = str(time.time() - 3600)
    assert post(test_client, body, signed(body, timestamp=old)).status_code == 401
    assert publisher.events == []


def test_webhook_closes_when_secret_is_not_configured() -> None:
    body = json.dumps(EVENT).encode()

    assert post(client(secret=None), body, signed(body)).status_code == 503


def test_signed_but_invalid_event_is_422() -> None:
    body = json.dumps({"event_id": "e1", "type": "drop_tables", "lead_id": "L1"}).encode()

    assert post(client(), body, signed(body)).status_code == 422


def auth() -> dict[str, str]:
    return bearer("rep-a")


def test_card_route_needs_a_token_and_access_to_the_lead() -> None:
    url = "/crm/leads/L1/card"

    assert client().get(url).status_code == 401
    assert client(lead_access=deny_all).get(url, headers=auth()).status_code == 403


def test_card_is_returned_with_staleness() -> None:
    store = FakeCrmStore(call())
    store.card = ContextCard("L1", "Client One", LEAD.stage, NOW, call().occurred_at)
    url = "/crm/leads/L1/card"

    fresh = client(store).get(url, headers=auth())
    store.interactions["c2"] = call("c2", "New", day=3)
    stale = client(store).get(url, headers=auth())

    assert fresh.status_code == 200
    assert fresh.json()["stale"] is False
    assert fresh.json()["card"]["display_name"] == "Client One"
    assert stale.json()["stale"] is True


def test_lead_without_card_is_404() -> None:
    assert client().get("/crm/leads/L1/card", headers=auth()).status_code == 404


def test_reps_list_their_own_leads_and_team_leads_list_all() -> None:
    store = FakeCrmStore()
    other_rep = client(store).get("/crm/leads", headers=bearer("rep-z"))
    team_lead = client(store).get("/crm/leads", headers=bearer("lead-a", Role.TEAM_LEAD))
    own = client(store).get("/crm/leads", headers=bearer(LEAD.assigned_rep))

    assert other_rep.json() == []
    assert [lead["id"] for lead in team_lead.json()] == [LEAD.id]
    assert own.json()[0]["stage"] == LEAD.stage.value
