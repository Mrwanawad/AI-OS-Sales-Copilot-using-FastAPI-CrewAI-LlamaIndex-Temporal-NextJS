import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.crm_copilot.adapters.inbound.api import create_router, sign
from ai_os.modules.crm_copilot.application.privacy import ErasureReport
from ai_os.modules.crm_copilot.domain.errors import LeadNotFound
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


class FakeRecordings:
    def __init__(self) -> None:
        self.saved: list[tuple[str, bytes, str]] = []

    async def put(self, filename: str, content_type: str, data: bytes, uploaded_by: str) -> str:
        self.saved.append((filename, data, uploaded_by))
        return f"stored:rec{len(self.saved)}"


def recording_client(store: FakeCrmStore, recordings: FakeRecordings, publisher: FakePublisher):
    app = FastAPI()
    app.include_router(
        create_router(
            lambda: store,
            lambda: publisher,
            lambda: SECRET,
            AUTH,
            _lead_access_for(store),
            lambda: recordings,
        )
    )
    return TestClient(app)


def _lead_access_for(store: FakeCrmStore):
    from ai_os.modules.crm_copilot.application.access import can_access_lead

    async def access(user, lead_id: str) -> bool:
        return await can_access_lead(store, user, lead_id)

    return access


AUDIO = {"file": ("call.mp3", b"ID3fake-audio", "audio/mpeg")}


def test_recording_upload_stores_the_file_and_starts_transcription() -> None:
    store = FakeCrmStore(call(recording_consent=True, recording_path=None))
    recordings, publisher = FakeRecordings(), FakePublisher()
    client = recording_client(store, recordings, publisher)

    response = client.post("/crm/interactions/c1/recording", files=AUDIO, headers=bearer("rep-a"))

    assert response.status_code == 202, response.text
    assert store.interactions["c1"].recording_path == "stored:rec1"
    assert recordings.saved[0][2] == "rep-a"
    event_id, kind, lead_id, interaction_id, *_ = publisher.events[0]
    assert (kind, lead_id, interaction_id) == ("call_ended", "L1", "c1")
    assert event_id == "recording-c1-rec1"


def test_recording_upload_needs_consent_access_a_call_and_audio() -> None:
    no_consent = FakeCrmStore(call(recording_consent=False))
    recordings, publisher = FakeRecordings(), FakePublisher()

    refused = recording_client(no_consent, recordings, publisher).post(
        "/crm/interactions/c1/recording", files=AUDIO, headers=bearer("rep-a")
    )
    consented = FakeCrmStore(call(recording_consent=True))
    client = recording_client(consented, recordings, publisher)
    other_rep = client.post("/crm/interactions/c1/recording", files=AUDIO, headers=bearer("rep-b"))
    unknown = client.post("/crm/interactions/nope/recording", files=AUDIO, headers=bearer("rep-a"))
    not_audio = client.post(
        "/crm/interactions/c1/recording",
        files={"file": ("notes.pdf", b"%PDF", "application/pdf")},
        headers=bearer("rep-a"),
    )

    assert refused.status_code == 409 and "consent" in refused.json()["detail"]
    assert other_rep.status_code == 404 and unknown.status_code == 404
    assert not_audio.status_code == 422
    assert recordings.saved == [] and publisher.events == []


class FakePrivacy:
    def __init__(self) -> None:
        self.erased: list[tuple[str, str]] = []

    async def erase_lead(self, lead_id: str, requested_by: str) -> ErasureReport:
        if lead_id != "L1":
            raise LeadNotFound(lead_id)
        self.erased.append((lead_id, requested_by))
        return ErasureReport(lead_id, 1, 2)


def test_only_a_manager_can_erase_a_lead() -> None:
    privacy = FakePrivacy()
    app = FastAPI()
    app.include_router(
        create_router(
            FakeCrmStore,
            FakePublisher,
            lambda: SECRET,
            AUTH,
            allow_all,
            get_privacy=lambda: privacy,
        )
    )
    test_client = TestClient(app)

    for role in (Role.REP, Role.TEAM_LEAD):
        response = test_client.delete("/crm/leads/L1", headers=bearer("x", role))
        assert response.status_code == 403
    assert privacy.erased == []

    manager = bearer("manager-a", Role.MANAGER)
    response = test_client.delete("/crm/leads/L1", headers=manager)
    assert response.status_code == 200
    assert response.json() == {
        "lead_id": "L1",
        "recordings_deleted": 1,
        "linked_records_deleted": 2,
    }
    assert privacy.erased == [("L1", "manager-a")]
    assert test_client.delete("/crm/leads/L9", headers=manager).status_code == 404


def test_erase_is_503_when_not_configured() -> None:
    response = client().delete("/crm/leads/L1", headers=bearer("manager-a", Role.MANAGER))

    assert response.status_code == 503
