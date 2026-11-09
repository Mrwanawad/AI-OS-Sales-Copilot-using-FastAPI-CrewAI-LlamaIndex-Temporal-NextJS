import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.escalation.adapters.inbound.api import create_router
from ai_os.modules.escalation.application.service import Escalations
from ai_os.modules.escalation.domain.errors import (
    CandidateNotFound,
    ContainsPersonalData,
    InvalidTransition,
    NotShareable,
    SameReviewer,
    TicketNotFound,
)
from ai_os.modules.escalation.domain.models import (
    CandidateStatus,
    KnowledgeCandidate,
    Ticket,
    TicketStatus,
)
from ai_os.modules.escalation.domain.personal_data import find_personal_data
from ai_os.shared.auth import Role

from .auth_helpers import AUTH, bearer

NOW = datetime(2026, 11, 7, tzinfo=UTC)


class FakeStore:
    def __init__(self) -> None:
        self.tickets: dict[str, Ticket] = {}
        self.candidates: dict[str, KnowledgeCandidate] = {}

    async def add_ticket(self, ticket: Ticket) -> None:
        self.tickets[ticket.id] = ticket

    async def get_ticket(self, ticket_id: str) -> Ticket | None:
        return self.tickets.get(ticket_id)

    async def list_tickets(self, status: TicketStatus, limit: int) -> list[Ticket]:
        return [t for t in self.tickets.values() if t.status is status][:limit]

    async def list_tickets_asked_by(self, asked_by: str, limit: int) -> list[Ticket]:
        return [t for t in self.tickets.values() if t.asked_by == asked_by][:limit]

    async def update_ticket(self, ticket: Ticket, expected: TicketStatus) -> bool:
        if self.tickets[ticket.id].status is not expected:
            return False
        self.tickets[ticket.id] = ticket
        return True

    async def delete_tickets_for_lead(self, lead_id: str) -> int:
        doomed = [t.id for t in self.tickets.values() if t.lead_id == lead_id]
        for ticket_id in doomed:
            del self.tickets[ticket_id]
        return len(doomed)

    async def delete_closed_tickets(self, before: datetime) -> int:
        doomed = [
            t.id
            for t in self.tickets.values()
            if t.status is not TicketStatus.OPEN and t.closed_at and t.closed_at < before
        ]
        for ticket_id in doomed:
            del self.tickets[ticket_id]
        return len(doomed)

    async def add_candidate(self, candidate: KnowledgeCandidate) -> None:
        self.candidates[candidate.id] = candidate

    async def get_candidate(self, candidate_id: str) -> KnowledgeCandidate | None:
        return self.candidates.get(candidate_id)

    async def list_candidates(
        self, status: CandidateStatus, limit: int
    ) -> list[KnowledgeCandidate]:
        return [c for c in self.candidates.values() if c.status is status][:limit]

    async def update_candidate(
        self, candidate: KnowledgeCandidate, expected: CandidateStatus
    ) -> bool:
        if self.candidates[candidate.id].status is not expected:
            return False
        self.candidates[candidate.id] = candidate
        return True


class FakePublisher:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.live: dict[str, KnowledgeCandidate] = {}

    async def publish(self, candidate: KnowledgeCandidate) -> None:
        if self.fail:
            raise ConnectionError("vector store down")
        self.live[candidate.id] = candidate

    async def remove(self, candidate: KnowledgeCandidate) -> None:
        if self.fail:
            raise ConnectionError("vector store down")
        del self.live[candidate.id]


class FakeNotifier:
    def __init__(self) -> None:
        self.opened: list[str] = []
        self.answered: list[str] = []

    async def ticket_opened(self, ticket: Ticket) -> None:
        self.opened.append(ticket.id)

    async def ticket_answered(self, ticket: Ticket) -> None:
        self.answered.append(ticket.id)


def build(publisher: FakePublisher | None = None) -> tuple[Escalations, FakeStore, FakePublisher]:
    store = FakeStore()
    live = publisher or FakePublisher()
    return Escalations(store, live, FakeNotifier(), lambda: NOW), store, live


async def proposed(service: Escalations, answer: str = "Yes, the clubhouse opens in 2027."):
    ticket = await service.open_ticket("When does the clubhouse open?", "no evidence found", "PH")
    _ticket, candidate = await service.answer_ticket(ticket.id, answer, "lead-a", "PH")
    return candidate


@pytest.mark.parametrize(
    ("text", "kinds"),
    [
        ("Call 01012345678 today", ["phone number"]),
        ("Call 010 1234 5678 today", ["phone number"]),
        ("Call +201012345678", ["phone number"]),
        ("اتصل على ٠١٠١٢٣٤٥٦٧٨", ["phone number"]),
        ("Mail client@example.com", ["email address"]),
        ("ID 29001011234567", ["national ID"]),
        ("Price is 5,750,000 EGP for unit A-101 in 2027", []),
        ("Price is 15000000 EGP", []),
    ],
)
def test_personal_data_screen(text: str, kinds: list[str]) -> None:
    assert find_personal_data(text) == kinds


async def test_opened_ticket_is_queued_and_the_team_lead_notified() -> None:
    store = FakeStore()
    notifier = FakeNotifier()
    service = Escalations(store, FakePublisher(), notifier, lambda: NOW)

    ticket = await service.open_ticket("Q?", "low confidence", "PH", "L1", "Card text")

    assert await service.list_tickets(TicketStatus.OPEN) == [ticket]
    assert notifier.opened == [ticket.id]
    assert ticket.context == "Card text"


async def test_answer_without_sharing_creates_no_candidate() -> None:
    service, store, _live = build()
    ticket = await service.open_ticket("Q?", "reason")

    answered, candidate = await service.answer_ticket(ticket.id, "A", "lead-a")

    assert answered.status is TicketStatus.ANSWERED
    assert answered.answer == "A"
    assert candidate is None
    assert store.candidates == {}


async def test_answering_twice_is_refused() -> None:
    service, _store, _live = build()
    ticket = await service.open_ticket("Q?", "reason")
    await service.answer_ticket(ticket.id, "A", "lead-a")

    with pytest.raises(InvalidTransition):
        await service.answer_ticket(ticket.id, "B", "lead-b")


async def test_nothing_reaches_knowledge_until_a_second_person_approves() -> None:
    service, _store, live = build()
    candidate = await proposed(service)
    assert live.live == {}

    with pytest.raises(SameReviewer):
        await service.approve(candidate.id, "lead-a")
    assert live.live == {}

    approved = await service.approve(candidate.id, "manager-b")

    assert approved.status is CandidateStatus.APPROVED
    assert approved.reviewed_by == "manager-b"
    assert list(live.live) == [candidate.id]


async def test_rejected_answer_never_goes_live_and_cannot_be_approved_later() -> None:
    service, _store, live = build()
    candidate = await proposed(service)

    rejected = await service.reject(candidate.id, "manager-b", "Wrong date")

    assert rejected.note == "Wrong date"
    with pytest.raises(InvalidTransition):
        await service.approve(candidate.id, "manager-b")
    assert live.live == {}


async def test_personal_data_blocks_proposal_and_leaves_ticket_open() -> None:
    service, store, _live = build()
    ticket = await service.open_ticket("Q?", "reason")

    with pytest.raises(ContainsPersonalData, match="phone number"):
        await service.answer_ticket(ticket.id, "Call 01012345678", "lead-a", "PH")

    assert store.tickets[ticket.id].status is TicketStatus.OPEN
    assert store.candidates == {}


async def test_personal_data_in_the_question_also_blocks_sharing() -> None:
    service, _store, _live = build()
    ticket = await service.open_ticket("Client on 01012345678 wants a refund?", "reason")

    with pytest.raises(ContainsPersonalData):
        await service.answer_ticket(ticket.id, "Yes", "lead-a", "PH")


async def test_lead_specific_answer_can_be_given_but_not_shared() -> None:
    service, _store, _live = build()
    ticket = await service.open_ticket("Q?", "reason", lead_id="L1")

    with pytest.raises(NotShareable):
        await service.answer_ticket(ticket.id, "A", "lead-a", "PH")

    answered, candidate = await service.answer_ticket(ticket.id, "A", "lead-a")
    assert answered.status is TicketStatus.ANSWERED
    assert candidate is None


async def test_failed_publish_returns_candidate_to_pending_for_a_retry() -> None:
    failing = FakePublisher(fail=True)
    service, store, _live = build(failing)
    candidate = await proposed(service)

    with pytest.raises(ConnectionError):
        await service.approve(candidate.id, "manager-b")
    assert store.candidates[candidate.id].status is CandidateStatus.PENDING
    assert store.candidates[candidate.id].reviewed_by is None

    failing.fail = False
    await service.approve(candidate.id, "manager-b")
    assert candidate.id in failing.live


async def test_revoke_takes_approved_answer_out_of_knowledge() -> None:
    service, store, live = build()
    candidate = await proposed(service)
    await service.approve(candidate.id, "manager-b")

    revoked = await service.revoke(candidate.id, "manager-b", "Date changed")

    assert revoked.status is CandidateStatus.REVOKED
    assert live.live == {}
    assert store.candidates[candidate.id].note == "Date changed"
    with pytest.raises(InvalidTransition):
        await service.revoke(candidate.id, "manager-b", "Again")


async def test_only_approved_answers_can_be_revoked() -> None:
    service, _store, _live = build()
    candidate = await proposed(service)

    with pytest.raises(InvalidTransition):
        await service.revoke(candidate.id, "manager-b", "Not live yet")


async def test_failed_removal_stays_approved_so_it_can_be_retried() -> None:
    failing = FakePublisher()
    service, store, _live = build(failing)
    candidate = await proposed(service)
    await service.approve(candidate.id, "manager-b")

    failing.fail = True
    with pytest.raises(ConnectionError):
        await service.revoke(candidate.id, "manager-b", "Date changed")

    assert store.candidates[candidate.id].status is CandidateStatus.APPROVED


async def test_approved_answer_expires_after_its_period() -> None:
    clock = [NOW]
    store, live = FakeStore(), FakePublisher()
    service = Escalations(store, live, FakeNotifier(), lambda: clock[0], answer_valid_days=30)
    candidate = await proposed(service)
    approved = await service.approve(candidate.id, "manager-b")
    assert approved.expires_at == NOW + timedelta(days=30)

    clock[0] = NOW + timedelta(days=29)
    assert await service.expire_due() == []
    assert list(live.live) == [candidate.id]

    clock[0] = NOW + timedelta(days=30)
    expired = await service.expire_due()

    assert [c.id for c in expired] == [candidate.id]
    assert store.candidates[candidate.id].status is CandidateStatus.EXPIRED
    assert live.live == {}


async def test_renewal_restarts_the_period() -> None:
    clock = [NOW]
    store, live = FakeStore(), FakePublisher()
    service = Escalations(store, live, FakeNotifier(), lambda: clock[0], answer_valid_days=30)
    candidate = await proposed(service)
    await service.approve(candidate.id, "manager-b", valid_days=10)

    clock[0] = NOW + timedelta(days=9)
    renewed = await service.renew(candidate.id, "lead-c")
    clock[0] = NOW + timedelta(days=12)

    assert renewed.expires_at == NOW + timedelta(days=39)
    assert await service.expire_due() == []
    assert list(live.live) == [candidate.id]


async def test_only_approved_answers_can_be_renewed() -> None:
    service, _store, _live = build()
    candidate = await proposed(service)

    with pytest.raises(InvalidTransition):
        await service.renew(candidate.id, "manager-b")


async def test_failed_removal_on_expiry_stays_approved_for_the_next_run() -> None:
    clock = [NOW]
    store, live = FakeStore(), FakePublisher()
    service = Escalations(store, live, FakeNotifier(), lambda: clock[0], answer_valid_days=1)
    candidate = await proposed(service)
    await service.approve(candidate.id, "manager-b")
    live.fail = True
    clock[0] = NOW + timedelta(days=2)

    with pytest.raises(ConnectionError):
        await service.expire_due()

    assert store.candidates[candidate.id].status is CandidateStatus.APPROVED


async def test_dismiss_closes_ticket_with_note() -> None:
    service, _store, _live = build()
    ticket = await service.open_ticket("Q?", "reason")

    dismissed = await service.dismiss_ticket(ticket.id, "lead-a", "Duplicate")

    assert dismissed.status is TicketStatus.DISMISSED
    assert dismissed.answer == "Duplicate"
    assert await service.list_tickets(TicketStatus.OPEN) == []


async def test_unknown_ids_raise() -> None:
    service, _store, _live = build()

    with pytest.raises(TicketNotFound):
        await service.get_ticket("nope")
    with pytest.raises(CandidateNotFound):
        await service.approve("nope", "manager-b")


REP = bearer("rep-a", Role.REP)
LEAD = bearer("lead-a", Role.TEAM_LEAD)
MANAGER = bearer("manager-b", Role.MANAGER)


def api() -> tuple[TestClient, Escalations, FakePublisher]:
    service, _store, live = build()
    app = FastAPI()
    app.include_router(create_router(lambda: service, AUTH))
    return TestClient(app), service, live


def test_every_route_needs_a_token() -> None:
    client, _service, _live = api()

    assert client.get("/escalations").status_code == 401
    assert client.get("/escalations/mine").status_code == 401
    assert client.get("/escalations/knowledge").status_code == 401
    assert client.post("/escalations/x/answer", json={}).status_code == 401
    assert client.post("/escalations/knowledge/x/approve").status_code == 401


def test_reps_cannot_use_the_queue_or_review() -> None:
    client, service, _live = api()
    ticket = asyncio.run(service.open_ticket("Q?", "reason"))

    assert client.get("/escalations", headers=REP).status_code == 403
    assert client.get("/escalations/knowledge", headers=REP).status_code == 403
    answer = {"answer": "A"}
    assert (
        client.post(f"/escalations/{ticket.id}/answer", json=answer, headers=REP).status_code == 403
    )
    assert client.post("/escalations/knowledge/x/approve", headers=REP).status_code == 403


def test_reps_see_only_their_own_tickets() -> None:
    client, service, _live = api()
    mine = asyncio.run(service.open_ticket("Mine?", "reason", asked_by="rep-a"))
    other = asyncio.run(service.open_ticket("Other?", "reason", asked_by="rep-b"))

    listed = client.get("/escalations/mine", headers=REP).json()

    assert [ticket["id"] for ticket in listed] == [mine.id]
    assert client.get(f"/escalations/{mine.id}", headers=REP).status_code == 200
    assert client.get(f"/escalations/{other.id}", headers=REP).status_code == 403
    assert client.get(f"/escalations/{other.id}", headers=LEAD).status_code == 200


def test_full_loop_over_http_with_identity_from_tokens() -> None:
    client, service, live = api()
    opened = asyncio.run(service.open_ticket("Clubhouse?", "no evidence found", "PH"))
    assert client.get("/escalations", headers=LEAD).json()[0]["id"] == opened.id

    answered = client.post(
        f"/escalations/{opened.id}/answer",
        # A name in the body is ignored: the token says who answered.
        json={"answer": "Opens in 2027.", "answered_by": "manager-b", "share_with_project": "PH"},
        headers=LEAD,
    )
    assert answered.status_code == 200
    assert answered.json()["ticket"]["answered_by"] == "lead-a"
    candidate_id = answered.json()["candidate"]["id"]
    assert client.get("/escalations/knowledge", headers=LEAD).json()[0]["id"] == candidate_id

    url = f"/escalations/knowledge/{candidate_id}"
    assert client.post(f"{url}/approve", headers=LEAD).status_code == 409
    approved = client.post(f"{url}/approve", headers=MANAGER)
    assert approved.json()["status"] == "approved"
    assert approved.json()["reviewed_by"] == "manager-b"
    assert candidate_id in live.live
    revoked = client.post(f"{url}/revoke", json={"note": "Changed"}, headers=MANAGER)
    assert revoked.json()["status"] == "revoked"
    assert live.live == {}


def test_errors_map_to_clean_statuses() -> None:
    client, _service, _live = api()

    assert client.get("/escalations/nope", headers=LEAD).status_code == 404
    assert client.post("/escalations/knowledge/nope/approve", headers=LEAD).status_code == 404
    assert client.post("/escalations/nope/answer", json={}, headers=LEAD).status_code == 422


def test_approval_period_and_renewal_over_http() -> None:
    client, service, live = api()
    candidate = asyncio.run(proposed(service))
    url = f"/escalations/knowledge/{candidate.id}"

    approved = client.post(f"{url}/approve", json={"valid_days": 14}, headers=MANAGER)
    assert approved.status_code == 200
    assert approved.json()["expires_at"].startswith("2026-11-21")
    assert client.post(f"{url}/renew", json={"valid_days": 400}, headers=LEAD).status_code == 422
    assert client.post(f"{url}/renew", headers=REP).status_code == 403

    renewed = client.post(f"{url}/renew", headers=LEAD)

    assert renewed.status_code == 200
    assert renewed.json()["expires_at"].startswith("2027-02-05")
    assert renewed.json()["reviewed_by"] == "lead-a"
    assert list(live.live) == [candidate.id]
