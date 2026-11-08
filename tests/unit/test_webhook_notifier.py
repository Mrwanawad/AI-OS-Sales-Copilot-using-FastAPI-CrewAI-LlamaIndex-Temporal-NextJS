import json
from datetime import UTC, datetime

import httpx

from ai_os.modules.escalation.adapters.outbound.webhook_notifier import (
    WebhookTicketNotifier,
    sign,
)
from ai_os.modules.escalation.domain.models import Ticket, TicketStatus

SECRET = "webhook-secret"
TICKET = Ticket(
    "t1",
    "Can Mr. Hassan (0100 123 4567) get 12% off unit B-204?",
    "low_confidence",
    datetime(2026, 11, 8, tzinfo=UTC),
    project="Demo Bay",
    lead_id="L1",
    asked_by="rep-a",
)


def notifier(handler) -> WebhookTicketNotifier:
    transport = httpx.MockTransport(handler)
    return WebhookTicketNotifier(
        "https://hooks.example/escalations",
        SECRET,
        "https://app.example/",
        lambda: httpx.AsyncClient(transport=transport),
    )


async def test_events_are_signed_and_carry_ids_but_never_the_question() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    await notifier(handler).ticket_opened(TICKET)
    answered = Ticket(**{**TICKET.__dict__, "status": TicketStatus.ANSWERED, "answer": "No."})
    await notifier(handler).ticket_answered(answered)

    opened, done = requests
    body = opened.content
    assert opened.headers["X-Signature"] == sign(SECRET, opened.headers["X-Timestamp"], body)
    payload = json.loads(body)
    assert payload == {
        "event": "escalation.opened",
        "ticket_id": "t1",
        "status": "open",
        "project": "Demo Bay",
        "lead_id": "L1",
        "notify": "team_leads",
        "url": "https://app.example/escalations/t1",
    }
    assert b"Hassan" not in body and b"0100" not in body
    assert json.loads(done.content)["notify"] == "rep-a"
    assert b"No." not in done.content


async def test_a_failing_endpoint_does_not_break_the_ticket_flow(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    await notifier(handler).ticket_opened(TICKET)

    assert "webhook failed for ticket t1" in caplog.text
