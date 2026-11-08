import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable

import httpx

from ai_os.modules.escalation.domain.models import Ticket

logger = logging.getLogger(__name__)


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """Same scheme as the CRM webhook: HMAC-SHA256 over `<timestamp>.<body>`."""
    return hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


class WebhookTicketNotifier:
    """Posts ticket events to an automation endpoint (n8n, Zapier, a WhatsApp gateway).

    The payload carries ids and a link, never the question or answer: those can hold client
    details, and the receiver fetches them through the authenticated API if it needs them.
    """

    def __init__(
        self,
        url: str,
        secret: str,
        app_url: str | None = None,
        client_factory: Callable[[], httpx.AsyncClient] | None = None,
    ) -> None:
        self._url = url
        self._secret = secret
        self._app_url = (app_url or "").rstrip("/")
        self._client_factory = client_factory or (lambda: httpx.AsyncClient(timeout=10))

    async def ticket_opened(self, ticket: Ticket) -> None:
        await self._send("escalation.opened", ticket, notify="team_leads")

    async def ticket_answered(self, ticket: Ticket) -> None:
        await self._send("escalation.answered", ticket, notify=ticket.asked_by)

    async def _send(self, event: str, ticket: Ticket, notify: str | None) -> None:
        payload = {
            "event": event,
            "ticket_id": ticket.id,
            "status": ticket.status.value,
            "project": ticket.project,
            "lead_id": ticket.lead_id,
            "notify": notify,
            "url": f"{self._app_url}/escalations/{ticket.id}" if self._app_url else None,
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))
        headers = {
            "Content-Type": "application/json",
            "X-Timestamp": timestamp,
            "X-Signature": sign(self._secret, timestamp, body),
        }
        try:
            async with self._client_factory() as client:
                response = await client.post(self._url, content=body, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as error:
            # The ticket is already stored and visible in the queue; a failed push loses nothing.
            logger.warning("Escalation webhook failed for ticket %s: %s", ticket.id, error)
