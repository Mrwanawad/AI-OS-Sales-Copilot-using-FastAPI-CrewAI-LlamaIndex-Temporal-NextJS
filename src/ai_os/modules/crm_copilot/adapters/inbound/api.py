import hashlib
import hmac
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, ValidationError

from ai_os.modules.crm_copilot.domain.models import ContextCard
from ai_os.modules.crm_copilot.ports.crm import CrmStore, EventPublisher
from ai_os.shared.auth import Authenticator, LeadAccess, Principal

_MAX_SIGNATURE_AGE_SECONDS = 300


class EventType(StrEnum):
    LEAD_OPENED = "lead_opened"
    LEAD_REASSIGNED = "lead_reassigned"
    CALL_ENDED = "call_ended"


class CrmWebhookEvent(BaseModel):
    event_id: str
    type: EventType
    lead_id: str
    interaction_id: str | None = None
    from_rep: str | None = None
    to_rep: str | None = None


class CardResponse(BaseModel):
    card: ContextCard
    stale: bool


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


def create_router(
    get_store: Callable[[], CrmStore],
    get_publisher: Callable[[], EventPublisher],
    get_webhook_secret: Callable[[], str | None],
    auth: Authenticator,
    lead_access: LeadAccess,
) -> APIRouter:
    router = APIRouter(prefix="/crm", tags=["crm"])

    @router.get("/leads/{lead_id}/card")
    async def get_card(
        lead_id: str,
        user: Annotated[Principal, Depends(auth)],
        store: Annotated[CrmStore, Depends(get_store)],
    ) -> CardResponse:
        if not await lead_access(user, lead_id):
            raise HTTPException(status_code=403, detail="This lead isn't assigned to you")
        card = await store.get_card(lead_id)
        if card is None:
            raise HTTPException(status_code=404, detail="No context card for this lead yet")
        interactions = await store.list_interactions(lead_id)
        latest = max((item.occurred_at for item in interactions), default=None)
        return CardResponse(card=card, stale=card.latest_interaction_at != latest)

    @router.post("/webhooks/events", status_code=202)
    async def receive_event(
        request: Request,
        publisher: Annotated[EventPublisher, Depends(get_publisher)],
        x_signature: Annotated[str | None, Header()] = None,
        x_timestamp: Annotated[str | None, Header()] = None,
    ) -> dict[str, str]:
        secret = get_webhook_secret()
        if not secret:
            raise HTTPException(status_code=503, detail="CRM_WEBHOOK_SECRET is not configured")
        body = await request.body()
        _verify(secret, x_timestamp, x_signature, body)
        try:
            event = CrmWebhookEvent.model_validate_json(body)
        except ValidationError as error:
            raise HTTPException(status_code=422, detail="Invalid event") from error
        await publisher.publish(
            event.event_id,
            event.type.value,
            event.lead_id,
            event.interaction_id,
            event.from_rep,
            event.to_rep,
        )
        return {"status": "accepted"}

    return router


def _verify(secret: str, timestamp: str | None, signature: str | None, body: bytes) -> None:
    if not timestamp or not signature:
        raise HTTPException(status_code=401, detail="Missing signature")
    try:
        age = abs(time.time() - float(timestamp))
    except ValueError as error:
        raise HTTPException(status_code=401, detail="Invalid timestamp") from error
    if age > _MAX_SIGNATURE_AGE_SECONDS:
        raise HTTPException(status_code=401, detail="Signature expired")
    if not hmac.compare_digest(sign(secret, timestamp, body), signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
