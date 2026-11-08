import hashlib
import hmac
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, ValidationError

from ai_os.modules.crm_copilot.domain.models import ContextCard, InteractionKind, LeadStage
from ai_os.modules.crm_copilot.ports.crm import CrmStore, EventPublisher, RecordingUploads
from ai_os.shared.auth import Authenticator, LeadAccess, Principal

_MAX_SIGNATURE_AGE_SECONDS = 300
_MAX_RECORDING_BYTES = 200 * 1024 * 1024
_AUDIO_SUFFIXES = frozenset(
    {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".opus", ".webm", ".flac", ".aac", ".mp4"}
)


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


class LeadSummary(BaseModel):
    id: str
    display_name: str
    stage: LeadStage
    assigned_rep: str


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
    get_recordings: Callable[[], RecordingUploads] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/crm", tags=["crm"])

    @router.get("/leads")
    async def list_leads(
        user: Annotated[Principal, Depends(auth)],
        store: Annotated[CrmStore, Depends(get_store)],
    ) -> list[LeadSummary]:
        # Same rule as card access: reps see their own leads, team leads see all.
        rep = None if user.leads_a_team else user.subject
        return [
            LeadSummary(
                id=lead.id,
                display_name=lead.display_name,
                stage=lead.stage,
                assigned_rep=lead.assigned_rep,
            )
            for lead in await store.list_leads(rep, 500)
        ]

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

    @router.post("/interactions/{interaction_id}/recording", status_code=202)
    async def upload_recording(
        interaction_id: str,
        user: Annotated[Principal, Depends(auth)],
        store: Annotated[CrmStore, Depends(get_store)],
        publisher: Annotated[EventPublisher, Depends(get_publisher)],
        file: Annotated[UploadFile, File()],
    ) -> dict[str, str]:
        """For dialers that can't host recordings: stores one and starts transcription."""
        if get_recordings is None:
            raise HTTPException(status_code=503, detail="Recording storage is not configured")
        call = await store.get_interaction(interaction_id)
        # Unknown ids and other reps' calls look the same, so ids don't leak.
        if call is None or not await lead_access(user, call.lead_id):
            raise HTTPException(status_code=404, detail="No such call")
        if call.kind is not InteractionKind.CALL:
            raise HTTPException(status_code=409, detail="Only calls have recordings")
        if not call.recording_consent:
            # Egypt PDPL: without the client's consent the recording isn't stored at all.
            raise HTTPException(status_code=409, detail="The client did not consent to recording")
        name = file.filename or ""
        if not any(name.lower().endswith(suffix) for suffix in _AUDIO_SUFFIXES):
            raise HTTPException(status_code=422, detail="Upload an audio file")
        data = await file.read(_MAX_RECORDING_BYTES + 1)
        if not data or len(data) > _MAX_RECORDING_BYTES:
            raise HTTPException(status_code=422, detail="The file is empty or over 200 MB")
        reference = await get_recordings().put(
            name, file.content_type or "application/octet-stream", data, user.subject
        )
        await store.set_recording(interaction_id, reference)
        # Keyed on the stored file, so a retried upload of a new file is processed again.
        event_id = f"recording-{interaction_id}-{reference.rsplit(':', 1)[-1]}"
        await publisher.publish(event_id, "call_ended", call.lead_id, interaction_id, None, None)
        return {"status": "accepted"}

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
