from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from ai_os.modules.coaching.domain.errors import (
    NothingToScore,
    PersonaNotFound,
    SessionClosed,
    SessionNotFound,
)
from ai_os.modules.coaching.domain.models import Persona, RolePlaySession
from ai_os.modules.coaching.domain.personas import PERSONAS
from ai_os.modules.coaching.ports.coaching import CoachingService
from ai_os.shared.auth import Authenticator, Principal

_STATUS_FOR = {
    PersonaNotFound: 404,
    SessionNotFound: 404,
    SessionClosed: 409,
    NothingToScore: 409,
}
_HANDLED = tuple(_STATUS_FOR)


class StartRequest(BaseModel):
    persona_id: str


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)


def create_router(get_service: Callable[[], CoachingService], auth: Authenticator) -> APIRouter:
    router = APIRouter(prefix="/coaching", tags=["coaching"])
    Service = Annotated[CoachingService, Depends(get_service)]
    User = Annotated[Principal, Depends(auth)]

    async def owned(service: CoachingService, session_id: str, user: Principal) -> RolePlaySession:
        session = await service.get(session_id)
        if session.rep_id != user.subject:
            raise HTTPException(status_code=403, detail="This is another rep's session")
        return session

    async def visible(
        service: CoachingService, session_id: str, user: Principal
    ) -> RolePlaySession:
        # Team leads may review a rep's practice, but only the rep may practice.
        if user.leads_a_team:
            return await service.get(session_id)
        return await owned(service, session_id, user)

    @router.get("/personas")
    async def personas(user: User) -> list[Persona]:
        return list(PERSONAS.values())

    @router.post("/sessions", status_code=201)
    async def start(request: StartRequest, user: User, service: Service) -> RolePlaySession:
        try:
            return await service.start(user.subject, request.persona_id)
        except _HANDLED as error:
            raise _http(error) from error

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str, user: User, service: Service) -> RolePlaySession:
        try:
            return await visible(service, session_id, user)
        except _HANDLED as error:
            raise _http(error) from error

    @router.post("/sessions/{session_id}/messages")
    async def send(
        session_id: str, request: MessageRequest, user: User, service: Service
    ) -> RolePlaySession:
        try:
            await owned(service, session_id, user)
            return await service.send(session_id, request.message)
        except _HANDLED as error:
            raise _http(error) from error

    @router.post("/sessions/{session_id}/finish")
    async def finish(session_id: str, user: User, service: Service) -> RolePlaySession:
        try:
            await owned(service, session_id, user)
            return await service.finish(session_id)
        except _HANDLED as error:
            raise _http(error) from error

    @router.get("/sessions/{session_id}/turns/{turn_index}/audio")
    async def audio(session_id: str, turn_index: int, user: User, service: Service) -> Response:
        try:
            await owned(service, session_id, user)
            content = await service.turn_audio(session_id, turn_index)
        except _HANDLED as error:
            raise _http(error) from error
        return Response(content=content, media_type="audio/mpeg")

    return router


def _http(error: Exception) -> HTTPException:
    return HTTPException(status_code=_STATUS_FOR[type(error)], detail=str(error))
