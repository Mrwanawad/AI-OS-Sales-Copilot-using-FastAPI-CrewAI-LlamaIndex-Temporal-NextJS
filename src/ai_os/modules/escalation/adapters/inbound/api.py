from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

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
from ai_os.shared.auth import Authenticator, Principal, Role

_STATUS_FOR = {
    TicketNotFound: 404,
    CandidateNotFound: 404,
    InvalidTransition: 409,
    SameReviewer: 409,
    NotShareable: 409,
    ContainsPersonalData: 422,
}
_HANDLED = tuple(_STATUS_FOR)

Note = Annotated[str, Field(min_length=1, max_length=2000)]


class AnswerRequest(BaseModel):
    answer: str = Field(min_length=1, max_length=5000)
    share_with_project: str | None = Field(default=None, min_length=1, max_length=200)


class AnswerResponse(BaseModel):
    ticket: Ticket
    candidate: KnowledgeCandidate | None


class NoteRequest(BaseModel):
    note: Note


class ValidityRequest(BaseModel):
    # Days the answer stays in search before it must be renewed. Default: the setting.
    valid_days: int | None = Field(default=None, ge=1, le=365)


def _http(error: Exception) -> HTTPException:
    return HTTPException(status_code=_STATUS_FOR[type(error)], detail=str(error))


def create_router(get_service: Callable[[], Escalations], auth: Authenticator) -> APIRouter:
    router = APIRouter(prefix="/escalations", tags=["escalations"])
    Service = Annotated[Escalations, Depends(get_service)]
    User = Annotated[Principal, Depends(auth)]
    Lead = Annotated[Principal, Depends(auth.require(Role.TEAM_LEAD, Role.MANAGER))]
    Limit = Annotated[int, Query(ge=1, le=500)]

    @router.get("")
    async def list_tickets(
        user: Lead,
        service: Service,
        status: TicketStatus = TicketStatus.OPEN,
        limit: Limit = 100,
    ) -> list[Ticket]:
        return await service.list_tickets(status, limit)

    @router.get("/mine")
    async def my_tickets(user: User, service: Service, limit: Limit = 100) -> list[Ticket]:
        return await service.tickets_asked_by(user.subject, limit)

    @router.get("/knowledge")
    async def list_candidates(
        user: Lead,
        service: Service,
        status: CandidateStatus = CandidateStatus.PENDING,
        limit: Limit = 100,
    ) -> list[KnowledgeCandidate]:
        return await service.list_candidates(status, limit)

    @router.post("/knowledge/{candidate_id}/approve")
    async def approve(
        candidate_id: str, user: Lead, service: Service, request: ValidityRequest | None = None
    ) -> KnowledgeCandidate:
        valid_days = request.valid_days if request else None
        try:
            return await service.approve(candidate_id, user.subject, valid_days)
        except _HANDLED as error:
            raise _http(error) from error

    @router.post("/knowledge/{candidate_id}/renew")
    async def renew(
        candidate_id: str, user: Lead, service: Service, request: ValidityRequest | None = None
    ) -> KnowledgeCandidate:
        valid_days = request.valid_days if request else None
        try:
            return await service.renew(candidate_id, user.subject, valid_days)
        except _HANDLED as error:
            raise _http(error) from error

    @router.post("/knowledge/{candidate_id}/reject")
    async def reject(
        candidate_id: str, request: NoteRequest, user: Lead, service: Service
    ) -> KnowledgeCandidate:
        try:
            return await service.reject(candidate_id, user.subject, request.note)
        except _HANDLED as error:
            raise _http(error) from error

    @router.post("/knowledge/{candidate_id}/revoke")
    async def revoke(
        candidate_id: str, request: NoteRequest, user: Lead, service: Service
    ) -> KnowledgeCandidate:
        try:
            return await service.revoke(candidate_id, user.subject, request.note)
        except _HANDLED as error:
            raise _http(error) from error

    @router.get("/{ticket_id}")
    async def get_ticket(ticket_id: str, user: User, service: Service) -> Ticket:
        try:
            ticket = await service.get_ticket(ticket_id)
        except _HANDLED as error:
            raise _http(error) from error
        if not user.leads_a_team and ticket.asked_by != user.subject:
            raise HTTPException(status_code=403, detail="This is another rep's ticket")
        return ticket

    @router.post("/{ticket_id}/answer")
    async def answer(
        ticket_id: str, request: AnswerRequest, user: Lead, service: Service
    ) -> AnswerResponse:
        try:
            ticket, candidate = await service.answer_ticket(
                ticket_id, request.answer, user.subject, request.share_with_project
            )
        except _HANDLED as error:
            raise _http(error) from error
        return AnswerResponse(ticket=ticket, candidate=candidate)

    @router.post("/{ticket_id}/dismiss")
    async def dismiss(ticket_id: str, request: NoteRequest, user: Lead, service: Service) -> Ticket:
        try:
            return await service.dismiss_ticket(ticket_id, user.subject, request.note)
        except _HANDLED as error:
            raise _http(error) from error

    return router
