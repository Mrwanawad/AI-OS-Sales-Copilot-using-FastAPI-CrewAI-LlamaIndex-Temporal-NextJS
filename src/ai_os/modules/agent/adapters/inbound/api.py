from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai_os.modules.agent.domain.errors import LLMUnavailable
from ai_os.modules.agent.domain.models import AgentAnswer
from ai_os.modules.agent.ports.tools import AnswerQuestion
from ai_os.shared.auth import Authenticator, LeadAccess, Principal


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    project: str | None = None
    lead_id: str | None = None


def create_router(
    get_ask: Callable[[], AnswerQuestion], auth: Authenticator, lead_access: LeadAccess
) -> APIRouter:
    router = APIRouter(prefix="/agent", tags=["agent"])

    @router.post("/ask")
    async def ask(
        request: AskRequest,
        user: Annotated[Principal, Depends(auth)],
        ask_agent: Annotated[AnswerQuestion, Depends(get_ask)],
    ) -> AgentAnswer:
        # A lead id reads that client's context card.
        if request.lead_id and not await lead_access(user, request.lead_id):
            raise HTTPException(status_code=403, detail="This lead isn't assigned to you")
        try:
            return await ask_agent(request.question, request.project, request.lead_id, user.subject)
        except LLMUnavailable as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    return router
