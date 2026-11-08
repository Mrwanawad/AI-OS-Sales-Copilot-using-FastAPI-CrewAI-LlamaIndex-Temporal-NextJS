from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from ai_os.modules.best_time.domain.models import BestTimeRecommendation
from ai_os.shared.auth import Authenticator, LeadAccess, Principal

Recommend = Callable[[str], Awaitable[BestTimeRecommendation | None]]


def create_router(
    get_recommend: Callable[[], Recommend], auth: Authenticator, lead_access: LeadAccess
) -> APIRouter:
    router = APIRouter(prefix="/best-time", tags=["best-time"])

    @router.get("/leads/{lead_id}")
    async def recommend(
        lead_id: str,
        user: Annotated[Principal, Depends(auth)],
        service: Annotated[Recommend, Depends(get_recommend)],
    ) -> BestTimeRecommendation:
        if not await lead_access(user, lead_id):
            raise HTTPException(status_code=403, detail="This lead isn't assigned to you")
        recommendation = await service(lead_id)
        if recommendation is None:
            raise HTTPException(status_code=404, detail="No model has been promoted yet")
        return recommendation

    return router
