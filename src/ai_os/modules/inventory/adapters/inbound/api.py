from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ai_os.modules.inventory.application.ask import AskInventory
from ai_os.modules.inventory.domain.calculator import (
    DiscountDecision,
    DiscountPolicy,
    Frequency,
    InstallmentPlan,
    MortgagePlan,
    Role,
    check_discount,
    installment_plan,
    mortgage_plan,
)
from ai_os.modules.inventory.domain.errors import UnanswerableQuestion
from ai_os.modules.inventory.domain.models import InventoryAnswer, Unit
from ai_os.modules.inventory.ports.store import UnitStore
from ai_os.shared.auth import Authenticator, Principal

_MAX_YEARS = 40


class InstallmentRequest(BaseModel):
    price: Decimal = Field(gt=0)
    down_payment_pct: Decimal = Field(ge=0, le=100)
    years: int = Field(gt=0, le=_MAX_YEARS)
    frequency: Frequency = Frequency.MONTHLY
    maintenance_pct: Decimal = Field(default=Decimal(0), ge=0, le=100)


class MortgageRequest(BaseModel):
    principal: Decimal = Field(gt=0)
    annual_rate_pct: Decimal = Field(ge=0, le=100)
    years: int = Field(gt=0, le=_MAX_YEARS)


class DiscountRequest(BaseModel):
    price: Decimal = Field(gt=0)
    requested_pct: Decimal = Field(ge=0, le=100)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class UnitResponse(BaseModel):
    code: str
    project: str
    unit_type: str
    bedrooms: int
    area_sqm: Decimal
    price_egp: Decimal
    status: str
    delivery_date: date | None


def create_router(
    get_store: Callable[[], UnitStore],
    get_ask: Callable[[], AskInventory],
    get_discount_policy: Callable[[], DiscountPolicy],
    auth: Authenticator,
) -> APIRouter:
    router = APIRouter(prefix="/inventory", tags=["inventory"], dependencies=[Depends(auth)])

    @router.get("/units/{code}", response_model=UnitResponse)
    async def get_unit(code: str, store: Annotated[UnitStore, Depends(get_store)]) -> Unit:
        found = await store.get_by_codes([code])
        if not found:
            raise HTTPException(status_code=404, detail=f"Unit {code} not found")
        return found[0]

    @router.post("/ask")
    async def ask(
        request: AskRequest, ask_inventory: Annotated[AskInventory, Depends(get_ask)]
    ) -> InventoryAnswer:
        try:
            return await ask_inventory(request.question)
        except UnanswerableQuestion as error:
            raise HTTPException(status_code=422, detail=error.reason) from error

    @router.post("/calculate/installments")
    async def installments(request: InstallmentRequest) -> InstallmentPlan:
        return installment_plan(
            request.price,
            request.down_payment_pct,
            request.years,
            request.frequency,
            request.maintenance_pct,
        )

    @router.post("/calculate/mortgage")
    async def mortgage(request: MortgageRequest) -> MortgagePlan:
        return mortgage_plan(request.principal, request.annual_rate_pct, request.years)

    @router.post("/calculate/discount")
    async def discount(
        request: DiscountRequest,
        user: Annotated[Principal, Depends(auth)],
        policy: Annotated[DiscountPolicy, Depends(get_discount_policy)],
    ) -> DiscountDecision:
        # The role comes from the token, so a rep can't claim a manager's limit.
        return check_discount(request.price, request.requested_pct, Role(user.role.value), policy)

    return router
