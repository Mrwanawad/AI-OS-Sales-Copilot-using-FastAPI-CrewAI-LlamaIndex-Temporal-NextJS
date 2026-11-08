from collections.abc import Sequence
from decimal import Decimal as D

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.inventory.adapters.inbound.api import create_router
from ai_os.modules.inventory.domain.calculator import DiscountPolicy
from ai_os.modules.inventory.domain.errors import UnanswerableQuestion
from ai_os.modules.inventory.domain.models import InventoryAnswer, Unit, UnitStatus
from ai_os.shared.auth import Role

from .auth_helpers import AUTH, bearer

UNIT = Unit("A-101", "Palm Hills", "villa", 4, D("300"), D("9000000"), UnitStatus.AVAILABLE)
POLICY = DiscountPolicy(D("2"), D("5"), D("10"))


class FakeStore:
    async def get_by_codes(self, codes: Sequence[str]) -> list[Unit]:
        return [UNIT] if UNIT.code in codes else []


class FakeAsk:
    async def __call__(self, question: str) -> InventoryAnswer:
        if question == "bad":
            raise UnanswerableQuestion(question, "only SELECT is allowed")
        return InventoryAnswer(question, "SELECT code FROM units", [{"code": "A-101"}])


def client() -> TestClient:
    app = FastAPI()
    app.include_router(create_router(FakeStore, FakeAsk, lambda: POLICY, AUTH))
    return TestClient(app, headers=bearer())


def test_get_unit_and_not_found() -> None:
    found = client().get("/inventory/units/A-101")

    assert found.status_code == 200
    assert found.json()["status"] == "available"
    assert client().get("/inventory/units/nope").status_code == 404


def test_ask_returns_rows_and_sql() -> None:
    response = client().post("/inventory/ask", json={"question": "Which units?"})

    assert response.json()["rows"] == [{"code": "A-101"}]
    assert response.json()["sql"] == "SELECT code FROM units"


def test_unanswerable_question_is_a_clean_422() -> None:
    response = client().post("/inventory/ask", json={"question": "bad"})

    assert response.status_code == 422
    assert response.json()["detail"] == "only SELECT is allowed"


def test_installment_amounts_are_exact_decimals() -> None:
    response = client().post(
        "/inventory/calculate/installments",
        json={"price": "5000000", "down_payment_pct": "10", "years": 8},
    )

    assert response.status_code == 200
    assert D(response.json()["installment"]) == D("46875.00")


def test_mortgage_and_discount() -> None:
    mortgage = client().post(
        "/inventory/calculate/mortgage",
        json={"principal": "1000000", "annual_rate_pct": "12", "years": 20},
    )
    discount = client().post(
        "/inventory/calculate/discount",
        json={"price": "5000000", "requested_pct": "4"},
    )

    assert D(mortgage.json()["monthly_payment"]) == D("11010.86")
    assert discount.json()["within_limit"] is False
    assert discount.json()["approver"] == "team_lead"


def test_discount_limit_follows_the_token_not_the_request() -> None:
    body = {"price": "5000000", "requested_pct": "4", "role": "manager"}
    url = "/inventory/calculate/discount"

    as_rep = client().post(url, json=body)
    as_lead = client().post(url, json=body, headers=bearer("lead-a", Role.TEAM_LEAD))

    assert as_rep.json()["within_limit"] is False
    assert as_lead.json()["within_limit"] is True


def test_every_route_needs_a_token() -> None:
    unsigned = {"Authorization": ""}

    assert client().get("/inventory/units/A-101", headers=unsigned).status_code == 401
    assert (
        client().post("/inventory/ask", json={"question": "x"}, headers=unsigned).status_code == 401
    )


def test_rejects_bad_calculator_input() -> None:
    bad = {"price": "-1", "down_payment_pct": "10", "years": 8}
    too_long = {"price": "100", "down_payment_pct": "10", "years": 500}

    assert client().post("/inventory/calculate/installments", json=bad).status_code == 422
    assert client().post("/inventory/calculate/installments", json=too_long).status_code == 422
