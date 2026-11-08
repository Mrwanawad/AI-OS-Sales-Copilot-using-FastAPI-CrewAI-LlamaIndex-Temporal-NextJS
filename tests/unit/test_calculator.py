from decimal import Decimal as D

import pytest

from ai_os.modules.inventory.domain.calculator import (
    DiscountPolicy,
    Frequency,
    Role,
    check_discount,
    installment_plan,
    mortgage_plan,
)
from ai_os.modules.inventory.domain.errors import InvalidCalculation

POLICY = DiscountPolicy(D("2"), D("5"), D("10"))


def test_installment_plan_worked_example() -> None:
    plan = installment_plan(D("5000000"), D("10"), years=8, maintenance_pct=D("8"))

    assert plan.down_payment == D("500000.00")
    assert plan.installment_count == 96
    assert plan.installment == D("46875.00")
    assert plan.final_installment == D("46875.00")
    assert plan.maintenance == D("400000.00")
    assert plan.total_payable == D("5400000.00")


@pytest.mark.parametrize(
    ("years", "frequency", "count"),
    [(8, Frequency.QUARTERLY, 32), (5, Frequency.SEMI_ANNUAL, 10), (7, Frequency.ANNUAL, 7)],
)
def test_installment_count_follows_frequency(years: int, frequency: Frequency, count: int) -> None:
    assert installment_plan(D("1000000"), D("0"), years, frequency).installment_count == count


@pytest.mark.parametrize("price", [D("1000000"), D("3333333.33"), D("7654321.09"), D("1000000.01")])
def test_payments_sum_to_price_exactly(price: D) -> None:
    plan = installment_plan(price, D("7.5"), years=7)

    paid = (
        plan.down_payment + plan.installment * (plan.installment_count - 1) + plan.final_installment
    )
    assert paid == plan.price
    assert abs(plan.final_installment - plan.installment) < D("1.00")


def test_full_down_payment_leaves_no_installments_owed() -> None:
    plan = installment_plan(D("1000000"), D("100"), years=1)

    assert plan.installment == D("0.00")
    assert plan.final_installment == D("0.00")


def test_mortgage_matches_published_amortization_formula() -> None:
    # 1,000,000 at 12% over 20 years: a standard table value of 11,010.86 per month.
    plan = mortgage_plan(D("1000000"), D("12"), years=20)

    assert plan.monthly_payment == D("11010.86")
    assert plan.months == 240


def test_mortgage_repays_principal_exactly() -> None:
    plan = mortgage_plan(D("2500000"), D("18.5"), years=15)

    total_paid = plan.monthly_payment * (plan.months - 1) + plan.final_payment

    assert total_paid == plan.principal + plan.total_interest
    assert abs(plan.final_payment - plan.monthly_payment) < D("5.00")


def test_zero_interest_mortgage() -> None:
    plan = mortgage_plan(D("1200000"), D("0"), years=10)

    assert plan.monthly_payment == D("10000.00")
    assert plan.total_interest == D("0.00")
    assert plan.final_payment == D("10000.00")


def test_discount_within_rep_limit() -> None:
    decision = check_discount(D("5000000"), D("2"), Role.REP, POLICY)

    assert decision.within_limit
    assert decision.discount_amount == D("100000.00")
    assert decision.final_price == D("4900000.00")
    assert decision.approver is Role.REP


def test_discount_above_rep_limit_names_lowest_approver() -> None:
    decision = check_discount(D("5000000"), D("4"), Role.REP, POLICY)

    assert not decision.within_limit
    assert decision.approver is Role.TEAM_LEAD


def test_discount_above_every_limit_has_no_approver() -> None:
    decision = check_discount(D("5000000"), D("11"), Role.MANAGER, POLICY)

    assert not decision.within_limit
    assert decision.approver is None


@pytest.mark.parametrize(
    "call",
    [
        lambda: installment_plan(D("0"), D("10"), 5),
        lambda: installment_plan(D("-1"), D("10"), 5),
        lambda: installment_plan(D("100"), D("101"), 5),
        lambda: installment_plan(D("100"), D("10"), 0),
        lambda: installment_plan(D("NaN"), D("10"), 5),
        lambda: mortgage_plan(D("100"), D("-1"), 5),
        lambda: mortgage_plan(D("100"), D("10"), 0),
        lambda: check_discount(D("100"), D("-1"), Role.REP, POLICY),
    ],
)
def test_invalid_input_is_rejected(call) -> None:
    with pytest.raises(InvalidCalculation):
        call()
