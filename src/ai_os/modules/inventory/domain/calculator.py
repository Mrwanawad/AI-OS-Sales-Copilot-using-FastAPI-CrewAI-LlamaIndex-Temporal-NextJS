from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import IntEnum, StrEnum

from ai_os.modules.inventory.domain.errors import InvalidCalculation

_CENTS = Decimal("0.01")
_HUNDRED = Decimal(100)


class Frequency(IntEnum):
    """Value is the number of months between installments."""

    MONTHLY = 1
    QUARTERLY = 3
    SEMI_ANNUAL = 6
    ANNUAL = 12


class Role(StrEnum):
    REP = "rep"
    TEAM_LEAD = "team_lead"
    MANAGER = "manager"


@dataclass(frozen=True)
class InstallmentPlan:
    price: Decimal
    down_payment: Decimal
    installment_count: int
    installment: Decimal
    # Absorbs rounding, so the payments add up to the price exactly.
    final_installment: Decimal
    maintenance: Decimal
    total_payable: Decimal


@dataclass(frozen=True)
class MortgagePlan:
    principal: Decimal
    monthly_payment: Decimal
    months: int
    # Absorbs rounding, so the principal is fully repaid.
    final_payment: Decimal
    total_interest: Decimal


@dataclass(frozen=True)
class DiscountPolicy:
    rep_limit_pct: Decimal
    team_lead_limit_pct: Decimal
    manager_limit_pct: Decimal

    def limit_for(self, role: Role) -> Decimal:
        return {
            Role.REP: self.rep_limit_pct,
            Role.TEAM_LEAD: self.team_lead_limit_pct,
            Role.MANAGER: self.manager_limit_pct,
        }[role]


@dataclass(frozen=True)
class DiscountDecision:
    requested_pct: Decimal
    final_price: Decimal
    discount_amount: Decimal
    within_limit: bool
    # Lowest role that may approve; None when no role may.
    approver: Role | None


def _money(value: Decimal) -> Decimal:
    return value.quantize(_CENTS, rounding=ROUND_HALF_UP)


def _require_positive(name: str, value: Decimal) -> None:
    if not value.is_finite() or value <= 0:
        raise InvalidCalculation(f"{name} must be positive")


def _require_percentage(name: str, value: Decimal) -> None:
    if not value.is_finite() or not 0 <= value <= 100:
        raise InvalidCalculation(f"{name} must be between 0 and 100")


def installment_plan(
    price: Decimal,
    down_payment_pct: Decimal,
    years: int,
    frequency: Frequency = Frequency.MONTHLY,
    maintenance_pct: Decimal = Decimal(0),
) -> InstallmentPlan:
    _require_positive("price", price)
    _require_percentage("down_payment_pct", down_payment_pct)
    _require_percentage("maintenance_pct", maintenance_pct)
    if years <= 0:
        raise InvalidCalculation("years must be positive")

    down_payment = _money(price * down_payment_pct / _HUNDRED)
    remaining = _money(price) - down_payment
    count = years * 12 // frequency
    installment = _money(remaining / count)
    final_installment = remaining - installment * (count - 1)
    maintenance = _money(price * maintenance_pct / _HUNDRED)
    return InstallmentPlan(
        price=_money(price),
        down_payment=down_payment,
        installment_count=count,
        installment=installment,
        final_installment=final_installment,
        maintenance=maintenance,
        total_payable=_money(price) + maintenance,
    )


def mortgage_plan(principal: Decimal, annual_rate_pct: Decimal, years: int) -> MortgagePlan:
    _require_positive("principal", principal)
    _require_percentage("annual_rate_pct", annual_rate_pct)
    if years <= 0:
        raise InvalidCalculation("years must be positive")

    months = years * 12
    balance = _money(principal)
    rate = annual_rate_pct / _HUNDRED / 12
    if rate == 0:
        payment = _money(balance / months)
    else:
        growth = (1 + rate) ** months
        payment = _money(balance * rate * growth / (growth - 1))

    total_interest = Decimal(0)
    for _ in range(months - 1):
        interest = _money(balance * rate)
        total_interest += interest
        balance = balance + interest - payment
    interest = _money(balance * rate)
    return MortgagePlan(
        principal=_money(principal),
        monthly_payment=payment,
        months=months,
        final_payment=balance + interest,
        total_interest=total_interest + interest,
    )


def check_discount(
    price: Decimal, requested_pct: Decimal, role: Role, policy: DiscountPolicy
) -> DiscountDecision:
    _require_positive("price", price)
    _require_percentage("requested_pct", requested_pct)

    discount = _money(price * requested_pct / _HUNDRED)
    approver = next(
        (candidate for candidate in Role if requested_pct <= policy.limit_for(candidate)), None
    )
    return DiscountDecision(
        requested_pct=requested_pct,
        final_price=_money(price) - discount,
        discount_amount=discount,
        within_limit=requested_pct <= policy.limit_for(role),
        approver=approver,
    )
