from dataclasses import asdict
from decimal import Decimal
from functools import partial

import ai_os.bootstrap.crewai_env  # noqa: F401
from ai_os.modules.agent.application.composer import CrewAnswerComposer
from ai_os.modules.agent.application.flow import AskAgent
from ai_os.modules.agent.domain.errors import ToolCouldNotAnswer
from ai_os.modules.agent.domain.models import (
    EvidenceItem,
    InstallmentQuery,
    MortgageQuery,
    SourceKind,
)
from ai_os.modules.crm_copilot.domain.models import ContextCard
from ai_os.modules.crm_copilot.ports.crm import CrmStore
from ai_os.modules.escalation.application.service import Escalations
from ai_os.modules.inventory.application.ask import AskInventory
from ai_os.modules.inventory.domain.calculator import (
    Frequency,
    installment_plan,
    mortgage_plan,
)
from ai_os.modules.inventory.domain.errors import (
    InvalidCalculation,
    UnanswerableQuestion,
)
from ai_os.modules.inventory.domain.models import Unit
from ai_os.modules.inventory.ports.store import UnitStore
from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.modules.knowledge.domain.models import SearchFilters
from ai_os.shared.config import Settings
from ai_os.shared.llm.crewai_adapter import fallback_llm, primary_llm
from ai_os.shared.llm.port import LLMRole
from ai_os.shared.llm.pydantic_ai_adapter import (
    PydanticAIStructuredLLM,
    build_model,
)


class KnowledgeToolAdapter:
    def __init__(self, retrieve: RetrieveKnowledge, store: UnitStore) -> None:
        self._retrieve = retrieve
        self._store = store

    async def search(self, query: str, project: str | None) -> list[EvidenceItem]:
        hits = await self._retrieve(query, filters=SearchFilters(project=project))
        evidence = [
            EvidenceItem(
                hit.chunk.text,
                SourceKind.DOCUMENT,
                _document_reference(
                    hit.chunk.citation.document_id,
                    hit.chunk.citation.document_version,
                    hit.chunk.citation.page,
                ),
            )
            for hit in hits
        ]
        # Documents can lag behind live inventory, so attach the live row for each unit they name.
        codes = list(dict.fromkeys(code for hit in hits for code in hit.chunk.unit_codes))
        return evidence + [_unit_evidence(unit) for unit in await self._store.get_by_codes(codes)]


class InventoryToolAdapter:
    def __init__(self, ask: AskInventory) -> None:
        self._ask = ask

    async def ask(self, question: str) -> list[EvidenceItem]:
        try:
            answer = await self._ask(question)
        except UnanswerableQuestion as error:
            raise ToolCouldNotAnswer(error.reason) from error
        reference = f"live inventory query: {answer.sql}"
        if not answer.rows:
            return [EvidenceItem("The query returned no units.", SourceKind.INVENTORY, reference)]
        return [
            EvidenceItem(
                ", ".join(f"{column}: {value}" for column, value in row.items()),
                SourceKind.INVENTORY,
                reference,
            )
            for row in answer.rows
        ]


class CalculatorToolAdapter:
    def __init__(self, store: UnitStore) -> None:
        self._store = store

    async def installments(self, query: InstallmentQuery) -> EvidenceItem:
        price, source = await self._price(query.price, query.unit_code)
        try:
            frequency = Frequency(query.months_between_payments)
            plan = installment_plan(
                price, query.down_payment_pct, query.years, frequency, query.maintenance_pct
            )
        except (InvalidCalculation, ValueError) as error:
            raise ToolCouldNotAnswer(str(error)) from error
        return _calculator_evidence("Installment plan", source, asdict(plan))

    async def mortgage(self, query: MortgageQuery) -> EvidenceItem:
        price, source = await self._price(query.price, query.unit_code)
        principal = price * (100 - query.down_payment_pct) / 100
        try:
            plan = mortgage_plan(principal, query.annual_rate_pct, query.years)
        except InvalidCalculation as error:
            raise ToolCouldNotAnswer(str(error)) from error
        return _calculator_evidence("Mortgage", source, asdict(plan))

    async def _price(self, price: Decimal | None, code: str | None) -> tuple[Decimal, str]:
        # A named unit always uses its live price, whatever else the question says.
        if code:
            units = await self._store.get_by_codes([code])
            if not units:
                raise ToolCouldNotAnswer(f"unit {code} was not found in inventory")
            return units[0].price_egp, f"live price of unit {code}"
        if price is None:
            raise ToolCouldNotAnswer("no price or unit code was given")
        return price, "price stated by the rep"


class ContextCardToolAdapter:
    def __init__(self, store: CrmStore) -> None:
        self._store = store

    async def card_evidence(self, lead_id: str) -> EvidenceItem | None:
        card = await self._store.get_card(lead_id)
        if card is None:
            return None
        return EvidenceItem(_card_text(card), SourceKind.CRM, f"context card for lead {lead_id}")


class EscalationToolAdapter:
    def __init__(self, escalations: Escalations, crm: CrmStore) -> None:
        self._escalations = escalations
        self._crm = crm

    async def create_ticket(
        self,
        question: str,
        reason: str,
        project: str | None,
        lead_id: str | None,
        asked_by: str | None,
    ) -> str:
        card = await self._crm.get_card(lead_id) if lead_id else None
        ticket = await self._escalations.open_ticket(
            question, reason, project, lead_id, _card_text(card) if card else "", asked_by
        )
        return ticket.id


def _card_text(card: ContextCard) -> str:
    def listed(title: str, items: list[str]) -> str | None:
        return f"{title}: " + "; ".join(items) if items else None

    lines = [
        f"Client: {card.display_name}. Stage: {card.stage.value}.",
        listed("Stated needs", [item.text for item in card.stated_needs]),
        listed("Closing drivers", [item.text for item in card.closing_drivers]),
        listed(
            "Open concerns",
            [
                f"{item.concern} (suggested answer: {item.suggested_answer})"
                if item.suggested_answer
                else item.concern
                for item in card.open_concerns
            ],
        ),
        listed("Promises", [item.text for item in card.promises]),
        listed("Talking points", [item.text for item in card.talking_points]),
        f"Best time to call: {card.best_time_to_call}." if card.best_time_to_call else None,
        listed(
            "Matched units",
            [
                f"{unit.code} ({unit.project}, {unit.bedrooms} bedrooms, {unit.price_egp} EGP)"
                for unit in card.matched_units
            ],
        ),
    ]
    return "\n".join(line for line in lines if line)


def _document_reference(document_id: str, version: int, page: int | None) -> str:
    location = f", page {page}" if page is not None else ""
    return f"{document_id} v{version}{location}"


def _unit_evidence(unit: Unit) -> EvidenceItem:
    return EvidenceItem(
        f"Live inventory for unit {unit.code}: price {unit.price_egp} EGP, status "
        f"{unit.status.value}, {unit.bedrooms} bedrooms, {unit.area_sqm} sqm",
        SourceKind.INVENTORY,
        f"unit {unit.code}",
    )


def _calculator_evidence(title: str, source: str, values: dict[str, object]) -> EvidenceItem:
    fields = ", ".join(f"{name}: {value}" for name, value in values.items())
    return EvidenceItem(f"{title} ({source}): {fields}", SourceKind.CALCULATOR, title.casefold())


def build_ask_agent(
    settings: Settings,
    retrieve: RetrieveKnowledge,
    store: UnitStore,
    ask_inventory: AskInventory,
    crm: CrmStore,
    escalations: Escalations,
) -> AskAgent:
    settings.require_anthropic_key()
    settings.require_openai_key()
    composer = CrewAnswerComposer(
        [primary_llm(settings, LLMRole.ANSWER), fallback_llm(settings, LLMRole.ANSWER)]
    )
    return AskAgent(
        PydanticAIStructuredLLM(partial(build_model, settings)),
        composer,
        KnowledgeToolAdapter(retrieve, store),
        InventoryToolAdapter(ask_inventory),
        CalculatorToolAdapter(store),
        EscalationToolAdapter(escalations, crm),
        ContextCardToolAdapter(crm),
        min_confidence=settings.agent_min_confidence,
    )
