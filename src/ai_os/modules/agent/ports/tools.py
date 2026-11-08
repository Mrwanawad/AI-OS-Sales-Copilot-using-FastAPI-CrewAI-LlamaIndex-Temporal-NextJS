from typing import Protocol

from ai_os.modules.agent.domain.models import (
    AgentAnswer,
    EvidenceItem,
    InstallmentQuery,
    MortgageQuery,
)


class KnowledgeTool(Protocol):
    async def search(self, query: str, project: str | None) -> list[EvidenceItem]: ...


class InventoryTool(Protocol):
    async def ask(self, question: str) -> list[EvidenceItem]: ...


class CalculatorTool(Protocol):
    async def installments(self, query: InstallmentQuery) -> EvidenceItem: ...

    async def mortgage(self, query: MortgageQuery) -> EvidenceItem: ...


class ContextCardTool(Protocol):
    async def card_evidence(self, lead_id: str) -> EvidenceItem | None: ...


class EscalationTool(Protocol):
    async def create_ticket(
        self,
        question: str,
        reason: str,
        project: str | None,
        lead_id: str | None,
        asked_by: str | None,
    ) -> str: ...


class AnswerQuestion(Protocol):
    async def __call__(
        self,
        question: str,
        project: str | None = None,
        lead_id: str | None = None,
        asked_by: str | None = None,
    ) -> AgentAnswer: ...
