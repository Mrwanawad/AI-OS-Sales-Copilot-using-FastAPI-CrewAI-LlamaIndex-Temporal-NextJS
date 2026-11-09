import logging
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.modules.escalation.adapters.outbound.postgres_store import PostgresEscalationStore
from ai_os.modules.escalation.adapters.outbound.webhook_notifier import WebhookTicketNotifier
from ai_os.modules.escalation.application.service import Escalations
from ai_os.modules.escalation.domain.models import KnowledgeCandidate, Ticket
from ai_os.modules.escalation.ports.escalation import TicketNotifier
from ai_os.modules.knowledge.application.approved_answers import ApprovedAnswers
from ai_os.shared.config import Settings

logger = logging.getLogger(__name__)


class KnowledgePublisherAdapter:
    def __init__(self, get_answers) -> None:
        # A factory, because the knowledge store loads model weights on creation.
        self._get_answers = get_answers

    async def publish(self, candidate: KnowledgeCandidate) -> None:
        await self._get_answers().publish(
            _document_id(candidate), candidate.project, candidate.question, candidate.answer
        )

    async def remove(self, candidate: KnowledgeCandidate) -> None:
        await self._get_answers().remove(_document_id(candidate))


class QueueOnlyNotifier:
    """Team leads read the open queue; set ESCALATION_WEBHOOK_URL to push tickets instead."""

    async def ticket_opened(self, ticket: Ticket) -> None:
        # The id only: questions can contain client details, so they stay out of logs.
        logger.info("Escalation ticket %s opened", ticket.id)

    async def ticket_answered(self, ticket: Ticket) -> None:
        logger.info("Escalation ticket %s answered", ticket.id)


def _document_id(candidate: KnowledgeCandidate) -> str:
    return f"approved-answer-{candidate.id}"


@lru_cache
def _approved_answers() -> ApprovedAnswers:
    from ai_os.bootstrap.knowledge import build_store

    return ApprovedAnswers(build_store(Settings()))


def build_notifier(settings: Settings) -> TicketNotifier:
    if settings.escalation_webhook_url is None:
        return QueueOnlyNotifier()
    if settings.escalation_webhook_secret is None:
        raise ValueError("ESCALATION_WEBHOOK_URL needs ESCALATION_WEBHOOK_SECRET")
    return WebhookTicketNotifier(
        settings.escalation_webhook_url,
        settings.escalation_webhook_secret.get_secret_value(),
        settings.app_url,
    )


def build_escalations(settings: Settings) -> Escalations:
    if not settings.async_database_url:
        raise ValueError("Set ASYNC_DATABASE_URL")
    return Escalations(
        PostgresEscalationStore(create_async_engine(settings.async_database_url)),
        KnowledgePublisherAdapter(_approved_answers),
        build_notifier(settings),
        lambda: datetime.now(UTC),
        settings.approved_answer_valid_days,
    )
