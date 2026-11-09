"""Egypt PDPL (Law 151 of 2020): erase a client on request, and keep personal data no longer
than needed. The retention periods are settings, for the brokerage's legal team to set."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from ai_os.modules.crm_copilot.domain.errors import LeadNotFound
from ai_os.modules.crm_copilot.ports.crm import (
    LinkedPersonalData,
    PersonalDataStore,
    RecordingEraser,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionPeriods:
    # Interaction text and call recordings.
    interaction_content_days: int
    # Context cards and handovers of leads with no recent interaction.
    dormant_card_days: int
    # Closed escalation tickets, which can quote a client's card.
    closed_ticket_days: int


@dataclass(frozen=True)
class RetentionReport:
    interactions_cleared: int
    recordings_deleted: int
    cards_deleted: int
    tickets_deleted: int


@dataclass(frozen=True)
class ErasureReport:
    lead_id: str
    recordings_deleted: int
    linked_records_deleted: int


class PersonalDataLifecycle:
    def __init__(
        self,
        store: PersonalDataStore,
        recordings: RecordingEraser,
        linked: Sequence[LinkedPersonalData],
        periods: RetentionPeriods,
        now: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._recordings = recordings
        self._linked = linked
        self._periods = periods
        self._now = now

    async def erase_lead(self, lead_id: str, requested_by: str) -> ErasureReport:
        """The client's right to erasure. The source CRM and any dialer copies must be erased
        there too; this covers what the AI OS holds."""
        references = await self._store.erase_lead(lead_id)
        if references is None:
            raise LeadNotFound(lead_id)
        for reference in references:
            await self._recordings.delete(reference)
        linked = 0
        for holder in self._linked:
            linked += await holder.erase_lead(lead_id)
        # Ids only: the audit trail must not keep what was just erased.
        logger.info("Lead %s erased at the request of %s", lead_id, requested_by)
        return ErasureReport(lead_id, len(references), linked)

    async def apply_retention(self) -> RetentionReport:
        now = self._now()
        periods = self._periods
        cleared, references = await self._store.purge_interaction_content(
            now - timedelta(days=periods.interaction_content_days)
        )
        for reference in references:
            await self._recordings.delete(reference)
        cards = await self._store.purge_dormant_cards(
            now - timedelta(days=periods.dormant_card_days)
        )
        tickets = 0
        for holder in self._linked:
            tickets += await holder.purge(now - timedelta(days=periods.closed_ticket_days))
        return RetentionReport(cleared, len(references), cards, tickets)
