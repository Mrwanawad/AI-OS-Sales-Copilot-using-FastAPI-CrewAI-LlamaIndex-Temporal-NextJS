from ai_os.modules.crm_copilot.ports.crm import CrmStore
from ai_os.shared.auth import Principal


async def can_access_lead(store: CrmStore, user: Principal, lead_id: str) -> bool:
    """Reps see their own leads, which also keeps unknown ids from revealing what exists."""
    if user.leads_a_team:
        return True
    lead = await store.get_lead(lead_id)
    return lead is not None and lead.assigned_rep == user.subject
