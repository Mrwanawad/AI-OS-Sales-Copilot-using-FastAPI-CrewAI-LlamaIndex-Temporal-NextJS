from collections.abc import Mapping, Sequence

from ai_os.modules.crm_copilot.domain.models import Interaction
from ai_os.shared.numbers import numbers_in


def source_text(interaction: Interaction) -> str:
    return f"{interaction.occurred_at:%Y-%m-%d %H:%M} {interaction.text}"


def is_supported(
    text: str, source_ids: Sequence[str], interactions: Mapping[str, Interaction]
) -> bool:
    """A claim needs real sources, and each of its numbers must appear in them."""
    if not source_ids or any(source_id not in interactions for source_id in source_ids):
        return False
    allowed = set().union(*(numbers_in(source_text(interactions[i])) for i in source_ids))
    return numbers_in(text) <= allowed
