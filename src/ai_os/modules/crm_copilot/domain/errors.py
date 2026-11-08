class LeadNotFound(Exception):
    def __init__(self, lead_id: str) -> None:
        super().__init__(f"Lead {lead_id} was not found")


class InteractionNotFound(Exception):
    def __init__(self, interaction_id: str) -> None:
        super().__init__(f"Interaction {interaction_id} was not found")
