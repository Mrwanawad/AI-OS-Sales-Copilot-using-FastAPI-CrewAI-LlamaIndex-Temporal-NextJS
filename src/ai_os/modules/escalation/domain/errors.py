class TicketNotFound(Exception):
    def __init__(self, ticket_id: str) -> None:
        super().__init__(f"Ticket {ticket_id} was not found")


class CandidateNotFound(Exception):
    def __init__(self, candidate_id: str) -> None:
        super().__init__(f"Knowledge candidate {candidate_id} was not found")


class InvalidTransition(Exception):
    """The item is not in a state that allows this, possibly because someone just acted on it."""


class SameReviewer(Exception):
    def __init__(self) -> None:
        super().__init__("A different person must review an answer before it reaches every rep")


class ContainsPersonalData(Exception):
    def __init__(self, kinds: list[str]) -> None:
        self.kinds = kinds
        super().__init__(f"Remove personal data before sharing: {', '.join(kinds)}")


class NotShareable(Exception):
    """The answer can't become shared knowledge as it stands."""
