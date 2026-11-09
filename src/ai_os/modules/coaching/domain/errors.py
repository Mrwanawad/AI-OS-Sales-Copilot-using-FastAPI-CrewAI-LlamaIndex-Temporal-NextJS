class SessionNotFound(Exception):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id} was not found")


class PersonaNotFound(Exception):
    def __init__(self, persona_id: str) -> None:
        super().__init__(f"Persona {persona_id} was not found")


class SessionClosed(Exception):
    """The session is finished or has reached its turn limit."""


class SessionChanged(Exception):
    """Another request updated the session first."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id} changed during this request; reload and retry")


class NothingToScore(Exception):
    """The rep has not said anything yet."""


class LLMUnavailable(Exception):
    """Every configured provider failed."""
