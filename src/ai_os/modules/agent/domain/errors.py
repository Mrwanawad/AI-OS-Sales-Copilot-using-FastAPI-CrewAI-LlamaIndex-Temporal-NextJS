class ToolCouldNotAnswer(Exception):
    """A tool ran correctly but has no answer, so the question goes to a person."""


class LLMUnavailable(Exception):
    """Every configured provider failed."""
