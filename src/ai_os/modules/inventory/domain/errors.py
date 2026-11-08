class UnsafeSql(Exception):
    """Generated SQL broke a guard rule. The reason is safe to feed back to the model."""


class UnanswerableQuestion(Exception):
    def __init__(self, question: str, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Could not answer {question!r}: {reason}")


class InvalidCalculation(ValueError):
    pass
