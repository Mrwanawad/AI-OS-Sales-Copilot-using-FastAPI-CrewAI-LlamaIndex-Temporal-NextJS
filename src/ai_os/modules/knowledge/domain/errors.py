class StaleDocumentVersion(Exception):
    def __init__(self, document_id: str, version: int, active_version: int) -> None:
        super().__init__(
            f"{document_id} version {version} is older than active version {active_version}"
        )


class EmptyDocument(Exception):
    def __init__(self, document_id: str) -> None:
        super().__init__(f"{document_id} produced no text; keeping the active version")


class InvalidPriceSheet(Exception):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))
