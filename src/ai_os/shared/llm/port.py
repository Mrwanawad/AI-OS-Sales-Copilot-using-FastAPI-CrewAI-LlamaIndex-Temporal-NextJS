from enum import StrEnum
from typing import Protocol, TypeVar

from pydantic import BaseModel

OutputT = TypeVar("OutputT", bound=BaseModel)


class LLMProviderError(Exception):
    """Every configured provider failed. Retrying later may work."""


class LLMRole(StrEnum):
    ROUTER = "router"
    ANSWER = "answer"
    EXTRACTION = "extraction"


class StructuredLLM(Protocol):
    async def extract(
        self, role: LLMRole, instructions: str, prompt: str, output_type: type[OutputT]
    ) -> OutputT: ...
