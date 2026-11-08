from collections.abc import Callable

from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelAPIError,
    UnexpectedModelBehavior,
)
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from ai_os.shared.config import Settings
from ai_os.shared.llm.port import LLMProviderError, LLMRole, OutputT


def build_model(settings: Settings, role: LLMRole) -> Model:
    route = settings.route_for(role)
    return FallbackModel(
        AnthropicModel(
            route.anthropic,
            provider=AnthropicProvider(api_key=settings.require_anthropic_key()),
        ),
        OpenAIChatModel(
            route.openai,
            provider=OpenAIProvider(api_key=settings.require_openai_key()),
        ),
    )


class PydanticAIStructuredLLM:
    def __init__(self, model_for: Callable[[LLMRole], Model]) -> None:
        self._model_for = model_for

    async def extract(
        self, role: LLMRole, instructions: str, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        agent = Agent(self._model_for(role), output_type=output_type, instructions=instructions)
        try:
            result = await agent.run(prompt)
        except (ModelAPIError, FallbackExceptionGroup, UnexpectedModelBehavior) as error:
            raise LLMProviderError(str(error)) from error
        return result.output
