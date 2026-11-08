from pydantic import BaseModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from ai_os.shared.config import Settings
from ai_os.shared.llm.port import LLMRole
from ai_os.shared.llm.pydantic_ai_adapter import PydanticAIStructuredLLM, build_model


class HandoverSummary(BaseModel):
    stated_needs: str


def test_build_model_falls_back_from_anthropic_to_openai() -> None:
    settings = Settings(_env_file=None, anthropic_api_key="a", openai_api_key="o")

    model = build_model(settings, LLMRole.ANSWER)

    assert isinstance(model, FallbackModel)
    primary, fallback = model.models
    assert isinstance(primary, AnthropicModel)
    assert primary.model_name == settings.anthropic_answer_model
    assert isinstance(fallback, OpenAIChatModel)
    assert fallback.model_name == settings.openai_answer_model


async def test_returns_schema_validated_output_for_role() -> None:
    seen_roles: list[LLMRole] = []

    def model_for(role: LLMRole) -> TestModel:
        seen_roles.append(role)
        return TestModel(custom_output_args={"stated_needs": "3-bedroom, delivery 2027"})

    llm = PydanticAIStructuredLLM(model_for)

    result = await llm.extract(
        LLMRole.EXTRACTION, "Summarize the lead.", "Transcript", HandoverSummary
    )

    assert result == HandoverSummary(stated_needs="3-bedroom, delivery 2027")
    assert seen_roles == [LLMRole.EXTRACTION]
