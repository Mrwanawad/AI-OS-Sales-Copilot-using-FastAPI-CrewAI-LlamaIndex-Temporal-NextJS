import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ai_os.bootstrap.app import create_app, get_ask_agent
from ai_os.modules.agent.domain.errors import LLMUnavailable
from ai_os.shared.config import MissingApiKey
from ai_os.shared.llm.port import LLMProviderError, LLMRole
from ai_os.shared.llm.pydantic_ai_adapter import PydanticAIStructuredLLM


class Output(BaseModel):
    value: str


def failing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=500, model_name="fake", body="provider down")


@pytest.mark.parametrize(
    "model",
    [FunctionModel(failing), FallbackModel(FunctionModel(failing), FunctionModel(failing))],
)
async def test_provider_failures_become_one_error_type(model: object) -> None:
    llm = PydanticAIStructuredLLM(lambda _role: model)

    with pytest.raises(LLMProviderError):
        await llm.extract(LLMRole.ROUTER, "Do it", "Prompt", Output)


@pytest.mark.parametrize(
    "error",
    [
        LLMProviderError("down"),
        LLMUnavailable("down"),
        MissingApiKey("ANTHROPIC_API_KEY is not set"),
    ],
)
def test_app_reports_llm_problems_as_503(error: Exception) -> None:
    class Failing:
        async def __call__(self, *args: object) -> None:
            raise error

    app = create_app()
    app.dependency_overrides[get_ask_agent] = Failing

    response = TestClient(app).post("/agent/ask", json={"question": "Hello?"})

    assert response.status_code == 503
