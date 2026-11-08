import pytest

from ai_os.shared.config import Settings
from ai_os.shared.llm.crewai_adapter import fallback_llm, primary_llm
from ai_os.shared.llm.port import LLMRole


def test_primary_is_anthropic_and_fallback_is_openai() -> None:
    settings = Settings(_env_file=None, anthropic_api_key="a", openai_api_key="o")

    primary = primary_llm(settings, LLMRole.ROUTER)
    fallback = fallback_llm(settings, LLMRole.ROUTER)

    assert primary.model == settings.anthropic_router_model
    assert type(primary).__name__ == "AnthropicCompletion"
    assert fallback.model == settings.openai_router_model
    assert type(fallback).__name__ == "OpenAICompletion"


def test_missing_key_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        primary_llm(settings, LLMRole.ROUTER)
