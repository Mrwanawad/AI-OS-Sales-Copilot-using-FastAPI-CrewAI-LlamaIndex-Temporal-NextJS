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


def test_gemini_provider_uses_one_gemini_model_everywhere() -> None:
    from ai_os.shared.llm.crewai_adapter import crew_llms
    from ai_os.shared.llm.pydantic_ai_adapter import build_model

    settings = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="g")

    llms = crew_llms(settings, LLMRole.ANSWER)
    model = build_model(settings, LLMRole.ROUTER)

    assert [llm.model for llm in llms] == [settings.gemini_model]
    assert model.model_name == settings.gemini_model
    settings.require_llm_keys()


def test_default_provider_needs_both_paid_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from ai_os.shared.llm.crewai_adapter import crew_llms

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings(_env_file=None, anthropic_api_key="a")

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        settings.require_llm_keys()
    with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
        Settings(_env_file=None, llm_provider="gemini").require_llm_keys()
    assert (
        len(
            crew_llms(
                Settings(_env_file=None, anthropic_api_key="a", openai_api_key="o"), LLMRole.ANSWER
            )
        )
        == 2
    )
