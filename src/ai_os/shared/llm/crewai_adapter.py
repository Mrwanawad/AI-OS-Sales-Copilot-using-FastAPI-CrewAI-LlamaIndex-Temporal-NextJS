from crewai import LLM

from ai_os.shared.config import Settings
from ai_os.shared.llm.port import LLMRole


def primary_llm(settings: Settings, role: LLMRole) -> LLM:
    return LLM(
        model=f"anthropic/{settings.route_for(role).anthropic}",
        api_key=settings.require_anthropic_key(),
    )


def crew_llms(settings: Settings, role: LLMRole) -> list[LLM]:
    """The models a crew tries in order."""
    if settings.llm_provider == "gemini":
        return [LLM(model=f"gemini/{settings.gemini_model}", api_key=settings.require_gemini_key())]
    return [primary_llm(settings, role), fallback_llm(settings, role)]


def fallback_llm(settings: Settings, role: LLMRole) -> LLM:
    return LLM(
        model=f"openai/{settings.route_for(role).openai}",
        api_key=settings.require_openai_key(),
    )
