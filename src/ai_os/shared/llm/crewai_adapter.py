from crewai import LLM

from ai_os.shared.config import Settings
from ai_os.shared.llm.port import LLMRole


def primary_llm(settings: Settings, role: LLMRole) -> LLM:
    return LLM(
        model=f"anthropic/{settings.route_for(role).anthropic}",
        api_key=settings.require_anthropic_key(),
    )


def fallback_llm(settings: Settings, role: LLMRole) -> LLM:
    return LLM(
        model=f"openai/{settings.route_for(role).openai}",
        api_key=settings.require_openai_key(),
    )
