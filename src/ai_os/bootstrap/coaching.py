from datetime import UTC, datetime
from functools import partial

from sqlalchemy.ext.asyncio import create_async_engine

import ai_os.bootstrap.crewai_env  # noqa: F401
from ai_os.modules.coaching.adapters.outbound.postgres_store import PostgresSessionStore
from ai_os.modules.coaching.application.buyer import CrewBuyer
from ai_os.modules.coaching.application.service import Coaching
from ai_os.modules.coaching.ports.coaching import SpeechSynthesizer
from ai_os.shared.config import MissingApiKey, Settings
from ai_os.shared.llm.crewai_adapter import crew_llms
from ai_os.shared.llm.port import LLMRole
from ai_os.shared.llm.pydantic_ai_adapter import PydanticAIStructuredLLM, build_model


def _speech(settings: Settings) -> SpeechSynthesizer:
    from ai_os.modules.coaching.adapters.outbound.elevenlabs_speech import ElevenLabsSpeech

    if not settings.elevenlabs_api_key or not settings.elevenlabs_voice_id:
        raise MissingApiKey("ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID are not set")
    return ElevenLabsSpeech(
        settings.elevenlabs_api_key.get_secret_value(), settings.elevenlabs_voice_id
    )


def build_coaching(settings: Settings) -> Coaching:
    if not settings.async_database_url:
        raise ValueError("Set ASYNC_DATABASE_URL")

    def buyer_llms() -> list:
        return crew_llms(settings, LLMRole.EXTRACTION)

    return Coaching(
        PostgresSessionStore(create_async_engine(settings.async_database_url)),
        CrewBuyer(buyer_llms),
        PydanticAIStructuredLLM(partial(build_model, settings)),
        partial(_speech, settings),
        lambda: datetime.now(UTC),
    )
