import logging
import os

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from ai_os.shared.config import Settings

logger = logging.getLogger(__name__)


def instrumentation_settings(capture_content: bool) -> InstrumentationSettings:
    return InstrumentationSettings(include_content=capture_content)


def configure_tracing(settings: Settings) -> bool:
    """Sends latency, tokens and cost to Langfuse when it's configured. Returns whether it is.

    Prompts and answers stay out of traces unless LANGFUSE_CAPTURE_CONTENT is on, because
    they can hold client details and a hosted Langfuse is a third party.
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return False
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key.get_secret_value()
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host

    from langfuse import get_client

    get_client()
    Agent.instrument_all(instrumentation_settings(settings.langfuse_capture_content))
    logger.info("Langfuse tracing is on")
    return True
