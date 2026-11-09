"""OpenTelemetry spans for CrewAI runs, so composer and buyer calls reach Langfuse too.

PydanticAI instruments its own calls; CrewAI's are invisible without this, which left cost per
answer incomplete. Spans follow the OpenTelemetry GenAI conventions, which Langfuse reads as a
generation with model and token usage, and so prices. Prompts and replies are never attached:
they can hold client details. Without a configured tracer provider the spans are no-ops.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from opentelemetry import trace

_TRACER_NAME = "ai_os.crewai"


class _Usage(Protocol):
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    successful_requests: int


class _Output(Protocol):
    token_usage: _Usage


class _Crew(Protocol):
    async def akickoff(self, inputs: Mapping[str, Any]) -> Any: ...


async def traced_kickoff(
    crew: _Crew,
    inputs: Mapping[str, Any],
    name: str,
    model: str,
    tracer: trace.Tracer | None = None,
) -> Any:
    """Runs the crew inside a GenAI span named after the caller, such as "composer"."""
    tracer = tracer or trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(f"{name} {model}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.system", _provider(model))
        span.set_attribute("ai_os.crew", name)
        result = await crew.akickoff(inputs=inputs)
        usage = getattr(result, "token_usage", None)
        if usage is not None:
            span.set_attribute("gen_ai.response.model", model)
            span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            span.set_attribute("gen_ai.usage.cache_read_input_tokens", usage.cached_prompt_tokens)
            span.set_attribute("ai_os.llm_requests", usage.successful_requests)
        return result


def _provider(model: str) -> str:
    lowered = model.lower()
    for prefix, provider in (("claude", "anthropic"), ("gpt", "openai"), ("gemini", "gemini")):
        if prefix in lowered:
            return provider
    return lowered.split("/", 1)[0] if "/" in lowered else "unknown"
