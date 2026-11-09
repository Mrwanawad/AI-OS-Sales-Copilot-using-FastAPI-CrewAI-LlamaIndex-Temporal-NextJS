import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

from ai_os.bootstrap.observability import configure_tracing, instrumentation_settings
from ai_os.shared.config import Settings
from ai_os.shared.llm.crew_tracing import traced_kickoff

SECRET_QUESTION = "Client Mona on 01012345678 wants a refund"


class Output(BaseModel):
    value: str


async def spans_for(settings: InstrumentationSettings, exporter: InMemorySpanExporter) -> str:
    agent = Agent(TestModel(custom_output_args={"value": "ok"}), output_type=Output)
    agent.instrument = settings
    await agent.run(SECRET_QUESTION)
    return " ".join(str(dict(span.attributes)) for span in exporter.get_finished_spans())


def exporter_and_provider() -> tuple[InMemorySpanExporter, TracerProvider]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


@pytest.mark.parametrize(("capture", "leaks"), [(False, False), (True, True)])
async def test_prompts_stay_out_of_traces_unless_turned_on(capture: bool, leaks: bool) -> None:
    exporter, provider = exporter_and_provider()
    base = instrumentation_settings(capture)
    settings = InstrumentationSettings(tracer_provider=provider, include_content=capture)
    assert base.include_content is capture

    captured = await spans_for(settings, exporter)

    assert captured, "no spans were recorded"
    assert ("01012345678" in captured) is leaks


def test_tracing_stays_off_without_keys() -> None:
    assert configure_tracing(Settings(_env_file=None)) is False


class FakeUsage:
    prompt_tokens = 120
    completion_tokens = 30
    cached_prompt_tokens = 0
    successful_requests = 1


class FakeCrewOutput:
    token_usage = FakeUsage()
    raw = "secret reply"


class FakeCrew:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    async def akickoff(self, inputs):
        if self.error:
            raise self.error
        return FakeCrewOutput()


async def test_crew_runs_are_traced_with_usage_and_no_content() -> None:
    exporter, provider = exporter_and_provider()

    result = await traced_kickoff(
        FakeCrew(),
        {"question": SECRET_QUESTION},
        "composer",
        "claude-opus-5-5",
        provider.get_tracer("test"),
    )

    assert result.raw == "secret reply"
    [span] = exporter.get_finished_spans()
    attributes = dict(span.attributes)
    assert attributes["gen_ai.request.model"] == "claude-opus-5-5"
    assert attributes["gen_ai.system"] == "anthropic"
    assert attributes["gen_ai.usage.input_tokens"] == 120
    assert attributes["gen_ai.usage.output_tokens"] == 30
    assert "Mona" not in str(attributes) and "secret reply" not in str(attributes)


async def test_failed_crew_run_is_recorded_as_an_error() -> None:
    exporter, provider = exporter_and_provider()

    with pytest.raises(ConnectionError):
        await traced_kickoff(
            FakeCrew(ConnectionError("down")), {}, "buyer", "gpt-5.5", provider.get_tracer("t")
        )

    [span] = exporter.get_finished_spans()
    assert span.status.status_code.name == "ERROR"
    assert dict(span.attributes)["gen_ai.system"] == "openai"
