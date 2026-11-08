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
