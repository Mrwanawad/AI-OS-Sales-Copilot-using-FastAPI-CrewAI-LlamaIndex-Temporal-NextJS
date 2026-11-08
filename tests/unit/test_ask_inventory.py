from collections.abc import Sequence

import pytest
from pydantic_ai.models.test import TestModel

from ai_os.modules.inventory.adapters.outbound.sql_guard import SqlGlotGuard
from ai_os.modules.inventory.application.ask import AskInventory, GeneratedSql
from ai_os.modules.inventory.domain.errors import UnanswerableQuestion
from ai_os.modules.inventory.domain.models import Unit
from ai_os.shared.llm.port import LLMRole, OutputT
from ai_os.shared.llm.pydantic_ai_adapter import PydanticAIStructuredLLM


class ScriptedLLM:
    def __init__(self, *sql: str) -> None:
        self.sql = list(sql)
        self.prompts: list[str] = []

    async def extract(
        self, role: LLMRole, instructions: str, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        self.prompts.append(prompt)
        return output_type.model_validate({"sql": self.sql.pop(0)})


class RecordingStore:
    def __init__(self) -> None:
        self.executed: list[str] = []

    async def get_by_codes(self, codes: Sequence[str]) -> list[Unit]:
        raise NotImplementedError

    async def find_available(self, max_price, bedrooms, project, limit) -> list[Unit]:
        raise NotImplementedError

    async def upsert(self, units: Sequence[Unit]) -> None:
        raise NotImplementedError

    async def run_read_only(self, sql: str) -> list[dict[str, object]]:
        self.executed.append(sql)
        return [{"code": "A-101"}]


async def test_runs_validated_sql_and_returns_rows() -> None:
    store = RecordingStore()
    ask = AskInventory(ScriptedLLM("SELECT code FROM units"), SqlGlotGuard(), store)

    answer = await ask("Which units are available?")

    assert answer.rows == [{"code": "A-101"}]
    assert answer.sql == "SELECT code FROM units LIMIT 50"
    assert store.executed == [answer.sql]


async def test_rejected_sql_never_reaches_store_and_retries_with_reason() -> None:
    store = RecordingStore()
    llm = ScriptedLLM("SELECT passwd FROM pg_shadow", "SELECT code FROM units")

    await AskInventory(llm, SqlGlotGuard(), store)("Show me passwords")

    assert len(llm.prompts) == 2
    assert "only the units table is allowed" in llm.prompts[1]
    assert store.executed == ["SELECT code FROM units LIMIT 50"]


async def test_gives_up_after_repeated_unsafe_sql_without_touching_store() -> None:
    store = RecordingStore()
    llm = ScriptedLLM("DROP TABLE units", "DELETE FROM units")

    with pytest.raises(UnanswerableQuestion):
        await AskInventory(llm, SqlGlotGuard(), store)("Delete everything")

    assert store.executed == []


async def test_works_through_the_real_pydantic_ai_adapter() -> None:
    store = RecordingStore()
    llm = PydanticAIStructuredLLM(
        lambda _role: TestModel(custom_output_args=GeneratedSql(sql="SELECT code FROM units"))
    )

    answer = await AskInventory(llm, SqlGlotGuard(), store)("Which units?")

    assert answer.rows == [{"code": "A-101"}]
