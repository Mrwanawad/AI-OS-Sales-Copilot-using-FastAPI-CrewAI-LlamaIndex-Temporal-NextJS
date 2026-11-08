"""End-to-end wiring check: real API and Temporal worker processes, real Postgres, real models.

Usage: python scripts/smoke_e2e.py [--vector-store qdrant|pgvector]
Qdrant mode needs a running server at QDRANT_URL. LLM calls use a scripted model, since this
check needs no API keys. Small models are used by default; override with EMBEDDING_MODEL,
EMBEDDING_DIM and RERANKER_MODEL to run the production ones.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import httpx  # noqa: E402
import pandas as pd
import pgserver
from crewai.llms.base_llm import BaseLLM
from fastmcp import Client as McpClient
from pydantic_ai.models.test import TestModel
from sqlalchemy import make_url, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from temporalio.client import WorkflowFailureError
from temporalio.testing import WorkflowEnvironment

from ai_os.bootstrap.agent import (
    CalculatorToolAdapter,
    ContextCardToolAdapter,
    EscalationToolAdapter,
    InventoryToolAdapter,
    KnowledgeToolAdapter,
)
from ai_os.bootstrap.crm import InventoryUnitMatcher
from ai_os.bootstrap.escalation import QueueOnlyNotifier
from ai_os.bootstrap.knowledge import build_retrieve
from ai_os.mcp_servers.inventory import create_inventory_server
from ai_os.modules.agent.application.composer import CrewAnswerComposer
from ai_os.modules.agent.application.flow import (
    AskAgent,
    CalculationKind,
    CalculationParams,
    RoutingDecision,
)
from ai_os.modules.agent.domain.models import AnswerStatus, Intent, SourceKind
from ai_os.modules.best_time.adapters.outbound.synthetic import generate_calls
from ai_os.modules.best_time.adapters.outbound.trainer import train_best_time
from ai_os.modules.coaching.adapters.outbound.postgres_store import PostgresSessionStore
from ai_os.modules.coaching.application.buyer import CrewBuyer
from ai_os.modules.coaching.application.service import Coaching, CriterionDraft, ScoreDraft
from ai_os.modules.coaching.domain.scoring import RUBRIC
from ai_os.modules.crm_copilot.adapters.inbound.api import sign
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.modules.crm_copilot.application.build_card import (
    BuildContextCard,
    CardDraft,
    PreferencesDraft,
    SourcedDraft,
)
from ai_os.modules.crm_copilot.domain.models import (
    Interaction,
    InteractionKind,
    Lead,
    LeadStage,
)
from ai_os.modules.escalation.adapters.outbound.postgres_store import PostgresEscalationStore
from ai_os.modules.escalation.application.service import Escalations
from ai_os.modules.inventory.adapters.outbound.postgres_store import PostgresUnitStore
from ai_os.modules.inventory.adapters.outbound.sql_guard import SqlGlotGuard
from ai_os.modules.inventory.application.ask import AskInventory, GeneratedSql
from ai_os.modules.inventory.domain.errors import UnanswerableQuestion
from ai_os.modules.inventory.domain.models import Unit, UnitStatus
from ai_os.shared.auth import Role, issue_token
from ai_os.shared.config import Settings
from ai_os.shared.llm.port import LLMRole, OutputT
from ai_os.shared.llm.pydantic_ai_adapter import PydanticAIStructuredLLM
from ai_os.workflows.ingest_document import IngestDocumentWorkflow, IngestRequest

ROOT = Path(__file__).parents[1]
WEBHOOK_SECRET = "smoke-webhook-secret"
AUTH_SECRET = "smoke-auth-secret-long-enough-for-hs256"
NO_TOKEN = {"Authorization": ""}
WHEN = datetime(2026, 11, 1, 10, 0, tzinfo=UTC)
PORT = 8765
API = f"http://127.0.0.1:{PORT}"
TASK_QUEUE = f"smoke-{uuid4().hex}"
failures: list[str] = []


def bearer(subject: str, role: Role) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(AUTH_SECRET, subject, role)}"}


def check(name: str, condition: object, detail: object = "") -> None:
    print(
        f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  -> {detail}" if not condition else "")
    )
    if not condition:
        failures.append(name)


def write_sheet(path: Path, prices: dict[str, object]) -> str:
    frame = pd.DataFrame({"Unit Code": list(prices), "Price": list(prices.values())})
    frame.to_excel(path, index=False)
    return str(path)


async def setup_database(async_url: str) -> str:
    admin = create_async_engine(async_url, poolclass=NullPool)
    await PostgresUnitStore(admin, admin).create_schema()
    async with admin.begin() as connection:
        for statement in (ROOT / "deploy" / "inventory_reader.sql").read_text().split(";"):
            lines = [line for line in statement.splitlines() if not line.startswith("--")]
            if "".join(lines).strip():
                await connection.execute(text("\n".join(lines)))
    await PostgresUnitStore(admin, admin).upsert(
        [
            Unit(
                "A-101",
                "Palm Hills",
                "apartment",
                3,
                Decimal("150"),
                Decimal("5000000"),
                UnitStatus.AVAILABLE,
            ),
            Unit(
                "A-102",
                "Palm Hills",
                "villa",
                4,
                Decimal("300"),
                Decimal("9000000"),
                UnitStatus.SOLD,
            ),
        ]
    )
    await admin.dispose()
    return make_url(async_url).set(username="inventory_reader").render_as_string(False)


async def wait_for_health(client: httpx.AsyncClient, process: subprocess.Popen) -> float:
    started = time.monotonic()
    while time.monotonic() - started < 240:
        if process.poll() is not None:
            raise RuntimeError(f"API exited with code {process.returncode}")
        try:
            if (await client.get("/health")).status_code == 200:
                return time.monotonic() - started
        except httpx.TransportError:
            pass
        await asyncio.sleep(1)
    raise TimeoutError("API did not become healthy")


async def search(client: httpx.AsyncClient, **body: object) -> list[dict]:
    response = await client.post("/knowledge/search", json=body)
    response.raise_for_status()
    return response.json()


async def expect_workflow_failure(run: Callable, kind: str, name: str) -> None:
    try:
        await run()
    except WorkflowFailureError as error:
        cause = error.cause.cause
        check(name, getattr(cause, "type", None) == kind, repr(cause))
    else:
        check(name, False, "workflow succeeded")


async def scenario(env: WorkflowEnvironment, client: httpx.AsyncClient, tmp: Path) -> None:
    async def ingest(path: str, document_id: str, kind: str, version: int, project: str = "PH"):
        return await env.client.execute_workflow(
            IngestDocumentWorkflow.run,
            IngestRequest(path, document_id, project, kind, version),
            id=str(uuid4()),
            task_queue=TASK_QUEUE,
        )

    print("\n-- Inventory over HTTP")
    response = await client.get("/inventory/units/A-101")
    check("unit lookup reads Postgres", Decimal(response.json()["price_egp"]) == 5000000, response)
    check("unknown unit is 404", (await client.get("/inventory/units/nope")).status_code == 404)
    response = await client.post(
        "/inventory/calculate/installments",
        json={"price": "5000000", "down_payment_pct": "10", "years": 8},
    )
    check("installments", Decimal(response.json()["installment"]) == 46875, response.text)
    response = await client.post(
        "/inventory/calculate/discount",
        json={"price": "5000000", "requested_pct": "4", "role": "rep"},
    )
    check("discount uses settings", response.json()["approver"] == "team_lead", response.text)
    response = await client.post("/inventory/ask", json={"question": "Which units?"})
    check("ask without keys is a clean 503", response.status_code == 503, response.text)
    started = time.monotonic()
    response = await client.post("/agent/ask", json={"question": "Which units?"})
    check("agent without keys is a clean 503", response.status_code == 503, response.text)
    check("key check does not load models", time.monotonic() - started < 15)

    print("\n-- Ingestion through Temporal, search over HTTP")
    v1 = write_sheet(tmp / "v1.xlsx", {"C-777": 5000000, "C-778": 6000000})
    summary = await ingest(v1, "prices", "price_sheet", 1)
    check("price sheet v1 ingested", summary.chunks == 2, summary)
    hits = await search(client, query="C-777", top_k=1)
    check("exact unit code found", hits and "C-777" in hits[0]["text"], hits)
    check("price cited", hits and "5,000,000" in hits[0]["text"], hits)
    check("citation version", hits and hits[0]["citation"]["document_version"] == 1, hits)
    check("citation row number", hits and hits[0]["citation"]["page"] == 2, hits)

    v2 = write_sheet(tmp / "v2.xlsx", {"C-777": 5750000, "C-778": 6000000})
    await ingest(v2, "prices", "price_sheet", 2)
    hits = await search(client, query="C-777 price", top_k=10)
    texts = " ".join(hit["text"] for hit in hits)
    check("new price live", "5,750,000" in texts, texts)
    check("old price retired", "5,000,000" not in texts, texts)

    await expect_workflow_failure(
        lambda: ingest(v1, "prices", "price_sheet", 1),
        "StaleDocumentVersion",
        "older version rejected",
    )
    bad = write_sheet(tmp / "bad.xlsx", {"C-777": -5})
    await expect_workflow_failure(
        lambda: ingest(bad, "prices", "price_sheet", 3),
        "InvalidPriceSheet",
        "bad sheet rejected",
    )
    hits = await search(client, query="C-777 price", top_k=10)
    check(
        "active version survives rejection",
        "5,750,000" in " ".join(h["text"] for h in hits),
        hits,
    )

    brochure = tmp / "brochure.md"
    brochure.write_text(
        "# Mountain View\n\nThe clubhouse has a gym and two swimming pools.\n", encoding="utf-8"
    )
    await ingest(str(brochure), "brochure", "brochure", 1, project="MV")
    hits = await search(client, query="swimming pools gym", top_k=1, project="MV")
    check("Docling document searchable", hits and "swimming" in hits[0]["text"], hits)
    hits = await search(client, query="swimming pools gym", top_k=5, project="PH")
    check("project filter holds", all(hit["project"] == "PH" for hit in hits), hits)
    hits = await search(client, query="gym", top_k=5, kinds=["policy"])
    check("kind filter holds", hits == [], hits)


async def inventory_with_scripted_llm(async_url: str, reader_url: str) -> None:
    print("\n-- Text-to-SQL with a scripted model, real guard, real reader role")
    admin = create_async_engine(async_url, poolclass=NullPool)
    reader = create_async_engine(reader_url, poolclass=NullPool)
    store = PostgresUnitStore(admin, reader)

    def asker(sql: str) -> AskInventory:
        llm = PydanticAIStructuredLLM(
            lambda _role: TestModel(custom_output_args=GeneratedSql(sql=sql))
        )
        return AskInventory(llm, SqlGlotGuard(), store)

    answer = await asker(
        "SELECT code, price_egp FROM units WHERE status = 'available' AND bedrooms = 3"
    )("3-bedroom units?")
    check(
        "safe SQL returns live rows",
        answer.rows == [{"code": "A-101", "price_egp": Decimal("5000000.00")}],
        answer,
    )
    try:
        await asker("DELETE FROM units")("Delete everything")
        check("unsafe SQL stopped", False)
    except UnanswerableQuestion:
        check("unsafe SQL stopped", True)
    check("units intact", len(await store.get_by_codes(["A-101", "A-102"])) == 2)

    print("\n-- MCP server against the real store")
    server = create_inventory_server(lambda: store, lambda: asker("SELECT code FROM units"))
    async with McpClient(server) as mcp:
        units = await mcp.call_tool("get_units", {"codes": ["A-101"]})
        check("MCP get_units", units.data[0]["code"] == "A-101", units.data)
        asked = await mcp.call_tool("ask_inventory", {"question": "Which units?"})
        check("MCP ask_inventory", len(asked.data["rows"]) == 2, asked.data)
    await admin.dispose()
    await reader.dispose()


class ScriptedAgentLLM:
    def __init__(self, intent: Intent, params: CalculationParams | None = None) -> None:
        self.intent = intent
        self.params = params

    async def extract(
        self, role: LLMRole, instructions: str, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        if output_type is RoutingDecision:
            return RoutingDecision(intent=self.intent)
        return self.params


class ScriptedBuyerLLM(BaseLLM):
    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        return "Thought: done\nFinal Answer: Too expensive."


class ScriptedCrewLLM(BaseLLM):
    """Plays the composer: reads the real prompt and cites the matching real evidence."""

    pattern: str = ""
    template: str = ""
    prompts: list[str] = []

    def call(
        self,
        messages,
        tools=None,
        callbacks=None,
        available_functions=None,
        from_task=None,
        from_agent=None,
        response_model=None,
    ) -> str:
        prompt = str(messages)
        self.prompts = [*self.prompts, prompt]
        found = re.search(rf"\[(E\d+)\][^\[]*{self.pattern}", prompt)
        payload = {
            "answer": self.template,
            "cited_evidence": [found.group(1)] if found else [],
            "confidence": 0.9,
        }
        return f"Thought: done\nFinal Answer: {json.dumps(payload)}"


async def agent_with_real_tools(async_url: str, reader_url: str) -> None:
    print("\n-- Agent flow with scripted LLMs and every real tool")
    admin = create_async_engine(async_url, poolclass=NullPool)
    reader = create_async_engine(reader_url, poolclass=NullPool)
    store = PostgresUnitStore(admin, reader)
    # The ingested price sheet says 5,750,000. Live inventory has moved on since.
    await store.upsert(
        [
            Unit(
                "C-777",
                "PH",
                "apartment",
                2,
                Decimal("90"),
                Decimal("5900000"),
                UnitStatus.AVAILABLE,
            )
        ]
    )
    retrieve = build_retrieve(Settings())
    sql = "SELECT code, price_egp FROM units WHERE code = 'A-101'"
    sql_llm = PydanticAIStructuredLLM(
        lambda _role: TestModel(custom_output_args=GeneratedSql(sql=sql))
    )
    knowledge = KnowledgeToolAdapter(retrieve, store)
    inventory = InventoryToolAdapter(AskInventory(sql_llm, SqlGlotGuard(), store))
    calculator = CalculatorToolAdapter(store)

    def agent(llm: ScriptedAgentLLM, crew_llm: ScriptedCrewLLM) -> AskAgent:
        return AskAgent(
            llm,
            CrewAnswerComposer([crew_llm]),
            knowledge,
            inventory,
            calculator,
            EscalationToolAdapter(queue(admin), PostgresCrmStore(admin)),
            ContextCardToolAdapter(PostgresCrmStore(admin)),
        )

    live = "Live inventory for unit C-777"
    crew_llm = ScriptedCrewLLM(model="s", pattern=live, template="C-777 is 5,900,000 EGP.")
    answer = await agent(ScriptedAgentLLM(Intent.KNOWLEDGE), crew_llm)("C-777 price", "PH")
    check("knowledge answer from live row", answer.status is AnswerStatus.ANSWERED, answer)
    check("stale document price reached composer", "5,750,000" in crew_llm.prompts[0])
    check("live price reached composer", "5900000" in crew_llm.prompts[0])
    check(
        "citation points at inventory",
        answer.citations and answer.citations[0].kind is SourceKind.INVENTORY,
        answer,
    )

    crew_llm = ScriptedCrewLLM(model="s", pattern=live, template="C-777 is 4,000,000 EGP.")
    answer = await agent(ScriptedAgentLLM(Intent.KNOWLEDGE), crew_llm)("C-777 price", "PH")
    check("invented price escalates", answer.status is AnswerStatus.ESCALATED, answer)
    check("invented price withheld", "4,000,000" not in answer.text, answer)

    params = CalculationParams(
        kind=CalculationKind.INSTALLMENTS, unit_code="A-101", down_payment_pct=10, years=8
    )
    crew_llm = ScriptedCrewLLM(
        model="s", pattern="Installment plan", template="You pay 46,875.00 monthly."
    )
    answer = await agent(ScriptedAgentLLM(Intent.CALCULATION, params), crew_llm)(
        "Monthly on A-101 with 10% down over 8 years?"
    )
    check("calculation uses live unit price", answer.status is AnswerStatus.ANSWERED, answer)
    check(
        "calculator evidence cited",
        answer.citations and answer.citations[0].kind is SourceKind.CALCULATOR,
        answer,
    )

    crew_llm = ScriptedCrewLLM(
        model="s", pattern="code: A-101", template="A-101 costs 5,000,000 EGP."
    )
    answer = await agent(ScriptedAgentLLM(Intent.INVENTORY), crew_llm)("Price of A-101?")
    check("inventory answer from SQL row", answer.status is AnswerStatus.ANSWERED, answer)

    answer = await agent(ScriptedAgentLLM(Intent.KNOWLEDGE), crew_llm)("Anything?", "NoSuchProject")
    check("no evidence escalates", answer.status is AnswerStatus.ESCALATED, answer)
    ticket = await queue(admin).get_ticket(answer.ticket_id)
    check("agent escalation stored as a real ticket", ticket.question == "Anything?", ticket)
    await admin.dispose()
    await reader.dispose()


def spoken_wav(path: Path, sentence: str) -> bool:
    if sys.platform != "win32":
        return False
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{path}'); $s.Speak('{sentence}'); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, timeout=60)
    return path.exists()


async def post_event(client: httpx.AsyncClient, **event: object) -> httpx.Response:
    body = json.dumps(event).encode()
    stamp = str(time.time())
    headers = {"X-Timestamp": stamp, "X-Signature": sign(WEBHOOK_SECRET, stamp, body)}
    return await client.post("/crm/webhooks/events", content=body, headers=headers)


def queue(engine: object) -> Escalations:
    """Opens tickets the way the agent does. This process never publishes knowledge."""
    return Escalations(
        PostgresEscalationStore(engine), None, QueueOnlyNotifier(), lambda: datetime.now(UTC)
    )


async def memory_loop_over_http(client: httpx.AsyncClient, async_url: str) -> None:
    print("\n-- Escalation and memory loop over HTTP, into real search")
    auth = bearer("lead-a", Role.TEAM_LEAD)
    manager = bearer("manager-b", Role.MANAGER)
    admin = create_async_engine(async_url, poolclass=NullPool)
    question = "Is there a rooftop padel court?"
    ticket = await queue(admin).open_ticket(question, "no evidence found", "PH")
    await admin.dispose()

    async def rooftop_hits() -> list[dict]:
        hits = await search(client, query="rooftop padel court", top_k=10, project="PH")
        return [hit for hit in hits if "padel" in hit["text"]]

    response = await client.get("/escalations", headers=NO_TOKEN)
    check("escalation queue needs a token", response.status_code == 401, response.text)
    response = await client.get("/escalations")
    check("reps cannot read the queue", response.status_code == 403, response.text)
    response = await client.get("/escalations", headers=auth)
    queued = [item["id"] for item in response.json()]
    check("ticket waits in the team-lead queue", ticket.id in queued, response.text)

    url = f"/escalations/{ticket.id}/answer"
    body = {"share_with_project": "PH"}
    response = await client.post(
        url, json={**body, "answer": "Ask 01012345678 about padel."}, headers=auth
    )
    check("client phone number blocks sharing", response.status_code == 422, response.text)
    response = await client.post(
        url, json={**body, "answer": "Yes, a rooftop padel court opens in phase two."}, headers=auth
    )
    check("clean answer proposed", response.status_code == 200, response.text)
    check("unreviewed answer is not searchable", await rooftop_hits() == [])

    candidate = f"/escalations/knowledge/{response.json()['candidate']['id']}"
    response = await client.post(f"{candidate}/approve", headers=auth)
    check("answerer cannot approve their own answer", response.status_code == 409, response.text)
    response = await client.post(f"{candidate}/approve", headers=manager)
    check("second person approves", response.status_code == 200, response.text)
    hits = await rooftop_hits()
    check("approved answer is searchable", len(hits) == 1, hits)
    check(
        "approved answer has a citation",
        hits and hits[0]["citation"]["document_id"].startswith("approved-answer-"),
        hits,
    )
    response = await client.post(
        f"{candidate}/revoke",
        json={"note": "Plan changed"},
        headers=manager,
    )
    check("approved answer can be revoked", response.status_code == 200, response.text)
    check("revoked answer leaves search", await rooftop_hits() == [])


async def coaching_over_http(client: httpx.AsyncClient) -> None:
    print("\n-- Coaching over HTTP without keys")
    auth = bearer("rep-a", Role.REP)
    response = await client.get("/coaching/personas", headers=NO_TOKEN)
    check("coaching needs a token", response.status_code == 401, response.text)
    response = await client.get("/coaching/personas", headers=auth)
    check("personas listed", len(response.json()) == 4, response.text)
    response = await client.post(
        "/coaching/sessions", json={"persona_id": "investor"}, headers=auth
    )
    check("session starts without keys", response.status_code == 201, response.text)
    url = f"/coaching/sessions/{response.json()['id']}"
    response = await client.post(f"{url}/messages", json={"message": "Hello"}, headers=auth)
    check("buyer reply without keys is a clean 503", response.status_code == 503, response.text)
    response = await client.get(url, headers=auth)
    check("failed reply left the session intact", len(response.json()["turns"]) == 1, response.text)
    response = await client.get(f"{url}/turns/0/audio", headers=auth)
    check("audio without a key is a clean 503", response.status_code == 503, response.text)


async def coaching_with_scripted_llms(async_url: str) -> None:
    print("\n-- Full role-play with scripted LLMs, real crew and real store")
    admin = create_async_engine(async_url, poolclass=NullPool)
    store = PostgresSessionStore(admin)
    rep_line = "What budget do you have in mind?"
    draft = ScoreDraft(
        scores=[
            CriterionDraft(criterion=name, score=4, quote=rep_line, feedback="Good")
            for name in RUBRIC
        ]
    )
    draft.scores[0] = CriterionDraft(criterion="discovery", score=5, quote="Words never said")
    llm = PydanticAIStructuredLLM(lambda _role: TestModel(custom_output_args=draft))
    buyer = CrewBuyer(lambda: [ScriptedBuyerLLM(model="s")])
    coaching = Coaching(store, buyer, llm, lambda: None, lambda: datetime.now(UTC))

    session = await coaching.start("rep-a", "investor")
    session = await coaching.send(session.id, rep_line)
    check("real crew replied in character", session.turns[-1].text == "Too expensive.", session)
    session = await coaching.finish(session.id)
    card = session.scorecard
    check("invented quote not assessed", card.criteria[0].score is None, card)
    check("overall computed in code", card.overall == 80, card)
    check("scored session persisted", await store.get(session.id) == session)
    await admin.dispose()


async def crm_over_http(
    env: WorkflowEnvironment, client: httpx.AsyncClient, async_url: str, tmp: Path
) -> None:
    print("\n-- CRM events: signed webhook, Temporal, worker, Postgres")
    admin = create_async_engine(async_url, poolclass=NullPool)
    crm = PostgresCrmStore(admin)
    await crm.create_schema()
    await PostgresSessionStore(admin).create_schema()
    await PostgresEscalationStore(admin).create_schema()
    await crm.add_lead(Lead("L1", "Synthetic Client", LeadStage.CONTACTED, "rep-a"))
    await crm.add_interaction(
        Interaction("no-text", "L1", InteractionKind.CALL, WHEN, answered=False)
    )

    async def finished(event_id: str) -> None:
        await asyncio.wait_for(
            env.client.get_workflow_handle(f"crm-event-{event_id}").result(), 300
        )

    body = b'{"event_id": "x", "type": "lead_opened", "lead_id": "L1"}'
    response = await client.post("/crm/webhooks/events", content=body)
    check("unsigned webhook rejected", response.status_code == 401, response.text)

    response = await post_event(client, event_id="e1", type="lead_opened", lead_id="L1")
    check("signed webhook accepted", response.status_code == 202, response.text)
    await finished("e1")
    url = "/crm/leads/L1/card"
    response = await client.get(url, headers=NO_TOKEN)
    check("card needs a token", response.status_code == 401, response.text)
    other_rep = bearer("rep-b", Role.REP)
    response = await client.get(url, headers=other_rep)
    check("another rep cannot read the card", response.status_code == 403, response.text)
    auth = bearer("rep-a", Role.REP)
    response = await client.get(url, headers=auth)
    check("card built by the worker", response.status_code == 200, response.text)
    card = response.json()
    check("card covers the interaction", len(card["card"]["timeline"]) == 1, card)
    check("card is fresh", card["stale"] is False, card)
    generated_at = card["card"]["generated_at"]
    check("card has a best time to call", card["card"]["best_time_to_call"], card)
    response = await client.get("/best-time/leads/L1", headers=NO_TOKEN)
    check("best time needs a token", response.status_code == 401, response.text)
    response = await client.get("/best-time/leads/L1", headers=auth)
    check("best time slots returned", len(response.json()["slots"]) == 3, response.text)
    check("few calls flagged as low data", response.json()["low_data"] is True, response.text)

    response = await post_event(client, event_id="e1", type="lead_opened", lead_id="L1")
    check("redelivered webhook is harmless", response.status_code == 202, response.text)
    response = await post_event(client, event_id="e2", type="lead_opened", lead_id="L1")
    await finished("e2")
    card = (await client.get(url, headers=auth)).json()
    check("fresh card is not rebuilt", card["card"]["generated_at"] == generated_at, card)

    recording = tmp / "call.wav"
    has_audio = spoken_wav(recording, "The client wants a three bedroom apartment with a garden.")
    await crm.add_interaction(
        Interaction(
            "no-consent",
            "L1",
            InteractionKind.CALL,
            WHEN.replace(day=2),
            recording_path=str(recording),
            recording_consent=False,
        )
    )
    await post_event(
        client, event_id="e3", type="call_ended", lead_id="L1", interaction_id="no-consent"
    )
    await finished("e3")
    stored = await crm.get_interaction("no-consent")
    check("recording without consent not transcribed", stored.text == "", stored)
    card = (await client.get(url, headers=auth)).json()
    check("call event refreshed the card", len(card["card"]["timeline"]) == 2, card)

    if has_audio:
        await crm.add_interaction(
            Interaction(
                "consented",
                "L1",
                InteractionKind.CALL,
                WHEN.replace(day=3),
                recording_path=str(recording),
                recording_consent=True,
            )
        )
        await post_event(
            client, event_id="e4", type="call_ended", lead_id="L1", interaction_id="consented"
        )
        await expect_workflow_failure(
            lambda: finished("e4"), "MissingApiKey", "post-call step stops cleanly without keys"
        )
        stored = await crm.get_interaction("consented")
        check("real Whisper transcript stored", "bedroom" in stored.text.casefold(), stored)
        card = (await client.get(url, headers=auth)).json()
        check("new transcript marks the card stale", card["stale"] is True, card)
    await admin.dispose()


async def crm_card_with_scripted_llm(async_url: str, reader_url: str) -> None:
    print("\n-- Context card and agent with a scripted model and real stores")
    admin = create_async_engine(async_url, poolclass=NullPool)
    reader = create_async_engine(reader_url, poolclass=NullPool)
    crm = PostgresCrmStore(admin)
    units = PostgresUnitStore(admin, reader)
    await crm.add_lead(Lead("L2", "Synthetic Client Two", LeadStage.QUALIFIED, "rep-a"))
    await crm.add_interaction(
        Interaction(
            "w1",
            "L2",
            InteractionKind.WHATSAPP,
            WHEN,
            "I want 3 bedrooms in Palm Hills, budget 6000000 EGP.",
        )
    )
    draft = CardDraft(
        stated_needs=[
            SourcedDraft(text="Wants 3 bedrooms", source_ids=["w1"]),
            SourcedDraft(text="Budget is 9999999", source_ids=["w1"]),
        ],
        preferences=PreferencesDraft(
            max_budget_egp=Decimal(6000000), bedrooms=3, project="Palm Hills", source_ids=["w1"]
        ),
    )
    llm = PydanticAIStructuredLLM(lambda _role: TestModel(custom_output_args=draft))
    build = BuildContextCard(llm, crm, InventoryUnitMatcher(units), lambda: datetime.now(UTC))
    card = await build("L2")
    check("invented claim dropped from card", card.dropped_claims == 1, card)
    check("card matches a live unit", [u.code for u in card.matched_units] == ["A-101"], card)
    check("sold unit not matched", all(u.code != "A-102" for u in card.matched_units), card)
    check("card persisted", await crm.get_card("L2") == card)

    crew_llm = ScriptedCrewLLM(
        model="s", pattern="Stated needs", template="They want 3 bedrooms, and A-101 fits."
    )
    agent = AskAgent(
        ScriptedAgentLLM(Intent.CONTEXT_CARD),
        CrewAnswerComposer([crew_llm]),
        KnowledgeToolAdapter(build_retrieve(Settings()), units),
        InventoryToolAdapter(AskInventory(llm, SqlGlotGuard(), units)),
        CalculatorToolAdapter(units),
        EscalationToolAdapter(queue(admin), crm),
        ContextCardToolAdapter(crm),
    )
    answer = await agent("Brief me on this client", lead_id="L2")
    check("agent answers from the stored card", answer.status is AnswerStatus.ANSWERED, answer)
    check("card cited", answer.citations and answer.citations[0].kind is SourceKind.CRM, answer)
    await admin.dispose()
    await reader.dispose()


async def start_temporal() -> WorkflowEnvironment:
    # The first run downloads the dev server, and downloads can fail transiently.
    for attempt in range(1, 9):
        try:
            return await WorkflowEnvironment.start_local()
        except RuntimeError as error:
            print(f"Temporal attempt {attempt} failed: {str(error)[:80]}")
            await asyncio.sleep(15)
    return await WorkflowEnvironment.start_local()


async def main(vector_store: str) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        tmp = Path(directory)
        database = pgserver.get_server(tmp / "pg")
        base = database.get_uri()
        sync_url = base.replace("postgresql://", "postgresql+psycopg2://")
        async_url = base.replace("postgresql://", "postgresql+asyncpg://")
        engine = create_async_engine(async_url, poolclass=NullPool)
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await engine.dispose()
        subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT,
            env={**os.environ, "DATABASE_URL": sync_url},
            check=True,
        )
        reader_url = await setup_database(async_url)

        print("Training the best-time model on synthetic calls")
        tz = ZoneInfo("Africa/Cairo")
        calls, _profiles = generate_calls(600, 7, tz, WHEN)
        model_path = tmp / "best_time.txt"
        report = train_best_time(
            calls, tz, model_path, f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", trials=3
        )
        check("model beats baseline and is promoted", report.promoted, report)

        env = await start_temporal()
        logs = open(tmp / "processes.log", "w", encoding="utf-8")  # noqa: SIM115
        environment = {
            **os.environ,
            "PYTHONUTF8": "1",
            "VECTOR_STORE": vector_store,
            "EMBEDDING_MODEL": os.environ.get(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "EMBEDDING_DIM": os.environ.get("EMBEDDING_DIM", "384"),
            "RERANKER_MODEL": os.environ.get(
                "RERANKER_MODEL", "cross-encoder/ms-marco-TinyBERT-L2-v2"
            ),
            "DATABASE_URL": sync_url,
            "ASYNC_DATABASE_URL": async_url,
            "INVENTORY_READ_ONLY_DATABASE_URL": reader_url,
            "TEMPORAL_ADDRESS": env.client.service_client.config.target_host,
            "TEMPORAL_TASK_QUEUE": TASK_QUEUE,
            "BEST_TIME_MODEL_PATH": str(model_path),
            "CRM_WEBHOOK_SECRET": WEBHOOK_SECRET,
            "AUTH_SECRET": AUTH_SECRET,
            "WHISPER_MODEL": os.environ.get("WHISPER_MODEL", "tiny"),
        }
        # The 503 check below needs both keys absent.
        environment.pop("ANTHROPIC_API_KEY", None)
        environment.pop("OPENAI_API_KEY", None)
        api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "ai_os.bootstrap.app:create_app",
                "--factory",
                "--port",
                str(PORT),
            ],
            env=environment,
            stdout=logs,
            stderr=subprocess.STDOUT,
        )
        worker = subprocess.Popen(
            [sys.executable, "-m", "ai_os.bootstrap.worker"],
            env=environment,
            stdout=logs,
            stderr=subprocess.STDOUT,
        )
        try:
            rep = bearer("rep-a", Role.REP)
            async with httpx.AsyncClient(base_url=API, timeout=300, headers=rep) as client:
                seconds = await wait_for_health(client, api)
                print(f"API healthy after {seconds:.0f}s (vector store: {vector_store})")
                await scenario(env, client, tmp)
                await crm_over_http(env, client, async_url, tmp)
                await coaching_over_http(client)
                await memory_loop_over_http(client, async_url)
            await inventory_with_scripted_llm(async_url, reader_url)
            os.environ.update(environment)
            await agent_with_real_tools(async_url, reader_url)
            await crm_card_with_scripted_llm(async_url, reader_url)
            await coaching_with_scripted_llms(async_url)
        except Exception as error:
            check("scenario completed", False, repr(error))
        finally:
            worker_alive = worker.poll() is None
            api.terminate()
            worker.terminate()
            api.wait(30)
            worker.wait(30)
            logs.close()
            await env.shutdown()
            database.cleanup()
            check("worker stayed up", worker_alive)
            if failures:
                print("\n--- process log tail ---")
                log_text = (tmp / "processes.log").read_text(encoding="utf-8", errors="replace")
                print(log_text[-4000:])

    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-store", choices=["qdrant", "pgvector"], default="pgvector")
    asyncio.run(main(parser.parse_args().vector_store))
