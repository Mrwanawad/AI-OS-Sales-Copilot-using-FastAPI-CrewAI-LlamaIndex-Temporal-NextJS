import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from crewai.llms.base_llm import BaseLLM
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_os.modules.coaching.adapters.inbound.api import create_router
from ai_os.modules.coaching.adapters.outbound.elevenlabs_speech import ElevenLabsSpeech
from ai_os.modules.coaching.application.buyer import CrewBuyer
from ai_os.modules.coaching.application.service import (
    MAX_REP_TURNS,
    Coaching,
    CriterionDraft,
    ScoreDraft,
)
from ai_os.modules.coaching.domain.errors import (
    LLMUnavailable,
    NothingToScore,
    PersonaNotFound,
    SessionChanged,
    SessionClosed,
    SessionNotFound,
)
from ai_os.modules.coaching.domain.models import (
    Persona,
    RolePlaySession,
    SessionStatus,
    Speaker,
    Turn,
)
from ai_os.modules.coaching.domain.personas import PERSONAS
from ai_os.modules.coaching.domain.scoring import RUBRIC, ProposedScore, build_scorecard
from ai_os.shared.auth import Role
from ai_os.shared.llm.port import LLMRole, OutputT

from .auth_helpers import AUTH, bearer

NOW = datetime(2026, 11, 7, tzinfo=UTC)
REP_LINE = "What budget do you have in mind, and when would you like to visit?"
TURNS = [
    Turn(Speaker.BUYER, "I'm only looking."),
    Turn(Speaker.REP, REP_LINE),
]


class FakeStore:
    def __init__(self) -> None:
        self.sessions: dict[str, RolePlaySession] = {}

    async def save(self, session: RolePlaySession) -> None:
        stored = self.sessions.get(session.id)
        if (stored.version if stored else -1) != session.version - 1:
            raise SessionChanged(session.id)
        self.sessions[session.id] = session

    async def get(self, session_id: str) -> RolePlaySession | None:
        return self.sessions.get(session_id)


class FakeBuyer:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.seen: list[Turn] = []

    async def reply(self, persona: Persona, turns: Sequence[Turn]) -> str:
        self.seen = list(turns)
        if self.error:
            raise self.error
        return "That sounds expensive."


class ScriptedLLM:
    def __init__(self, draft: ScoreDraft | None = None) -> None:
        self.draft = draft or ScoreDraft()
        self.calls = 0

    async def extract(
        self, role: LLMRole, instructions: str, prompt: str, output_type: type[OutputT]
    ) -> OutputT:
        self.calls += 1
        return self.draft


class FakeSpeech:
    async def synthesize(self, text: str) -> bytes:
        return f"audio:{text}".encode()


def service(
    store: FakeStore | None = None, buyer: FakeBuyer | None = None, llm: ScriptedLLM | None = None
) -> Coaching:
    return Coaching(
        store or FakeStore(), buyer or FakeBuyer(), llm or ScriptedLLM(), FakeSpeech, lambda: NOW
    )


def full_marks(score: int = 4) -> ScoreDraft:
    return ScoreDraft(
        scores=[
            CriterionDraft(criterion=name, score=score, quote=REP_LINE, feedback="Good")
            for name in RUBRIC
        ]
    )


def test_rubric_weights_sum_to_one() -> None:
    assert sum(weight for _, weight in RUBRIC.values()) == pytest.approx(1)


def test_overall_is_computed_from_verified_scores() -> None:
    proposed = [ProposedScore(name, 4, REP_LINE) for name in RUBRIC]

    assert build_scorecard(proposed, TURNS).overall == 80


def test_quote_matching_ignores_case_and_spacing() -> None:
    proposed = [ProposedScore("discovery", 5, "what   BUDGET do you have")]

    assert build_scorecard(proposed, TURNS).criteria[0].score == 5


def test_invented_quote_means_not_assessed() -> None:
    proposed = [ProposedScore("discovery", 5, "I always ask about budget first")]

    assert build_scorecard(proposed, TURNS).criteria[0].score is None


def test_quote_from_the_buyer_does_not_count_for_the_rep() -> None:
    proposed = [ProposedScore("discovery", 5, "I'm only looking")]

    assert build_scorecard(proposed, TURNS).criteria[0].score is None


def test_score_above_zero_without_quote_is_not_assessed() -> None:
    assert build_scorecard([ProposedScore("rapport", 3)], TURNS).criteria[-1].score is None


def test_zero_needs_no_quote_and_lowers_the_overall() -> None:
    proposed = [ProposedScore(name, 0, feedback="Missing") for name in RUBRIC]

    card = build_scorecard(proposed, TURNS)

    assert card.overall == 0
    assert card.criteria[0].feedback == "Missing"


@pytest.mark.parametrize("score", [-1, 6, 100])
def test_out_of_range_score_is_not_assessed(score: int) -> None:
    proposed = [ProposedScore("discovery", score, REP_LINE)]

    assert build_scorecard(proposed, TURNS).criteria[0].score is None


def test_unknown_criterion_is_ignored() -> None:
    card = build_scorecard([ProposedScore("charisma", 5, REP_LINE)], TURNS)

    assert {c.criterion for c in card.criteria} == set(RUBRIC)


def test_too_little_verified_gives_no_overall() -> None:
    assert build_scorecard([ProposedScore("rapport", 5, REP_LINE)], TURNS).overall is None


def test_overall_weights_only_assessed_criteria() -> None:
    proposed = [
        ProposedScore("objection_handling", 5, REP_LINE),
        ProposedScore("discovery", 0),
    ]

    assert build_scorecard(proposed, TURNS).overall == 55


async def test_start_opens_with_persona_line_and_no_model_call() -> None:
    llm = ScriptedLLM()

    session = await service(llm=llm).start("rep-a", "investor")

    assert session.turns == [Turn(Speaker.BUYER, PERSONAS["investor"].opening_line)]
    assert llm.calls == 0


async def test_unknown_persona_raises() -> None:
    with pytest.raises(PersonaNotFound):
        await service().start("rep-a", "nobody")


async def test_send_appends_both_turns_and_gives_buyer_the_message() -> None:
    buyer = FakeBuyer()
    coaching = service(buyer=buyer)
    session = await coaching.start("rep-a", "investor")

    updated = await coaching.send(session.id, "Hello")

    assert [t.text for t in updated.turns[-2:]] == ["Hello", "That sounds expensive."]
    assert buyer.seen[-1] == Turn(Speaker.REP, "Hello")


class SlowBuyer(FakeBuyer):
    async def reply(self, persona: Persona, turns: Sequence[Turn]) -> str:
        await asyncio.sleep(0)
        return await super().reply(persona, turns)


async def test_two_messages_at_once_cannot_overwrite_each_other() -> None:
    store = FakeStore()
    coaching = service(store=store, buyer=SlowBuyer())
    session = await coaching.start("rep-a", "investor")

    results = await asyncio.gather(
        coaching.send(session.id, "First"),
        coaching.send(session.id, "Second"),
        return_exceptions=True,
    )

    assert sum(isinstance(result, SessionChanged) for result in results) == 1
    saved = store.sessions[session.id]
    assert len(saved.turns) == 3
    assert saved.version == 1


async def test_provider_failure_loses_nothing_so_the_rep_can_resend() -> None:
    store = FakeStore()
    session = await service(store).start("rep-a", "investor")

    with pytest.raises(LLMUnavailable):
        await service(store, FakeBuyer(LLMUnavailable("down"))).send(session.id, "Hello")

    assert len(store.sessions[session.id].turns) == 1


async def test_turn_limit_stops_the_session() -> None:
    store = FakeStore()
    coaching = service(store)
    session = await coaching.start("rep-a", "investor")
    for _ in range(MAX_REP_TURNS):
        await coaching.send(session.id, "Hello")

    with pytest.raises(SessionClosed, match="turn limit"):
        await coaching.send(session.id, "One more")


async def test_finish_scores_once_and_closes_the_session() -> None:
    llm = ScriptedLLM(full_marks())
    coaching = service(llm=llm)
    session = await coaching.start("rep-a", "investor")
    await coaching.send(session.id, REP_LINE)

    finished = await coaching.finish(session.id)
    again = await coaching.finish(session.id)

    assert finished.status is SessionStatus.FINISHED
    assert finished.scorecard.overall == 80
    assert again == finished
    assert llm.calls == 1
    with pytest.raises(SessionClosed, match="finished"):
        await coaching.send(session.id, "Wait")


async def test_finish_before_rep_speaks_is_refused_without_a_model_call() -> None:
    llm = ScriptedLLM()
    coaching = service(llm=llm)
    session = await coaching.start("rep-a", "investor")

    with pytest.raises(NothingToScore):
        await coaching.finish(session.id)

    assert llm.calls == 0


async def test_audio_is_only_for_buyer_turns() -> None:
    coaching = service()
    session = await coaching.start("rep-a", "investor")
    await coaching.send(session.id, "Hello")

    assert await coaching.turn_audio(session.id, 2) == b"audio:That sounds expensive."
    with pytest.raises(SessionNotFound):
        await coaching.turn_audio(session.id, 1)
    with pytest.raises(SessionNotFound):
        await coaching.turn_audio(session.id, 99)


class FakeCrewLLM(BaseLLM):
    fail: bool = False
    reply_text: str = "That sounds expensive."
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
        self.prompts = [*self.prompts, str(messages)]
        if self.fail:
            raise RuntimeError("provider down")
        return f"Thought: done\nFinal Answer: {self.reply_text}"


async def test_real_crew_plays_the_persona_from_the_conversation() -> None:
    llm = FakeCrewLLM(model="fake")

    reply = await CrewBuyer(lambda: [llm]).reply(PERSONAS["investor"], TURNS)

    assert reply == "That sounds expensive."
    assert REP_LINE in llm.prompts[0]
    assert PERSONAS["investor"].objections[0] in llm.prompts[0]


async def test_buyer_falls_back_to_second_provider() -> None:
    primary = FakeCrewLLM(model="a", fail=True)
    fallback = FakeCrewLLM(model="b")

    reply = await CrewBuyer(lambda: [primary, fallback]).reply(PERSONAS["investor"], TURNS)

    assert reply == "That sounds expensive."


async def test_all_buyer_providers_down_raises() -> None:
    llms = [FakeCrewLLM(model="a", fail=True), FakeCrewLLM(model="b", fail=True)]

    with pytest.raises(LLMUnavailable):
        await CrewBuyer(lambda: llms).reply(PERSONAS["investor"], TURNS)


class FakeTextToSpeech:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def convert(self, **kwargs: object):
        self.kwargs = kwargs
        yield b"abc"
        yield b"def"


class FakeElevenLabs:
    text_to_speech = FakeTextToSpeech()


async def test_elevenlabs_adapter_joins_audio_chunks() -> None:
    client = FakeElevenLabs()

    audio = await ElevenLabsSpeech("key", "voice-1", client=client).synthesize("Hello")

    assert audio == b"abcdef"
    assert client.text_to_speech.kwargs["voice_id"] == "voice-1"
    assert client.text_to_speech.kwargs["text"] == "Hello"


REP = bearer("rep-a", Role.REP)
OTHER_REP = bearer("rep-b", Role.REP)
LEAD = bearer("lead-a", Role.TEAM_LEAD)


def api(coaching: Coaching | None = None) -> TestClient:
    app = FastAPI()
    built = coaching or service()
    app.include_router(create_router(lambda: built, AUTH))
    return TestClient(app)


def test_every_route_needs_a_token() -> None:
    client = api()

    assert client.get("/coaching/personas").status_code == 401
    assert client.post("/coaching/sessions", json={}).status_code == 401
    assert client.get("/coaching/sessions/x/turns/0/audio").status_code == 401


def start_session(client: TestClient) -> str:
    started = client.post("/coaching/sessions", json={"persona_id": "investor"}, headers=REP)
    assert started.status_code == 201
    return started.json()["id"]


def test_full_session_over_http() -> None:
    client = api(service(llm=ScriptedLLM(full_marks())))

    assert len(client.get("/coaching/personas", headers=REP).json()) == 4
    session_id = start_session(client)
    url = f"/coaching/sessions/{session_id}"

    sent = client.post(f"{url}/messages", json={"message": REP_LINE}, headers=REP)
    assert sent.json()["turns"][-1]["speaker"] == "buyer"
    audio = client.get(f"{url}/turns/2/audio", headers=REP)
    assert audio.headers["content-type"] == "audio/mpeg"
    finished = client.post(f"{url}/finish", headers=REP)
    assert finished.json()["scorecard"]["overall"] == 80
    assert client.post(f"{url}/messages", json={"message": "x"}, headers=REP).status_code == 409
    assert client.get(url, headers=REP).json()["status"] == "finished"


def test_session_belongs_to_the_token_holder_whatever_the_body_says() -> None:
    client = api()

    started = client.post(
        "/coaching/sessions", json={"rep_id": "someone-else", "persona_id": "investor"}, headers=REP
    )

    assert started.json()["rep_id"] == "rep-a"


def test_other_reps_cannot_read_or_use_a_session() -> None:
    client = api(service(llm=ScriptedLLM(full_marks())))
    url = f"/coaching/sessions/{start_session(client)}"

    assert client.get(url, headers=OTHER_REP).status_code == 403
    assert (
        client.post(f"{url}/messages", json={"message": "x"}, headers=OTHER_REP).status_code == 403
    )
    assert client.post(f"{url}/finish", headers=OTHER_REP).status_code == 403
    assert client.get(f"{url}/turns/0/audio", headers=OTHER_REP).status_code == 403


def test_team_lead_can_review_but_not_play_a_session() -> None:
    client = api()
    url = f"/coaching/sessions/{start_session(client)}"

    assert client.get(url, headers=LEAD).status_code == 200
    assert client.post(f"{url}/messages", json={"message": "x"}, headers=LEAD).status_code == 403


def test_errors_map_to_clean_statuses() -> None:
    client = api()

    assert client.get("/coaching/sessions/nope", headers=REP).status_code == 404
    bad_persona = {"persona_id": "nobody"}
    assert client.post("/coaching/sessions", json=bad_persona, headers=REP).status_code == 404
    long_message = {"message": "x" * 1001}
    assert (
        client.post("/coaching/sessions/x/messages", json=long_message, headers=REP).status_code
        == 422
    )
