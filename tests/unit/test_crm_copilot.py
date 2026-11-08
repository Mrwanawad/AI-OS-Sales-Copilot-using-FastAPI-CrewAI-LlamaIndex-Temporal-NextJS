from datetime import datetime, timedelta
from decimal import Decimal as D
from pathlib import Path

import pytest

from ai_os.modules.crm_copilot.adapters.outbound.speech import (
    PyannoteDiarizer,
    WhisperTranscriber,
)
from ai_os.modules.crm_copilot.application.build_card import (
    BuildContextCard,
    CardDraft,
    ConcernDraft,
    PreferencesDraft,
    SourcedDraft,
    TimelineDraft,
)
from ai_os.modules.crm_copilot.application.handover import BuildHandover, HandoverDraft
from ai_os.modules.crm_copilot.application.history import format_history
from ai_os.modules.crm_copilot.application.post_call import PostCallDraft, ProcessPostCall
from ai_os.modules.crm_copilot.application.process_call import ProcessCallRecording
from ai_os.modules.crm_copilot.domain.errors import InteractionNotFound, LeadNotFound
from ai_os.modules.crm_copilot.domain.models import (
    ContextCard,
    Lead,
    LeadStage,
    MatchedUnit,
    Preferences,
    SpeakerTurn,
    SpeechSegment,
)
from ai_os.modules.crm_copilot.domain.transcript import UNKNOWN_SPEAKER, format_transcript

from .crm_fakes import LEAD, NOW, FakeCrmStore, FakeMatcher, ScriptedLLM, call


def builder(llm: ScriptedLLM, store: FakeCrmStore, matcher: FakeMatcher | None = None):
    return BuildContextCard(llm, store, matcher or FakeMatcher(), lambda: NOW)


async def test_card_keeps_only_claims_their_sources_support() -> None:
    store = FakeCrmStore(call())
    draft = CardDraft(
        stated_needs=[
            SourcedDraft(text="Wants 3 bedrooms", source_ids=["c1"]),
            SourcedDraft(text="Budget is 9000000", source_ids=["c1"]),
            SourcedDraft(text="Has no source"),
            SourcedDraft(text="Cites a missing interaction", source_ids=["nope"]),
        ],
        timeline=[TimelineDraft(interaction_id="c1", summary="Discussed 3 bedrooms")],
    )

    card = await builder(ScriptedLLM(draft), store)("L1")

    assert [item.text for item in card.stated_needs] == ["Wants 3 bedrooms"]
    assert card.dropped_claims == 3
    assert card.timeline[0].summary == "Discussed 3 bedrooms"
    assert store.card == card


async def test_concern_survives_when_only_its_suggested_answer_is_unsupported() -> None:
    draft = CardDraft(
        open_concerns=[
            ConcernDraft(
                concern="Worried about price",
                suggested_answer="Offer 4,000,000",
                source_ids=["c1"],
            )
        ]
    )

    card = await builder(ScriptedLLM(draft), FakeCrmStore(call()))("L1")

    assert card.open_concerns[0].concern == "Worried about price"
    assert card.open_concerns[0].suggested_answer is None
    assert card.dropped_claims == 1


async def test_stated_preferences_drive_unit_matching() -> None:
    unit = MatchedUnit("A-101", "Palm Hills", D(5000000), 3)
    matcher = FakeMatcher([unit])
    draft = CardDraft(
        preferences=PreferencesDraft(
            max_budget_egp=D(5000000), bedrooms=3, project="Palm Hills", source_ids=["c1"]
        )
    )

    card = await builder(ScriptedLLM(draft), FakeCrmStore(call()), matcher)("L1")

    assert matcher.preferences == [Preferences(D(5000000), 3, "Palm Hills")]
    assert card.matched_units == [unit]


async def test_unstated_preferences_are_dropped_and_do_not_match_units() -> None:
    matcher = FakeMatcher()
    draft = CardDraft(
        preferences=PreferencesDraft(
            max_budget_egp=D(7000000), bedrooms=9, project="Other", source_ids=["c1"]
        )
    )

    card = await builder(ScriptedLLM(draft), FakeCrmStore(call()), matcher)("L1")

    assert card.preferences == Preferences()
    assert card.dropped_claims == 3
    assert matcher.preferences == []


async def test_lead_with_no_text_builds_empty_card_without_calling_the_model() -> None:
    llm = ScriptedLLM()

    card = await builder(llm, FakeCrmStore(call(text="")))("L1")

    assert llm.prompts == []
    assert card.stated_needs == []
    assert len(card.timeline) == 1
    assert card.timeline[0].summary is None


class FakeBestTime:
    async def best_time(self, lead_id: str) -> str | None:
        return "Tuesday around 19:00"


async def test_card_includes_best_time_when_available() -> None:
    build = BuildContextCard(
        ScriptedLLM(CardDraft()),
        FakeCrmStore(call()),
        FakeMatcher(),
        lambda: NOW,
        best_time=FakeBestTime(),
    )

    card = await build("L1")

    assert card.best_time_to_call == "Tuesday around 19:00"


async def test_unknown_lead_raises() -> None:
    with pytest.raises(LeadNotFound):
        await builder(ScriptedLLM(), FakeCrmStore())("missing")


async def test_only_if_stale_reuses_card_until_a_new_interaction_arrives() -> None:
    store = FakeCrmStore(call())
    llm = ScriptedLLM(CardDraft(), CardDraft())
    build = builder(llm, store)
    await build("L1")

    await build("L1", only_if_stale=True)
    assert len(llm.prompts) == 1

    store.interactions["c2"] = call("c2", "New message", day=2)
    await build("L1", only_if_stale=True)
    assert len(llm.prompts) == 2


def test_history_keeps_newest_within_budget_and_reports_coverage() -> None:
    interactions = [call("old", "x" * 500, day=1), call("new", "recent", day=2)]

    text, covered = format_history(interactions, max_chars=200)

    assert [item.id for item in covered] == ["new"]
    assert "recent" in text and "xxx" not in text


async def test_card_counts_interactions_left_out_of_the_prompt() -> None:
    store = FakeCrmStore(call("old", "x" * 500, day=1), call("new", "recent", day=2))
    build = BuildContextCard(
        ScriptedLLM(CardDraft()), store, FakeMatcher(), lambda: NOW, max_history_chars=200
    )

    card = await build("L1")

    assert card.omitted_interactions == 1


async def test_claim_citing_an_omitted_interaction_is_dropped() -> None:
    store = FakeCrmStore(call("old", "x" * 500, day=1), call("new", "recent", day=2))
    draft = CardDraft(stated_needs=[SourcedDraft(text="Old fact", source_ids=["old"])])
    build = BuildContextCard(
        ScriptedLLM(draft), store, FakeMatcher(), lambda: NOW, max_history_chars=200
    )

    card = await build("L1")

    assert card.stated_needs == []


def test_transcript_uses_most_overlapping_speaker_and_merges_lines() -> None:
    segments = [
        SpeechSegment(0, 2, " Hello"),
        SpeechSegment(2, 4, " how are you"),
        SpeechSegment(4, 6, " Fine"),
    ]
    turns = [SpeakerTurn(0, 3.9, "SPEAKER_00"), SpeakerTurn(3.9, 6, "SPEAKER_01")]

    assert format_transcript(segments, turns) == ("SPEAKER_00: Hello how are you\nSPEAKER_01: Fine")


def test_transcript_without_turns_marks_speaker_unknown() -> None:
    text = format_transcript([SpeechSegment(0, 1, "Hi")], [])

    assert text == f"{UNKNOWN_SPEAKER}: Hi"


class FakeSpeech:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, path: Path) -> list[SpeechSegment]:
        self.calls += 1
        return [SpeechSegment(0, 1, "Hello")]

    async def diarize(self, path: Path) -> list[SpeakerTurn]:
        return [SpeakerTurn(0, 1, "SPEAKER_00")]


async def test_recording_with_consent_is_transcribed_and_stored() -> None:
    speech = FakeSpeech()
    store = FakeCrmStore(call(text="", recording_path="c1.wav", recording_consent=True))

    stored = await ProcessCallRecording(store, speech, speech)("c1")

    assert stored
    assert store.interactions["c1"].text == "SPEAKER_00: Hello"


async def test_recording_without_consent_is_never_touched() -> None:
    speech = FakeSpeech()
    store = FakeCrmStore(call(text="", recording_path="c1.wav", recording_consent=False))

    stored = await ProcessCallRecording(store, speech, speech)("c1")

    assert not stored
    assert speech.calls == 0
    assert store.interactions["c1"].text == ""


async def test_call_without_recording_is_skipped_and_unknown_call_raises() -> None:
    speech = FakeSpeech()
    process = ProcessCallRecording(FakeCrmStore(call()), speech, speech)

    assert not await process("c1")
    with pytest.raises(InteractionNotFound):
        await process("missing")


def post_call(store: FakeCrmStore, draft: PostCallDraft) -> ProcessPostCall:
    return ProcessPostCall(ScriptedLLM(draft), store, lambda: NOW)


async def test_post_call_applies_allowed_supported_fields_only() -> None:
    store = FakeCrmStore(call())
    draft = PostCallDraft(
        summary="Wants 3 bedrooms.",
        fields={"budget_egp": "5000000", "preferred_bedrooms": "9", "phone": "0100", "owner": "me"},
    )

    result = await post_call(store, draft)("c1")

    assert store.field_updates == {"budget_egp": "5000000"}
    assert sorted(result.rejected_fields) == ["owner", "phone", "preferred_bedrooms"]
    assert result.summary == "Wants 3 bedrooms."


@pytest.mark.parametrize(
    ("current", "proposed", "applied"),
    [
        (LeadStage.CONTACTED, LeadStage.QUALIFIED, LeadStage.QUALIFIED),
        (LeadStage.CONTACTED, LeadStage.NEW, None),
        (LeadStage.NEGOTIATION, LeadStage.QUALIFIED, None),
        (LeadStage.NEGOTIATION, LeadStage.WON, None),
        (LeadStage.NEGOTIATION, LeadStage.LOST, None),
        (LeadStage.NEGOTIATION, LeadStage.RESERVED, None),
        (LeadStage.LOST, LeadStage.QUALIFIED, None),
    ],
)
async def test_automation_only_moves_stage_forward_and_never_closes_a_deal(
    current: LeadStage, proposed: LeadStage, applied: LeadStage | None
) -> None:
    lead = Lead("L1", "Client", current, "rep-a")
    store = FakeCrmStore(call(), lead=lead)

    result = await post_call(store, PostCallDraft(summary="Fine", stage=proposed))("c1")

    assert result.new_stage is applied
    assert store.stage_updates == ([applied] if applied else [])


@pytest.mark.parametrize(
    ("at", "kept"),
    [
        (NOW + timedelta(days=3), True),
        (NOW - timedelta(days=1), False),
        (NOW + timedelta(days=400), False),
    ],
)
async def test_follow_up_must_be_in_a_sensible_future_window(at: datetime, kept: bool) -> None:
    store = FakeCrmStore(call())

    await post_call(store, PostCallDraft(summary="Fine", follow_up_at=at))("c1")

    assert bool(store.follow_ups) is kept


async def test_call_with_no_transcript_makes_no_model_call_or_update() -> None:
    llm = ScriptedLLM()
    store = FakeCrmStore(call(text=""))

    result = await ProcessPostCall(llm, store, lambda: NOW)("c1")

    assert result.summary is None
    assert llm.prompts == []
    assert store.stage_updates == []


async def test_summary_with_invented_number_is_withheld_but_updates_stand() -> None:
    store = FakeCrmStore(call())
    draft = PostCallDraft(summary="Agreed on 1000000", fields={"budget_egp": "5000000"})

    result = await post_call(store, draft)("c1")

    assert result.summary is None
    assert result.applied_fields == {"budget_egp": "5000000"}


def stored_card() -> ContextCard:
    return ContextCard("L1", "Client One", LEAD.stage, NOW, None)


async def test_handover_saved_when_grounded() -> None:
    store = FakeCrmStore(call())
    store.card = stored_card()
    build = BuildHandover(ScriptedLLM(HandoverDraft(summary="Early stage.")), store, lambda: NOW)

    handover = await build("L1", "rep-a", "rep-b")

    assert handover.summary == "Early stage."
    assert (handover.from_rep, handover.to_rep) == ("rep-a", "rep-b")
    assert store.assigned_to == "rep-b"
    assert store.handovers == [handover]


async def test_handover_with_invented_number_falls_back_to_card_facts() -> None:
    store = FakeCrmStore(call())
    store.card = stored_card()
    llm = ScriptedLLM(HandoverDraft(summary="Promised 5% off."))

    handover = await BuildHandover(llm, store, lambda: NOW)("L1", "rep-a", "rep-b")

    assert "5%" not in handover.summary
    assert "Client One" in handover.summary


async def test_handover_without_card_raises() -> None:
    with pytest.raises(LeadNotFound):
        await BuildHandover(ScriptedLLM(), FakeCrmStore(), lambda: NOW)("L1", "a", "b")


class FakeWhisper:
    def transcribe(self, path: str) -> dict:
        return {"segments": [{"start": 0.0, "end": 1.5, "text": " Hi"}]}


class _Turn:
    start = 0.0
    end = 1.5


class FakeAnnotation:
    def itertracks(self, yield_label: bool):
        yield _Turn(), "A", "SPEAKER_00"


class FakeOutput:
    speaker_diarization = FakeAnnotation()


async def test_whisper_adapter_maps_segments() -> None:
    segments = await WhisperTranscriber(model=FakeWhisper()).transcribe(Path("a.wav"))

    assert segments == [SpeechSegment(0.0, 1.5, " Hi")]


@pytest.mark.parametrize("output", [FakeOutput(), FakeAnnotation()])
async def test_pyannote_adapter_handles_both_output_shapes(output: object) -> None:
    diarizer = PyannoteDiarizer("token", pipeline=lambda path: output)

    assert await diarizer.diarize(Path("a.wav")) == [SpeakerTurn(0.0, 1.5, "SPEAKER_00")]
