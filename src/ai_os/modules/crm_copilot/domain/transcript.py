from collections.abc import Sequence

from ai_os.modules.crm_copilot.domain.models import SpeakerTurn, SpeechSegment

UNKNOWN_SPEAKER = "Unknown speaker"


def format_transcript(segments: Sequence[SpeechSegment], turns: Sequence[SpeakerTurn]) -> str:
    """Labels each segment with the speaker who overlaps it most, merging consecutive lines."""
    lines: list[tuple[str, str]] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        speaker = _speaker_for(segment, turns)
        if lines and lines[-1][0] == speaker:
            lines[-1] = (speaker, f"{lines[-1][1]} {text}")
        else:
            lines.append((speaker, text))
    return "\n".join(f"{speaker}: {text}" for speaker, text in lines)


def _speaker_for(segment: SpeechSegment, turns: Sequence[SpeakerTurn]) -> str:
    best_speaker = UNKNOWN_SPEAKER
    best_overlap = 0.0
    for turn in turns:
        overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker
    return best_speaker
