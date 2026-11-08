import asyncio
from collections.abc import Sequence
from pathlib import Path

from ai_os.modules.crm_copilot.domain.models import SpeakerTurn, SpeechSegment


class WhisperTranscriber:
    """Runs locally, so call audio stays on our infrastructure. Needs ffmpeg on the PATH."""

    def __init__(self, model_name: str = "large-v3", model: object | None = None) -> None:
        self._model_name = model_name
        self._model = model

    async def transcribe(self, path: Path) -> Sequence[SpeechSegment]:
        return await asyncio.to_thread(self._transcribe, path)

    def _transcribe(self, path: Path) -> list[SpeechSegment]:
        if self._model is None:
            import whisper

            self._model = whisper.load_model(self._model_name)
        result = self._model.transcribe(str(path))
        return [
            SpeechSegment(segment["start"], segment["end"], segment["text"])
            for segment in result["segments"]
        ]


class PyannoteDiarizer:
    """The model is gated: accept its terms on Hugging Face and supply a token."""

    def __init__(
        self,
        token: str | None,
        model_name: str = "pyannote/speaker-diarization-community-1",
        pipeline: object | None = None,
    ) -> None:
        self._token = token
        self._model_name = model_name
        self._pipeline = pipeline

    async def diarize(self, path: Path) -> Sequence[SpeakerTurn]:
        return await asyncio.to_thread(self._diarize, path)

    def _diarize(self, path: Path) -> list[SpeakerTurn]:
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            self._pipeline = Pipeline.from_pretrained(self._model_name, token=self._token)
        output = self._pipeline(str(path))
        # pyannote 4 returns several views; the exclusive one has no overlaps, for transcription.
        annotation = getattr(
            output, "exclusive_speaker_diarization", getattr(output, "speaker_diarization", output)
        )
        return [
            SpeakerTurn(turn.start, turn.end, str(speaker))
            for turn, _track, speaker in annotation.itertracks(yield_label=True)
        ]


class NoDiarizer:
    """Used without a Hugging Face token. Transcripts keep working, without speaker labels."""

    async def diarize(self, path: Path) -> Sequence[SpeakerTurn]:
        return []
