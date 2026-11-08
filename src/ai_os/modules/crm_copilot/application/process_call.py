import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from ai_os.modules.crm_copilot.domain.errors import InteractionNotFound
from ai_os.modules.crm_copilot.domain.transcript import format_transcript
from ai_os.modules.crm_copilot.ports.crm import CrmStore, Diarizer, RecordingFiles, Transcriber

logger = logging.getLogger(__name__)


class LocalRecordings:
    """Recordings referenced by a path this process can read."""

    @asynccontextmanager
    async def local_copy(self, reference: str) -> AsyncIterator[Path]:
        yield Path(reference)


class ProcessCallRecording:
    def __init__(
        self,
        store: CrmStore,
        transcriber: Transcriber,
        diarizer: Diarizer,
        files: RecordingFiles | None = None,
    ) -> None:
        self._store = store
        self._transcriber = transcriber
        self._diarizer = diarizer
        self._files = files or LocalRecordings()

    async def __call__(self, interaction_id: str) -> bool:
        """Returns whether a transcript was stored."""
        call = await self._store.get_interaction(interaction_id)
        if call is None:
            raise InteractionNotFound(interaction_id)
        if call.recording_path is None:
            return False
        if not call.recording_consent:
            # Egypt PDPL: the recording is not processed without the client's consent.
            logger.warning("Skipping recording for %s: no consent on file", interaction_id)
            return False

        async with self._files.local_copy(call.recording_path) as path:
            segments = await self._transcriber.transcribe(path)
            turns = await self._diarizer.diarize(path)
        await self._store.set_interaction_text(interaction_id, format_transcript(segments, turns))
        return True
