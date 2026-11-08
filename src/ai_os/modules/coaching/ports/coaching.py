from collections.abc import Sequence
from typing import Protocol

from ai_os.modules.coaching.domain.models import Persona, RolePlaySession, Turn


class SessionStore(Protocol):
    async def save(self, session: RolePlaySession) -> None: ...

    async def get(self, session_id: str) -> RolePlaySession | None: ...


class BuyerActor(Protocol):
    async def reply(self, persona: Persona, turns: Sequence[Turn]) -> str: ...


class SpeechSynthesizer(Protocol):
    async def synthesize(self, text: str) -> bytes: ...


class CoachingService(Protocol):
    async def start(self, rep_id: str, persona_id: str) -> RolePlaySession: ...

    async def send(self, session_id: str, message: str) -> RolePlaySession: ...

    async def finish(self, session_id: str) -> RolePlaySession: ...

    async def get(self, session_id: str) -> RolePlaySession: ...

    async def turn_audio(self, session_id: str, turn_index: int) -> bytes: ...
