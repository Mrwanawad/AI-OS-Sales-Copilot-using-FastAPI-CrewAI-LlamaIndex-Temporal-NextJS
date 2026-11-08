import asyncio


class ElevenLabsSpeech:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_multilingual_v2",
        client: object | None = None,
    ) -> None:
        if client is None:
            from elevenlabs.client import ElevenLabs

            client = ElevenLabs(api_key=api_key)
        self._client = client
        self._voice_id = voice_id
        self._model_id = model_id

    async def synthesize(self, text: str) -> bytes:
        return await asyncio.to_thread(self._synthesize, text)

    def _synthesize(self, text: str) -> bytes:
        audio = self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
        )
        return b"".join(audio)
