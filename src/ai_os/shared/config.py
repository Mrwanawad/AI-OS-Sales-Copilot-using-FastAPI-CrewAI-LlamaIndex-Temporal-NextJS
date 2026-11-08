from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_os.shared.llm.port import LLMRole


@dataclass(frozen=True)
class ModelRoute:
    anthropic: str
    openai: str


class MissingApiKey(ValueError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    anthropic_router_model: str = "claude-haiku-5-5"
    anthropic_answer_model: str = "claude-opus-5-5"
    anthropic_extraction_model: str = "claude-sonnet-5-5"

    openai_router_model: str = "gpt-5.4-mini"
    openai_answer_model: str = "gpt-5.5"
    openai_extraction_model: str = "gpt-5.5"

    vector_store: Literal["qdrant", "pgvector"] = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    # SQLAlchemy URLs, e.g. postgresql+psycopg2://... and postgresql+asyncpg://...
    database_url: str | None = None
    async_database_url: str | None = None

    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Connects with a database role that has SELECT on units only.
    inventory_read_only_database_url: str | None = None

    # Placeholder limits for the demo. Replace with the brokerage's real policy.
    discount_limit_rep_pct: Decimal = Decimal("2")
    discount_limit_team_lead_pct: Decimal = Decimal("5")
    discount_limit_manager_pct: Decimal = Decimal("10")

    # Answers below this self-reported confidence go to a person. Calibrate on the eval set.
    agent_min_confidence: float = 0.5

    whisper_model: str = "large-v3"
    # Without it, transcripts still work but have no speaker labels.
    hugging_face_token: SecretStr | None = None
    crm_webhook_secret: SecretStr | None = None
    # Signs API tokens. At least 32 characters. The API stays closed without it.
    auth_secret: SecretStr | None = None

    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str | None = None

    business_timezone: str = "Africa/Cairo"
    # Written by scripts/train_best_time.py. Cards omit the estimate until it exists.
    best_time_model_path: str | None = None

    business_timezone: str = "Africa/Cairo"
    # Written by scripts/train_best_time.py. Cards omit the estimate until it exists.
    best_time_model_path: str | None = None

    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    # Prompts can hold client details, so traces leave them out by default.
    langfuse_capture_content: bool = False

    temporal_address: str = "localhost:7233"
    temporal_task_queue: str = "ai-os"

    def route_for(self, role: LLMRole) -> ModelRoute:
        routes = {
            LLMRole.ROUTER: ModelRoute(self.anthropic_router_model, self.openai_router_model),
            LLMRole.ANSWER: ModelRoute(self.anthropic_answer_model, self.openai_answer_model),
            LLMRole.EXTRACTION: ModelRoute(
                self.anthropic_extraction_model, self.openai_extraction_model
            ),
        }
        return routes[role]

    def require_anthropic_key(self) -> str:
        if self.anthropic_api_key is None:
            raise MissingApiKey("ANTHROPIC_API_KEY is not set")
        return self.anthropic_api_key.get_secret_value()

    def require_openai_key(self) -> str:
        if self.openai_api_key is None:
            raise MissingApiKey("OPENAI_API_KEY is not set")
        return self.openai_api_key.get_secret_value()
