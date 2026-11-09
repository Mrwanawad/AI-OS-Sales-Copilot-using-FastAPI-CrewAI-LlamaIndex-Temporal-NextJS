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

    # anthropic_openai: Anthropic first, OpenAI as fallback (production).
    # gemini: Google's free tier, for evaluation runs without paid keys.
    llm_provider: Literal["anthropic_openai", "gemini"] = "anthropic_openai"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.1-flash-lite"

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

    # When reps may call, in BUSINESS_TIMEZONE. Placeholder: set the brokerage's policy.
    call_window_first_hour: int = 9
    call_window_last_hour: int = 21
    # Comma-separated weekday numbers, Monday = 0. Example without Friday: 0,1,2,3,5,6
    call_window_days: str = "0,1,2,3,4,5,6"

    # Answers below this self-reported confidence go to a person. Calibrate on the eval set.
    agent_min_confidence: float = 0.5

    whisper_model: str = "large-v3"
    # Without it, transcripts still work but have no speaker labels.
    hugging_face_token: SecretStr | None = None
    crm_webhook_secret: SecretStr | None = None
    # Signs API tokens. At least 32 characters. The API stays closed without it.
    auth_secret: SecretStr | None = None
    # Set to the old AUTH_SECRET while rotating, until tokens it signed have expired.
    auth_previous_secret: SecretStr | None = None
    auth_token_hours: int = 8

    # Ticket events are POSTed here, signed like the CRM webhook. Without it, tickets only queue.
    escalation_webhook_url: str | None = None
    escalation_webhook_secret: SecretStr | None = None
    # Approved answers leave search after this many days unless a team lead renews them.
    approved_answer_valid_days: int = 90
    # Where the web app runs, for links in notifications and CORS.
    app_url: str | None = None
    # Extra comma-separated browser origins allowed to call the API, e.g. Vercel previews.
    cors_origins: str | None = None

    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str | None = None

    # Egypt PDPL retention. Placeholders: the brokerage's legal team sets the real periods.
    # Interaction text and call recordings; times and call outcomes stay for best-time training.
    interaction_content_retention_days: int = 730
    # Context cards and handovers of leads with no interaction for this long.
    dormant_card_retention_days: int = 365
    # Answered and dismissed escalation tickets, which can quote a client's card.
    closed_ticket_retention_days: int = 365

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

    def require_gemini_key(self) -> str:
        if self.gemini_api_key is None:
            raise MissingApiKey("GEMINI_API_KEY is not set")
        return self.gemini_api_key.get_secret_value()

    def require_llm_keys(self) -> None:
        """Fails fast when the configured provider's keys are missing."""
        if self.llm_provider == "gemini":
            self.require_gemini_key()
            return
        self.require_anthropic_key()
        self.require_openai_key()

    def require_openai_key(self) -> str:
        if self.openai_api_key is None:
            raise MissingApiKey("OPENAI_API_KEY is not set")
        return self.openai_api_key.get_secret_value()
