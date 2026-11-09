import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ai_os.bootstrap.observability import configure_tracing
from ai_os.modules.agent.adapters.inbound.api import create_router as agent_router
from ai_os.modules.agent.domain.errors import LLMUnavailable as AgentLLMUnavailable
from ai_os.modules.agent.ports.tools import AnswerQuestion
from ai_os.modules.best_time.adapters.inbound.api import create_router as best_time_router
from ai_os.modules.coaching.adapters.inbound.api import create_router as coaching_router
from ai_os.modules.coaching.domain.errors import LLMUnavailable as CoachingLLMUnavailable
from ai_os.modules.coaching.ports.coaching import CoachingService
from ai_os.modules.crm_copilot.adapters.inbound.api import create_router as crm_router
from ai_os.modules.crm_copilot.application.access import can_access_lead
from ai_os.modules.crm_copilot.ports.crm import CrmStore, EventPublisher
from ai_os.modules.escalation.adapters.inbound.api import create_router as escalation_router
from ai_os.modules.escalation.application.service import Escalations
from ai_os.modules.identity.adapters.inbound.api import create_router as identity_router
from ai_os.modules.identity.application.service import Identity
from ai_os.modules.inventory.adapters.inbound.api import create_router as inventory_router
from ai_os.modules.inventory.application.ask import AskInventory
from ai_os.modules.inventory.domain.calculator import DiscountPolicy
from ai_os.modules.inventory.domain.errors import InvalidCalculation
from ai_os.modules.inventory.ports.store import UnitStore
from ai_os.modules.knowledge.adapters.inbound.api import create_documents_router
from ai_os.modules.knowledge.adapters.inbound.api import create_router as knowledge_router
from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.modules.knowledge.application.upload import UploadDocument
from ai_os.modules.knowledge.ports.retrieval import VectorStore
from ai_os.shared.auth import Authenticator, Principal, Revocations
from ai_os.shared.config import MissingApiKey, Settings
from ai_os.shared.llm.port import LLMProviderError


@lru_cache
def get_vector_store() -> VectorStore:
    # Imported here because this loads model weights; /health must not wait for them.
    from ai_os.bootstrap.knowledge import build_store

    return build_store(Settings())


@lru_cache
def get_retrieve() -> RetrieveKnowledge:
    from ai_os.bootstrap.knowledge import build_retrieve

    return build_retrieve(Settings(), get_vector_store())


@lru_cache
def _upload_document() -> UploadDocument:
    from ai_os.bootstrap.uploads import build_upload_document

    return build_upload_document(Settings(), get_vector_store)


@lru_cache
def get_recording_uploads():
    from sqlalchemy.ext.asyncio import create_async_engine

    from ai_os.bootstrap.uploads import StoredUploads
    from ai_os.shared.files import PostgresFileStore

    url = Settings().async_database_url
    if not url:
        raise HTTPException(status_code=503, detail="Set ASYNC_DATABASE_URL")
    return StoredUploads(PostgresFileStore(create_async_engine(url)))


def get_upload_document() -> UploadDocument:
    try:
        return _upload_document()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@lru_cache
def get_unit_store() -> UnitStore:
    from ai_os.bootstrap.inventory import build_unit_store

    return build_unit_store(Settings())


@lru_cache
def get_crm_store() -> CrmStore:
    from ai_os.bootstrap.crm import build_crm_store

    return build_crm_store(Settings())


@lru_cache
def _privacy():
    from ai_os.bootstrap.crm import build_privacy

    return build_privacy(Settings())


def get_privacy():
    try:
        return _privacy()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@lru_cache
def get_event_publisher() -> EventPublisher:
    from ai_os.bootstrap.crm import build_event_publisher

    return build_event_publisher(Settings())


@lru_cache
def get_coaching() -> CoachingService:
    from ai_os.bootstrap.coaching import build_coaching

    return build_coaching(Settings())


def get_recommend_best_time():
    from ai_os.bootstrap.best_time import build_recommend

    return build_recommend(Settings(), get_crm_store())


@lru_cache
def get_escalations() -> Escalations:
    from ai_os.bootstrap.escalation import build_escalations

    return build_escalations(Settings())


def get_webhook_secret() -> str | None:
    secret = Settings().crm_webhook_secret
    return secret.get_secret_value() if secret else None


def get_auth_secret() -> str | None:
    secret = Settings().auth_secret
    return secret.get_secret_value() if secret else None


def get_previous_auth_secret() -> str | None:
    secret = Settings().auth_previous_secret
    return secret.get_secret_value() if secret else None


@lru_cache
def _identity() -> Identity:
    from ai_os.bootstrap.identity import build_identity

    return build_identity(Settings())


def get_identity() -> Identity:
    try:
        return _identity()
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@lru_cache
def get_revocations() -> Revocations | None:
    from ai_os.bootstrap.identity import IdentityRevocations

    settings = Settings()
    # Without a database there is no user table, so only signatures are checked.
    if not settings.async_database_url or settings.auth_secret is None:
        return None
    return IdentityRevocations(_identity())


async def lead_access(user: Principal, lead_id: str) -> bool:
    return await can_access_lead(get_crm_store(), user, lead_id)


def get_ask_inventory() -> AskInventory:
    from ai_os.bootstrap.inventory import build_ask_inventory

    try:
        return build_ask_inventory(Settings(), get_unit_store())
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_ask_agent() -> AnswerQuestion:
    try:
        from ai_os.bootstrap.agent import build_ask_agent

        # Resolved first because it checks the API keys; get_retrieve loads model weights.
        ask_inventory = get_ask_inventory()
        return build_ask_agent(
            Settings(),
            get_retrieve(),
            get_unit_store(),
            ask_inventory,
            get_crm_store(),
            get_escalations(),
        )
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


def get_discount_policy() -> DiscountPolicy:
    settings = Settings()
    return DiscountPolicy(
        settings.discount_limit_rep_pct,
        settings.discount_limit_team_lead_pct,
        settings.discount_limit_manager_pct,
    )


logger = logging.getLogger(__name__)


def _load_models() -> None:
    try:
        get_retrieve()
    except Exception:
        # Search routes report the problem on first use; startup must not fail over it.
        logger.exception("Warm-up of the retrieval models failed")


@asynccontextmanager
async def _warm_up(_app: FastAPI) -> AsyncIterator[None]:
    """Loads model weights in the background, so /health answers at once and the first
    search or upload doesn't wait for them."""
    task = asyncio.create_task(asyncio.to_thread(_load_models))
    yield
    task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(title="AI OS", lifespan=_warm_up)
    settings = Settings()
    configure_tracing(settings)
    origins = [o.strip().rstrip("/") for o in (settings.cors_origins or "").split(",") if o.strip()]
    if settings.app_url:
        origins.append(settings.app_url.rstrip("/"))
    if origins:
        # Tokens travel in the Authorization header, not cookies, so no credentials mode.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def llm_problem(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(error)})

    for llm_error in (
        LLMProviderError,
        MissingApiKey,
        AgentLLMUnavailable,
        CoachingLLMUnavailable,
    ):
        app.add_exception_handler(llm_error, llm_problem)

    @app.exception_handler(InvalidCalculation)
    async def invalid_calculation(_request: Request, error: InvalidCalculation) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    auth = Authenticator(get_auth_secret, get_revocations, get_previous_auth_secret)
    app.include_router(identity_router(get_identity, auth))
    app.include_router(agent_router(get_ask_agent, auth, lead_access))
    app.include_router(coaching_router(get_coaching, auth))
    app.include_router(best_time_router(get_recommend_best_time, auth, lead_access))
    app.include_router(escalation_router(get_escalations, auth))
    app.include_router(
        crm_router(
            get_crm_store,
            get_event_publisher,
            get_webhook_secret,
            auth,
            lead_access,
            get_recording_uploads,
            get_privacy,
        )
    )
    app.include_router(knowledge_router(get_retrieve, auth))
    app.include_router(create_documents_router(get_upload_document, auth))
    app.include_router(
        inventory_router(get_unit_store, get_ask_inventory, get_discount_policy, auth)
    )
    return app
