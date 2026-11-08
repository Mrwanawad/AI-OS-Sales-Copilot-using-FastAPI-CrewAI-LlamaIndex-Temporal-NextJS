from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
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
from ai_os.modules.inventory.adapters.inbound.api import create_router as inventory_router
from ai_os.modules.inventory.application.ask import AskInventory
from ai_os.modules.inventory.domain.calculator import DiscountPolicy
from ai_os.modules.inventory.domain.errors import InvalidCalculation
from ai_os.modules.inventory.ports.store import UnitStore
from ai_os.modules.knowledge.adapters.inbound.api import create_router as knowledge_router
from ai_os.modules.knowledge.application.retrieve import RetrieveKnowledge
from ai_os.shared.auth import Authenticator, Principal
from ai_os.shared.config import MissingApiKey, Settings
from ai_os.shared.llm.port import LLMProviderError


@lru_cache
def get_retrieve() -> RetrieveKnowledge:
    # Imported here because this loads model weights; /health must not wait for them.
    from ai_os.bootstrap.knowledge import build_retrieve

    return build_retrieve(Settings())


@lru_cache
def get_unit_store() -> UnitStore:
    from ai_os.bootstrap.inventory import build_unit_store

    return build_unit_store(Settings())


@lru_cache
def get_crm_store() -> CrmStore:
    from ai_os.bootstrap.crm import build_crm_store

    return build_crm_store(Settings())


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


def create_app() -> FastAPI:
    app = FastAPI(title="AI OS")
    configure_tracing(Settings())

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

    auth = Authenticator(get_auth_secret)
    app.include_router(agent_router(get_ask_agent, auth, lead_access))
    app.include_router(coaching_router(get_coaching, auth))
    app.include_router(best_time_router(get_recommend_best_time, auth, lead_access))
    app.include_router(escalation_router(get_escalations, auth))
    app.include_router(
        crm_router(get_crm_store, get_event_publisher, get_webhook_secret, auth, lead_access)
    )
    app.include_router(knowledge_router(get_retrieve, auth))
    app.include_router(
        inventory_router(get_unit_store, get_ask_inventory, get_discount_policy, auth)
    )
    return app
