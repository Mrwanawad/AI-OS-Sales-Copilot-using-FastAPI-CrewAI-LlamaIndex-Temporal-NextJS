# AI OS: Sales Team-Lead Copilot and CRM Copilot

An internal AI service for a real estate brokerage's sales team. Reps ask about projects, prices,
payment plans and live inventory and get answers with citations. Before a call, they see a
context card built from every past call and message with the client. Questions the system can't
answer safely go to a team lead, and approved answers become shared knowledge.

Design and decisions: [architecture.md](architecture.md). Build status: [CHECKPOINT.md](CHECKPOINT.md).
All projects, units and clients in `demo_data/` are invented.

## What it does

| Area | What a user gets |
|---|---|
| Agent (`/agent/ask`) | Routes a question to documents, inventory SQL, the calculator, the context card or a person. Every number must appear in cited evidence, or the answer is withheld and escalated. |
| Knowledge (`/knowledge/*`) | Hybrid dense and sparse retrieval with reranking, Arabic normalization, versioned documents. Team leads upload files; ingestion runs on a Temporal worker. |
| Inventory (`/inventory/*`) | Guarded read-only text-to-SQL, installment, mortgage and discount calculations in `Decimal`. |
| CRM Copilot (`/crm/*`) | Signed CRM webhooks, Whisper transcription with consent checks, context cards, handover and post-call summaries. |
| Coaching (`/coaching/*`) | Role-play with invented buyers and rubric scores that must quote the rep. |
| Best time to call (`/best-time/*`) | LightGBM model trained on call outcomes, published only if it beats a baseline. |
| Escalations (`/escalations/*`) | Ticket queue, answers back to the rep, two-person approval before an answer becomes knowledge. Optional signed webhook push. |
| Accounts (`/auth/*`) | Username and password login, logout, ending a user's sessions, lockout after repeated failures. |

## Stack

FastAPI, CrewAI, PydanticAI, LlamaIndex, Qdrant or PGVector, Temporal, PostgreSQL, LightGBM and
MLflow, Whisper, FastMCP, Next.js. LLMs: Anthropic first, OpenAI as fallback.

## Run locally

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra agents --extra retrieval --extra ingestion --extra arabic --extra workflows --extra inventory --extra mcp --extra ml
uv run pytest
uv run python scripts/smoke_e2e.py
```

The tests and the smoke check start their own Postgres (`pgserver`) and Temporal dev server and
need no API keys or Docker.

Full stack with Docker:

```bash
cp .env.example .env
# Set AUTH_SECRET (32+ characters), CRM_WEBHOOK_SECRET and the LLM keys in .env
POSTGRES_PASSWORD=... INVENTORY_READER_PASSWORD=... docker compose up --build
docker compose exec -e DEMO_PASSWORD=... api python scripts/seed_demo.py
```

Web app:

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Sign in at http://localhost:3000 as `rep-a`, `lead-a` or `manager-a` with `DEMO_PASSWORD`.

## Accounts

Create the first manager with `python scripts/create_user.py <username> "<Name>" manager`.
Managers then add, disable and reset users through `/auth/users`. Disabling a user or changing a
password ends that user's existing sessions.

## Deploy

- API, Temporal worker, retrain and cleanup jobs: `render.yaml` (Render Blueprint).
- Web app: Vercel with root directory `frontend` and `NEXT_PUBLIC_API_URL` set to the API URL.
  Put the Vercel URL in the API's `APP_URL`.
- Uploaded files and the promoted best-time model are stored in Postgres, so the API and worker
  need no shared disk.
