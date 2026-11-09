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
| Escalations (`/escalations/*`) | Ticket queue, answers back to the rep, two-person approval before an answer becomes knowledge. Shared answers leave search after a review period (90 days by default) unless a team lead renews them. Optional signed webhook push. |
| Privacy (`DELETE /crm/leads/{id}`, `scripts/apply_retention.py`) | Egypt PDPL: a manager erases a client and everything linked to them; a daily job clears old interaction text, recordings, dormant cards and closed tickets. |
| MCP (`scripts/mcp_server.py`) | Inventory, calculator and CRM tools over MCP stdio for clients such as Claude Code. |
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

## Evaluation results

All cases are synthetic and run on the invented demo data, so the numbers show how the
pipeline behaves, not how it performs for a real brokerage. Full reports are in `evals/results/`.

**Offline (no LLM)**, `python scripts/run_offline_evals.py --real-models`:

| Check | Cases | Result |
|---|---|---|
| SQL guard blocks unsafe queries | 50 attacks, 24 valid | precision 1.0, recall 1.0 |
| Grounding check rejects unsupported numbers | 30 answers | precision 1.0, recall 1.0 |
| Non-answer detector ("the documents don't mention it") | 63 replies, 49 from real Gemini runs | precision 1.0, recall 1.0 |
| Judgment-question detector (competitor offers, exceptions, promises, forecasts) | 22 to escalate, 28 policy questions to leave alone | precision 1.0, recall 1.0 |

Retrieval on 40 queries (paraphrases, typos, unit codes, Arabic, cross-lingual):

| Metric | Hashing embedder (CI) | all-MiniLM-L6-v2 + TinyBERT | bge-m3 + bge-reranker-v2-m3 (production) |
|---|---|---|---|
| Recall@5 | 0.91 | 0.93 | 1.00 |
| MRR | 0.80 | 0.90 | 1.00 |
| Cross-lingual recall@5 | 0.60 | 0.40 | 1.00 |
| Typo recall@5 | 0.50 | 1.00 | 1.00 |

There are only 7 demo documents, so the production models saturate; a real document set is
needed to tell them apart. The judgment detector was written against its own 50 cases, so its
score shows it covers them, not how it does on new phrasings. CI gates the SQL guard, grounding,
non-answer and judgment scores
(`evals/offline_thresholds.json`) and the hashing-embedder retrieval scores.

The first grounding run missed 2 of 30: "8%" passed because "8 years" was in the evidence, and
"5.2M" was rejected because the evidence says "5,200,000". Numbers now carry their scale
("5.2M", "3.9 million", "مليون") and percentages must match percentages.

**Answer gate on a real model**, `LLM_PROVIDER=gemini python scripts/run_answer_eval.py`,
40 cases on `gemini-3.1-flash-lite` (free tier) with small local embedding models:

| Metric | First run | After fixes |
|---|---|---|
| Pass rate | 75% | 85% |
| Routing accuracy | 89% | 89% |
| Escalation recall | 29% | 57% |
| Answered when it should have escalated | 6 | 3 |
| Inventory questions | 90% | 100% |

The first run found two bugs, both fixed:
- When the documents didn't cover a question, the agent replied "the documents don't mention
  it" as an answer, so no ticket reached a team lead. The composer now reports whether the
  evidence answers the question, and the agent escalates when it doesn't.
- The text-to-SQL prompt always filtered on available units, so "Is DB-210 still available?"
  couldn't see that it was sold. Questions naming unit codes now return their status.

The 3 remaining "answered when it should have escalated" cases were replies saying the
evidence is silent. A rule-based detector now escalates those even when the model marks them
answered. Re-checking the recorded Gemini replies with it gives 90% pass rate and 100%
escalation recall; that is a replay of the same replies, not a new model run. Production uses Anthropic with OpenAI as fallback, and this gate hasn't run on
those models yet.

## Deploy

- API, Temporal worker, retrain and cleanup jobs: `render.yaml` (Render Blueprint).
- Web app: Vercel with root directory `frontend` and `NEXT_PUBLIC_API_URL` set to the API URL.
  Put the Vercel URL in the API's `APP_URL`.
- Uploaded files and the promoted best-time model are stored in Postgres, so the API and worker
  need no shared disk.
- Daily cron jobs expire approved answers past their review period and apply PDPL retention.
  The retention periods (`*_RETENTION_DAYS`) are placeholders for the brokerage's legal team.
