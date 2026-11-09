# AI OS — Sales Team-Lead Clone + CRM Copilot — Architecture

## Scenario
New sales reps in a real estate brokerage take 5–6 months to close their first deal. They are new to the compounds, developers, payment plans, and objection handling, so they keep interrupting the team lead and dig through dozens of Excel sheets and PDFs. AI OS is one internal AI service for 100% of the sales team: a 24/7 "team-lead twin" that knows every project, unit, price, policy, and the team lead's playbook. When a new project launches, its documents are ingested once and every rep can sell it the same day.

CRM Copilot module: when a client is reassigned between follow-ups, the new rep has no context. Before the rep clicks "Call", the CRM shows a client context card built from every past call, message, and meeting.

## Input
- Rep question (text or voice), from web UI, WhatsApp, or CRM sidebar.
- Knowledge: developer brochures, price sheets, payment plans, master plans, FAQs, policies, team-lead playbook.
- Live inventory database (units, prices, availability).
- CRM: lead records, stage, notes, site visits, assignments.
- Interactions: call recordings, WhatsApp/email threads, meeting notes, call logs (answered/missed timestamps).

## Output
- Grounded answer with citations (unit IDs, document pages), confidence score.
- Recommended units, installment/mortgage calculations, next best action.
- Role-play session with scoring (onboarding mode), text or voice.
- Escalation ticket to team lead with summarized context.
- Client context card (JSON → CRM sidebar / dialer pop-up): profile, interaction timeline, stated needs, closing drivers, open concerns with suggested answers, promises made, best time to call, top matched units, talking points.
- Handover summary on reassignment; post-call summary + auto-updated CRM fields + scheduled follow-up.

## Application Architecture
Modular monolith with Hexagonal (Ports & Adapters) modules. One deployable, split by business area; each module keeps its domain free of frameworks and vendors.

```
src/ai_os/
  bootstrap/            # composition root: settings pick adapters
  shared/               # config, errors, tracing, LLM port + adapters
  modules/
    agent/              # CrewAI router flow
    knowledge/          # RAG + ingestion
    inventory/          # guarded text-to-SQL, calculator
    crm_copilot/        # context card, handover, post-call
    coaching/           # role-play
    best_time/          # LightGBM + survival model
    escalation/         # tickets, knowledge approval
  workflows/            # Temporal workflows and activities
  mcp_servers/          # FastMCP servers

Each module:
  domain/               # pure Python entities and rules
  ports/                # Protocols the module needs from the outside
  application/          # use cases; imports domain and ports, never adapters
  adapters/
    inbound/            # FastAPI routers
    outbound/           # vendor implementations of ports
```

Rules, enforced by `import-linter` in CI:
- Dependencies point inward: adapters > application > ports > domain.
- Modules do not import each other. The agent module defines ports for what it needs; `bootstrap` wires them to other modules.
- `domain` imports no frameworks or vendor SDKs. `shared` imports no modules.
- A port exists only for a real second implementation or a test fake.
- CrewAI and PydanticAI are deliberate direct dependencies of `application`; wrapping them adds nothing.

## System Architecture

```
 Channels: Web UI | WhatsApp | CRM sidebar / dialer
        |
        v
 [API Gateway - FastAPI]  auth, role-based access (rep / team lead / manager)
        |
        v
 [Router flow - CrewAI Flows]
   intent: knowledge | inventory | lead coaching | calculation | role-play | context card | escalate
        |
        +--> [Knowledge RAG]     LlamaIndex retrievers over Qdrant or PGVector -> cross-encoder rerank
        +--> [Inventory tool]    guarded text-to-SQL over live units (PydanticAI)
        +--> [CRM tool]          lead history, stage, notes, assignments (FastMCP server)
        +--> [Calculator tool]   installments, mortgage, discount limits (deterministic Python)
        +--> [Role-play]         points to /coaching sessions: buyer crew + scoring; ElevenLabs voice
        +--> [Context builder]   transcripts + threads + notes -> structured client card (PydanticAI)
        +--> [Best-time model]   LightGBM / survival on answered vs. missed calls
        +--> [Escalation tool]   ticket + summary to team lead
        |
        v
 [Answer composer]  citations, confidence, guardrails (no invented prices/policies)
        |
        v
 [Memory loop]  team-lead answers to escalations -> reviewed -> knowledge base

 Ingestion pipeline (new project onboarding, Temporal workflow):
   PDFs / Excel / images -> Docling parse -> pandera validation -> chunk + embed (bge-m3)
   -> LlamaIndex index into active vector store
   -> version tag (old price sheets retired) -> link to inventory rows

 Interaction pipeline (CRM Copilot, Temporal workflow):
   CRM webhook (lead opened / reassigned / call ended)
   -> Whisper transcription + pyannote diarization -> PydanticAI structured summary
   -> context card + handover summary -> CRM widget; post-call CRM field update

 Observability: Langfuse traces, eval dashboard, cost per answer
```

## Framework Responsibilities
- CrewAI: routing and multi-step agent work (router flow, role-play crew).
- PydanticAI: typed single-agent tasks with schema-validated output (context card, handover and post-call summaries, CRM field extraction, text-to-SQL step).
- LlamaIndex: retrieval layer. Not a vector database; it runs on top of Qdrant and PGVector behind the `VectorStore` port.
- Temporal: durable background pipelines. CrewAI handles in-request agent work.

## Retrieval Notes
- Qdrant hybrid search: bge-m3 dense vectors plus a BM25-style sparse encoder built on camel-tools normalization, so Arabic spelling variants, Arabic digits, and unit codes match exactly.
- PGVector hybrid search uses Postgres full-text search (`simple` config), which does not apply that normalization. Qdrant is the primary store for this reason.
- Price sheets skip Docling: pandera validates each workbook, and each row becomes one chunk, so a price never separates from its unit. One bad row rejects the file and the active version stays live.

## Inventory Safety
Text-to-SQL has three independent layers:
- Guard (`sqlglot`): one SELECT on `units` only, allowed columns, functions and casts only, row cap, comments stripped.
- Transaction: `SET TRANSACTION READ ONLY` with a statement timeout.
- Database role: `inventory_reader` (`deploy/inventory_reader.sql`) has SELECT on `units` and nothing else.

Prices and calculations come from the database and `Decimal` code, never from model output. Live inventory is the source of truth over any price found in a document.

## Answer Grounding
Deterministic checks run on every composed answer, after the model and before the rep:
- It must cite at least one evidence item, and every cited id must exist.
- Every number must appear in the cited evidence or the rep's question.
- It gets one repair attempt. After that the text is withheld and the question escalates.
- Answers below `agent_min_confidence` also escalate, as do questions with no evidence.

These checks cover numbers and citations only. Non-numeric claims are measured by the eval gate.

Two rule-based backstops escalate what a model tends to answer anyway:
- Replies that only say the evidence is silent ("the documents don't mention it").
- Questions asking for a person's decision: matching a competitor, a discount or exception to approve, a promise beyond the contract, or a price forecast. These skip the router, so policy text can't be stretched into a decision.

## CRM Copilot Safety
- Every card claim cites source interactions, and its numbers must appear in them. Unsupported claims are dropped and counted.
- Post-call updates touch allowlisted fields only. Automation moves a stage forward only, and never to reserved, won or lost.
- A recording without consent on file is not transcribed.
- Webhooks are HMAC-signed with a replay window. The event id makes redelivery harmless.
- Reps reach only leads assigned to them; team leads and managers reach all.

## Personal Data (Egypt PDPL)
- Erasure: a manager's `DELETE /crm/leads/{id}` removes the lead, its interactions, card, handovers, follow-ups, stored recordings and escalation tickets in one request. The source CRM and dialer copies must be erased there too.
- Retention, applied daily: interaction text and recordings after `INTERACTION_CONTENT_RETENTION_DAYS` (call times and outcomes stay for the best-time model), cards and handovers of leads dormant for `DORMANT_CARD_RETENTION_DAYS`, closed tickets after `CLOSED_TICKET_RETENTION_DAYS`. The defaults are placeholders for the legal team.
- Prompts and answers stay out of traces unless `LANGFUSE_CAPTURE_CONTENT` is on.

## Coaching Scoring
- Personas are invented, so practice uses no client data.
- A score above 0 must quote the rep's exact words, or that criterion is not assessed.
- The overall score is computed in code from fixed weights, and withheld if under half the rubric weight is verified.

## Best-Time Model
- Training and serving share one feature function, which only sees earlier calls.
- Data splits by time, so the model always predicts calls newer than those it learned from.
- A model is promoted only if it beats an hour-and-weekday baseline on the newest calls.
- Synthetic results prove the pipeline only. Real uplift needs a live experiment against random times.

## Memory Loop Safety
- Escalations are stored as tickets, and questions stay out of logs.
- An answer reaches shared knowledge only after a different person approves it.
- Phone numbers, emails and national IDs block sharing, and answers about a specific lead can't be shared.
- Approved answers can be revoked, which removes them from search.
- Approved answers leave search after a review period (`APPROVED_ANSWER_VALID_DAYS`, 90 by default) unless a team lead renews them, so a quoted price can't stay live forever.

## Access Control
- Signed tokens carry identity and role on every route except `/health` and the HMAC-signed webhook. Names and roles in request bodies are ignored.
- The API stays closed if `AUTH_SECRET` is missing or short.
- Discount limits, the two-person review rule and session ownership all rest on the token.

## LLM Service
- Providers: Anthropic (primary) and OpenAI (fallback) only.
- One LLM port with one adapter per framework: PydanticAI uses `FallbackModel`; CrewAI uses its `LLM` class with the same provider settings. CrewAI's `LLM` has no provider failover, so the agent module retries with the fallback LLM.
- Model per role comes from settings, so the router can use a small Anthropic model and answers a larger one.
- Tests use PydanticAI `TestModel` and fakes, not a local LLM.

## Tools Used
- LLM & agents: Anthropic, OpenAI, CrewAI, PydanticAI, FastMCP (CRM, inventory, docs servers)
- Retrieval: LlamaIndex, Qdrant (primary), PostgreSQL + PGVector (second adapter), bge-m3 embeddings, cross-encoder reranker (bge-reranker)
- Arabic: camel-tools normalization
- Documents: Docling / Unstructured, pandas, openpyxl, pandera
- Speech: Whisper large-v3 (or API), pyannote.audio, ElevenLabs (role-play voice)
- ML: LightGBM, lifelines, Optuna, MLflow (best-time-to-call)
- Backend: FastAPI, Pydantic, Temporal, Redis (cache, rate limits), PostgreSQL, SQLAlchemy, sqlglot (SQL guard)
- Integrations: CRM APIs/webhooks (Salesforce / HubSpot / Zoho / in-house), Twilio or local cloud telephony, WhatsApp Business API
- Frontend: Next.js (chat + CRM widget mock)
- Evaluation & observability: own retrieval metrics (recall@k, MRR, per-tag gates), DeepEval (pytest release gate), custom gold Q&A set, LLM-as-judge, Langfuse
- DevOps: Docker, GitHub Actions, pytest, ruff, import-linter

## Tools Not Used (and why)
- LangChain, LangGraph: replaced by CrewAI for agent work.
- Ragas: dropped. It pulled LangChain and LangGraph in, and `scripts/run_eval.py` already computes the retrieval metrics against the gold set.
- AutoGen, Google ADK, Microsoft Agent Framework: would duplicate CrewAI.
- Celery: Temporal gives durable retries across multi-step pipelines.
- Groq, Ollama and other LLM providers: LLM service is limited to Anthropic and OpenAI.
- LangSmith, Weights & Biases: duplicate Langfuse and MLflow.
- Pinecone, ChromaDB, FAISS: two vector stores are enough to prove the port.
- Fine-tuning an LLM on company data: knowledge changes daily; RAG + tools stay fresh and auditable.
- LLM arithmetic: all prices/installments come from the calculator tool.
- Unrestricted text-to-SQL: schema-limited, read-only queries only.
- External profiling of clients (social scraping): only data the client shared with the company.
- Pure vector search: Arabic/English unit codes and compound names need sparse or BM25 search too.
- AWS, Terraform, Kubernetes: cost without benefit for a demo deployment.

## Methodologies
- Agentic RAG with tool routing; citation-first answers.
- Human-in-the-loop escalation and knowledge approval.
- Offline eval set (gold Q&A graded by team leads) gating every release.
- Role-based access control; PII minimization; Egypt PDPL (Law 151/2020) consent and retention; call-recording consent.

## Evaluation Metrics
- Answer accuracy on gold Q&A, citation faithfulness, hallucination rate.
- Retrieval: recall@k, MRR, NDCG@10; exact-match recall on Arabic/English unit codes (decides whether Qdrant sparse search is enough or BM25 is needed).
- Context card: summary faithfulness (human-graded), field completeness.
- Best-time model: answer-rate uplift at predicted times vs. random.
- Ops: p95 latency, cost per answer, escalation rate.
- Business: time-to-first-close for new reps, team-lead interruptions per rep per week, reassigned-lead conversion, rep prep time before calls.

## Production Monitoring
- Langfuse traces: latency, token cost, tool errors.
- Unanswered / low-confidence question log → knowledge gaps.
- Stale-document detector (price sheets older than active version).
- Thumbs up/down feedback, escalation trends, drift in question topics.

## Deployment Plan
- Backend: Render (Docker web service + Temporal worker + Temporal server + Qdrant + managed Postgres with PGVector, Redis).
- Frontend: Vercel (Next.js chat UI + CRM sidebar demo).
- CI/CD: GitHub Actions (tests, import-linter, retrieval gate). Deploy on main is not set up.
- Migrations: Alembic, run before each deploy.
- Demo data: sample compounds, units, synthetic leads and conversations clearly marked synthetic.

## Build Order
1. Scaffold modules, ports, import-linter.
2. Knowledge (ingestion + retrieval).
3. Inventory (text-to-SQL + calculator).
4. Agent (router flow + answer composer).
5. CRM Copilot.
6. Coaching.
7. Best-time model.
8. Escalation + memory loop.
9. Eval gate and deployment.
