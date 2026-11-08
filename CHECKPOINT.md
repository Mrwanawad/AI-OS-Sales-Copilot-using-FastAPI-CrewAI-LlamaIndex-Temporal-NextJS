# Checkpoint — AI OS

Last updated: 2026-11-08, during step 9. Full design: `architecture.md`.

## Current State
- Build steps 1 (scaffold), 2 (Knowledge), 3 (Inventory), 4 (Agent), 5 (CRM Copilot), 6 (Coaching), 7 (Best-time model) and 8 (Escalation and memory loop) of 9 are done. Step 9 is partly done: everything buildable without API keys or your accounts is built, and nothing is deployed.
- 2026-11-08 session: added login, uploads, Postgres file store, webhook notifier, model publishing and the Next.js web app (see Step 9). The new and changed tests pass (unit suite 316 before the session, plus 30+ new unit and 5 new Postgres tests); a final full-suite run was cut short when the C: drive filled up (see Gotchas).
- 383 default tests passed before this session. 3 more use `-m models`; the embedding and Docling ones last ran in step 2, the Whisper one in step 5.
- `scripts/smoke_e2e.py` passes all 97 checks (84 before, plus login, logout, upload, worker ingestion from Postgres and re-upload versioning) on PGVector: real API and Temporal worker processes, real Postgres, real small models, over HTTP, with scripted LLMs. Qdrant server mode has not been run.
- Postgres tests run on an embedded Postgres 16 with pgvector (`pgserver`, dev dependency), or on `TEST_DATABASE_URL` when set. Docker is not needed.
- `ruff check` and all 6 `lint-imports` contracts pass.
- Local git repository on `main`, 2 commits, authored by Mrwan only. Not pushed; CI has never run. GitHub `AIOS-CRM-Copilot` holds an older, different version.
- No paid API keys. `LLM_PROVIDER=gemini` runs the answer gate on Gemini's free tier: 40 synthetic cases, 75% before and 85% after two fixes it surfaced (README, `evals/results/`). Offline evals: SQL guard 74/74, grounding 28/30, retrieval saturated at 1.0 on 15 queries.
- Public repo: https://github.com/Mrwanawad/AI-OS-Sales-Copilot-using-FastAPI-CrewAI-LlamaIndex-Temporal-NextJS

## Decisions
- **Application architecture:** modular monolith with Hexagonal (Ports & Adapters) modules, enforced by `import-linter`.
- **Agents:** CrewAI for routing and multi-step work, replacing LangGraph and LangChain.
- **Typed single-agent tasks:** PydanticAI.
- **Retrieval:** LlamaIndex on top of Qdrant (primary) and PGVector (second adapter). LlamaIndex is the retrieval layer, not a vector database.
- **LLM providers:** Anthropic (primary) and OpenAI (fallback) only.
- **Workflows:** Temporal instead of Celery; Redis stays for cache and rate limits.
- **Deployment:** Render and Vercel. No AWS, Terraform or Kubernetes.
- **Sparse search:** own BM25-style encoder with camel-tools normalization, instead of bge-m3 sparse output.

## Work Done

### Step 1: Scaffold
- 7 modules (`agent`, `knowledge`, `inventory`, `crm_copilot`, `coaching`, `best_time`, `escalation`), each with `domain`, `ports`, `application` and `adapters`. Only `knowledge`, `inventory` and `agent` have code.
- `src/ai_os/shared/config.py`: settings, including a model per LLM role.
- `src/ai_os/shared/llm/`: LLM port, PydanticAI adapter with `FallbackModel`, CrewAI adapter.
- `pyproject.toml` with extras, `uv.lock`, `.env.example`, `.github/workflows/ci.yml`, FastAPI `/health`.

### Step 2: Knowledge
- **Domain:** models, errors, chunking with stable chunk IDs.
- **Use cases:** `IngestDocument`, `RetrieveKnowledge`.
- **Adapters:** `LlamaIndexVectorStore`, Qdrant and PGVector stores, sparse encoder, cross-encoder reranker, Docling parser, price sheet parser, routing parser.
- **API:** `POST /knowledge/search`.
- **Temporal:** `IngestDocumentWorkflow`, activities, worker in `src/ai_os/bootstrap/worker.py`.
- **Tested behaviour:** new versions replace old ones, older versions are rejected, failed or empty parses keep the active version, bad price sheets fail without retry, Arabic variants and exact unit codes match.

### Step 3: Inventory
- **Calculator (`Decimal`):** installment plans, mortgages, discount limits by role. Final payments absorb rounding, so totals are exact.
- **Text-to-SQL:** `AskInventory` generates SQL through PydanticAI, validates it with `SqlGlotGuard`, gives the model one retry with the rejection reason, then runs it read-only.
- **Store:** `PostgresUnitStore`; `get_by_codes` is the link from chunk `unit_codes` to live rows.
- **Least-privilege role:** `deploy/inventory_reader.sql`, tested on real Postgres.
- **API:** `GET /inventory/units/{code}`, `POST /inventory/ask`, `POST /inventory/calculate/{installments,mortgage,discount}`.
- **MCP:** `src/ai_os/mcp_servers/inventory.py`, with unit lookup, ask, and calculator tools.
- **Tested behaviour:** 40+ SQL attack strings rejected, writes refused by the database even when the guard is bypassed, slow queries cancelled, reader role denied other tables and writes.

### Step 4: Agent
- **Flow:** `AnswerFlow` (CrewAI Flow) classifies each question through PydanticAI, then routes to knowledge, inventory, calculation, or escalation. Coaching, role-play and context cards reply that they aren't available yet.
- **Composer:** `CrewAnswerComposer` runs a CrewAI crew on the Anthropic LLM, then the OpenAI LLM if it fails.
- **Grounding:** `domain/grounding.py` rejects an answer with no citation, an unknown citation, or a number not in the cited evidence or the question. One repair attempt, then the text is withheld and the question escalates.
- **Calculation inputs:** numbers and unit codes the rep never stated are dropped, and the agent asks for them instead. A named unit always uses its live price.
- **Live over documents:** the knowledge tool attaches the live inventory row for each unit a document names.
- **API:** `POST /agent/ask`. Missing keys give `503`, as do provider outages.
- **Telemetry:** CrewAI telemetry and tracing are off, so client questions don't reach its service.

### Step 5: CRM Copilot
- **Context card:** every claim must cite source interactions, and its numbers must appear in them; unsupported claims are dropped and counted. Stated preferences drive matching against live available units.
- **Post-call:** allowlisted fields only. Stage moves forward only, never to reserved, won or lost. Follow-ups must fall within 180 days.
- **Recordings:** no consent means no transcription. Whisper ran for real (`tiny`, English synthesized audio).
- **Events:** HMAC-signed webhook with replay window, then a Temporal workflow keyed on event id, so redelivery is harmless.
- **Access:** reps reach only leads assigned to them. Reassignment moves access to the new rep.
- **Agent:** context card and lead coaching questions now answer from the stored card. Role-play questions point the rep to a practice session.
- **MCP:** `src/ai_os/mcp_servers/crm.py`.

### Step 6: Coaching
- **Personas:** 4 invented buyers in `domain/personas.py`. Practice never uses real client data.
- **Buyer:** `CrewBuyer` plays the persona through a CrewAI crew, moving to OpenAI if Anthropic fails. A failed reply saves nothing, so the rep can resend. Sessions cap at 20 rep turns.
- **Scoring:** the model proposes scores through PydanticAI. A score above 0 must quote the rep's exact words, or that criterion counts as not assessed. The overall score is computed in code and withheld if under half the rubric weight is verified.
- **Voice:** `ElevenLabsSpeech` speaks buyer turns on request. It is optional.
- **API:** `/coaching/*`. Sessions belong to the token holder; team leads can review but not play them.
- **Fix from step 4:** provider failures through PydanticAI returned `500`. Every LLM failure and missing key now returns `503` through app-level handlers.

### Step 7: Best-time model
- **Features:** `domain/features.py` is shared by training and serving, uses only earlier calls, and works in Cairo time with a Friday and Saturday weekend.
- **Training:** `scripts/train_best_time.py` splits by time, tunes LightGBM with Optuna, and logs to MLflow. It reads real CRM calls, or takes `--synthetic`.
- **Promotion gate:** the model is written and registered as `champion` only if its AUC beats an hour-and-weekday baseline by 0.01. Pure noise and too few calls are refused.
- **Serving:** `GET /best-time/leads/{id}` and the context card. A missing or broken model only leaves the estimate out.
- **Survival curve:** a lifelines Kaplan-Meier curve of attempts until first answer is logged to MLflow. Nothing in the app uses it yet.
- **Synthetic result (1,500 leads, 20 trials):** AUC 0.619 against a 0.521 baseline, top-decile lift 1.35. These measure a pattern planted in invented data, so they say nothing about real reps.

### Step 8: Escalation and memory loop
- **Tickets:** the agent's escalations are stored in Postgres with the project, lead and context card. Questions no longer go into logs.
- **Memory loop:** a lead answers a ticket and can propose the answer for shared knowledge. It goes live only after a different person approves it.
- **Screening:** phone numbers, emails and national IDs block sharing, at proposal and again at approval. Answers about a specific lead can't be shared.
- **Revoke:** an approved answer can be taken back out of search, leaving other knowledge alone.
- **Concurrency:** status-checked updates mean only one of two simultaneous reviews lands.
- **General knowledge:** project `general` now reaches searches for every project.
- **API:** `/escalations/*`. The queue and reviews need team lead or manager; reps see their own tickets at `/escalations/mine`.

### Step 9: Partly done
- **Auth:** signed tokens (HS256, expiry required, unique `jti`) carry identity and role on every route but `/health`, `/auth/login` and the HMAC-signed webhook. Roles and names in request bodies are ignored. The API stays closed if `AUTH_SECRET` is missing or short. `AUTH_PREVIOUS_SECRET` is accepted during rotation.
- **Accounts (`identity` module):** scrypt password hashes, `/auth/login`, `/auth/logout` (revokes that token), `/auth/me`, `/auth/password`. Managers create, disable, enable and reset users and end sessions. 5 failures lock an account for 15 minutes. Unknown users, disabled users and tokens issued before a reset are refused. `scripts/create_user.py` makes the first manager; `seed_demo.py` creates demo logins when `DEMO_PASSWORD` is set.
- **Uploads:** `POST /knowledge/documents` (team lead or manager) stores the file in Postgres (`stored_files`), reserves the next version atomically (`knowledge_documents`), and starts `IngestDocumentWorkflow` with `stored:<id>`. The worker copies it to a temp file. Status at `/knowledge/documents/{id}/versions/{v}`. Call recordings can use `stored:` references too. The API and worker need no shared disk.
- **Best-time model:** `train_best_time.py --publish` stores a promoted model as `stored:best-time-champion`; the API and worker check its hash every 10 minutes and reload it. Render cron retrains weekly (Fridays 01:00 UTC).
- **Escalation push:** `ESCALATION_WEBHOOK_URL` gets HMAC-signed `escalation.opened` and `escalation.answered` events with ids and a link, never the question or answer. Failures are logged; the ticket stays in the queue.
- **Web app (`frontend/`):** Next.js 16, login, workspace (leads, agent chat with citations, context card), escalation queue and knowledge approvals, document upload with status polling. `npm run build` passes. CI job added.
- **Migrations:** Alembic in `migrations/`. A test proves upgrade, no drift from the code, and downgrade.
- **Demo data:** `demo_data/` holds invented projects, units and clients. `scripts/seed_demo.py` loads them through the real pipeline.
- **Retrieval gate:** `scripts/run_eval.py` fails below `evals/thresholds.json`, and CI runs it with a hashing embedder.
- **Answer gate:** `scripts/run_answer_eval.py` and `evals/gold_answers.jsonl`, written but never run: it needs keys.
- **Tracing:** Langfuse through PydanticAI, off without keys. Prompts stay out of traces by default.
- **Packaging:** `Dockerfile`, `docker-compose.yml`, `render.yaml` (API, worker, two cron jobs, Qdrant). Linux now gets CPU-only PyTorch, so no CUDA wheels. The image built once on 2026-11-08 but filled the disk; the compose stack has not been run.

## Due

### Build Steps
- [ ] 9. Remaining: run the answer gate with keys, replace synthetic gold sets with team-lead-graded ones, decide on Ragas, run the Docker stack, push to GitHub, deploy to Render and Vercel.
- [x] Next.js frontend. Done 2026-11-08; not yet tried against a live API with real LLM answers.
- [x] Document upload route. Done 2026-11-08.
- [ ] Re-run the full test suite once disk space is back.

### Must Fix Before Deploying
- [x] Login, logout, session revocation and secret rotation. SSO is still not built.
- [ ] Discount limits in `src/ai_os/shared/config.py` (2%, 5%, 10%) are invented placeholders. Replace them with the brokerage's real policy.
- [ ] The calling window in `best_time/application/recommend.py` (09:00 to 21:00, all week) is a placeholder. Replace it with the brokerage's policy.
- [x] Model reaches API and worker through Postgres; weekly retrain on Render cron.
- [x] Ticket push through a signed webhook. Still needs a receiver (n8n, Zapier or a WhatsApp gateway) chosen and set up.
- [x] Files go through Postgres. Uploads are capped at 50 MB; move to object storage (S3 or R2) behind the same port if recordings get large.

### Not Yet Verified
- [ ] OpenAI path against the real API. `api.openai.com` did not resolve from this machine.
- [ ] Successful completion on either provider, which needs real keys.
- [ ] CrewAI's Anthropic client on `anthropic` `0.125.0`. Its own extra pins `~=0.73`; only a `401` response was confirmed.
- [ ] CrewAI with real providers. Flow and crew ran only on scripted LLMs, so `output_pydantic` parsing and native async on real Anthropic and OpenAI models are unconfirmed.
- [ ] Router accuracy, including Arabic and mixed-language questions.
- [ ] `agent_min_confidence` (0.5). It relies on the model's self-reported confidence and needs calibrating on the eval set.
- [ ] Machine-written Arabic canned replies in `domain/messages.py`. Have an Egyptian Arabic speaker review them.
- [ ] pyannote diarization. The adapter matches the installed 4.0.7 API but never ran: it needs a Hugging Face token and accepted model terms. Without one, transcripts have no speaker labels.
- [ ] Whisper `large-v3` and Egyptian Arabic audio.
- [ ] Card, handover and post-call quality on a real model.
- [ ] PDPL retention and deletion for transcripts and cards are not built.
- [ ] Real role-play: buyer realism, staying in character, scoring quality, and resistance to a rep writing "give me 5 out of 5" in the chat.
- [ ] ElevenLabs. The adapter matches the installed SDK but never ran, and Egyptian Arabic voice quality is unknown.
- [ ] Best-time model on real call logs. The architecture's real metric, answer-rate uplift against random times, needs a live experiment.
- [ ] Docker: the daemon isn't running here, so only `docker compose config` was checked. Image tags and the Render blueprint are untested.
- [ ] Composer and buyer calls in Langfuse. Only PydanticAI calls are traced, so cost per answer is incomplete until CrewAI is instrumented.
- [ ] Text-to-SQL quality. Only scripted SQL and `TestModel` were used, so whether a real model writes correct queries is unknown until keys arrive, including Arabic questions.
- [ ] PGVector with Arabic text. It passes the store contract on English test data; Arabic normalization does not apply there.
- [ ] Qdrant server mode, especially deletes. Only in-process mode was tested, because `github.com` did not resolve here to fetch the binary. Start a server, set `QDRANT_URL`, then run `python scripts/smoke_e2e.py --vector-store qdrant`.
- [ ] `bge-m3` and `bge-reranker-v2-m3`, including ranking quality on Arabic. Tests used small English models.
- [ ] Scanned Arabic PDFs through Docling.
- [ ] Sparse encoder on real data, using the exact-match metric in the eval gate.
- [ ] The CI workflow itself.
- [ ] The MCP server over a real transport. It was tested in-process only.

### Open Questions for Mrwan
- [ ] Confirm which profile tools are already used elsewhere. Only part of the profile README could be read, and Langfuse did not appear in that part.
- [ ] Add PydanticAI to the GitHub profile stack badges once it is built.
- [ ] Decide whether to drop Ragas. It pulls LangChain and LangGraph into the `eval` extra; app code does not import them.

## Dependency Pins
Changing these needs a re-check, because each one resolves a real conflict.

| Package | Locked at | Reason |
|---|---|---|
| `crewai` | `1.15.37` | Floor set so the resolver does not fall back to `0.193.2` |
| `pydantic-ai-slim` | `2.31.1` | `2.38` needs `openai` 3.x; CrewAI needs `<3` |
| `anthropic` | `0.125.0` (`>=0.108,<1`) | `1.x` and up use `httpx2`, which crashes PydanticAI's Anthropic client |
| `pydantic` | `2.12.5` | CrewAI requires `<2.13` |
| `fastmcp` | `3.4.11` | `fastmcp` 4 conflicts with CrewAI's `mcp` requirement |

## Run It
Set `PYTHONUTF8=1` on Windows, because CrewAI's emoji logging throws `charmap` errors otherwise.

```bash
uv sync --extra agents --extra retrieval --extra ingestion --extra arabic --extra workflows --extra inventory --extra mcp
uv run ruff check .
uv run lint-imports
uv run pytest              # default suite; the Temporal tests download a test server on first run
uv run pytest -m models    # real weights and a real PDF; took about 7 minutes with downloads
uv run python scripts/smoke_e2e.py   # full wiring check; add --vector-store qdrant with a server running
```

Network from this machine has intermittent DNS failures. If `uv` fails with `dns error`, retry it.

## Known Limits
- Grounding checks numbers and citations only. A wrong claim with no number, such as an invented policy, can still pass. The eval gate (step 9) has to measure this.
- Grounding can over-escalate: "5.75 million" doesn't match `5750000`. It fails towards a person, not a wrong number.
- Calculations ask again when a value is written in words ("5 million", "ten percent"), because only digits count as stated.
- Discount checks are not routed through the agent, because they need a trusted role from auth.

- Coaching scores cannot check product accuracy, because the rubric has no reference facts. Reps can state a wrong price and not lose points.
- The retrieval gate scores 1.0 on an easy synthetic set with a hashing embedder. It catches broken wiring, not poor retrieval quality.
- Two simultaneous messages in one session can overwrite each other, because saves are last-write-wins.
- All personas share one ElevenLabs voice, and audio is not cached, so each request spends credits.

- Top-decile lift is observational: reps chose when to call, so it isn't the effect of following the advice.
- Best-time probabilities in the API are not calibrated, so the card shows only the time, not a percentage.
- The card's best time is set when the card is built, so it can age until the next interaction.

- The personal data screen catches phones, emails and national IDs only. Client names depend on the human reviewer.
- A crash between claiming an approval and publishing leaves an answer marked approved but not searchable. A publish that fails normally returns it to pending.
- Approved answers don't expire, so one quoting a price can go stale until someone revokes it.

## Gotchas
- 2026-11-08: the first Docker build used `chown -R app /app`, which copied the whole venv into a second layer. Docker's disk (`docker_data.vhdx`, 36.6 GB) filled the C: drive and Docker reported I/O errors. Docker and WSL were shut down. The `chown` is removed from the Dockerfile. Before building again, prune Docker and compact its disk.
- LlamaIndex overwrites a `document_id` metadata key with its own reference ID, so the store uses `source_document_id`.
- Qdrant's `scroll` defaults to 10 points, so `active_version` passes an explicit limit.
- CrewAI's `LLM` has no provider failover, so the agent module must retry with OpenAI itself.
- PGVector hybrid search does not apply the Arabic normalization used for Qdrant.
- PGVector's `aget_nodes` has no `limit` argument, unlike Qdrant's, so the adapter checks the signature.
- `sqlglot` models `AND` and `OR` as functions, so the guard's function allowlist must include them.
- CrewAI's `Flow` is a Pydantic model: dependencies must be `PrivateAttr`s set after `super().__init__()`.
- CrewAI retries a failing LLM 3 times before raising, so failover to OpenAI happens after those retries.
- MLflow 3 can only register logged models, so `runs:/` URIs to plain artifacts fail with `Unable to find a logged_model`.
- Never set `OTEL_SDK_DISABLED`: it stops Langfuse silently. `CREWAI_DISABLE_TELEMETRY` alone turns off CrewAI's telemetry.
- `pgserver`'s `psql()` helper breaks on the space in this folder's path. Connect with a driver instead.
