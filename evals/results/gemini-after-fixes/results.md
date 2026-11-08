# Answer evaluation

Run 2026-11-08T18:19:47+00:00 on `gemini-3.1-flash-lite` (gemini), embeddings `sentence-transformers/all-MiniLM-L6-v2`. Synthetic gold set and demo data.

| Metric | Value |
|---|---|
| Cases | 40 |
| Pass rate | 85% (34/40) |
| Routing accuracy | 89% |
| Escalation recall | 57% |
| Answered when it should not | 3 |
| Provider errors | 0 |
| Latency p50 / p95 | 3.73s / 17.58s |

| Group | Pass rate |
|---|---|
| calculation | 86% |
| escalate | 50% |
| escalated | 60% |
| inventory | 100% |
| knowledge | 86% |
| role_play | 100% |

## Failures

- Does Demo Gardens have a helipad on the roof of every villa?: status answered, expected escalated
- Do Demo Bay chalets have a sea view?: status escalated, expected answered; intent inventory, expected knowledge; no document citation
- Who approves a discount above my limit?: status escalated, expected answered; intent escalate, expected knowledge; no document citation
- What's the down payment on DB-120?: status escalated, expected needs_clarification; intent inventory, expected calculation
- What is Demo Gardens' price per square meter in 2030?: status answered, expected escalated
- Our client says a competitor offered 20% off. What should I do?: status answered, expected escalated; intent knowledge, expected escalate
