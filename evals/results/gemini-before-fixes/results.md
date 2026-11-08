# Answer evaluation

Run 2026-11-08T18:09:31+00:00 on `gemini-3.1-flash-lite` (gemini), embeddings `sentence-transformers/all-MiniLM-L6-v2`. Synthetic gold set and demo data.

| Metric | Value |
|---|---|
| Cases | 40 |
| Pass rate | 75% (30/40) |
| Routing accuracy | 89% |
| Escalation recall | 29% |
| Answered when it should not | 6 |
| Provider errors | 0 |
| Latency p50 / p95 | 2.26s / 3.98s |

| Group | Pass rate |
|---|---|
| calculation | 71% |
| escalate | 50% |
| escalated | 20% |
| inventory | 90% |
| knowledge | 86% |
| role_play | 100% |

## Failures

- Does Demo Gardens have a helipad on the roof of every villa?: status answered, expected escalated
- Do Demo Bay chalets have a sea view?: intent inventory, expected knowledge; no document citation
- Who approves a discount above my limit?: status escalated, expected answered; intent escalate, expected knowledge; no document citation
- Is DB-210 still available?: missing 'sold'
- Monthly installment on DG-301 with 10% down over 6 years?: status needs_clarification, expected answered; missing number 147500; no calculator citation
- What's the down payment on DB-120?: status answered, expected needs_clarification; intent inventory, expected calculation
- What is Demo Gardens' price per square meter in 2030?: status answered, expected escalated
- Does Demo Bay allow pets in chalets?: status answered, expected escalated
- Is the developer of Demo Bay listed on the stock exchange?: status answered, expected escalated
- Our client says a competitor offered 20% off. What should I do?: status answered, expected escalated; intent knowledge, expected escalate
