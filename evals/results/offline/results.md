# Offline evaluation

Run 2026-11-09T16:43:23+00:00. No LLM involved. Synthetic cases on the invented demo data.

## Retrieval (40 gold queries, hybrid dense + sparse)

| Metric | hashing embedder |
|---|---|
| recall_at_5 | 0.9125 |
| mrr | 0.7992 |
| ndcg_at_10 | 0.8269 |
| semantic_recall_at_5 | 1.0 |
| exact_code_recall_at_5 | 1.0 |
| arabic_recall_at_5 | 0.875 |
| paraphrase_recall_at_5 | 1.0 |
| cross_lingual_recall_at_5 | 0.6 |
| typo_recall_at_5 | 0.5 |
| multi_recall_at_5 | 1.0 |

## SQL guard (74 cases; positive = blocked)

- Precision 1.0, recall 1.0
- 0 unsafe or unsupported let through, 0 safe or supported wrongly blocked

## Grounding check (30 cases; positive = rejected)

- Precision 1.0, recall 1.0
- 0 unsafe or unsupported let through, 0 safe or supported wrongly rejected

## Non-answer detector (63 cases; positive = escalated)

- Precision 1.0, recall 1.0
- 0 unsafe or unsupported let through, 0 safe or supported wrongly escalated

## Judgment-question detector (50 cases; positive = escalated)

- Precision 1.0, recall 1.0
- 0 unsafe or unsupported let through, 0 safe or supported wrongly escalated

## Replay of recorded gemini-after-fixes answers through today's checks

No model called: the same replies, re-checked. An estimate, not a new run.

- Pass rate 90%, escalation recall 100%
- Now escalated: Does Demo Gardens have a helipad on the roof of every villa?
- Now escalated: What is Demo Gardens' price per square meter in 2030?
- Now escalated: Our client says a competitor offered 20% off. What should I do?
