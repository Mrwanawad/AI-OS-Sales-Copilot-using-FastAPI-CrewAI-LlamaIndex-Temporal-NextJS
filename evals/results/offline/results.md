# Offline evaluation

Run 2026-11-08T17:38:08+00:00. No LLM involved. Synthetic cases on the invented demo data.

## Retrieval (15 gold queries, hybrid dense + BM25-style sparse)

| Metric | hashing embedder | all-MiniLM-L6-v2 + reranker |
|---|---|---|
| recall_at_5 | 1.0 | 1.0 |
| mrr | 1.0 | 1.0 |
| ndcg_at_10 | 1.0 | 1.0 |
| exact_code_recall_at_5 | 1.0 | 1.0 |
| arabic_recall_at_5 | 1.0 | 1.0 |

## SQL guard (74 cases; positive = blocked)

- Precision 1.0, recall 1.0
- 0 unsafe or unsupported let through, 0 safe or supported wrongly blocked

## Grounding check (30 cases; positive = rejected)

- Precision 0.933, recall 0.933
- 1 unsafe or unsupported let through, 1 safe or supported wrongly rejected
- Miss: `{"answer": "It costs EGP 5.2M... specifically 5,200,000.", "expected": "pass", "problems": ["numbers not found in the cited evidence: 5.2"]}`
- Miss: `{"answer": "Maintenance deposit is 8%.", "expected": "reject", "problems": []}`
