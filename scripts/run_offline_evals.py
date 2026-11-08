"""Evaluations that need no LLM: retrieval, the SQL guard and the grounding check.

Usage: python scripts/run_offline_evals.py [--real-models] [--output DIR]

All cases are synthetic and run on the invented demo data. Writes results.json and
results.md to --output.
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from run_eval import evaluate as evaluate_retrieval

from ai_os.modules.agent.domain.grounding import find_grounding_problems
from ai_os.modules.agent.domain.models import Evidence, EvidenceItem, SourceKind
from ai_os.modules.inventory.adapters.outbound.sql_guard import SqlGlotGuard
from ai_os.modules.inventory.domain.errors import UnsafeSql

ROOT = Path(__file__).parents[1]
EVALS = ROOT / "evals"


def _lines(name: str) -> list[dict]:
    text = (EVALS / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _confusion(rows: list[tuple[bool, bool]]) -> dict:
    """rows: (should_flag, flagged). Flagging is the positive class."""
    tp = sum(s and f for s, f in rows)
    fp = sum(not s and f for s, f in rows)
    fn = sum(s and not f for s, f in rows)
    tn = sum(not s and not f for s, f in rows)
    return {
        "cases": len(rows),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
    }


def evaluate_sql_guard() -> dict:
    guard = SqlGlotGuard(max_rows=50)
    rows, misses = [], []
    for case in _lines("sql_guard_cases.jsonl"):
        try:
            guard.validate(case["sql"])
            blocked = False
        except UnsafeSql:
            blocked = True
        should_block = case["expect"] == "block"
        rows.append((should_block, blocked))
        if blocked != should_block:
            misses.append(case["sql"])
    return {**_confusion(rows), "misses": misses}


def evaluate_grounding() -> dict:
    rows, misses = [], []
    for case in _lines("grounding_cases.jsonl"):
        evidence = [
            Evidence(eid, EvidenceItem(text, SourceKind.DOCUMENT, eid))
            for eid, text in case["evidence"].items()
        ]
        problems = find_grounding_problems(
            case["answer"], case["cited"], evidence, case["question"]
        )
        should_reject = case["expect"] == "reject"
        rows.append((should_reject, bool(problems)))
        if bool(problems) != should_reject:
            misses.append(
                {"answer": case["answer"], "expected": case["expect"], "problems": problems}
            )
    return {**_confusion(rows), "misses": misses}


def write_report(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    retrieval = report["retrieval"]
    sql, grounding = report["sql_guard"], report["grounding"]
    lines = [
        "# Offline evaluation",
        "",
        f"Run {report['run_at']}. No LLM involved. Synthetic cases on the invented demo data.",
        "",
        "## Retrieval (15 gold queries, hybrid dense + BM25-style sparse)",
        "",
        "| Metric | " + " | ".join(retrieval) + " |",
        "|---|" + "---|" * len(retrieval),
    ]
    metrics = next(iter(retrieval.values())).keys()
    for metric in metrics:
        lines.append(
            f"| {metric} | " + " | ".join(str(r[metric]) for r in retrieval.values()) + " |"
        )
    for title, block, positive in (
        ("SQL guard", sql, "blocked"),
        ("Grounding check", grounding, "rejected"),
    ):
        lines += [
            "",
            f"## {title} ({block['cases']} cases; positive = {positive})",
            "",
            f"- Precision {block['precision']}, recall {block['recall']}",
            f"- {block['false_negatives']} unsafe or unsupported let through, "
            f"{block['false_positives']} safe or supported wrongly {positive}",
        ]
        lines += [f"- Miss: `{json.dumps(miss, ensure_ascii=False)}`" for miss in block["misses"]]
    (directory / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-models", action="store_true", help="also run the configured models")
    parser.add_argument("--output", type=Path, default=EVALS / "results" / "offline")
    arguments = parser.parse_args()

    retrieval = {"hashing embedder": asyncio.run(evaluate_retrieval(False))}
    if arguments.real_models:
        from ai_os.shared.config import Settings

        name = Settings().embedding_model.split("/")[-1]
        retrieval[f"{name} + reranker"] = asyncio.run(evaluate_retrieval(True))
    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "retrieval": retrieval,
        "sql_guard": evaluate_sql_guard(),
        "grounding": evaluate_grounding(),
    }
    write_report(report, arguments.output)
    print((arguments.output / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
