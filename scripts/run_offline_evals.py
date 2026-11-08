"""Evaluations that need no LLM: retrieval, the SQL guard and the grounding check.

Usage: python scripts/run_offline_evals.py [--real-models] [--output DIR] [--gate]

--gate exits non-zero when a score falls below evals/offline_thresholds.json (CI runs it).

All cases are synthetic and run on the invented demo data. Writes results.json and
results.md to --output.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_eval import evaluate as evaluate_retrieval

from ai_os.modules.agent.domain.grounding import find_grounding_problems
from ai_os.modules.agent.domain.models import Evidence, EvidenceItem, SourceKind
from ai_os.modules.agent.domain.non_answers import says_evidence_is_silent
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


def evaluate_non_answers() -> dict:
    rows, misses = [], []
    for case in _lines("non_answer_cases.jsonl"):
        flagged = says_evidence_is_silent(case["text"])
        rows.append((case["non_answer"], flagged))
        if flagged != case["non_answer"]:
            misses.append(case["text"])
    return {**_confusion(rows), "misses": misses}


def replay_recorded_answers(run: str = "gemini-after-fixes") -> dict | None:
    """Re-applies today's rule-based checks to answers a real model gave in an earlier run.

    No model is called, so this estimates the effect of the non-answer backstop on those exact
    replies; a fresh model run would give different replies.
    """
    path = EVALS / "results" / run / "results.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    changed, passed = [], 0
    for result in results:
        status = result["status"]
        if status == "answered" and says_evidence_is_silent(result["answer"]):
            status = "escalated"
            changed.append(result["question"])
        other_problems = [p for p in result["problems"] if not p.startswith("status")]
        if status == result["status"]:
            passed += result["passed"]
        else:
            # An escalated reply carries no citations or numbers, so only status and intent count.
            other_problems = [p for p in other_problems if p.startswith("intent")]
            passed += status == result["expected_status"] and not other_problems
    should_escalate = [r for r in results if r["expected_status"] == "escalated"]
    escalated = [
        r for r in should_escalate if r["status"] == "escalated" or r["question"] in changed
    ]
    return {
        "source_run": run,
        "cases": len(results),
        "pass_rate": round(passed / len(results), 3),
        "escalation_recall": round(len(escalated) / len(should_escalate), 3),
        "newly_escalated": changed,
    }


def write_report(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    retrieval = report["retrieval"]
    sql, grounding, silent = report["sql_guard"], report["grounding"], report["non_answers"]
    lines = [
        "# Offline evaluation",
        "",
        f"Run {report['run_at']}. No LLM involved. Synthetic cases on the invented demo data.",
        "",
        f"## Retrieval ({report['retrieval_queries']} gold queries, hybrid dense + sparse)",
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
        ("Non-answer detector", silent, "escalated"),
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
    replay = report.get("replay")
    if replay:
        lines += [
            "",
            f"## Replay of recorded {replay['source_run']} answers through today's checks",
            "",
            "No model called: the same replies, re-checked. An estimate, not a new run.",
            "",
            f"- Pass rate {replay['pass_rate']:.0%}, escalation recall "
            f"{replay['escalation_recall']:.0%}",
            *[f"- Now escalated: {question}" for question in replay["newly_escalated"]],
        ]
    (directory / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def gate_failures(report: dict) -> list[str]:
    thresholds = json.loads((EVALS / "offline_thresholds.json").read_text(encoding="utf-8"))
    problems = []
    for name, minimum in thresholds.items():
        section, metric = name.split(".")
        value = report[section][metric] or 0.0
        if value < minimum:
            problems.append(f"{name}: {value} is below {minimum}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-models", action="store_true", help="also run the configured models")
    parser.add_argument("--output", type=Path, default=EVALS / "results" / "offline")
    parser.add_argument("--gate", action="store_true", help="fail below offline_thresholds.json")
    arguments = parser.parse_args()

    retrieval = {"hashing embedder": asyncio.run(evaluate_retrieval(False))}
    if arguments.real_models:
        from ai_os.shared.config import Settings

        name = Settings().embedding_model.split("/")[-1]
        retrieval[f"{name} + reranker"] = asyncio.run(evaluate_retrieval(True))
    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "retrieval_queries": len(_lines("gold_retrieval.jsonl")),
        "retrieval": retrieval,
        "sql_guard": evaluate_sql_guard(),
        "grounding": evaluate_grounding(),
        "non_answers": evaluate_non_answers(),
        "replay": replay_recorded_answers(),
    }
    write_report(report, arguments.output)
    print((arguments.output / "results.md").read_text(encoding="utf-8"))
    if arguments.gate:
        problems = gate_failures(report)
        for problem in problems:
            print("FAILED", problem)
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
