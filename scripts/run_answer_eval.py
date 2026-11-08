"""Answer release gate on the real agent. Each case makes real LLM calls.

Needs a migrated, seeded database (scripts/seed_demo.py or scripts/dev_stack.py) and keys for
the configured provider. LLM_PROVIDER=gemini runs on Google's free tier; --delay paces calls
to stay under its rate limit.

Usage: python scripts/run_answer_eval.py [--min-pass-rate 0.9] [--delay 0] [--output DIR]
       [--deepeval]

Writes results.json and results.md to --output. --deepeval adds an LLM-as-judge relevancy
score (needs the eval extra). The gold set is synthetic; replace it with questions graded by
team leads.
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.bootstrap.agent import build_ask_agent
from ai_os.bootstrap.answer_grading import AnswerCase, grade
from ai_os.bootstrap.escalation import build_escalations
from ai_os.bootstrap.inventory import build_ask_inventory, build_unit_store
from ai_os.bootstrap.knowledge import build_retrieve
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.shared.config import Settings

ROOT = Path(__file__).parents[1]
CASES = ROOT / "evals" / "gold_answers.jsonl"


def relevancy(question: str, answer: str, model: str) -> float:
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    metric = AnswerRelevancyMetric(model=model, include_reason=False)
    metric.measure(LLMTestCase(input=question, actual_output=answer))
    return metric.score


TRANSIENT = ("503", "429", "high demand", "RESOURCE_EXHAUSTED", "ConnectError", "ReadError")


async def ask_with_retry(agent, case: AnswerCase, attempts: int = 4):
    """Free-tier overloads and network blips are retried; anything else fails the case."""
    for attempt in range(1, attempts + 1):
        try:
            return await agent(case.question, case.project, asked_by="answer-eval")
        except Exception as error:
            text = f"{type(error).__name__}: {error}"
            if attempt == attempts or not any(mark in text for mark in TRANSIENT):
                raise
            print(f"  retry {attempt} after {text[:80]}", flush=True)
            await asyncio.sleep(20 * attempt)


def model_name(settings: Settings) -> str:
    if settings.llm_provider == "gemini":
        return settings.gemini_model
    return f"{settings.anthropic_answer_model} (fallback {settings.openai_answer_model})"


async def run(judge: bool, delay: float) -> dict:
    settings = Settings()
    units = build_unit_store(settings)
    engine = create_async_engine(settings.async_database_url)
    agent = build_ask_agent(
        settings,
        build_retrieve(settings),
        units,
        build_ask_inventory(settings, units),
        PostgresCrmStore(engine),
        build_escalations(settings),
    )
    lines = CASES.read_text(encoding="utf-8").splitlines()
    cases = [AnswerCase(**json.loads(line)) for line in lines if line.strip()]
    results = []
    for index, case in enumerate(cases):
        if index and delay:
            await asyncio.sleep(delay)
        started = time.perf_counter()
        try:
            answer = await ask_with_retry(agent, case)
            problems = grade(case, answer)
            row = {
                "status": answer.status.value,
                "intent": answer.intent.value,
                "answer": answer.text,
                "citations": [f"{c.kind.value}:{c.reference}" for c in answer.citations],
                "confidence": answer.confidence,
            }
        except Exception as error:  # A provider error fails the case, not the run.
            problems = [f"error {type(error).__name__}: {str(error)[:200]}"]
            row = {"status": "error", "intent": None, "answer": "", "citations": []}
        seconds = round(time.perf_counter() - started, 2)
        result = {
            "question": case.question,
            "expected_status": case.status,
            "expected_intent": case.intent,
            **row,
            "seconds": seconds,
            "passed": not problems,
            "problems": problems,
        }
        if judge and row["status"] == "answered":
            result["relevancy"] = relevancy(
                case.question, row["answer"], settings.openai_answer_model
            )
        results.append(result)
        print(
            "PASS" if not problems else f"FAIL {problems}",
            f"({seconds}s) -",
            case.question,
            flush=True,
        )
    await engine.dispose()
    return summarize(results, settings)


def summarize(results: list[dict], settings: Settings) -> dict:
    by_intent: dict[str, list[bool]] = defaultdict(list)
    for result in results:
        by_intent[result["expected_intent"] or result["expected_status"]].append(result["passed"])
    graded_intents = [r for r in results if r["expected_intent"]]
    routed = sum(r["intent"] == r["expected_intent"] for r in graded_intents)
    should_escalate = [r for r in results if r["expected_status"] == "escalated"]
    answered_wrongly = [
        r for r in results if r["expected_status"] != "answered" and r["status"] == "answered"
    ]
    seconds = sorted(r["seconds"] for r in results if r["status"] != "error")
    return {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "provider": settings.llm_provider,
        "model": model_name(settings),
        "embedding_model": settings.embedding_model,
        "cases": len(results),
        "passed": sum(r["passed"] for r in results),
        "pass_rate": round(sum(r["passed"] for r in results) / len(results), 3),
        "routing_accuracy": round(routed / len(graded_intents), 3) if graded_intents else None,
        "escalation_recall": round(
            sum(r["status"] == "escalated" for r in should_escalate) / len(should_escalate), 3
        )
        if should_escalate
        else None,
        "answered_when_it_should_not": len(answered_wrongly),
        "provider_errors": sum(r["status"] == "error" for r in results),
        "statuses": dict(Counter(r["status"] for r in results)),
        "pass_rate_by_group": {
            group: round(sum(flags) / len(flags), 3) for group, flags in sorted(by_intent.items())
        },
        "latency_p50_seconds": seconds[len(seconds) // 2] if seconds else None,
        "latency_p95_seconds": seconds[int(len(seconds) * 0.95) - 1] if seconds else None,
        "results": results,
    }


def write_report(report: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Answer evaluation",
        "",
        f"Run {report['run_at']} on `{report['model']}` ({report['provider']}), "
        f"embeddings `{report['embedding_model']}`. Synthetic gold set and demo data.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Cases | {report['cases']} |",
        f"| Pass rate | {report['pass_rate']:.0%} ({report['passed']}/{report['cases']}) |",
        f"| Routing accuracy | {_pct(report['routing_accuracy'])} |",
        f"| Escalation recall | {_pct(report['escalation_recall'])} |",
        f"| Answered when it should not | {report['answered_when_it_should_not']} |",
        f"| Provider errors | {report['provider_errors']} |",
        f"| Latency p50 / p95 | {report['latency_p50_seconds']}s / "
        f"{report['latency_p95_seconds']}s |",
        "",
        "| Group | Pass rate |",
        "|---|---|",
        *[f"| {g} | {r:.0%} |" for g, r in report["pass_rate_by_group"].items()],
        "",
        "## Failures",
        "",
    ]
    failures = [r for r in report["results"] if not r["passed"]]
    lines += [f"- {r['question']}: {'; '.join(r['problems'])}" for r in failures] or ["None."]
    (directory / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between cases")
    parser.add_argument("--output", type=Path, default=ROOT / "evals" / "results" / "latest")
    parser.add_argument("--deepeval", action="store_true")
    arguments = parser.parse_args()

    report = asyncio.run(run(arguments.deepeval, arguments.delay))
    write_report(report, arguments.output)
    print(f"Pass rate: {report['pass_rate']:.0%}  ->  {arguments.output}")
    sys.exit(0 if report["pass_rate"] >= arguments.min_pass_rate else 1)


if __name__ == "__main__":
    main()
