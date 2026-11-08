"""Answer release gate on the real agent. Costs API credits, so it runs on demand.

Needs API keys, a migrated database and `scripts/seed_demo.py` already run.
Usage: python scripts/run_answer_eval.py [--min-pass-rate 0.9] [--deepeval]

--deepeval adds an LLM-as-judge relevancy score through DeepEval (needs the eval extra).
The gold set is synthetic. Replace it with questions graded by team leads.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.bootstrap.agent import build_ask_agent
from ai_os.bootstrap.answer_grading import AnswerCase, grade
from ai_os.bootstrap.escalation import build_escalations
from ai_os.bootstrap.inventory import build_ask_inventory, build_unit_store
from ai_os.bootstrap.knowledge import build_retrieve
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.shared.config import Settings

CASES = Path(__file__).parents[1] / "evals" / "gold_answers.jsonl"


def relevancy(question: str, answer: str, model: str) -> float:
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    metric = AnswerRelevancyMetric(model=model, include_reason=False)
    metric.measure(LLMTestCase(input=question, actual_output=answer))
    return metric.score


async def run(judge: bool) -> float:
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
    cases = [AnswerCase(**json.loads(line)) for line in lines]
    passed = 0
    for case in cases:
        answer = await agent(case.question, case.project, asked_by="answer-eval")
        problems = grade(case, answer)
        passed += not problems
        print("PASS" if not problems else f"FAIL {problems}", "-", case.question)
        if judge and answer.status.value == "answered":
            score = relevancy(case.question, answer.text, settings.openai_answer_model)
            print(f"     relevancy {score:.2f}")
    await engine.dispose()
    return passed / len(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--deepeval", action="store_true")
    arguments = parser.parse_args()

    rate = asyncio.run(run(arguments.deepeval))
    print(f"Pass rate: {rate:.0%}")
    sys.exit(0 if rate >= arguments.min_pass_rate else 1)


if __name__ == "__main__":
    main()
