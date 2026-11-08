import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

from ai_os.bootstrap.answer_grading import AnswerCase, grade
from ai_os.modules.agent.domain.models import (
    AgentAnswer,
    AnswerCitation,
    AnswerStatus,
    Intent,
    SourceKind,
)

ROOT = Path(__file__).parents[2]
ANSWER = AgentAnswer(
    "DG-102 costs 5,200,000 EGP.",
    AnswerStatus.ANSWERED,
    Intent.INVENTORY,
    [AnswerCitation(SourceKind.INVENTORY, "unit DG-102")],
)


def test_matching_answer_passes() -> None:
    case = AnswerCase(
        "Price?",
        "answered",
        intent="inventory",
        must_contain=["dg-102"],
        must_contain_numbers=["5200000"],
        must_not_contain=["DB-210"],
        citation_kind="inventory",
    )

    assert grade(case, ANSWER) == []


def test_each_mismatch_is_reported() -> None:
    case = AnswerCase(
        "Price?",
        "escalated",
        intent="knowledge",
        must_contain=["garden"],
        must_contain_numbers=["999"],
        must_not_contain=["costs"],
        citation_kind="document",
    )

    assert len(grade(case, ANSWER)) == 6


def test_gold_answer_set_loads_with_valid_expectations() -> None:
    lines = (ROOT / "evals" / "gold_answers.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [AnswerCase(**json.loads(line)) for line in lines]

    assert len(cases) >= 10
    assert {case.status for case in cases} <= {status.value for status in AnswerStatus}
    assert {case.intent for case in cases} - {None} <= {intent.value for intent in Intent}


def test_crewai_env_keeps_opentelemetry_on_and_crewai_telemetry_off() -> None:
    """Disabling the OpenTelemetry SDK would silently stop Langfuse from receiving anything."""
    script = (
        "import ai_os.bootstrap.crewai_env, os\n"
        "from crewai.telemetry.telemetry import Telemetry\n"
        "print(Telemetry._is_telemetry_disabled(), os.environ.get('OTEL_SDK_DISABLED'))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in {"OTEL_SDK_DISABLED"}}
    env.pop("CREWAI_DISABLE_TELEMETRY", None)

    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120
    )

    assert result.stdout.split()[-2:] == ["True", "None"], result.stderr[-500:]


def test_run_eval_gate_reports_each_metric_below_threshold() -> None:
    gate = runpy.run_path(str(ROOT / "scripts" / "run_eval.py"))["failures"]

    problems = gate(
        {"mrr": 0.5, "recall_at_5": 1.0}, {"mrr": 0.7, "recall_at_5": 0.9, "other": 0.1}
    )

    assert len(problems) == 2
    assert any("mrr" in problem for problem in problems)
    assert any("other" in problem for problem in problems)
