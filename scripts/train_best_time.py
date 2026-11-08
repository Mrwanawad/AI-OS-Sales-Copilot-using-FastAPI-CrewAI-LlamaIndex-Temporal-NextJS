"""Trains the best-time model and writes it only if it beats the baseline.

Usage:
  python scripts/train_best_time.py --synthetic [--leads 1500] --output models/best_time.txt
  python scripts/train_best_time.py --output models/best_time.txt   # real calls from the CRM

Synthetic results prove the pipeline works, not that recommendations help real reps.
"""

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import create_async_engine

from ai_os.modules.best_time.adapters.outbound.synthetic import generate_calls
from ai_os.modules.best_time.adapters.outbound.trainer import train_best_time
from ai_os.modules.best_time.domain.models import CallRecord
from ai_os.modules.crm_copilot.adapters.outbound.postgres_store import PostgresCrmStore
from ai_os.shared.config import Settings


async def real_calls(settings: Settings) -> list[CallRecord]:
    if not settings.async_database_url:
        raise SystemExit("Set ASYNC_DATABASE_URL, or pass --synthetic")
    engine = create_async_engine(settings.async_database_url)
    records = await PostgresCrmStore(engine).list_answer_outcomes()
    await engine.dispose()
    return [CallRecord(lead_id, at, answered) for lead_id, at, answered in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--leads", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    arguments = parser.parse_args()

    settings = Settings()
    tz = ZoneInfo(settings.business_timezone)
    if arguments.synthetic:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        calls, _profiles = generate_calls(arguments.leads, 7, tz, start)
    else:
        calls = asyncio.run(real_calls(settings))
    report = train_best_time(calls, tz, arguments.output, arguments.tracking_uri, arguments.trials)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
