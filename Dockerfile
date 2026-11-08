FROM python:3.12-slim

# ffmpeg is how Whisper reads call recordings.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.7

WORKDIR /app
COPY pyproject.toml uv.lock README.md* ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
COPY deploy ./deploy
COPY demo_data ./demo_data
COPY scripts ./scripts
RUN uv sync --frozen --no-dev --extra agents --extra retrieval --extra ingestion --extra arabic \
    --extra workflows --extra inventory --extra mcp --extra speech --extra ml --extra observability

# The app only reads /app; its home holds downloaded model weights.
RUN useradd --create-home --uid 10001 app
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "ai_os.bootstrap.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
