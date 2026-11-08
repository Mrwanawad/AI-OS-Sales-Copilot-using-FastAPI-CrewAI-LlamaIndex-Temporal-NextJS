"""Import before anything that imports CrewAI, which reads these at import."""

import os

# Client questions and practice transcripts must not reach CrewAI's telemetry service.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
