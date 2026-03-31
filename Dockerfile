# ---- Token Analysis A2A Agent ----
# Multi-stage build using uv for fast, reproducible installs.

FROM python:3.13-slim-bookworm AS builder

# Install uv via standalone installer (no ghcr.io dependency)
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install dependencies first (cached unless lockfile changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy source and install the project itself
COPY src/ src/
COPY oasf/ oasf/
RUN uv sync --frozen --no-dev


# ---- Runtime stage ----
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

# Copy the entire virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application source
COPY src/ src/
COPY oasf/ oasf/
COPY .env.example .env.example

# Put the venv on PATH so `python` resolves to the right interpreter
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# A2A server port (matches TOKEN_ANALYSIS_PORT default)
EXPOSE 8883

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8883/.well-known/agent.json')" || exit 1

ENTRYPOINT ["python", "-m", "server.server"]
