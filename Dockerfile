# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage Dockerfile
# Stage 1 (builder): install deps in an isolated layer
# Stage 2 (runtime): copy only what's needed — no build tools in prod image
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS builder

WORKDIR /build

# Pin exact version from requirements.txt — no open ranges
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user — never run as root in production
RUN useradd --create-home appuser
WORKDIR /app
USER appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --chown=appuser:appuser . .

# Secrets are NEVER baked in at build time.
# They are injected at container startup via platform secret store.
# (e.g.  flyctl secrets set OPENAI_API_KEY=sk-... )

EXPOSE 8000

# Health check — used by the orchestrator to detect unhealthy containers
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["python", "-m", "uvicorn", "part3_cicd.app:app", "--host", "0.0.0.0", "--port", "8000"]
