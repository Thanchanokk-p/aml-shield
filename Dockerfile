# ── Base image ────────────────────────────────────────────────────
# Python 3.12 — matches the version on the host machine
# (confirmed by shap==0.52.0 requiring Python >=3.12)
FROM python:3.12-slim

# ── Working directory ─────────────────────────────────────────────
# Every command from this point on runs inside this path in the
# container (equivalent to "cd /app" before running anything else)
WORKDIR /app

# ── Install dependencies FIRST — order matters here (layer caching) ──
# Copying requirements.txt and installing before copying the rest of
# the code means Docker can reuse this layer on future builds as long
# as requirements.txt hasn't changed — even if app code changes a lot.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code + model files ────────────────────────────
COPY src/        ./src/
COPY mlflow.db   ./mlflow.db
COPY mlruns/     ./mlruns/

# ── Expose port ──────────────────────────────────────────────────
# Documents that this container listens on port 8000
# (doesn't actually open the port — that happens at "docker run -p")
EXPOSE 8000

# ── Startup command ──────────────────────────────────────────────
# Runs automatically when the container starts.
# Using 0.0.0.0 (not 127.0.0.1) so the API accepts connections from
# outside the container, not just from within it.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
