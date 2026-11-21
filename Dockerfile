# Pinned base image: digest-locked python:3.12-slim.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

ARG APP_VERSION=2.0.0
ARG APP_GIT_SHA=unknown
ARG BAKE_WEIGHTS=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC \
    PROJECT_ENV=production \
    APP_VERSION=${APP_VERSION} \
    APP_GIT_SHA=${APP_GIT_SHA} \
    HF_HOME=/opt/hf-cache \
    HF_HUB_CACHE=/opt/hf-cache \
    RERANK_CACHE_DIR=/opt/hf-cache \
    EMBEDDING_CACHE_DIR=/opt/hf-cache \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt requirements-rag.lock.txt ./
RUN python -m pip install --no-cache-dir --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.lock.txt

COPY medcopilot ./medcopilot
RUN chmod 0755 /app/medcopilot/deploy/docker-entrypoint.sh

RUN if [ "$BAKE_WEIGHTS" = "1" ]; then \
        python -m pip install --no-cache-dir --require-hashes \
            --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements-rag.lock.txt && \
        python -m medcopilot.scripts.prepare_weights \
            --cache-dir /opt/hf-cache \
            --fingerprint /opt/hf-cache/weights_fingerprint.json; \
    fi

ENV MODEL_WEIGHTS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

RUN mkdir -p /opt/hf-cache \
    && useradd --create-home appuser \
    && chown -R appuser:appuser /app /opt/hf-cache
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4)"]

ENTRYPOINT ["/app/medcopilot/deploy/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "medcopilot.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
