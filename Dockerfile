# Root-level Dockerfile used by Hugging Face Spaces.
# Hugging Face requires `Dockerfile` at the repo root; for local dev and
# docker-compose, `backend/Dockerfile` is the equivalent build (kept in
# sync — change both if you change either).

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

COPY backend/ /app/backend/

# Model artefacts + raw trajectories are fetched from S3 at startup by
# backend/storage.py — the image stays slim.
RUN mkdir -p /app/models /app/data/processed /app/data/raw

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
