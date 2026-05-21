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

# Hugging Face Spaces runs containers as UID 1000.  Create that user,
# own /app, and switch to it so storage.py can write the model files it
# downloads from S3 at startup.
RUN useradd --create-home --uid 1000 user \
 && mkdir -p /app /app/models /app/data/processed /app/data/raw \
 && chown -R user:user /app

WORKDIR /app

COPY --chown=user:user backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

COPY --chown=user:user backend/ /app/backend/

USER user
EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
