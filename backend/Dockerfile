FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages required by tensorflow/shap (libgomp), pandas/openpyxl, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so the layer caches across code edits.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/backend/requirements.txt

# Application code
COPY backend/ /app/backend/

# Model artefacts and processed metadata are fetched from object storage
# at startup when MODELS_BUCKET is set (see backend/storage.py).  For
# local development you can still mount or COPY them, but in cloud
# deployments (Fly.io, Cloud Run) the image stays slim and the heavy
# files live in S3/R2.
RUN mkdir -p /app/models /app/data/processed /app/data/raw

EXPOSE 8000

WORKDIR /app/backend
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/status || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
