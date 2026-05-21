"""
One-shot upload of model artefacts + raw trajectories to S3.

Reads AWS credentials from environment variables and uploads the files
that `backend/storage.py` expects to find at startup.  Idempotent — boto3
overwrites existing keys silently.

Usage:
    python scripts/upload_to_s3.py
"""
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config


# ── Required env vars ─────────────────────────────────────────────────────────
BUCKET   = os.environ["AWS_BUCKET_NAME"]
REGION   = os.environ["AWS_REGION"]
KEY_ID   = os.environ["AWS_ACCESS_KEY_ID"]
SECRET   = os.environ["AWS_SECRET_ACCESS_KEY"]

# Optional — only set for R2 / MinIO.  For native AWS S3, leave empty.
ENDPOINT = os.environ.get("S3_ENDPOINT_URL") or None


# ── File map: local path → S3 key ─────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent

UPLOADS = {
    # Model artefacts
    HERE / "models" / "bilstm_v2_classifier.h5":   "bilstm_v2_classifier.h5",
    HERE / "models" / "bilstm_v2_thresholds.json": "bilstm_v2_thresholds.json",
    HERE / "models" / "lstm_digital_twin.h5":      "lstm_digital_twin.h5",
    HERE / "models" / "twin_meta.json":            "twin_meta.json",
    HERE / "models" / "shap_cache.json":           "shap_cache.json",
    HERE / "data" / "processed" / "dataset_meta.json": "dataset_meta.json",

    # Raw trajectories (paths resolved via the docker-compose symlink layout)
    HERE.parent.parent / "gengine1" / "raw_000053_measurement_000000.csv":
        "data/raw/gengine1/raw_000053_measurement_000000.csv",
    HERE.parent.parent / "gengine2" / "raw_000016_measurement_000000.csv":
        "data/raw/gengine2/raw_000016_measurement_000000.csv",
    HERE.parent.parent / "pengines" / "engine1_normalized.xlsx":
        "data/raw/pengines/engine1_normalized.xlsx",
}


def main():
    client = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        aws_access_key_id=KEY_ID,
        aws_secret_access_key=SECRET,
        config=Config(signature_version="s3v4"),
    )

    print(f"Uploading {len(UPLOADS)} files to s3://{BUCKET}/ in {REGION}\n")

    total = 0
    for src, key in UPLOADS.items():
        if not src.exists():
            print(f"  [MISS] {src} not found — skipping")
            continue
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"  -> {key:<55s} ({size_mb:6.2f} MB) ", end="", flush=True)
        client.upload_file(str(src), BUCKET, key)
        total += size_mb
        print("ok")

    print(f"\nDone. Uploaded ~{total:.1f} MB total.")


if __name__ == "__main__":
    sys.exit(main())
