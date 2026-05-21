"""
storage.py — Pull model artefacts and raw trajectory files from S3-compatible
object storage (AWS S3, Cloudflare R2, MinIO) at startup.

If MODELS_BUCKET is unset, the fetch is skipped and the backend falls back
to files already on disk — keeps local development unchanged.
"""
import os
from pathlib import Path

import config


# Files the backend needs at runtime.  Keys are the object names in the
# bucket; values are the on-disk destinations.
_REQUIRED_FILES = {
    "bilstm_v2_classifier.h5":       Path(config.MODEL_DIR) / "bilstm_v2_classifier.h5",
    "bilstm_v2_thresholds.json":     Path(config.MODEL_DIR) / "bilstm_v2_thresholds.json",
    "lstm_digital_twin.h5":          Path(config.MODEL_DIR) / "lstm_digital_twin.h5",
    "twin_meta.json":                Path(config.MODEL_DIR) / "twin_meta.json",
    "shap_cache.json":               Path(config.MODEL_DIR) / "shap_cache.json",
    "dataset_meta.json":             Path(config.PROC_DIR)  / "dataset_meta.json",
    # Raw trajectories used by SimulationSession._init_trajectory
    "data/raw/gengine1/raw_000053_measurement_000000.csv":
        Path(config.DATA_DIR) / "gengine1" / "raw_000053_measurement_000000.csv",
    "data/raw/gengine2/raw_000016_measurement_000000.csv":
        Path(config.DATA_DIR) / "gengine2" / "raw_000016_measurement_000000.csv",
    "data/raw/pengines/engine1_normalized.xlsx":
        Path(config.DATA_DIR) / "pengines" / "engine1_normalized.xlsx",
}


def ensure_artefacts() -> None:
    bucket = os.getenv("MODELS_BUCKET")
    if not bucket:
        print("[storage] MODELS_BUCKET not set — using on-disk files.")
        return

    import boto3
    from botocore.client import Config as BotoConfig

    endpoint = os.getenv("S3_ENDPOINT_URL")  # set this for Cloudflare R2 / MinIO
    region   = os.getenv("AWS_REGION", "auto")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=BotoConfig(signature_version="s3v4"),
    )

    for key, dest in _REQUIRED_FILES.items():
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[storage] downloading s3://{bucket}/{key} -> {dest}")
        client.download_file(bucket, key, str(dest))
    print("[storage] all required artefacts present on disk.")
