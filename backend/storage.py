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

    endpoint = os.getenv("S3_ENDPOINT_URL") or None
    region   = os.getenv("AWS_REGION", "auto")
    key_id   = os.getenv("AWS_ACCESS_KEY_ID")
    secret   = os.getenv("AWS_SECRET_ACCESS_KEY")

    # Diagnostic — names only, never the values
    print(f"[storage] bucket={bucket!r} region={region!r} "
          f"endpoint={endpoint!r} "
          f"key_id_set={bool(key_id)} secret_set={bool(secret)}")

    if not key_id or not secret:
        raise RuntimeError(
            "AWS credentials not found in env. Set AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY in the Space's Variables and secrets."
        )

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
        config=BotoConfig(signature_version="s3v4"),
    )

    # Probe with head_bucket first so we can distinguish a bad-credential
    # error from a missing-object error.  The bare HeadObject path swallows
    # the AWS error code (just says "403 Forbidden"), which is useless for
    # debugging.  head_bucket surfaces the proper code.
    from botocore.exceptions import ClientError
    try:
        client.head_bucket(Bucket=bucket)
        print(f"[storage] head_bucket OK — credentials accepted and bucket reachable.")
    except ClientError as e:
        err = e.response.get("Error", {})
        code = err.get("Code", "?")
        msg  = err.get("Message", "?")
        http = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", "?")
        print(f"[storage] head_bucket FAILED status={http} code={code!r} message={msg!r}")
        raise

    for key, dest in _REQUIRED_FILES.items():
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"[storage] downloading s3://{bucket}/{key} -> {dest}")
        try:
            client.download_file(bucket, key, str(dest))
        except ClientError as e:
            err = e.response.get("Error", {})
            print(f"[storage] download of {key!r} FAILED "
                  f"code={err.get('Code','?')!r} message={err.get('Message','?')!r}")
            raise
    print("[storage] all required artefacts present on disk.")
