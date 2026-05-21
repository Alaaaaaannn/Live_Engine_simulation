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

    endpoint = (os.getenv("S3_ENDPOINT_URL") or "").strip() or None
    region   = (os.getenv("AWS_REGION", "auto") or "").strip()
    key_id   = (os.getenv("AWS_ACCESS_KEY_ID")     or "").strip()
    secret   = (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()

    # Length diagnostic — AWS key IDs are exactly 20 chars, secrets 40 chars.
    # Mismatches usually mean the value was pasted with quotes / whitespace
    # or got truncated.  Prefix/suffix of the key id is safe to log (visible
    # in CloudTrail anyway); the secret is never logged.
    key_id_preview = (key_id[:4] + "..." + key_id[-4:]) if len(key_id) >= 8 else "?"
    print(f"[storage] bucket={bucket!r} region={region!r} endpoint={endpoint!r}")
    print(f"[storage] access_key_id_len={len(key_id)} (expect 20) "
          f"preview={key_id_preview} secret_len={len(secret)} (expect 40)")

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

    # Probe credential validity with STS first.  GetCallerIdentity is a
    # POST that returns a full XML body, so AWS error codes
    # (InvalidClientTokenId, SignatureDoesNotMatch, etc.) come through
    # cleanly — unlike HeadBucket which is a HEAD with empty body.
    from botocore.exceptions import ClientError
    try:
        sts = boto3.client(
            "sts",
            region_name=region if region != "auto" else "us-east-1",
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )
        ident = sts.get_caller_identity()
        print(f"[storage] sts.get_caller_identity OK "
              f"account={ident.get('Account')!r} "
              f"arn={ident.get('Arn')!r}")
    except ClientError as e:
        err = e.response.get("Error", {})
        print(f"[storage] STS auth FAILED code={err.get('Code','?')!r} "
              f"message={err.get('Message','?')!r}")
        raise

    # Now verify bucket access with a body-bearing call (list_objects_v2).
    try:
        resp = client.list_objects_v2(Bucket=bucket, MaxKeys=1)
        n = resp.get("KeyCount", 0)
        print(f"[storage] list_objects_v2 OK — bucket reachable, found {n} object(s).")
    except ClientError as e:
        err = e.response.get("Error", {})
        print(f"[storage] list_objects_v2 FAILED code={err.get('Code','?')!r} "
              f"message={err.get('Message','?')!r}")
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
