"""
upload_to_r2.py
===============
Uploads all generated PNG maps and meta.json to Cloudflare R2.

Upload structure:
  latest/                          ← always overwritten, frontend reads this
    meta.json
    GFS/india_fxx024.png
    GFS/gujarat_plain_fxx024.png
    GFS/gujarat_district_fxx024.png
    ECMWF/india_fxx024.png
    ...

  archive/2026-07-11_12z/          ← permanent copy per run
    meta.json
    GFS/india_fxx024.png
    ...

R2 lifecycle rule (set once in Cloudflare dashboard):
  Prefix: archive/
  Action: Delete after 30 days

Credentials from environment variables (GitHub Secrets):
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_ENDPOINT_URL      e.g. https://xxxx.r2.cloudflarestorage.com
  R2_BUCKET_NAME       e.g. gujaratweatherman-maps
  R2_PUBLIC_URL        e.g. https://pub-xxxx.r2.dev
"""

import os
import sys
import json
import boto3
from botocore.config import Config
from pathlib import Path
from datetime import datetime, timezone

# ── Config ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path("output")

def get_env(key):
    val = os.environ.get(key)
    if not val:
        print(f"ERROR: Environment variable '{key}' not set.")
        sys.exit(1)
    return val

def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=get_env("R2_ENDPOINT_URL"),
        aws_access_key_id=get_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=get_env("R2_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )

def content_type(path):
    ext = Path(path).suffix.lower()
    return {
        ".png":  "image/png",
        ".json": "application/json",
        ".html": "text/html",
    }.get(ext, "application/octet-stream")

def cache_control(r2_key):
    """
    Cache-Control per file type:
      meta.json   — 30 minutes  (needs to be reasonably fresh for new model runs)
      archive/*   — 15 days     (archive files never change once written)
      *.png       — 6 hours     (PNG content fixed until next run overwrites)
    """
    if r2_key.endswith('meta.json'):
        return 'public, max-age=1800'           # 30 minutes
    if r2_key.startswith('archive/'):
        return 'public, max-age=1296000'        # 15 days
    return 'public, max-age=21600'              # 6 hours (PNGs)

def upload_file(client, bucket, local_path, r2_key):
    client.upload_file(
        str(local_path),
        bucket,
        r2_key,
        ExtraArgs={
            "ContentType": content_type(local_path),
            "CacheControl": cache_control(r2_key),
        }
    )

def upload_all(client, bucket, files, prefix):
    """Upload all files under a given R2 prefix. Returns (uploaded, failed) lists."""
    uploaded = []
    failed   = []
    for local_path in sorted(files):
        # R2 key = prefix + path relative to output/
        rel      = str(local_path.relative_to(OUT_DIR)).replace("\\", "/")
        r2_key   = f"{prefix}/{rel}"
        try:
            upload_file(client, bucket, local_path, r2_key)
            print(f"  ✓ {r2_key}")
            uploaded.append(r2_key)
        except Exception as e:
            print(f"  ✗ {r2_key} — {e}")
            failed.append(r2_key)
    return uploaded, failed

def get_run_stamp():
    """
    Derive the archive folder name from meta.json run times.
    Uses the earliest run time across all models.
    Falls back to current UTC time if meta.json is unavailable.
    Format: YYYY-MM-DD_HHz  e.g. 2026-07-11_12z
    """
    meta_path = OUT_DIR / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # Find earliest run time across all models
            run_times = []
            for model_data in meta.get("models", {}).values():
                rt = model_data.get("run_time_utc", "")
                if rt:
                    run_times.append(rt)
            if run_times:
                # Parse earliest run time
                earliest = sorted(run_times)[0]
                dt = datetime.strptime(earliest, "%Y-%m-%d %H:%M UTC")
                return dt.strftime("%Y-%m-%d_%Hz")
        except Exception as e:
            print(f"  WARNING: Could not parse meta.json for run stamp: {e}")

    # Fallback to current UTC time
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%Hz")

def main():
    bucket  = get_env("R2_BUCKET_NAME")
    pub_url = get_env("R2_PUBLIC_URL").rstrip("/")
    client  = get_r2_client()

    if not OUT_DIR.exists():
        print(f"ERROR: Output directory '{OUT_DIR}' not found.")
        sys.exit(1)

    # Collect all files — exclude existing_meta.json (temp file, never upload)
    files = [f for f in list(OUT_DIR.rglob("*.png")) + list(OUT_DIR.rglob("*.json"))
             if f.name != 'existing_meta.json']
    if not files:
        print("WARNING: No files found to upload.")
        sys.exit(0)

    run_stamp = get_run_stamp()

    # Read meta.json to get per-model run times for clean per-model archives
    meta_path = OUT_DIR / 'meta.json'
    meta = {}
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[R2] WARNING: Could not read meta.json: {e}")

    # Inject cache_version into meta.json before uploading
    if meta:
        try:
            meta['cache_version'] = run_stamp
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            print(f"[R2] Injected cache_version: {run_stamp} into meta.json")
        except Exception as e:
            print(f"[R2] WARNING: Could not inject cache_version: {e}")

    # Build per-model archive stamps from each model's own run_time_utc
    # e.g. GFS 00z → archive/2026-07-12_00z/GFS/
    #      ECMWF 18z → archive/2026-07-11_18z/ECMWF/
    model_archive_stamps = {}
    for mk, model_data in meta.get('models', {}).items():
        rt = model_data.get('run_time_utc', '')
        if rt:
            try:
                dt = datetime.strptime(rt, "%Y-%m-%d %H:%M UTC")
                model_archive_stamps[mk] = dt.strftime("%Y-%m-%d_%Hz")
            except Exception:
                model_archive_stamps[mk] = run_stamp  # fallback

    print(f"\n[R2] Run stamp:  {run_stamp}")
    print(f"[R2] Bucket:     {bucket}")
    print(f"[R2] Files:      {len(files)}")
    print(f"[R2] Model archive stamps: {model_archive_stamps}")
    print(f"[R2] Uploading to 'latest/' and per-model archives...\n")

    total_uploaded = []
    total_failed   = []

    # ── Upload everything to latest/ ─────────────────────────────────────────
    print("[R2] → latest/")
    up, fail = upload_all(client, bucket, files, "latest")
    total_uploaded += up
    total_failed   += fail

    # ── Upload per-model files to their own dated archive folder ─────────────
    # meta.json goes to each model's archive folder so each archive is self-contained
    # Model PNGs go only to their own archive: archive/YYYY-MM-DD_HHz/MODEL/
    for mk, stamp in model_archive_stamps.items():
        archive_prefix = f"archive/{stamp}"
        model_files = [f for f in files
                       if f.name == 'meta.json'           # always archive meta.json
                       or f'/{mk}/' in str(f).replace('\\', '/')]  # model-specific PNGs
        if not model_files:
            continue
        print(f"\n[R2] → archive/{stamp}/{mk}/ ({len(model_files)} files)")
        up, fail = upload_all(client, bucket, model_files, archive_prefix)
        total_uploaded += up
        total_failed   += fail

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n[R2] Complete — {len(total_uploaded)} uploaded, {len(total_failed)} failed")

    if total_failed:
        print(f"[R2] Failed: {total_failed}")
        sys.exit(1)

    # Print useful URLs
    print(f"\n[R2] Public URLs:")
    print(f"  Latest meta.json:  {pub_url}/latest/meta.json")
    print(f"  Archive meta.json: {pub_url}/{archive_prefix}/meta.json")
    print(f"\n[R2] Frontend config:")
    print(f"  META_URL = '{pub_url}/latest/meta.json'")
    print(f"  MAP_BASE = '{pub_url}/latest/'")

if __name__ == "__main__":
    main()
