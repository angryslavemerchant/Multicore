"""Build the FineWeb-Edu token cache and upload it to Google Drive via rclone.

Designed to run on a vast instance via launch.py:
  python vast/launch.py launch --profile 5090 --keep-alive \
      --train-script scripts/build_upload_cache.py \
      --train-args "--shards 5"

The instance must have RCLONE_DRIVE_TOKEN_B64 (or RCLONE_DRIVE_TOKEN) set.
"""
import argparse, base64, json, os, subprocess, sys, tempfile, time


def install_rclone():
    if subprocess.run(["which", "rclone"], capture_output=True).returncode == 0:
        print("[cache-upload] rclone already installed", flush=True)
        return
    print("[cache-upload] installing rclone...", flush=True)
    subprocess.run(
        "curl -fsSL https://rclone.org/install.sh | bash",
        shell=True, check=True)


def get_drive_token():
    tok = os.environ.get("RCLONE_DRIVE_TOKEN")
    if tok:
        return tok
    b64 = os.environ.get("RCLONE_DRIVE_TOKEN_B64")
    if b64:
        return base64.b64decode(b64).decode()
    raise RuntimeError("RCLONE_DRIVE_TOKEN / _B64 not set")


def make_rclone_conf(token):
    fd, path = tempfile.mkstemp(suffix=".conf")
    with os.fdopen(fd, "w") as f:
        f.write(f"[gdrive]\ntype = drive\ntoken = {token}\n")
    return path


def rclone_upload(local_path, remote_folder, conf_path):
    # 256M chunks, NOT the 8M default. Drive cannot resume a resumable-upload
    # session mid-file, so one failed chunk restarts the whole transfer -- and
    # at 8M a 7 GB cache is ~900 sequential chunks, which reliably lost the
    # race. Measured 2026-07-28: the default burned all three retries (21 GB
    # transferred, nothing committed) in 12 minutes; 256M committed in 2m23s
    # at 64 MB/s on the first attempt.
    dest = f"gdrive:{remote_folder}/"
    cmd = ["rclone", "copy", local_path, dest,
           "--config", conf_path, "--drive-chunk-size", "256M",
           "-v", "--stats", "15s"]
    print(f"[cache-upload] uploading {local_path} -> {dest}", flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - t0
    size_gb = os.path.getsize(local_path) / 1e9
    print(f"[cache-upload] upload done: {size_gb:.2f} GB in {elapsed:.0f}s "
          f"({size_gb * 8 / elapsed * 1000:.0f} Mbps)", flush=True)
    return elapsed


def rclone_verify(local_path, remote_folder, conf_path):
    fname = os.path.basename(local_path)
    cmd = ["rclone", "lsjson", f"gdrive:{remote_folder}/{fname}",
           "--config", conf_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[cache-upload] WARNING: verify failed: {result.stderr}",
              flush=True)
        return False
    items = json.loads(result.stdout)
    if not items:
        print("[cache-upload] WARNING: file not found on Drive", flush=True)
        return False
    remote_size = items[0].get("Size", 0)
    local_size = os.path.getsize(local_path)
    ok = remote_size == local_size
    print(f"[cache-upload] verify: local={local_size:,} remote={remote_size:,} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--drive-folder", default="multicore-cache")
    ap.add_argument("--skip-upload", action="store_true")
    args, _ = ap.parse_known_args()

    print(f"\n{'='*60}", flush=True)
    print(f"[cache-upload] building token cache: {args.shards} shards",
          flush=True)
    print(f"{'='*60}\n", flush=True)

    from m5_data import build_token_cache

    t0 = time.time()
    path = build_token_cache(n_shards=args.shards)
    build_time = time.time() - t0
    size = os.path.getsize(path)

    print(f"\n{'='*60}", flush=True)
    print(f"CACHE_BUILT path={path}", flush=True)
    print(f"CACHE_SIZE  bytes={size:,}  gb={size / 1e9:.2f}", flush=True)
    print(f"BUILD_TIME  seconds={build_time:.0f}  minutes={build_time/60:.1f}",
          flush=True)
    print(f"{'='*60}\n", flush=True)

    if args.skip_upload:
        print("[cache-upload] --skip-upload set, done.", flush=True)
        return

    install_rclone()
    token = get_drive_token()
    conf = make_rclone_conf(token)
    try:
        rclone_upload(path, args.drive_folder, conf)
        rclone_verify(path, args.drive_folder, conf)
    finally:
        os.unlink(conf)

    print("\nCACHE_UPLOAD_OK", flush=True)


if __name__ == "__main__":
    main()
