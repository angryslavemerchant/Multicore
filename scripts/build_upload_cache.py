"""scripts/build_upload_cache.py -- build the token cache once, publish it.

Run this on ONE rented box; every later instance then gets the cache from
Drive in ~90 seconds (scripts/drive_cache.py, called from m5_data) instead
of spending ~58 minutes tokenising it again.

    python vast/launch.py launch --profile cheap --keep-alive --hedge 1 \
        --branch <branch> --thresholds vast/thresholds_cheap.json \
        --train-script scripts/build_upload_cache.py --train-args "--shards 5"

Rent for the CPU, not the GPU: this workload never touches CUDA, and
tokenising is single-machine CPU-bound. Needs RCLONE_DRIVE_TOKEN (or _B64)
in the environment, which launch.py injects when secrets.env has it.

Prints CACHE_BUILT / CACHE_SIZE / BUILD_TIME and then DRIVE_PUSH_OK.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from drive_cache import push                              # noqa: E402
from m5_data import DEFAULT_SHARDS, build_token_cache     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--skip-upload", action="store_true",
                    help="build and measure only; leave Drive alone")
    args, _ = ap.parse_known_args()

    print(f"[cache] building the token cache: {args.shards} shard(s)",
          flush=True)
    t0 = time.time()
    # A cache already on Drive short-circuits this inside build_token_cache,
    # so re-running the publisher is cheap rather than another hour of CPU.
    path = build_token_cache(n_shards=args.shards, data_dir=args.data_dir)
    build_s = time.time() - t0
    size = os.path.getsize(path)

    print(f"CACHE_BUILT path={path}", flush=True)
    print(f"CACHE_SIZE  bytes={size:,} gb={size / 1e9:.2f}", flush=True)
    print(f"BUILD_TIME  seconds={build_s:.0f} minutes={build_s / 60:.1f}",
          flush=True)

    if args.skip_upload:
        print("[cache] --skip-upload set; not publishing", flush=True)
        return
    push(path)


if __name__ == "__main__":
    main()
