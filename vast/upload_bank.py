"""
vast/upload_bank.py — publish the jpeg cache to the Drive dataset bank.
Run once wherever a complete cache exists (instance or local):

    python vast/upload_bank.py [--jpeg_cache_dir ./jpeg_cache]

Needs RCLONE_DRIVE_TOKEN (or _B64) in the environment; prints
BANK_UPLOAD_OK on size-verified success (see bank.py).
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bank import upload_jpeg_cache  # noqa: E402


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jpeg_cache_dir", type=str, default="./jpeg_cache")
    upload_jpeg_cache(Path(p.parse_args().jpeg_cache_dir))
