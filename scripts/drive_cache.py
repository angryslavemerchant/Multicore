"""scripts/drive_cache.py -- the FineWeb token cache, parked on Google Drive.

Building the 5-shard token cache costs 58 minutes of tokenising (measured
2026-07-28 on a Ryzen 9 5950X); pulling the finished file back off Drive
costs 90 seconds. So it is built once, pushed, and every later instance
downloads it instead.

`try_pull` is an optimisation and NEVER raises. A missing token, a missing
rclone, a Drive outage, a truncated transfer -- all return False and leave
the caller to tokenise locally. A Drive problem must cost 58 minutes, not a
run. Only `push` raises, because a silent upload failure would leave the
whole fleet quietly rebuilding the cache forever.

Needs RCLONE_DRIVE_TOKEN (or RCLONE_DRIVE_TOKEN_B64) in the environment;
vast/launch.py base64s the raw token into the container because the JSON
would be mangled by the docker-style --env string.
"""
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time

REMOTE = "gdrive"
REMOTE_FOLDER = "multicore-cache"

# 256M, NOT the 8M default. Drive cannot resume a resumable-upload session
# mid-file, so a single failed chunk restarts the entire transfer -- and at
# 8M a 7 GB cache is ~900 sequential chunks, which reliably lost the race.
# Measured: the default burned all three retries (21 GB moved, nothing
# committed) in 12 minutes; 256M committed in 2m23s at 64 MB/s first try.
UPLOAD_CHUNK = "256M"
# Drive will not serve parallel range reads the way S3 does, but rclone still
# gets a useful multiple of single-stream throughput out of 8.
PULL_STREAMS = "8"


def _token():
    """The rclone Drive OAuth blob, or None if this box has no credentials."""
    tok = os.environ.get("RCLONE_DRIVE_TOKEN")
    if tok:
        return tok
    b64 = os.environ.get("RCLONE_DRIVE_TOKEN_B64")
    if not b64:
        return None
    try:
        return base64.b64decode(b64).decode()
    except Exception:
        return None


def _rclone_ready():
    """True once an rclone binary exists, installing one if it does not."""
    if shutil.which("rclone"):
        return True
    try:
        subprocess.run("curl -fsSL https://rclone.org/install.sh | bash",
                       shell=True, check=True, capture_output=True,
                       timeout=300)
    except Exception as e:
        print(f"[drive] rclone install failed ({type(e).__name__}); "
              f"falling back to a local build", flush=True)
        return False
    return shutil.which("rclone") is not None


def _conf(token):
    """A 0600 temp rclone.conf holding `token`. Caller unlinks it."""
    fd, path = tempfile.mkstemp(suffix=".conf")       # mkstemp is 0600
    with os.fdopen(fd, "w") as f:
        f.write(f"[{REMOTE}]\ntype = drive\ntoken = {token}\n")
    return path


def _remote_size(name, conf):
    """Bytes of `name` on the remote, or None if absent/unreachable."""
    proc = subprocess.run(
        ["rclone", "lsjson", f"{REMOTE}:{REMOTE_FOLDER}/{name}",
         "--config", conf],
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return None
    try:
        items = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return items[0].get("Size") if items else None


def try_pull(local_path, folder=REMOTE_FOLDER):
    """Fetch `local_path`'s basename from Drive. True iff it now exists.

    Downloads to a sibling .partial and renames only after the size matches
    what the remote reports, so a killed transfer can never leave a truncated
    file that the next run mistakes for a finished cache.
    """
    name = os.path.basename(local_path)
    token = _token()
    if not token:
        return False
    if not _rclone_ready():
        return False
    conf = _conf(token)
    # NOT .partial -- build_token_cache resumes an interrupted local build
    # from exactly that path, and a half-downloaded file landing there would
    # be read as "shard N is already written" and silently corrupt the cache.
    tmp = local_path + ".drivepull"
    try:
        want = _remote_size(name, conf)
        if not want:
            print(f"[drive] {name} not on Drive -- building locally",
                  flush=True)
            return False
        print(f"[drive] pulling {name} ({want / 1e9:.2f} GB)", flush=True)
        t0 = time.time()
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        proc = subprocess.run(
            ["rclone", "copyto", f"{REMOTE}:{folder}/{name}", tmp,
             "--config", conf, "--multi-thread-streams", PULL_STREAMS,
             "--stats", "30s", "-v"],
            timeout=7200)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"[drive] pull failed (rclone {proc.returncode}) after "
                  f"{dt:.0f}s -- building locally", flush=True)
            return False
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if got != want:
            print(f"[drive] size mismatch: got {got:,}, want {want:,} "
                  f"-- discarding, building locally", flush=True)
            return False
        os.replace(tmp, local_path)
        print(f"[drive] DRIVE_PULL_OK {name} in {dt:.0f}s "
              f"({got * 8 / dt / 1e6:.0f} Mbps)", flush=True)
        return True
    except Exception as e:
        print(f"[drive] pull errored ({type(e).__name__}: {e}) -- "
              f"building locally", flush=True)
        return False
    finally:
        os.unlink(conf)
        if os.path.exists(tmp):
            os.remove(tmp)


def push(local_path, folder=REMOTE_FOLDER):
    """Upload `local_path` to Drive and verify the committed size. Raises."""
    name = os.path.basename(local_path)
    token = _token()
    if not token:
        raise RuntimeError("RCLONE_DRIVE_TOKEN / _B64 not set")
    if not _rclone_ready():
        raise RuntimeError("no rclone binary available")
    conf = _conf(token)
    try:
        local = os.path.getsize(local_path)
        print(f"[drive] pushing {name} ({local / 1e9:.2f} GB)", flush=True)
        t0 = time.time()
        subprocess.run(
            ["rclone", "copy", local_path, f"{REMOTE}:{folder}/",
             "--config", conf, "--drive-chunk-size", UPLOAD_CHUNK,
             "--stats", "30s", "-v"],
            check=True, timeout=7200)
        dt = time.time() - t0
        remote = _remote_size(name, conf)
        if remote != local:
            raise RuntimeError(
                f"upload verify failed: local {local:,}, remote {remote!r}")
        print(f"[drive] DRIVE_PUSH_OK {name} in {dt:.0f}s "
              f"({local * 8 / dt / 1e6:.0f} Mbps)", flush=True)
    finally:
        os.unlink(conf)
