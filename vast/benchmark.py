"""
vast/benchmark.py — system-wide health benchmark for a Vast.ai instance.

Measures the factors that make two instances with the same GPU perform very
differently: internet download speed, disk write throughput, CPU image-decode
throughput, host->device (PCIe) bandwidth, and raw GPU bf16 matmul compute.

Usage (on the instance):
    python vast/benchmark.py --out /workspace/benchmark.json
    python vast/benchmark.py --gate vast/thresholds.json   # exit 1 on failure

Always prints a single machine-parseable line:
    BENCHMARK_JSON {...}
so the local orchestrator (vast/launch.py) can scrape it from `vastai logs`.
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
import urllib.request


# ---------------------------------------------------------------------------
# Individual tests — each returns a dict of metrics, exceptions -> error field
# ---------------------------------------------------------------------------

_UA = {"User-Agent": "Mozilla/5.0 (vast-bench)"}


def _timed_read(url: str, size_mb: int, headers: dict = None,
                deadline_s: float = 40.0) -> tuple[int, float]:
    """Read up to size_mb from url, return (bytes, seconds).

    deadline_s is a wall-clock cap on the read loop: the urlopen timeout only
    bounds each individual recv(), so a connection that trickles bytes can
    otherwise block forever at zero CPU.
    """
    req = urllib.request.Request(url, headers={**_UA, **(headers or {})})
    t0 = time.perf_counter()
    n = 0
    with urllib.request.urlopen(req, timeout=20) as r:
        while n < size_mb * (1 << 20):
            chunk = r.read(256 << 10)
            if not chunk:
                break
            n += len(chunk)
            if time.perf_counter() - t0 > deadline_s:
                break
    return n, time.perf_counter() - t0


def bench_download(size_mb: int = 200) -> dict:
    """
    Megabits/sec pulling from the HuggingFace CDN — a real shard of the
    training dataset, i.e. exactly the path the dataset download takes.
    Falls back to Cloudflare's speed-test endpoint.
    """
    # HF increasingly requires auth from datacenter IPs — use the token the
    # instance already has. Cloudflare's speed endpoint 403s DC ranges, so
    # OVH's public test file is the anonymous fallback.
    hf_auth = {}
    token = os.environ.get("HF_TOKEN")
    if token:
        hf_auth = {"Authorization": f"Bearer {token}"}

    urls = []
    try:
        api = ("https://huggingface.co/api/datasets/clane9/imagenet-100/"
               "parquet/default/train")
        with urllib.request.urlopen(
                urllib.request.Request(api, headers={**_UA, **hf_auth}),
                timeout=30) as r:
            shard_urls = json.load(r)
        if shard_urls:
            urls.append(("hf", shard_urls[0], hf_auth))
    except Exception:
        pass
    urls.append(("ovh", "https://proof.ovh.net/files/1Gb.dat", {}))
    urls.append(("cloudflare",
                 f"https://speed.cloudflare.com/__down?bytes={size_mb * (1 << 20)}",
                 {}))

    last_err: Exception = RuntimeError("no download source available")
    for src, url, headers in urls:
        try:
            n, dt = _timed_read(url, size_mb, headers)
            if n < 10 * (1 << 20):
                raise RuntimeError(f"only {n} bytes in {dt:.0f}s from {src}")
            return {"download_mbps": round(n * 8 / 1e6 / dt, 1),
                    "download_src": src}
        except Exception as e:
            last_err = e
    raise last_err


def bench_bank(size_mb: int = 64) -> dict:
    """Throughput on the ACTUAL boot dependency: a ranged read of the
    dataset tar from the Drive bank via rclone. Generic download speed
    does not predict this path (Drive peering/throttling varies per
    host — a 3 MB/s host cost a boot 10 extra minutes on 2026-07-19).
    Requires the bank token (launch.py always forwards it); a missing
    token surfaces as a gate failure, which is the correct loud response
    to a misconfigured boot."""
    import subprocess
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import bank
    if not bank._token():
        raise RuntimeError("RCLONE_DRIVE_TOKEN not configured")
    bank._rclone_ready()
    t0 = time.perf_counter()
    proc = subprocess.run(
        ["rclone", "cat", "--count", str(size_mb * (1 << 20)),
         f"{bank.BANK_REMOTE}/{bank.BANK_TAR}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=100)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"rclone cat failed: {proc.stderr[-200:]!r}")
    return {"drive_pull_mbps": round(size_mb * 8 / dt, 1)}


def bench_disk(path: str = "/workspace/.bench_tmp", size_mb: int = 1024) -> dict:
    """Sequential write throughput with fsync (matters for JPEG cache build)."""
    buf = os.urandom(1 << 20)
    t0 = time.perf_counter()
    with open(path, "wb") as f:
        for _ in range(size_mb):
            f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    dt = time.perf_counter() - t0
    os.remove(path)
    return {"disk_write_mbps": round(size_mb / dt, 1)}


def _jpeg_worker(args) -> int:
    jpeg_bytes, seconds = args
    from PIL import Image
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        Image.open(io.BytesIO(jpeg_bytes)).convert("RGB").load()
        n += 1
    return n


def bench_cpu(seconds: float = 5.0) -> dict:
    """
    JPEG decode throughput across all cores (the shape of dataloader CPU work)
    plus a multithreaded BLAS matmul as a general compute number.
    """
    import numpy as np
    from PIL import Image

    # Build a representative jpeg in memory (~500x375 photo-like noise)
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (375, 500, 3), dtype=np.uint8))
    bio = io.BytesIO()
    img.save(bio, format="JPEG", quality=95)
    jpeg_bytes = bio.getvalue()

    cores = os.cpu_count() or 1
    with mp.Pool(cores) as pool:
        counts = pool.map(_jpeg_worker, [(jpeg_bytes, seconds)] * cores)
    jpeg_per_sec = sum(counts) / seconds

    n = 4096
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b  # warmup
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        a @ b
    dt = time.perf_counter() - t0
    gflops = 2 * n**3 * reps / dt / 1e9

    return {
        "cpu_cores": cores,
        "cpu_jpeg_per_sec": round(jpeg_per_sec, 1),
        "np_gflops": round(gflops, 1),
    }


def bench_pcie(size_mb: int = 256, reps: int = 20) -> dict:
    """Pinned host->device and device->host copy bandwidth in GB/s."""
    import torch
    x = torch.empty(size_mb * (1 << 20) // 4, dtype=torch.float32).pin_memory()
    d = torch.empty_like(x, device="cuda")
    d.copy_(x, non_blocking=True)  # warmup
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(reps):
        d.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    h2d = size_mb * reps / 1024 / (time.perf_counter() - t0)

    t0 = time.perf_counter()
    for _ in range(reps):
        x.copy_(d, non_blocking=True)
    torch.cuda.synchronize()
    d2h = size_mb * reps / 1024 / (time.perf_counter() - t0)

    return {"pcie_h2d_gbps": round(h2d, 2), "pcie_d2h_gbps": round(d2h, 2)}


def bench_gpu(n: int = 8192, reps: int = 30) -> dict:
    """bf16 matmul throughput (TFLOPS) + device name."""
    import torch
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_bf16_tflops": round(2 * n**3 * reps / dt / 1e12, 1),
    }


# ---------------------------------------------------------------------------

# name -> (fn, hard timeout in seconds). GPU tests include torch import +
# CUDA context creation, which alone can take ~30s on a cold instance.
TESTS = {
    "download": (bench_download, 240),
    "bank": (bench_bank, 150),
    "disk": (bench_disk, 180),
    "cpu": (bench_cpu, 120),
    "pcie": (bench_pcie, 180),
    "gpu": (bench_gpu, 240),
}


def _test_child(name: str, q) -> None:
    try:
        q.put(("ok", TESTS[name][0]()))
    except Exception as e:
        q.put(("err", f"{type(e).__name__}: {e}"))


def run_test(name: str) -> dict:
    """Run one test in a subprocess with a hard timeout, so a wedged test
    (stalled network read, hung CUDA init) can never freeze the whole gate."""
    fn, timeout_s = TESTS[name]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_test_child, args=(name, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join(10)
        if p.is_alive():
            p.kill()
            p.join()
        raise TimeoutError(f"exceeded {timeout_s}s hard limit")
    if q.empty():
        raise RuntimeError(f"test process died (exit code {p.exitcode})")
    status, payload = q.get()
    if status != "ok":
        raise RuntimeError(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Vast instance health benchmark")
    parser.add_argument("--out", type=str, default=None, help="write JSON here")
    parser.add_argument("--gate", type=str, default=None,
                        help="thresholds JSON; exit 1 if any metric falls below")
    parser.add_argument("--skip", type=str, default="",
                        help="comma-separated test names to skip")
    args = parser.parse_args()

    skip = set(filter(None, args.skip.split(",")))
    results: dict = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    for name in TESTS:
        if name in skip:
            continue
        try:
            print(f"[bench] {name}: running (limit {TESTS[name][1]}s)",
                  file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            results.update(run_test(name))
            print(f"[bench] {name}: ok ({time.perf_counter() - t0:.1f}s)",
                  file=sys.stderr)
        except Exception as e:  # a failed test is a data point, not a crash
            results[f"{name}_error"] = f"{type(e).__name__}: {e}"
            print(f"[bench] {name}: FAILED — {e}", file=sys.stderr)

    print("BENCHMARK_JSON " + json.dumps(results), flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)

    if args.gate:
        with open(args.gate) as f:
            thresholds = json.load(f)
        failed = []
        for key, minimum in thresholds.items():
            if key.startswith("_"):   # comment/metadata keys
                continue
            value = results.get(key)
            if value is None or value < minimum:
                failed.append(f"{key}: {value} < {minimum}")
                print(f"GATE FAIL  {key} = {value}  (need >= {minimum})")
            else:
                print(f"GATE pass  {key} = {value}  (need >= {minimum})")
        if failed:
            print("BENCHMARK_GATE FAIL " + "; ".join(failed), flush=True)
            sys.exit(1)
        print("BENCHMARK_GATE PASS", flush=True)


if __name__ == "__main__":
    main()
