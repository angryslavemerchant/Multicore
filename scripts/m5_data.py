"""Deterministic, cached byte-level data for the M5 runs.

Replaces the old live `load_dataset(..., streaming=True)` path over
HuggingFaceFW/fineweb-edu, which was (a) fragile -- the hub connection drops
mid-run ("peer closed connection without sending complete message body",
"Cannot send a request, as the client has been closed"), which has already
cost two runs including a rented GPU -- and (b) non-deterministic, so two
model configs never saw the same bytes and their losses were not comparable.

Pipeline:
  1. hf_hub_download the deterministic first N parquet shards of the
     `sample-10BT` config (resumable + cached on disk, and every hub call is
     wrapped in an exponential-backoff retry, since the hub is the flaky bit).
  2. Convert ONCE to a flat uint8 file runs/data/fineweb_<N>shards.bin: each
     document's utf-8 bytes (errors="ignore") followed by a single 0
     separator. Rebuilt only if that file is missing/empty.
     Measured: each 2.15 GB parquet shard is 726k docs -> 3.47 GB of bytes
     (~25 s to convert), so the default 4 shards ~= 13.9 GB, about 9x the
     ~1.5e9 bytes a full run consumes. Use --data-shards 1 (3.47 GB, still
     2x a run) when disk is tight; the parquet shards can be deleted from
     the HF cache once the .bin exists.
  3. Sample batches as B random contiguous (T+1)-byte windows drawn from a
     seeded numpy Generator, so a given (seed, B, T) yields the identical
     sequence of batches for every model config. The last EVAL_FRAC of the
     file is eval-only; training samples strictly from the head, so eval
     bytes are never trained on.

Standalone use (do this once on a fresh machine, before launching runs):
  python scripts/m5_data.py --shards 4
"""
import argparse, os, time

import numpy as np
import torch

REPO_ID = "HuggingFaceFW/fineweb-edu"
SUBDIR = "sample/10BT"            # the `sample-10BT` config: 14 parquet shards
DEFAULT_SHARDS = 4
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "data")
EVAL_FRAC = 0.05                  # last 5% of the byte file is eval-only
SEP = 0                           # document separator byte


# ---------------------------------------------------------------- hub fetch
def _retry(fn, what, attempts=5, base_delay=2.0):
    """Run fn(), retrying transient hub/IO failures with exponential backoff."""
    for a in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if a == attempts:
                raise
            delay = base_delay * 2 ** (a - 1)
            print(f"[data] {what}: attempt {a}/{attempts} failed "
                  f"({type(e).__name__}: {e}); retrying in {delay:.0f}s",
                  flush=True)
            time.sleep(delay)


def shard_names(n_shards, repo_id=REPO_ID, subdir=SUBDIR):
    """The first `n_shards` parquet shard names, as listed by the hub."""
    names = None
    try:
        from huggingface_hub import list_repo_files
        files = _retry(lambda: list_repo_files(repo_id, repo_type="dataset"),
                       "list_repo_files", attempts=3)
        names = sorted(f for f in files
                       if f.startswith(subdir + "/") and f.endswith(".parquet"))
    except Exception as e:
        print(f"[data] list_repo_files failed ({type(e).__name__}: {e}); "
              f"falling back to the known {subdir} naming", flush=True)
    if not names:
        names = [f"{subdir}/{i:03d}_00000.parquet" for i in range(n_shards)]
    if len(names) < n_shards:
        raise RuntimeError(f"{repo_id}/{subdir} has only {len(names)} shards, "
                           f"asked for {n_shards}")
    return names[:n_shards]


def download_shards(n_shards=DEFAULT_SHARDS, repo_id=REPO_ID, subdir=SUBDIR):
    """Download (or reuse from the HF cache) the first n_shards parquet files."""
    from huggingface_hub import hf_hub_download
    paths = []
    for name in shard_names(n_shards, repo_id, subdir):
        print(f"[data] fetching {name}", flush=True)
        p = _retry(lambda n=name: hf_hub_download(repo_id=repo_id, filename=n,
                                                  repo_type="dataset"),
                   f"download {name}")
        paths.append(p)
    return paths


# ---------------------------------------------------------------- byte cache
def byte_cache_path(n_shards=DEFAULT_SHARDS, data_dir=None):
    return os.path.join(data_dir or DEFAULT_DATA_DIR,
                        f"fineweb_{n_shards}shards.bin")


def build_byte_cache(n_shards=DEFAULT_SHARDS, data_dir=None, repo_id=REPO_ID,
                     subdir=SUBDIR):
    """Return the path to the flat uint8 byte file, building it if needed."""
    out = byte_cache_path(n_shards, data_dir)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        n = os.path.getsize(out)
        print(f"[data] cache hit: {out} ({n} bytes, {n / 1e9:.2f} GB)",
              flush=True)
        return out
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp = out + ".partial"
    names = shard_names(n_shards, repo_id, subdir)
    print(f"[data] building {out} from {len(names)} shard(s): "
          f"{', '.join(names)}", flush=True)
    from huggingface_hub import hf_hub_download
    total = ndocs = 0
    reported = 0
    t0 = time.time()
    with open(tmp, "wb", buffering=1 << 22) as f:
        for name in names:
            print(f"[data] fetching {name}", flush=True)
            path = _retry(lambda n=name: hf_hub_download(
                repo_id=repo_id, filename=n, repo_type="dataset"),
                f"download {name}")
            pf = pq.ParquetFile(path)
            for rb in pf.iter_batches(batch_size=4096, columns=["text"]):
                docs = [t.encode("utf-8", "ignore") + bytes([SEP])
                        for t in rb.column("text").to_pylist() if t]
                blob = b"".join(docs)
                f.write(blob)
                total += len(blob)
                ndocs += len(docs)
                if total - reported > (1 << 29):     # every ~512 MB
                    reported = total
                    print(f"[data]   {total / 1e9:.2f} GB, {ndocs} docs, "
                          f"{time.time() - t0:.0f}s", flush=True)
    if total == 0:
        os.remove(tmp)
        raise RuntimeError("no bytes written -- parquet had no 'text' column?")
    os.replace(tmp, out)
    print(f"[data] wrote {out}: {total} bytes ({total / 1e9:.2f} GB) from "
          f"{ndocs} docs in {time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- induction
def induction_mask_np(b, window, n=8):
    """b: (T,) uint8 array. True at positions whose length-n context last
    occurred more than `window` bytes earlier -- the recall-dependent slice.

    Vectorized equivalent of the per-position dict loop this replaced (see
    tests/test_data.py, which checks them byte-for-byte): pack each n-gram
    into an exact uint64 key, stable-argsort so equal keys form runs in
    ascending position order, and read each position's nearest earlier
    occurrence off its predecessor in that run.
    """
    if n > 8:
        raise ValueError("n-gram longer than 8 bytes does not fit a uint64 key")
    T = int(b.shape[0])
    mask = np.zeros(T, dtype=bool)
    m = T - n + 1                      # one key per position n-1 .. T-1
    if m <= 0:
        return mask
    b64 = np.asarray(b, dtype=np.uint8).astype(np.uint64)
    keys = np.zeros(m, dtype=np.uint64)
    for i in range(n):
        keys = (keys << np.uint64(8)) | b64[i:i + m]
    order = np.argsort(keys, kind="stable")
    same = np.empty(m, dtype=bool)
    same[0] = False
    same[1:] = keys[order][1:] == keys[order][:-1]
    cand = np.empty(m, dtype=np.int64)
    cand[0] = -1
    cand[1:] = order[:-1]              # nearest earlier position, same key
    prev = np.empty(m, dtype=np.int64)
    prev[order] = np.where(same, cand, -1)
    j = np.arange(m, dtype=np.int64)
    mask[n - 1:] = (prev >= 0) & ((j - prev) > window)
    return mask


def induction_mask_batch(arr, window, n=8):
    """arr: (B, T) uint8 array -> (B, T) bool array."""
    return np.stack([induction_mask_np(arr[i], window, n)
                     for i in range(arr.shape[0])])


def induction_mask(chunk, window, n=8):
    """chunk: (T,) byte-valued tensor -> (T,) bool tensor."""
    b = chunk.detach().to("cpu").numpy().astype(np.uint8)
    return torch.from_numpy(induction_mask_np(b, window, n))


# ---------------------------------------------------------------- sampler
class ByteData:
    """Seeded random contiguous windows over the flat uint8 byte file.

    The last `eval_frac` of the file is reserved for eval; train windows lie
    entirely in the head, so the two splits share no bytes.
    """

    def __init__(self, path, eval_frac=EVAL_FRAC):
        self.path = path
        self.arr = np.memmap(path, dtype=np.uint8, mode="r")
        self.n = int(self.arr.shape[0])
        self.n_train = int(self.n * (1.0 - eval_frac))

    def bounds(self, split):
        if split == "train":
            return 0, self.n_train
        if split == "eval":
            return self.n_train, self.n
        raise ValueError(f"unknown split {split!r}")

    def window_at(self, start, T):
        return np.asarray(self.arr[start:start + T])

    def batches(self, B, T, window, device, seed=0, split="train",
                with_masks=False):
        """Yields (idx (B,T+1) long on device, ind_mask (B,T) bool or None)."""
        lo, hi = self.bounds(split)
        last = hi - (T + 1)
        if last < lo:
            raise RuntimeError(
                f"{split} split of {self.path} holds {hi - lo} bytes, too few "
                f"for a {T + 1}-byte window (build more shards)")
        rng = np.random.default_rng(seed)
        buf = np.empty((B, T + 1), dtype=np.uint8)
        while True:
            starts = rng.integers(lo, last, size=B, endpoint=True)
            for i, s in enumerate(starts):
                buf[i] = self.arr[s:s + T + 1]
            # ship the bytes, widen on the far side (8x less PCIe traffic)
            idx = torch.from_numpy(buf).to(device).long()
            masks = None
            if with_masks:
                masks = torch.from_numpy(
                    induction_mask_batch(buf[:, :-1], window)).to(device)
            yield idx, masks


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description="build and sanity-check the local FineWeb-Edu byte cache")
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS,
                    help="parquet shards to download (3.47 GB of bytes each)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    path = build_byte_cache(args.shards, args.data_dir)
    d = ByteData(path)
    print(f"[data] {path}: {d.n} bytes ({d.n / 1e9:.3f} GB); "
          f"train [0, {d.n_train}), eval [{d.n_train}, {d.n})")

    def first(split):
        return next(ByteData(path).batches(args.batch, args.seq_len, 256,
                                           "cpu", seed=args.seed,
                                           split=split))[0]

    a, b = first("train"), first("train")
    print(f"[data] determinism (train, seed={args.seed}): "
          f"{'IDENTICAL' if torch.equal(a, b) else 'MISMATCH'}")
    e1, e2 = first("eval"), first("eval")
    print(f"[data] determinism (eval,  seed={args.seed}): "
          f"{'IDENTICAL' if torch.equal(e1, e2) else 'MISMATCH'}")
    sample = bytes(a[0, :200].to(torch.uint8).numpy())
    print("[data] first 200 bytes of a training sample:")
    print(repr(sample.decode("utf-8", "replace")))


if __name__ == "__main__":
    main()
