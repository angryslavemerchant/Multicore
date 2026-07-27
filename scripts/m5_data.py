"""Deterministic, cached FineWeb-Edu for the M5 runs, in bytes or in tokens.

Replaces the old live `load_dataset(..., streaming=True)` path over
HuggingFaceFW/fineweb-edu, which was (a) fragile -- the hub connection drops
mid-run ("peer closed connection without sending complete message body",
"Cannot send a request, as the client has been closed"), which has already
cost two runs including a rented GPU -- and (b) non-deterministic, so two
model configs never saw the same bytes and their losses were not comparable.

TWO CORPORA, one pipeline:

  bytes   `sample/10BT`, utf-8 bytes + a 0x00 document separator, flat uint8.
          Every result before 2026-07-27 was produced this way. Kept working
          and bit-identical so those runs stay reproducible.
  tokens  `sample/100BT`, GPT-NeoX subword ids + `<|endoftext|>` (id 0) as the
          separator, flat uint16. This is what the comparisons against
          published models need: open-sci-ref-0.01 trained on FineWeb-Edu with
          the GPT-NeoX tokenizer at vocab 50304, and a byte-level loss cannot
          be put on the same axis as a token-level one at all. `sample/100BT`
          is the same document pool as `sample/10BT`, just a bigger draw --
          10BT would not cover a single 8.3B-token reference checkpoint.

uint16 caps the vocabulary at 65535; the NeoX vocab is 50277 (50304 padded),
and `build_token_cache` asserts every id fits rather than trusting that.

Pipeline (both modes):
  1. hf_hub_download the deterministic first N parquet shards (resumable +
     cached on disk, and every hub call is wrapped in an exponential-backoff
     retry, since the hub is the flaky bit).
  2. Convert ONCE to a flat file under runs/data/. Measured: each 2.15 GB
     `sample/10BT` shard is 726k docs -> 3.47 GB of bytes (~25 s), which is
     ~0.75B NeoX tokens (1.5 GB as uint16). So a 1.5e9-BYTE run wants ~1 shard
     and a 8.3e9-TOKEN run wants ~11. Tokenising is the slow half (minutes per
     shard, not seconds), so the token build records its progress and resumes
     at the next shard instead of restarting.
  3. Sample batches as B random contiguous (T+1)-symbol windows drawn from a
     seeded numpy Generator, so a given (seed, B, T) yields the identical
     sequence of batches for every model config. The last EVAL_FRAC of the
     file is eval-only; training samples strictly from the head, so eval
     symbols are never trained on.

Standalone use (do this once on a fresh machine, before launching runs):
  python scripts/m5_data.py --shards 4                 # bytes
  python scripts/m5_data.py --mode tokens --shards 11  # tokens
"""
import argparse, json, os, time

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

REPO_ID = "HuggingFaceFW/fineweb-edu"
SUBDIR = "sample/10BT"            # the `sample-10BT` config: 14 parquet shards
SUBDIR_TOKENS = "sample/100BT"    # same pool, big enough for a real token run
TOKENIZER = "EleutherAI/gpt-neox-20b"
DEFAULT_SHARDS = 4
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "data")
EVAL_FRAC = 0.05                  # last 5% of the file is eval-only
SEP = 0                           # document separator byte

# Induction n-gram length. The key is packed into ONE uint64, so it is 8
# symbols of 8 bits or 4 of 16 -- and 4 subword tokens is a longer span of text
# than 8 bytes anyway. Byte-era and token-era induction numbers are measuring
# different position sets and must not be compared.
IND_NGRAM_BYTES = 8
IND_NGRAM_TOKENS = 4


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


def _fetch(name, repo_id):
    from huggingface_hub import hf_hub_download
    print(f"[data] fetching {name}", flush=True)
    return _retry(lambda: hf_hub_download(repo_id=repo_id, filename=name,
                                          repo_type="dataset"),
                  f"download {name}")


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
    total = ndocs = 0
    reported = 0
    t0 = time.time()
    with open(tmp, "wb", buffering=1 << 22) as f:
        for name in names:
            path = _fetch(name, repo_id)
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


# --------------------------------------------------------------- token cache
def token_cache_path(n_shards=DEFAULT_SHARDS, data_dir=None,
                     tokenizer=TOKENIZER):
    tag = tokenizer.split("/")[-1].replace(".", "-")
    return os.path.join(data_dir or DEFAULT_DATA_DIR,
                        f"fineweb100_{tag}_{n_shards}shards_u16.bin")


def build_token_cache(n_shards=DEFAULT_SHARDS, data_dir=None, repo_id=REPO_ID,
                      subdir=SUBDIR_TOKENS, tokenizer=TOKENIZER,
                      doc_batch=1024):
    """Return the path to the flat uint16 token file, building it if needed.

    Documents are concatenated as `ids + [eos]`, exactly as the byte cache
    concatenates `utf-8 + 0x00`, so a sampled window may span a document
    boundary and the model sees the separator as an ordinary token. That is
    what open-sci-ref (and essentially every pretraining pipeline) does.

    Tokenising is minutes per shard rather than the byte path's seconds, so
    progress is recorded per shard in a `.progress` sidecar and a restart
    truncates the partial file back to the last completed shard instead of
    starting over. The sidecar carries the shard LIST, so changing --shards or
    the tokenizer invalidates it rather than silently appending the wrong data.

    `tokenizer` is a hub name, or an already-loaded object exposing
    `__call__(list_of_str) -> {"input_ids": ...}` and `eos_token_id` — which
    is how tests drive this without a network round trip or a 300 MB import.
    """
    name = tokenizer if isinstance(tokenizer, str) else getattr(
        tokenizer, "name_or_path", "custom")
    out = token_cache_path(n_shards, data_dir, name)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        n = os.path.getsize(out) // 2
        print(f"[data] cache hit: {out} ({n} tokens, {n / 1e9:.2f}B)",
              flush=True)
        return out
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmp, prog = out + ".partial", out + ".progress"
    names = shard_names(n_shards, repo_id, subdir)
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer)
    else:
        tok = tokenizer
    eos = tok.eos_token_id
    assert eos is not None, f"{name} has no eos_token_id to separate docs"

    done, total, ndocs = 0, 0, 0
    if os.path.exists(prog) and os.path.exists(tmp):
        st = json.load(open(prog))
        if st.get("names") == names and st.get("tokenizer") == name:
            done, total, ndocs = st["shards"], st["tokens"], st["docs"]
            with open(tmp, "r+b") as f:          # drop a half-written shard
                f.truncate(total * 2)
            print(f"[data] resuming at shard {done}/{len(names)} "
                  f"({total / 1e9:.2f}B tokens already written)", flush=True)
        else:
            os.remove(prog)
    print(f"[data] building {out} from {len(names)} shard(s) of {subdir} "
          f"with {name}", flush=True)
    t0 = time.time()
    reported = total
    with open(tmp, "r+b" if done else "wb", buffering=1 << 22) as f:
        f.seek(total * 2)
        for si, name in enumerate(names):
            if si < done:
                continue
            path = _fetch(name, repo_id)
            pf = pq.ParquetFile(path)
            for rb in pf.iter_batches(batch_size=doc_batch, columns=["text"]):
                texts = [t for t in rb.column("text").to_pylist() if t]
                if not texts:
                    continue
                ids = tok(texts, add_special_tokens=False)["input_ids"]
                arr = np.concatenate(
                    [np.asarray(d + [eos], dtype=np.int64) for d in ids])
                hi = int(arr.max())
                assert hi < 65536, (
                    f"token id {hi} does not fit uint16; {tokenizer} is too "
                    f"large for this cache format")
                f.write(arr.astype(np.uint16).tobytes())
                total += int(arr.size)
                ndocs += len(texts)
                if total - reported > (1 << 27):    # every ~134M tokens
                    reported = total
                    print(f"[data]   {total / 1e9:.3f}B tokens, {ndocs} docs, "
                          f"{time.time() - t0:.0f}s", flush=True)
            f.flush()
            json.dump({"names": names, "tokenizer": name, "shards": si + 1,
                       "tokens": total, "docs": ndocs}, open(prog, "w"))
    if total == 0:
        os.remove(tmp)
        raise RuntimeError("no tokens written -- parquet had no 'text' column?")
    os.replace(tmp, out)
    if os.path.exists(prog):
        os.remove(prog)
    print(f"[data] wrote {out}: {total} tokens ({total / 1e9:.3f}B, "
          f"{total * 2 / 1e9:.2f} GB) from {ndocs} docs in "
          f"{time.time() - t0:.0f}s", flush=True)
    return out


# ---------------------------------------------------------------- induction
def induction_mask_np(b, window, n=IND_NGRAM_BYTES, bits=8):
    """b: (T,) integer array. True at positions whose length-n context last
    occurred more than `window` symbols earlier -- the recall-dependent slice.

    `bits` is the width of one symbol: 8 for bytes, 16 for uint16 subword ids.
    n*bits must fit the uint64 key, so n <= 8 for bytes and n <= 4 for tokens.
    A symbol wider than `bits` raises rather than silently wrapping, which is
    what an unchecked uint8 cast used to do to token ids.

    Vectorized equivalent of the per-position dict loop this replaced (see
    tests/test_data.py, which checks them symbol-for-symbol): pack each n-gram
    into an exact uint64 key, stable-argsort so equal keys form runs in
    ascending position order, and read each position's nearest earlier
    occurrence off its predecessor in that run.
    """
    if n * bits > 64:
        raise ValueError(f"{n} symbols x {bits} bits exceeds a uint64 key")
    a = np.asarray(b)
    T = int(a.shape[0])
    # checked before the short-sequence early return, so the contract holds
    # whatever the length: a symbol too wide for `bits` is always an error
    if T and int(a.max()) >= (1 << bits):
        raise ValueError(f"symbol {int(a.max())} does not fit {bits} bits")
    mask = np.zeros(T, dtype=bool)
    m = T - n + 1                      # one key per position n-1 .. T-1
    if m <= 0:
        return mask
    b64 = a.astype(np.uint64)
    keys = np.zeros(m, dtype=np.uint64)
    for i in range(n):
        keys = (keys << np.uint64(bits)) | b64[i:i + m]
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


def induction_mask_batch(arr, window, n=IND_NGRAM_BYTES, bits=8):
    """arr: (B, T) integer array -> (B, T) bool array."""
    return np.stack([induction_mask_np(arr[i], window, n, bits)
                     for i in range(arr.shape[0])])


def induction_mask(chunk, window, n=IND_NGRAM_BYTES, bits=8):
    """chunk: (T,) integer tensor -> (T,) bool tensor."""
    b = chunk.detach().to("cpu").numpy()
    return torch.from_numpy(induction_mask_np(b, window, n, bits))


# ---------------------------------------------------------------- sampler
class PackedData:
    """Seeded random contiguous windows over a flat symbol file.

    The last `eval_frac` of the file is reserved for eval; train windows lie
    entirely in the head, so the two splits share no symbols.
    """

    dtype = np.uint8
    bits = 8
    ngram = IND_NGRAM_BYTES

    def __init__(self, path, eval_frac=EVAL_FRAC):
        self.path = path
        self.arr = np.memmap(path, dtype=self.dtype, mode="r")
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

    def _to_device(self, buf, device):
        # torch has no from_numpy path for uint16, and the widening is 262 KB
        # per batch at B=16, T=4096 -- nothing next to the step it feeds. The
        # uint8 branch keeps the byte path shipping one byte per symbol.
        if buf.dtype == np.uint8:
            return torch.from_numpy(buf).to(device).long()
        return torch.from_numpy(buf.astype(np.int32)).to(device).long()

    def batches(self, B, T, window, device, seed=0, split="train",
                with_masks=False, ngram=None, skip=0, stride=1):
        """Yields (idx (B,T+1) long on device, ind_mask (B,T) bool or None).

        `skip` advances the generator past the first `skip` batches WITHOUT
        reading them. A resumed run has to continue the stream, not replay it
        -- a WSD cooldown branching at 80M tokens must train on 80M..100M, so
        that the finished model has seen 0..100M exactly once. Only the RNG
        draw is repeated, never the gather, so fast-forwarding 15k batches
        costs microseconds rather than the 1B symbols of I/O it stands for.

        `stride` skips (stride - 1) batches after each yield. With
        skip=rank, stride=world_size, the N ranks of a data-parallel run
        partition ONE stream between them -- rank r takes batches r, r+W,
        r+2W. Their union is exactly the single-GPU stream, so an 8-GPU run
        and a 1-GPU run at the same token count see the same data. Without
        this every rank draws the identical batch and 8 GPUs buy 8x the cost
        for 1x the data, which trains and converges and looks completely
        normal on a loss curve.
        """
        lo, hi = self.bounds(split)
        last = hi - (T + 1)
        if last < lo:
            raise RuntimeError(
                f"{split} split of {self.path} holds {hi - lo} symbols, too "
                f"few for a {T + 1}-symbol window (build more shards)")
        n = self.ngram if ngram is None else ngram
        rng = np.random.default_rng(seed)
        for _ in range(skip):
            rng.integers(lo, last, size=B, endpoint=True)
        buf = np.empty((B, T + 1), dtype=self.dtype)
        while True:
            starts = rng.integers(lo, last, size=B, endpoint=True)
            for i, s in enumerate(starts):
                buf[i] = self.arr[s:s + T + 1]
            idx = self._to_device(buf, device)
            masks = None
            if with_masks:
                masks = torch.from_numpy(
                    induction_mask_batch(buf[:, :-1], window, n,
                                         self.bits)).to(device)
            yield idx, masks
            for _ in range(stride - 1):        # the other ranks' batches
                rng.integers(lo, last, size=B, endpoint=True)


class ByteData(PackedData):
    """uint8 utf-8 bytes, 0x00 document separator (`sample/10BT`)."""


class TokenData(PackedData):
    """uint16 GPT-NeoX subword ids, `<|endoftext|>` separator
    (`sample/100BT`)."""

    dtype = np.uint16
    bits = 16
    ngram = IND_NGRAM_TOKENS


def open_data(tokens, n_shards=DEFAULT_SHARDS, data_dir=None):
    """The cache for the corpus `tokens` selects, building it if needed."""
    if tokens:
        return TokenData(build_token_cache(n_shards, data_dir))
    return ByteData(build_byte_cache(n_shards, data_dir))


# ---------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(
        description="build and sanity-check the local FineWeb-Edu cache")
    ap.add_argument("--mode", choices=("bytes", "tokens"), default="bytes")
    ap.add_argument("--shards", type=int, default=DEFAULT_SHARDS,
                    help="parquet shards to download; a sample/10BT shard is "
                         "3.47 GB of bytes, or ~0.75B NeoX tokens")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()

    tokens = args.mode == "tokens"
    d = open_data(tokens, args.shards, args.data_dir)
    unit = "tokens" if tokens else "bytes"
    print(f"[data] {d.path}: {d.n} {unit} ({d.n / 1e9:.3f}B); "
          f"train [0, {d.n_train}), eval [{d.n_train}, {d.n})")

    cls = type(d)

    def first(split):
        return next(cls(d.path).batches(args.batch, args.seq_len, 256,
                                        "cpu", seed=args.seed,
                                        split=split))[0]

    a, b = first("train"), first("train")
    print(f"[data] determinism (train, seed={args.seed}): "
          f"{'IDENTICAL' if torch.equal(a, b) else 'MISMATCH'}")
    e1, e2 = first("eval"), first("eval")
    print(f"[data] determinism (eval,  seed={args.seed}): "
          f"{'IDENTICAL' if torch.equal(e1, e2) else 'MISMATCH'}")
    print(f"[data] first 200 {unit} of a training sample:")
    if tokens:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(TOKENIZER)
        print(repr(tk.decode(a[0, :200].tolist())))
    else:
        print(repr(bytes(a[0, :200].to(torch.uint8).numpy())
                   .decode("utf-8", "replace")))


if __name__ == "__main__":
    main()
