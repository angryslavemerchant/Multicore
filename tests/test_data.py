"""Data-pipeline gates for scripts/m5_data.py. No network required.

IND — the vectorized induction_mask is byte-for-byte identical to the
      per-position dict loop it replaced (that loop is copied verbatim below
      as the ground truth).
DET — the byte sampler is deterministic: a given (seed, B, T, split) yields
      the identical sequence of batches every time, from a fresh process
      state, so two model configs are compared on the same bytes.
SPL — train windows lie entirely in the head of the byte file and eval
      windows in the held-out tail; the splits share no bytes.
BLD — the parquet -> flat-uint8 conversion writes each doc's utf-8 bytes plus
      one 0 separator, and a second call reuses the cache instead of
      rebuilding (run against a local parquet, with the hub stubbed out).

Runnable via pytest or `python tests/test_data.py`.
"""
import os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from m5_data import (ByteData, induction_mask, induction_mask_batch,
                     induction_mask_np)


def induction_mask_loop(chunk, window, n=8):
    """The original scripts/m5_arch.py implementation, verbatim: ground truth.

    chunk: (T,) uint8 tensor. True at positions whose length-n context
    reoccurs from an earlier occurrence > window back.
    """
    T = chunk.shape[0]
    mask = torch.zeros(T, dtype=torch.bool)
    last = {}
    b = chunk.tolist()
    for t in range(n - 1, T):
        key = tuple(b[t - n + 1:t + 1])
        prev = last.get(key)
        if prev is not None and t - prev > window:
            mask[t] = True
        last[key] = t
    return mask


def test_induction_mask_matches_loop():
    g = torch.Generator().manual_seed(0)
    cases = []
    # small alphabets force plenty of repeated 8-grams (a 256-way alphabet
    # almost never repeats, which would make the test vacuous)
    for vocab in (2, 4, 17, 256):
        for T in (64, 257, 1024):
            for window in (0, 1, 8, 64, 256):
                cases.append((vocab, T, window))
    hits = 0
    for vocab, T, window in cases:
        x = torch.randint(0, vocab, (T,), generator=g)
        ref = induction_mask_loop(x, window)
        fast = induction_mask(x, window)
        assert fast.dtype == torch.bool and fast.shape == ref.shape
        assert torch.equal(fast, ref), (
            f"IND FAILED at vocab={vocab} T={T} window={window}: "
            f"{int((fast ^ ref).sum())} positions differ")
        hits += int(ref.sum())
    assert hits > 0, "IND vacuous: no induction positions in any case"

    # edge cases: shorter than the n-gram, and a degenerate all-equal stream
    for T in (0, 1, 7, 8, 9):
        x = torch.zeros(T, dtype=torch.long)
        assert torch.equal(induction_mask(x, 4), induction_mask_loop(x, 4)), \
            f"IND FAILED on all-zero T={T}"
    x = torch.randint(0, 3, (512,), generator=g)
    for n in (2, 3, 8):
        assert torch.equal(induction_mask(x, 16, n=n),
                           induction_mask_loop(x, 16, n=n)), \
            f"IND FAILED at n={n}"

    # batched helper agrees row-by-row
    arr = torch.randint(0, 5, (4, 300), generator=g)
    b = induction_mask_batch(arr.numpy().astype(np.uint8), 32)
    for i in range(arr.shape[0]):
        assert np.array_equal(b[i], induction_mask_loop(arr[i], 32).numpy()), \
            f"IND FAILED: batch row {i}"
    print(f"IND PASSED: vectorized induction_mask == dict loop on "
          f"{len(cases)} random cases ({hits} induction positions)")


def _fake_cache(tmp, n=200000, seed=7):
    """A byte file that stands in for the FineWeb cache (no network)."""
    path = os.path.join(tmp, "fake.bin")
    rng = np.random.default_rng(seed)
    rng.integers(0, 256, size=n, dtype=np.uint8).tofile(path)
    return path


def test_sampler_deterministic():
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_cache(tmp)
        B, T = 3, 64

        def first_k(seed, split, k=4, with_masks=False):
            # a FRESH ByteData each time: this is what two separate runs do
            it = ByteData(path).batches(B, T, 16, "cpu", seed=seed,
                                        split=split, with_masks=with_masks)
            return [next(it) for _ in range(k)]

        a, b = first_k(0, "train"), first_k(0, "train")
        for i, ((x, _), (y, _)) in enumerate(zip(a, b)):
            assert x.shape == (B, T + 1) and x.dtype == torch.long, \
                f"DET FAILED: batch {i} has shape {tuple(x.shape)} {x.dtype}"
            assert torch.equal(x, y), f"DET FAILED: train batch {i} differs"
        c = first_k(1, "train")
        assert not torch.equal(a[0][0], c[0][0]), \
            "DET vacuous: a different seed gave the same batch"
        # the eval set (masks on) is reproducible too
        e1 = first_k(0, "eval", k=2, with_masks=True)
        e2 = first_k(0, "eval", k=2, with_masks=True)
        for i, ((x, mx), (y, my)) in enumerate(zip(e1, e2)):
            assert torch.equal(x, y), f"DET FAILED: eval batch {i} differs"
            assert mx.shape == (B, T) and mx.dtype == torch.bool
            assert torch.equal(mx, my), f"DET FAILED: eval mask {i} differs"
        print("DET PASSED: same (seed, B, T, split) -> identical batches "
              "from independent ByteData instances")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_split_disjoint_and_contiguous():
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_cache(tmp)
        raw = open(path, "rb").read()
        d = ByteData(path)
        assert 0 < d.n_train < d.n, f"SPL FAILED: bounds {d.n_train}/{d.n}"
        B, T = 4, 64
        for split, (lo, hi) in (("train", (0, d.n_train)),
                                ("eval", (d.n_train, d.n))):
            it = ByteData(path).batches(B, T, 16, "cpu", seed=3, split=split)
            for _ in range(6):
                idx, _ = next(it)
                for r in range(B):
                    row = bytes(idx[r].to(torch.uint8).numpy())
                    off = raw.find(row)
                    # random bytes: a 65-byte window occurs exactly once
                    assert off >= 0, \
                        f"SPL FAILED: {split} row is not a contiguous slice"
                    assert lo <= off and off + T + 1 <= hi, \
                        (f"SPL FAILED: {split} window at [{off}, "
                         f"{off + T + 1}) escapes [{lo}, {hi})")
        print(f"SPL PASSED: train windows within [0, {d.n_train}), eval "
              f"within [{d.n_train}, {d.n}), all contiguous slices")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_build_byte_cache_offline():
    """The conversion itself, with the hub stubbed out by a local parquet."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    import m5_data

    tmp = tempfile.mkdtemp()
    docs = ["hello world", "déjà vu — café", "x" * 5000, ""]
    real_dl, real_names = huggingface_hub.hf_hub_download, m5_data.shard_names
    try:
        src = os.path.join(tmp, "shard.parquet")
        pq.write_table(pa.table({"text": docs, "id": ["a", "b", "c", "d"]}), src)
        huggingface_hub.hf_hub_download = lambda **kw: src
        m5_data.shard_names = lambda n, *a, **k: ["fake/000.parquet"] * n

        out = m5_data.build_byte_cache(2, data_dir=tmp)
        blob = open(out, "rb").read()
        want = b"".join(d.encode("utf-8") + b"\x00" for d in docs if d) * 2
        assert blob == want, (f"BLD FAILED: {len(blob)} bytes, expected "
                              f"{len(want)} (2 shards x {len(docs) - 1} docs)")
        assert blob.count(b"\x00") == 6, "BLD FAILED: wrong separator count"
        assert not os.path.exists(out + ".partial"), "BLD FAILED: .partial left"

        # second call must be a cache hit, not a rebuild
        stamp = os.stat(out).st_mtime_ns
        huggingface_hub.hf_hub_download = lambda **kw: (_ for _ in ()).throw(
            AssertionError("BLD FAILED: rebuilt despite a valid cache"))
        again = m5_data.build_byte_cache(2, data_dir=tmp)
        assert again == out and os.stat(out).st_mtime_ns == stamp
        print(f"BLD PASSED: {len(blob)} bytes written (utf-8 + 0 separator "
              f"per doc), rebuild skipped on cache hit")
    finally:
        huggingface_hub.hf_hub_download = real_dl
        m5_data.shard_names = real_names
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_induction_mask_matches_loop()
    test_sampler_deterministic()
    test_split_disjoint_and_contiguous()
    test_build_byte_cache_offline()
    print("ALL DATA GATES PASSED")
