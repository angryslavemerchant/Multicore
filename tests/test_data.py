"""Data-pipeline gates for scripts/m5_data.py. No network required.

IND — the vectorized induction_mask is symbol-for-symbol identical to the
      per-position dict loop it replaced (that loop is copied verbatim below
      as the ground truth), for 8-bit bytes and for 16-bit token ids.
DET — the sampler is deterministic: a given (seed, B, T, split) yields the
      identical sequence of batches every time, from a fresh process state,
      so two model configs are compared on the same symbols. Checked for the
      uint8 byte file and the uint16 token file.
SPL — train windows lie entirely in the head of the file and eval windows in
      the held-out tail; the splits share no symbols.
BLD — the parquet -> flat-uint8 conversion writes each doc's utf-8 bytes plus
      one 0 separator, and a second call reuses the cache instead of
      rebuilding (run against a local parquet, with the hub stubbed out).
TOK — the parquet -> flat-uint16 conversion writes each doc's ids plus one
      eos, and a build killed mid-way resumes at the next SHARD rather than
      restarting or appending a duplicate.

Runnable via pytest or `python tests/test_data.py`.
"""
import json, os, shutil, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import torch

from m5_data import (ByteData, TokenData, build_token_cache, induction_mask,
                     induction_mask_batch, induction_mask_np)


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


def test_induction_mask_token_space():
    """The same equivalence at 16 bits, which is the token corpus's key width.

    The old implementation cast to uint8 unconditionally, so a subword id
    would have wrapped mod 256 and silently reported the induction positions
    of a different sequence. That is now an error, and 4x16 bits is the widest
    n-gram a uint64 key holds.
    """
    g = torch.Generator().manual_seed(1)
    hits = 0
    # a SMALL alphabet of LARGE ids: 4-grams over 50304 uniform ids never
    # repeat in a few hundred positions, so a uniform draw would make this
    # vacuous. Every id here needs more than 8 bits, which is the point.
    for pool in (torch.tensor([300, 50303]),
                 torch.tensor([256, 999, 4096, 65535]),
                 torch.tensor([511, 512, 32767, 32768, 50304 - 1])):
        for T in (128, 777):
            for window in (0, 16, 128):
                x = pool[torch.randint(0, len(pool), (T,), generator=g)]
                ref = induction_mask_loop(x, window, n=4)
                fast = induction_mask(x, window, n=4, bits=16)
                assert torch.equal(fast, ref), (pool.tolist(), T, window)
                hits += int(ref.sum())
    assert hits > 0, "IND vacuous: no token induction positions"
    # a key that does not fit must raise, not wrap
    for bad in ((torch.tensor([256, 1, 2, 3, 4]), 8, 8),
                (torch.tensor([65536, 1, 2, 3, 4]), 4, 16)):
        x, n, bits = bad
        try:
            induction_mask(x, 1, n=n, bits=bits)
            raise AssertionError(f"IND FAILED: {bits}-bit overflow not caught")
        except ValueError:
            pass
    try:
        induction_mask(torch.zeros(9, dtype=torch.long), 1, n=5, bits=16)
        raise AssertionError("IND FAILED: 5x16 bits accepted for a uint64 key")
    except ValueError:
        pass
    print(f"IND-TOK PASSED: 4-token induction_mask == dict loop over 16-bit "
          f"ids ({hits} positions); overflow and >64-bit keys rejected")


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


def test_sampler_skip_continues_stream():
    """SKP: `skip=k` yields exactly the stream from batch k onward.

    This is what makes a WSD cooldown honest. A branch resuming at 0.8B tokens
    must train on 0.8B..1.0B, so that the finished model has seen 0..1.0B once;
    if `skip` were off by anything the cooldown would re-train on the trunk's
    own data and the endpoint would be a 0.8B model with a 0.2B second epoch.
    """
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_cache(tmp, n=400000)
        B, T = 3, 32
        full = ByteData(path).batches(B, T, 8, "cpu", seed=5)
        ref = [next(full)[0] for _ in range(12)]
        for k in (0, 1, 7, 11):
            it = ByteData(path).batches(B, T, 8, "cpu", seed=5, skip=k)
            for j in range(12 - k):
                got = next(it)[0]
                assert torch.equal(got, ref[k + j]), (k, j)
        # and it must actually SKIP, not restart
        assert not torch.equal(
            next(ByteData(path).batches(B, T, 8, "cpu", seed=5, skip=3))[0],
            ref[0]), "SKP vacuous: skip=3 returned the first batch"
        print("SKP PASSED: skip=k == the stream from batch k, for k in "
              "{0,1,7,11}; no overlap and no gap at the seam")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_sampler_stride_partitions_the_stream():
    """SHD: N ranks with skip=r, stride=N partition ONE stream exactly.

    This is the data-parallel correctness gate. Without it every rank draws
    the same batch and N GPUs buy N x the cost for 1 x the data — a bug that
    trains, converges, and looks entirely normal on a loss curve.
    """
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_cache(tmp, n=400000)
        B, T, W = 3, 32, 4
        ref = [next(ByteData(path).batches(B, T, 8, "cpu", seed=5))[0]
               for _ in range(1)]
        it = ByteData(path).batches(B, T, 8, "cpu", seed=5)
        ref = [next(it)[0] for _ in range(12)]
        for r in range(W):
            rk = ByteData(path).batches(B, T, 8, "cpu", seed=5, skip=r,
                                        stride=W)
            for j in range(3):
                got = next(rk)[0]
                assert torch.equal(got, ref[r + j * W]), (r, j)
        # and the ranks are disjoint: no two see the same batch
        firsts = [next(ByteData(path).batches(B, T, 8, "cpu", seed=5, skip=r,
                                              stride=W))[0] for r in range(W)]
        for a in range(W):
            for b in range(a + 1, W):
                assert not torch.equal(firsts[a], firsts[b]), (a, b)
        print(f"SHD PASSED: {W} ranks at stride {W} reproduce the single-GPU "
              f"stream exactly and share no batch")
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


class _StubTokenizer:
    """Deterministic stand-in for the NeoX tokenizer: no network, no import.

    Ids are spread across the whole uint16 range on purpose — a 1-byte-ish
    vocabulary would let a truncating write pass.
    """
    eos_token_id = 0
    name_or_path = "stub"

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [[1 + (hash_(t) + i) % 50000
                               for i in range(3 + len(t) % 5)]
                              for t in texts]}


def hash_(s):
    h = 0
    for c in s:
        h = (h * 131 + ord(c)) % 65521
    return h


def _stub_hub(tmp, docs, n_shards):
    """Point m5_data's shard list + download at one local parquet."""
    import huggingface_hub
    import pyarrow as pa
    import pyarrow.parquet as pq
    import m5_data
    src = os.path.join(tmp, "shard.parquet")
    pq.write_table(pa.table({"text": docs}), src)
    real = (huggingface_hub.hf_hub_download, m5_data.shard_names)
    huggingface_hub.hf_hub_download = lambda **kw: src
    m5_data.shard_names = lambda n, *a, **k: [f"fake/{i:03d}.parquet"
                                              for i in range(n)]
    return real


def test_build_token_cache_offline_and_resume():
    """TOK: uint16 layout, eos separators, and shard-level resume."""
    import huggingface_hub
    import m5_data
    tmp = tempfile.mkdtemp()
    docs = ["hello world", "déjà vu — café", "x" * 900, ""]
    tok = _StubTokenizer()
    real = _stub_hub(tmp, docs, 3)
    try:
        out = build_token_cache(3, data_dir=tmp, tokenizer=tok, doc_batch=2)
        arr = np.fromfile(out, dtype=np.uint16)
        want = np.concatenate([
            np.asarray(d + [0], dtype=np.uint16)
            for _ in range(3)
            for d in tok([t for t in docs if t])["input_ids"]])
        assert np.array_equal(arr, want), (arr[:20], want[:20])
        assert int((arr == 0).sum()) == 3 * 3, "wrong separator count"
        assert arr.max() > 255, "TOK vacuous: no id needs more than 8 bits"
        assert not os.path.exists(out + ".partial")
        assert not os.path.exists(out + ".progress")

        # a build killed after shard 1 must resume, not restart or duplicate
        os.rename(out, out + ".partial")
        with open(out + ".partial", "r+b") as f:      # a torn trailing write
            f.truncate((len(arr) // 3) * 2 + 6)
        names = m5_data.shard_names(3)
        json.dump({"names": names, "tokenizer": "stub", "shards": 1,
                   "tokens": len(arr) // 3, "docs": 3},
                  open(out + ".progress", "w"))
        again = build_token_cache(3, data_dir=tmp, tokenizer=tok, doc_batch=2)
        assert np.array_equal(np.fromfile(again, dtype=np.uint16), arr), \
            "TOK FAILED: resumed file differs from a clean build"

        # a cache hit must not rebuild
        huggingface_hub.hf_hub_download = lambda **kw: (_ for _ in ()).throw(
            AssertionError("TOK FAILED: rebuilt despite a valid cache"))
        assert build_token_cache(3, data_dir=tmp, tokenizer=tok) == out
        print(f"TOK PASSED: {len(arr)} uint16 tokens (max id {arr.max()}), "
              f"eos separator per doc, resume from shard 1 byte-identical")
    finally:
        huggingface_hub.hf_hub_download, m5_data.shard_names = real
        shutil.rmtree(tmp, ignore_errors=True)


def test_token_sampler():
    """DET/SPL for the uint16 path, incl. the widening torch cannot do."""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "fake_u16.bin")
        rng = np.random.default_rng(11)
        raw = rng.integers(0, 50304, size=200000, dtype=np.uint16)
        raw.tofile(path)
        B, T = 3, 64

        def first_k(seed, split, k=3, with_masks=False):
            it = TokenData(path).batches(B, T, 16, "cpu", seed=seed,
                                         split=split, with_masks=with_masks)
            return [next(it) for _ in range(k)]

        a, b = first_k(0, "train"), first_k(0, "train")
        for i, ((x, _), (y, _)) in enumerate(zip(a, b)):
            assert x.shape == (B, T + 1) and x.dtype == torch.long
            assert torch.equal(x, y), f"DET FAILED: token batch {i} differs"
            # ids above 32767 must survive the trip: torch has no uint16, and
            # a naive int16 view would make them negative
            assert int(x.min()) >= 0 and int(x.max()) < 50304, \
                (int(x.min()), int(x.max()))
        assert not torch.equal(a[0][0], first_k(1, "train")[0][0]), \
            "DET vacuous: a different seed gave the same batch"
        d = TokenData(path)
        for split, (lo, hi) in (("train", (0, d.n_train)),
                                ("eval", (d.n_train, d.n))):
            for idx, _ in first_k(3, split, k=4):
                for r in range(B):
                    row = idx[r].numpy().astype(np.uint16)
                    off = next(i for i in range(lo, hi - T)
                               if np.array_equal(raw[i:i + T + 1], row))
                    assert lo <= off and off + T + 1 <= hi, (split, off)
        e = first_k(0, "eval", k=1, with_masks=True)[0][1]
        assert e.shape == (B, T) and e.dtype == torch.bool
        big = int((raw > 32767).sum())
        print(f"DET/SPL PASSED (tokens): uint16 windows reproducible and "
              f"split-confined; {big} of {raw.size} ids exceed int16")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_induction_mask_matches_loop()
    test_induction_mask_token_space()
    test_sampler_deterministic()
    test_sampler_skip_continues_stream()
    test_sampler_stride_partitions_the_stream()
    test_split_disjoint_and_contiguous()
    test_build_byte_cache_offline()
    test_build_token_cache_offline_and_resume()
    test_token_sampler()
    print("ALL DATA GATES PASSED")
