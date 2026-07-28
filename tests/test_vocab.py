"""Vocabulary and cross-entropy gates for scripts/m5_arch.py.

VOC — a freshly initialised model's loss is ln(vocab), measured through the
      SAME functions that train and evaluate. This exists because of a bug
      that survived every previous gate: the loss was computed as
      `logits.reshape(-1, 256)`, a hard-coded byte vocabulary, in two places.
      At vocab 50304 that reshape does not crash — it silently folds 196.5
      token-logits into each row and returns a finite, plausible number. No
      shape assertion catches it; only the VALUE does.
CHK — the chunked+recomputed cross-entropy equals the unchunked one, in the
      loss and in the gradient. The chunking is not optional at vocab 50304
      (824 MB of fp32 logits per sequence), so it has to be exact.
ACC — gradient accumulation over G micro-batches equals one batch G times as
      large, so "batch size" becomes a free variable and our runs can match a
      reference's 4.13M-token step instead of reporting a 65k-token one at the
      same token count and calling it the same experiment.

Runnable via pytest or `python tests/test_vocab.py`.
"""
import math, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import torch
import torch.nn.functional as F

from core import CoreConfig, ModelConfig, SWTransformer
from m5_arch import (ce_chunk_default, ce_per_token, ce_sum, evaluate,
                     lr_schedule, train_loss)
from m5_data import induction_mask


def _cfg(vocab, cores=True):
    rc = CoreConfig(K=3, d_core=48, n_heads=4, n_core_layers=1,
                    routing="top1_recurrent", n_loops=2, ffn_hidden=64,
                    inter_core_window=8, residual_scale_init=0.1,
                    router_aux_weight=0.01)
    return ModelConfig(vocab_size=vocab, d_model=48, n_layers=3, n_heads=4,
                       window=8, max_seq_len=64, core_layer=1,
                       cores=[rc] * 4 if cores else [])


def _batch(vocab, B=3, T=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T + 1), generator=g)


def test_loss_is_ln_vocab():
    """VOC, at a byte-ish vocab and at the real one, train path and eval path."""
    for vocab in (37, 50304):
        torch.manual_seed(3)
        model = SWTransformer(_cfg(vocab))
        head_w = model.head.weight
        idx = _batch(vocab)
        want = math.log(vocab)
        for chunk in (0, 8):
            ce, _, _ = train_loss(model, head_w, idx, chunk)
            got = float(ce.detach())
            assert abs(got - want) < 0.15, (vocab, chunk, got, want)

        # and the eval path, which computes the loss a second time (per
        # position, for the induction slice) and so could disagree
        mask = torch.stack([induction_mask(idx[b, :-1], 4, 4, 16)
                            for b in range(idx.shape[0])])
        m = evaluate(model, [(idx, mask)], "cpu",
                     chunk=ce_chunk_default(vocab))
        assert abs(m["eval_loss"] - want) < 0.15, (vocab, m["eval_loss"])
        assert m["eval_induction_frac"] >= 0.0
    # the default must actually engage at the vocabulary that needs it
    assert ce_chunk_default(256) == 0 and ce_chunk_default(50304) > 0
    print(f"VOC PASSED: init loss == ln(vocab) at 37 ({math.log(37):.3f}) and "
          f"50304 ({math.log(50304):.3f}), through train_loss and evaluate")


def test_chunked_ce_matches():
    """CHK: same loss and same gradient, chunked or not."""
    torch.manual_seed(4)
    V, d, N = 50304, 48, 300
    w = (torch.randn(V, d) * 0.02).requires_grad_(True)
    tgt = torch.randint(0, V, (N,))

    def run(chunk):
        h = torch.randn(N, d, generator=torch.Generator().manual_seed(5))
        h.requires_grad_(True)
        if w.grad is not None:
            w.grad = None
        loss = ce_sum(h, w, tgt, chunk) / N
        loss.backward()
        return float(loss), h.grad.clone(), w.grad.clone()

    l0, gh0, gw0 = run(0)
    for chunk in (1, 7, 64, 299, 300, 4096):
        l1, gh1, gw1 = run(chunk)
        assert abs(l1 - l0) < 1e-5, (chunk, l1, l0)
        assert torch.allclose(gh1, gh0, atol=1e-6), chunk
        assert torch.allclose(gw1, gw0, atol=1e-6), chunk
    # and the per-position eval variant
    h = torch.randn(N, d, generator=torch.Generator().manual_seed(5))
    ref = ce_per_token(h, w.detach(), tgt, 0)
    for chunk in (1, 7, 64, 4096):
        assert torch.allclose(ce_per_token(h, w.detach(), tgt, chunk), ref,
                              atol=1e-6), chunk
    assert ref.shape == (N,)
    print(f"CHK PASSED: chunked CE == unchunked in loss ({l0:.6f}) and in "
          f"both gradients, over chunk sizes 1..4096 at vocab {V}")


def test_grad_accum_matches_big_batch():
    """ACC: G micro-batches of B == one batch of B*G, in the gradient.

    Exact for the cross-entropy, which is a mean over an equal number of
    tokens in every micro-batch. NOT exact for the router balance losses,
    which are nonlinear functions of per-batch routing statistics — those get
    computed on micro-batch statistics, as they are in every MoE
    implementation, so the model here is dense to keep the claim clean.
    """
    torch.manual_seed(6)
    V, G, B, T = 37, 4, 2, 24
    cfg = _cfg(V, cores=False)
    big = _batch(V, B * G, T, seed=1)

    def grads(model, chunks):
        model.zero_grad(set_to_none=True)
        for idx in chunks:
            ce, aux, _ = train_loss(model, model.head.weight, idx, 0)
            ((ce + aux) / len(chunks)).backward()
        return {n: p.grad.clone() for n, p in model.named_parameters()
                if p.grad is not None}

    torch.manual_seed(7)
    m1 = SWTransformer(cfg)
    torch.manual_seed(7)
    m2 = SWTransformer(cfg)
    one = grads(m1, [big])
    many = grads(m2, [big[i * B:(i + 1) * B] for i in range(G)])
    assert one.keys() == many.keys() and len(one) > 10
    worst, where = 0.0, ""
    for n in one:
        d = float((one[n] - many[n]).abs().max())
        scale = max(float(one[n].abs().max()), 1e-9)
        if d / scale > worst:
            worst, where = d / scale, n
    assert worst < 1e-4, (where, worst)
    print(f"ACC PASSED: {G} micro-batches of {B} == one batch of {B * G}; "
          f"worst relative gradient difference {worst:.2e} ({where})")


def test_wsd_schedule_shapes():
    """WSD: trunk is flat, cooldown is linear to zero, and they meet."""
    lr, N = 2e-3, 1000
    # a --decay-frac 0 trunk holds the peak from the end of warmup onward
    trunk = [lr_schedule(i, lr, N, 80, "wsd", 0.0) for i in range(N + 1)]
    assert abs(trunk[0]) < 1e-12 and abs(trunk[80] - lr) < 1e-12
    assert all(abs(v - lr) < 1e-12 for v in trunk[80:]), "trunk is not flat"
    # a cooldown branch starts AT the peak (no second warmup) and hits zero
    cd = [lr_schedule(i, lr, N, 0, "cooldown") for i in range(N + 1)]
    assert abs(cd[0] - lr) < 1e-12 and abs(cd[N]) < 1e-12
    assert abs(cd[N // 2] - lr / 2) < 1e-9, cd[N // 2]
    assert all(cd[i] > cd[i + 1] for i in range(N)), "cooldown not monotone"
    # the one-run form: WSD with a 20% cooldown must trace the same path as
    # trunk-then-branch, or the ladder is not measuring the same schedule
    one = [lr_schedule(i, lr, N, 80, "wsd", 0.2) for i in range(N + 1)]
    assert abs(one[N]) < 1e-12 and abs(one[800] - lr) < 1e-9
    assert abs(one[900] - lr / 2) < 1e-3, one[900]
    # and cosine, for contrast: its midpoint is at HALF the peak, which is
    # exactly why a mid-run cosine checkpoint cannot be read as an endpoint
    cos = [lr_schedule(i, lr, N, 0, "cosine") for i in range(N + 1)]
    assert abs(cos[N // 2] - lr / 2) < 1e-9 and abs(cos[N]) < 1e-9
    print("WSD PASSED: decay_frac 0 trunk flat at the peak, cooldown linear "
          "peak->0 with no second warmup, one-shot wsd traces both")


def test_checkpoint_resume_restores_optimizer():
    """RSM: a branch resumes weights AND Adam moments AND the step count.

    Restarting Adam at zero moments puts a bias-correction transient exactly
    where the cooldown is supposed to be measuring, so the checkpoint has to
    carry optimizer state — this asserts the resumed optimizer takes the same
    step the uninterrupted one would.
    """
    import copy
    torch.manual_seed(9)
    V, B, T = 37, 2, 24
    cfg = _cfg(V, cores=False)
    m = SWTransformer(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1,
                            betas=(0.9, 0.95))
    batches = [_batch(V, B, T, seed=s) for s in range(5)]

    def step(model, o, idx):
        o.zero_grad(set_to_none=True)
        ce, aux, _ = train_loss(model, model.head.weight, idx, 0)
        (ce + aux).backward()
        o.step()

    for idx in batches[:3]:
        step(m, opt, idx)
    ck = {"model": copy.deepcopy(m.state_dict()),
          "opt": copy.deepcopy(opt.state_dict())}
    for idx in batches[3:]:                       # uninterrupted continuation
        step(m, opt, idx)
    want = {n: p.detach().clone() for n, p in m.named_parameters()}

    m2 = SWTransformer(cfg)
    m2.load_state_dict(ck["model"])
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=0.1,
                             betas=(0.9, 0.95))
    opt2.load_state_dict(ck["opt"])
    assert opt2.state_dict()["state"], "RSM FAILED: no optimizer state carried"
    for idx in batches[3:]:
        step(m2, opt2, idx)
    worst = max(float((want[n] - p.detach()).abs().max())
                for n, p in m2.named_parameters())
    assert worst < 1e-6, worst

    # and the control: without the moments the same two steps land elsewhere,
    # so the assertion above is not passing for free
    m3 = SWTransformer(cfg)
    m3.load_state_dict(ck["model"])
    opt3 = torch.optim.AdamW(m3.parameters(), lr=1e-3, weight_decay=0.1,
                             betas=(0.9, 0.95))
    for idx in batches[3:]:
        step(m3, opt3, idx)
    naive = max(float((want[n] - p.detach()).abs().max())
                for n, p in m3.named_parameters())
    assert naive > 100 * max(worst, 1e-9), (worst, naive)
    print(f"RSM PASSED: resumed run == uninterrupted run to {worst:.1e}; "
          f"dropping the Adam moments diverges by {naive:.1e} ({naive/max(worst,1e-12):.0f}x)")


def test_wrapped_model_paths():
    """WRP: every helper survives a WRAPPED model, not just a bare one.

    Under DDP + compile the chain is OptimizedModule -> DDP -> SWTransformer.
    A helper that unwraps only `_orig_mod` lands on the DDP wrapper, and
    `model.cores` raises AttributeError. This cost two failed launches on a
    rented 8x5090 because no single-GPU test has a wrapper to trip over: the
    bug is invisible until eight processes are already running.

    Stands in for DDP with a plain nn.Module wrapper exposing `.module`, which
    is the attribute the unwrap chain has to strip. No GPUs required.
    """
    import torch.nn as nn
    from m5_arch import (evaluate, flops_per_token, flops_per_token_executed,
                         unwrap)

    class FakeDDP(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.module = m

        def forward(self, *a, **k):
            return self.module(*a, **k)

    class FakeCompiled(nn.Module):
        def __init__(self, m):
            super().__init__()
            self._orig_mod = m

        def forward(self, *a, **k):
            return self._orig_mod(*a, **k)

    V, T = 37, 24
    cfg = _cfg(V)
    bare = SWTransformer(cfg)
    for name, wrapped in (("DDP", FakeDDP(bare)),
                          ("compile(DDP)", FakeCompiled(FakeDDP(bare))),
                          ("DDP(compile)", FakeDDP(FakeCompiled(bare)))):
        assert unwrap(wrapped) is bare, f"unwrap failed for {name}"
        # the helpers that reach for .cores / .num_params / .tok_emb / .head
        f = flops_per_token(wrapped, cfg, T)
        assert f == flops_per_token(bare, cfg, T), name
        ex, sem = flops_per_token_executed(wrapped, cfg, T)
        assert ex >= sem > 0, name
        idx = _batch(V, 2, T, seed=2)
        mask = torch.zeros(2, T, dtype=torch.bool)
        m = evaluate(wrapped, [(idx, mask)], "cpu", cfg, T, chunk=0)
        assert abs(m["eval_loss"] - math.log(V)) < 0.3, (name, m["eval_loss"])
    print("WRP PASSED: unwrap, flops_per_token, flops_per_token_executed and "
          "evaluate all survive DDP, compile(DDP) and DDP(compile) wrappers")


def test_accumulation_uses_no_sync():
    """SYN: the TRAINER suppresses DDP's all-reduce during accumulation.

    DDP reduces gradients at the end of every backward unless wrapped in
    no_sync(). Without it, one optimizer step costs `grad_accum` all-reduces
    of the WHOLE parameter set instead of one. At 621.5M params that is
    2.49 GB ring-reduced to 4.35 GB moved per GPU, per micro-step, over PCIe
    with GeForce P2P disabled -- measured 89,960 tok/s on 8x5090 against ~260k
    expected, with every GPU at 100% util and 285 W of a 575 W card: busy
    moving bytes, not computing.

    I had written exactly this guard in scripts/bench_batch.py, with a comment
    explaining it, and never put it in scripts/m5_arch.py. So this gate reads
    the TRAINER's source: a benchmark that gets it right proves nothing about
    the thing that trains.
    """
    import inspect
    import m5_arch
    src = inspect.getsource(m5_arch.main)
    loop = src[src.index("for a in range(args.grad_accum)"):]
    body = loop[:loop.index("clip_grad_norm_")]
    assert "no_sync()" in body, \
        "SYN FAILED: the accumulation loop does not use model.no_sync()"
    assert "not last" in body or "a < args.grad_accum - 1" in body, \
        "SYN FAILED: no_sync must be skipped on the LAST micro-step, or the " \
        "gradients are never all-reduced at all"
    assert "world > 1" in body, \
        "SYN FAILED: no_sync must be conditional on world > 1; a bare module " \
        "has no no_sync()"
    # and the benchmark, which is where the pattern already lived
    import bench_batch
    bsrc = inspect.getsource(bench_batch.measure)
    assert "no_sync()" in bsrc, "SYN FAILED: bench_batch lost its no_sync"
    print("SYN PASSED: both the trainer and the benchmark suppress DDP "
          "all-reduce on every micro-step but the last")


class _FakeParam:
    def __init__(self, n, esize=4, requires_grad=True):
        self._n, self._e = n, esize
        self.requires_grad = requires_grad

    def numel(self):
        return self._n

    def element_size(self):
        return self._e


class _FakeModule:
    def __init__(self, *params):
        self._p = params

    def parameters(self):
        return iter(self._p)


def test_ddp_buckets_do_not_shred_the_graph():
    """BKT: DDP's gradient buckets are sized for FUSION, not just overlap.

    Wrapping DDP inside torch.compile turns on dynamo's DDPOptimizer, which
    splits the compiled graph once per gradient bucket so each bucket's
    all-reduce can overlap the next bucket's backward. Every split is also a
    hard fusion barrier, and `bucket_cap_mb` defaults to 25 -- ~99 splits at
    cores_620m's 621.5M params. Measured cost on 8x5090: 18,992 tok/s per
    GPU against 50,921 compiled on a single card and 18,938 eager, with the
    GPUs at 380-408 W of 575 and 84-98% util. Compute-bound and running at
    eager speed, which is what a shredded graph looks like -- the comms-bound
    failure this is often confused with looked completely different on the
    same box (285 W, see SYN).

    Same family of bug as SYN and WRP: correct on one GPU, wrong the moment
    a wrapper appears, and invisible to any single-process test.
    """
    import inspect
    import m5_arch

    # 621.5M fp32 params, the real cores_620m gradient footprint
    big = _FakeModule(_FakeParam(621_500_000))
    mb = m5_arch.ddp_bucket_mb(big)
    grad_mb = 621_500_000 * 4 / (1024 * 1024)
    buckets = grad_mb / mb
    assert abs(buckets - m5_arch.DDP_TARGET_BUCKETS) <= 1, (
        f"BKT FAILED: {buckets:.0f} buckets, wanted "
        f"~{m5_arch.DDP_TARGET_BUCKETS}")
    assert grad_mb / 25 > 50, "BKT test is vacuous: the 25 MB default no " \
        "longer shreds this model, so the gate proves nothing"

    # frozen params must not inflate the buckets -- they never all-reduce
    mixed = _FakeModule(_FakeParam(621_500_000),
                        _FakeParam(10_000_000_000, requires_grad=False))
    assert m5_arch.ddp_bucket_mb(mixed) == mb, \
        "BKT FAILED: ddp_bucket_mb counts params that carry no gradient"

    # and it may only ever raise the cap, never lower it below DDP's own
    assert m5_arch.ddp_bucket_mb(_FakeModule(_FakeParam(1000))) == 25, \
        "BKT FAILED: tiny models must fall back to DDP's 25 MB, not lower"

    # the TRAINER has to actually pass it -- a helper nothing calls is a
    # no-op, which is exactly how the no_sync bug survived (see SYN)
    src = inspect.getsource(m5_arch.main)
    ddp = src[src.index("DistributedDataParallel"):]
    ddp = ddp[:ddp.index("if args.compile")]
    assert "bucket_cap_mb=" in ddp, \
        "BKT FAILED: the trainer builds DDP without bucket_cap_mb, so the " \
        "25 MB default shreds the compiled graph again"
    print(f"BKT PASSED: DDP buckets sized to {mb} MB -> ~{buckets:.0f} graph "
          f"splits, not the ~{grad_mb / 25:.0f} the default gives")


if __name__ == "__main__":
    test_loss_is_ln_vocab()
    test_wrapped_model_paths()
    test_accumulation_uses_no_sync()
    test_ddp_buckets_do_not_shred_the_graph()
    test_chunked_ce_matches()
    test_grad_accum_matches_big_batch()
    test_wsd_schedule_shapes()
    test_checkpoint_resume_restores_optimizer()
    print("ALL VOCAB GATES PASSED")
