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


if __name__ == "__main__":
    test_loss_is_ln_vocab()
    test_chunked_ce_matches()
    test_grad_accum_matches_big_batch()
    test_wsd_schedule_shapes()
    test_checkpoint_resume_restores_optimizer()
    print("ALL VOCAB GATES PASSED")
