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
                     train_loss)
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


if __name__ == "__main__":
    test_loss_is_ln_vocab()
    test_chunked_ce_matches()
    test_grad_accum_matches_big_batch()
    print("ALL VOCAB GATES PASSED")
