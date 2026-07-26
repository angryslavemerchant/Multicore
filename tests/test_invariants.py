"""M0/M1/M2 gates. Runnable via pytest or `python tests/test_invariants.py`.

M0 — zero-init cores leave logits bit-identical to the core-free model.
M1 — gather->window->scatter agrees with the O(T^2) reference mask spec.
M2 — full prefill logits match token-by-token incremental decode (the cache
     test: if this fails, an invariant in CORE_ROUTING_PLAN.md section 6 is
     broken).
MB — MultiCore (M cores batched into one pass) == M independent Cores summed,
     and its own prefill/decode cache test.
SW — banded base sliding-window attention == the dense (T,T) mask it replaced,
     in forward AND in gradients.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core import CoreConfig, ModelConfig, SWTransformer, Core, MultiCore
from core.base_model import SWAttention, _rope_cos_sin, _rope_apply
from core.resident import resident_mask_reference, compact_indices, window_mask

torch.manual_seed(0)


def small_cfg(n_cores=2, adapter=False):
    return ModelConfig(
        vocab_size=96, d_model=64, n_layers=3, n_heads=4, window=16,
        max_seq_len=256, core_layer=1, adapter=adapter,
        cores=[CoreConfig(K=4, d_core=32, n_heads=2, target_rate=0.3),
               CoreConfig(K=8, d_core=32, n_heads=2, target_rate=0.1)][:n_cores])


def randomize_core_outputs(model, scale=0.1):
    """Make deltas nonzero so M1/M2 actually test the core path."""
    for c in model.cores:
        if isinstance(c, MultiCore):
            randomize_multicore(c, scale)
            continue
        torch.nn.init.normal_(c.out_proj.weight, std=scale)
        torch.nn.init.normal_(c.out_proj.bias, std=scale)
        # push gate scores around so a healthy fraction of tokens are admitted
        torch.nn.init.normal_(c.k_dir, std=1.0)


def randomize_multicore(mc, scale=0.1):
    torch.nn.init.normal_(mc.out_w, std=scale)
    torch.nn.init.normal_(mc.out_b, std=scale)
    torch.nn.init.normal_(mc.k_dir, std=1.0)


def test_m0_bit_identical():
    model = SWTransformer(small_cfg()).eval()
    idx = torch.randint(0, 96, (4, 128))
    with torch.no_grad():
        logits_with = model(idx)
        model.cores_enabled = False
        logits_without = model(idx)
    assert torch.equal(logits_with, logits_without), "M0 FAILED: zero-init cores changed logits"
    print("M0 PASSED: zero-init cores are bit-identical to base")


def test_m1_mask_equivalence():
    torch.manual_seed(1)
    for K in (1, 2, 4, 16):
        for rate in (0.05, 0.3, 0.9):
            core = Core(32, CoreConfig(K=K, d_core=16, n_heads=2)).eval()
            torch.nn.init.normal_(core.out_proj.weight, std=0.1)
            torch.nn.init.normal_(core.k_dir, std=2.0)
            core.tau.data.fill_(torch.quantile(
                torch.randn(1000) * 2.0, 1 - rate))
            h = torch.randn(3, 48, 32)
            with torch.no_grad():
                fast, _ = core(h)
                ref, _ = core.forward_reference(h)
            assert torch.allclose(fast, ref, atol=1e-5), \
                f"M1 FAILED at K={K} rate~{rate}: max err {(fast-ref).abs().max()}"
    # also test the pure mask logic on random booleans
    for K in (1, 3, 7):
        m = torch.rand(64) < 0.4
        ref = resident_mask_reference(m, K)
        idx, valid = compact_indices(m[None])
        wm = window_mask(idx.shape[1], K, valid)[0]
        # map compacted mask back to full coordinates and compare
        full = torch.zeros(64, 64, dtype=torch.bool)
        pos = idx[0]
        for a in range(idx.shape[1]):
            if not valid[0, a]:
                continue
            for b in range(idx.shape[1]):
                if valid[0, b] and wm[a, b]:
                    full[pos[a], pos[b]] = True
        assert torch.equal(full, ref), f"M1 mask logic FAILED at K={K}"
    print("M1 PASSED: gather-window-scatter == reference spec")


def test_m2_cache_correctness():
    torch.manual_seed(2)
    for adapter in (False, True):
        model = SWTransformer(small_cfg(adapter=adapter)).eval()
        randomize_core_outputs(model)
        B, T = 3, 200  # > max ring turnover: rate*T >> K for both cores
        idx = torch.randint(0, 96, (B, T))
        with torch.no_grad():
            full_logits = model(idx)
            caches = model.init_caches(B, idx.device)
            steps = [model.forward_step(idx[:, t], caches) for t in range(T)]
        inc_logits = torch.stack(steps, dim=1)
        err = (full_logits - inc_logits).abs().max().item()
        assert err < 1e-4, f"M2 FAILED (adapter={adapter}): max logit err {err}"
        # confirm cores actually turned over (test would be vacuous otherwise)
        if not adapter:
            counts = [int(r["count"].max()) for r in caches["rings"]]
            assert any(c > k for c, k in zip(counts, (4, 8))), \
                f"M2 vacuous: rings never filled (counts={counts})"
    print("M2 PASSED: prefill == incremental decode, cores turning over")


def test_mb_multicore_equivalence():
    """MultiCore(M=3) == three independent Cores whose deltas are summed."""
    torch.manual_seed(3)
    M, d = 3, 48
    cc = CoreConfig(K=8, d_core=32, n_heads=2, n_core_layers=2,
                    target_rate=0.2)
    mc = MultiCore(d, cc, M).eval()
    randomize_multicore(mc, scale=0.1)
    for lay in mc.layers:                      # rank bias starts at zero
        torch.nn.init.normal_(lay.rel_bias, std=0.5)
    torch.nn.init.normal_(mc.ln_w, mean=1.0, std=0.1)
    torch.nn.init.normal_(mc.ln_b, std=0.1)

    h = torch.randn(4, 64, d)
    # eval mode -> tau is static; set it to the per-core (1-rate) quantile so
    # each core admits a realistic (and different) slice of tokens. Take the
    # midpoint between the two neighbouring order statistics: a tau that sits
    # exactly ON a score would make membership hinge on the last bit of the
    # score (batched einsum vs h @ k_dir differ there), which is a float tie,
    # not a semantic difference.
    s = torch.einsum('btd,md->btm', h, mc.k_dir)
    n = s[..., 0].numel()
    k = int(n * (1 - cc.target_rate))
    for i in range(M):
        srt = s[..., i].flatten().sort().values
        mc.tau[i] = 0.5 * (srt[k - 1] + srt[k])
    assert torch.isfinite(mc.tau).all()

    cores = []
    for i in range(M):
        c = Core(d, cc).eval()
        c.load_state_dict(mc.core_state_dict(i))
        cores.append(c)

    with torch.no_grad():
        delta, auxes = mc(h)
        ref = torch.zeros_like(h)
        ref_aux = []
        for c in cores:
            dl, a = c(h)                       # all cores see the SAME h
            ref = ref + dl
            ref_aux.append(a)
    assert delta.abs().max() > 1e-3, "MB vacuous: MultiCore delta is ~zero"
    err = (delta - ref).abs().max().item()
    assert torch.allclose(delta, ref, atol=1e-4), \
        f"MB FAILED: batched vs per-core delta max err {err}"
    for i in range(M):
        assert torch.allclose(auxes[i]["rate"], ref_aux[i]["rate"]), \
            f"MB FAILED: core {i} aux rate mismatch"
        assert torch.equal(auxes[i]["m"], ref_aux[i]["m"])
    rates = [float(a["rate"]) for a in auxes]
    print(f"MB PASSED: MultiCore == {M} summed Cores "
          f"(max err {err:.2e}, rates {[round(r, 3) for r in rates]})")


def test_mb_multicore_cache():
    """M2 for the batched path: identical core configs -> MultiCore."""
    torch.manual_seed(4)
    cc = CoreConfig(K=4, d_core=32, n_heads=2, n_core_layers=2,
                    target_rate=0.3)
    cfg = ModelConfig(vocab_size=96, d_model=64, n_layers=3, n_heads=4,
                      window=16, max_seq_len=256, core_layer=1,
                      cores=[cc, CoreConfig(K=4, d_core=32, n_heads=2,
                                            target_rate=0.3)])
    model = SWTransformer(cfg).eval()
    assert model.batched_cores and isinstance(model.cores[0], MultiCore), \
        "MB FAILED: identical core configs did not take the MultiCore path"
    randomize_core_outputs(model)
    B, T = 3, 200
    idx = torch.randint(0, 96, (B, T))
    with torch.no_grad():
        full_logits, auxes = model(idx, collect_aux=True)
        caches = model.init_caches(B, idx.device)
        steps = [model.forward_step(idx[:, t], caches) for t in range(T)]
    inc_logits = torch.stack(steps, dim=1)
    err = (full_logits - inc_logits).abs().max().item()
    assert err < 1e-4, f"MB cache FAILED: max logit err {err}"
    assert len(auxes) == 2, f"MB FAILED: expected 2 auxes, got {len(auxes)}"
    counts = caches["rings"][0]["count"]        # (M, B)
    assert int(counts.max()) > cc.K, \
        f"MB cache vacuous: rings never filled (counts={counts.tolist()})"
    print(f"MB PASSED: MultiCore prefill == incremental decode "
          f"(err {err:.2e}, ring counts up to {int(counts.max())} > K={cc.K})")


def _sw_forward_dense(attn, x):
    """Reference SWAttention.forward: build the dense (T,T) bool mask and let
    scaled_dot_product_attention do it. This is verbatim the O(T^2) code the
    banded path replaced — the ground truth for test_sw_attention_equivalence.
    """
    B, T, C = x.shape
    q, k, v = attn.qkv(x).chunk(3, dim=-1)

    def split(t):
        return t.view(B, T, attn.n_heads, attn.d_head).transpose(1, 2)

    q, k, v = split(q), split(k), split(v)
    if attn.rope:
        cos, sin = _rope_cos_sin(attn.d_head, torch.arange(T, device=x.device))
        q, k = _rope_apply(q, cos, sin), _rope_apply(k, cos, sin)
    i = torch.arange(T, device=x.device)
    mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < attn.window)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return attn.proj(out.transpose(1, 2).reshape(B, T, C))


def test_sw_attention_equivalence():
    """SW — banded sliding-window attention == the dense-mask reference, both
    forward and (via autograd) in d/dx and d/d{qkv,proj} weights."""
    torch.manual_seed(5)
    # (B, T, window, d_model, n_heads, rope)
    combos = [
        (2, 64, 16, 32, 4, True),     # T a whole number of windows
        (2, 70, 16, 32, 4, True),     # T NOT a multiple of window
        (3, 37, 8, 32, 2, True),      # ragged last block, B>1, H>1
        (2, 40, 100, 32, 4, True),    # window > T  -> causal fast path
        (2, 40, 40, 32, 4, True),     # window == T -> causal fast path
        (2, 33, 1, 32, 4, True),      # window == 1 -> self-attention only
        (1, 129, 64, 64, 8, True),    # block == window, ragged tail
        (2, 100, 16, 32, 4, False),   # learned-position variant (no rope)
        (4, 200, 96, 48, 6, True),    # B>1, block floor, many blocks
        (2, 257, 3, 32, 4, True),     # window << SW_MIN_BLOCK
    ]
    # float64 pins the maths (errors ~1e-15 if the attended sets are equal and
    # ~1e-1 if they are not); float32 is the dtype that actually ships.
    tols = {torch.float64: 1e-10, torch.float32: 1e-5}
    worst_f = worst_g = 0.0
    for B, T, window, d, H, rope in combos:
        cfg = ModelConfig(vocab_size=32, d_model=d, n_heads=H, window=window,
                          max_seq_len=max(T, 8), rope=rope, cores=[])
        attn0 = SWAttention(cfg)
        for p in (attn0.qkv.weight, attn0.proj.weight):
            torch.nn.init.normal_(p, std=0.1)
        for p in (attn0.qkv.bias, attn0.proj.bias):
            torch.nn.init.normal_(p, std=0.1)
        x0 = torch.randn(B, T, d)
        # cotangent normalised by sqrt(B*T) so weight grads stay O(1) as the
        # combos grow — atol on a quantity whose scale rides with B*T would
        # otherwise just be a rescaled rtol.
        cot = torch.randn(B, T, d) / (B * T) ** 0.5

        for dtype, atol in tols.items():
            tag = (f"B={B} T={T} W={window} d={d} H={H} rope={rope} "
                   f"{str(dtype).split('.')[-1]}")
            attn = SWAttention(cfg).to(dtype)
            attn.load_state_dict({k: v.to(dtype)
                                  for k, v in attn0.state_dict().items()})
            got = {}
            for name in ("banded", "dense"):
                x = x0.to(dtype).clone().requires_grad_(True)
                attn.zero_grad(set_to_none=True)
                out = (attn(x) if name == "banded"
                       else _sw_forward_dense(attn, x))
                (out * cot.to(dtype)).sum().backward()
                got[name] = (out.detach(), x.grad.clone(),
                             attn.qkv.weight.grad.clone(),
                             attn.qkv.bias.grad.clone(),
                             attn.proj.weight.grad.clone(),
                             attn.proj.bias.grad.clone())

            names = ("out", "d/dx", "d/dqkv.w", "d/dqkv.b",
                     "d/dproj.w", "d/dproj.b")
            for i, nm in enumerate(names):
                a, b = got["banded"][i], got["dense"][i]
                assert torch.isfinite(a).all(), \
                    f"SW FAILED [{tag}]: {nm} not finite"
                err = (a - b).abs().max().item()
                if dtype is torch.float32:
                    worst_f, worst_g = ((max(worst_f, err), worst_g) if i == 0
                                        else (worst_f, max(worst_g, err)))
                assert err < atol, f"SW FAILED [{tag}]: {nm} max err {err:.2e}"
            # a vacuous test would pass on all-zero tensors
            assert got["dense"][0].abs().max() > 1e-3, f"SW vacuous [{tag}]"

    # the band must really be a band: a token `window` back must not leak in
    cfg = ModelConfig(vocab_size=32, d_model=32, n_heads=4, window=8,
                      max_seq_len=64, cores=[])
    attn = SWAttention(cfg)
    x = torch.randn(1, 40, 32, requires_grad=True)
    attn(x)[0, 30].sum().backward()
    reach = x.grad[0].abs().sum(-1) > 0
    assert reach[23:31].all() and not reach[:23].any() and not reach[31:].any(), \
        f"SW FAILED: receptive field of token 30 is {reach.nonzero().flatten().tolist()}"
    print(f"SW PASSED: banded == dense (T,T) mask, fwd err {worst_f:.2e}, "
          f"grad err {worst_g:.2e}; receptive field exactly [i-W+1, i]")


if __name__ == "__main__":
    test_m0_bit_identical()
    test_m1_mask_equivalence()
    test_m2_cache_correctness()
    test_mb_multicore_equivalence()
    test_mb_multicore_cache()
    test_sw_attention_equivalence()
    print("ALL GATES PASSED")
