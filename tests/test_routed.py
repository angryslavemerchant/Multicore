"""Invariants for the top-1 recurrent eight-core middle stack."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core import CoreConfig, ModelConfig, SWTransformer


def routed_cfg(M=4):
    rc = CoreConfig(K=3, d_core=48, n_heads=4, n_core_layers=1,
                    routing="top1_recurrent", n_loops=2, ffn_hidden=64,
                    inter_core_window=8, residual_scale_init=0.1,
                    router_aux_weight=0.01)
    return ModelConfig(vocab_size=37, d_model=48, n_layers=3, n_heads=4,
                       window=8, max_seq_len=64, core_layer=1,
                       cores=[rc] * M)


def test_top1_exact_and_gradients():
    torch.manual_seed(10)
    model = SWTransformer(routed_cfg())
    idx = torch.randint(0, 37, (4, 24))
    logits, aux = model(idx, collect_aux=True)
    masks = torch.stack([a["m"] for a in aux], dim=1)  # (L,M,B,T)
    assert torch.equal(masks.sum(1), torch.ones_like(masks[:, 0], dtype=torch.long))
    assert abs(sum(float(a["rate"]) for a in aux) - 1.0) < 1e-6
    ce = F.cross_entropy(logits.reshape(-1, 37), idx.reshape(-1))
    loss = ce + aux[0]["router_aux_loss"]
    loss.backward()
    routed = model.cores[0]
    assert routed.router_w.grad is not None and routed.router_w.grad.norm() > 0
    assert routed.expert.q_w.grad is not None and routed.expert.q_w.grad.norm() > 0
    with torch.no_grad():
        routed.router_w[1].copy_(routed.router_w[0])
    model.reproject_gates()
    rw = routed.router_w.detach()
    rn = rw / rw.norm(dim=1, keepdim=True)
    off = ~torch.eye(rw.shape[0], dtype=torch.bool)
    assert float((rn @ rn.t()).abs()[off].max()) < 1e-5
    assert routed.expert.o_w.grad is not None and routed.expert.o_w.grad.norm() > 0
    assert routed.expert.f1_w.grad is not None and routed.expert.f1_w.grad.norm() > 0
    print("RT PASSED: exactly one expert/token/loop; router and expert gradients live")


def test_top1_prefill_decode():
    torch.manual_seed(11)
    model = SWTransformer(routed_cfg()).eval()
    idx = torch.randint(0, 37, (3, 31))
    with torch.no_grad():
        full = model(idx)
        cache = model.init_caches(3, "cpu")
        step = torch.stack([model.forward_step(idx[:, t], cache)
                            for t in range(idx.shape[1])], 1)
    err = float((full - step).abs().max())
    assert err < 3e-6, err
    ring = cache["rings"][0]
    assert ring["count"].shape == (2, 4, 3)
    assert torch.equal(ring["count"].sum((1, 2)),
                       torch.full((2,), 3 * 31, dtype=torch.long))
    assert len({id(c) for c in ring["mixer"]}) == 2
    assert all(c["k"].shape[2] == 8 for c in ring["mixer"])
    print(f"RT-CACHE PASSED: prefill == decode ({err:.2e}); loop caches independent")


def test_top1_compute_match():
    from scripts.m5_arch import presets, flops_per_token
    T = 2048
    routed_cfg_ = presets(T)["smoke_cores_top1_loopmix"]
    dense_cfg = presets(T)["smoke_dense_local"]
    routed = SWTransformer(routed_cfg_)
    dense = SWTransformer(dense_cfg)
    rf = flops_per_token(routed, routed_cfg_, T)
    df = flops_per_token(dense, dense_cfg, T)
    rel = abs(rf - df) / df
    assert rel < 0.001, (rf, df, rel)
    assert routed.num_params() > dense.num_params()
    print(f"RT-FLOPS PASSED: routed {rf:,} vs dense {df:,} ({rel:.3%})")


def _prefill_decode_err(cfg, seed):
    torch.manual_seed(seed)
    model = SWTransformer(cfg).eval()
    idx = torch.randint(0, 37, (3, 31))
    with torch.no_grad():
        full = model(idx)
        cache = model.init_caches(3, "cpu")
        step = torch.stack([model.forward_step(idx[:, t], cache)
                            for t in range(idx.shape[1])], 1)
    return float((full - step).abs().max()), model


def test_ablation_nomix():
    """inter_core_window=0 removes the mixer's params, FLOPs and cache."""
    from dataclasses import replace
    base = routed_cfg()
    cfg = replace(base, cores=[replace(base.cores[0], inter_core_window=0)] * 4)
    err, model = _prefill_decode_err(cfg, 12)
    assert err < 3e-6, err
    routed = model.cores[0]
    assert not routed.use_mixer
    # the module must be GONE, not merely unused — otherwise its parameters
    # still get gradient and the "removed" FLOPs are still spent.
    assert not hasattr(routed, "mixer") and not hasattr(routed, "mix_scale")
    assert routed.estimated_flops_parts(2048)[1] == 0
    full = SWTransformer(routed_cfg())
    assert model.num_params() < full.num_params()
    print(f"RT-NOMIX PASSED: prefill == decode ({err:.2e}); mixer absent, "
          f"{full.num_params() - model.num_params():,} params dropped")


def test_ablation_unfold():
    """tie_loops=False gives each depth its own expert weights, same FLOPs."""
    from dataclasses import replace
    base = routed_cfg()
    cfg = replace(base, cores=[replace(base.cores[0], tie_loops=False)] * 4)
    err, model = _prefill_decode_err(cfg, 13)
    assert err < 3e-6, err          # catches a _slice mismatch packed vs step
    routed, tied = model.cores[0], SWTransformer(routed_cfg()).cores[0]
    L, M = base.cores[0].n_loops, 4
    assert routed.expert.n_sets == L and routed.expert.q_w.shape[0] == L * M
    assert routed.estimated_flops_parts(2048) == tied.estimated_flops_parts(2048)
    assert model.num_params() > SWTransformer(routed_cfg()).num_params()

    # every depth's weights must receive their OWN gradient: if _slice were
    # constant the later blocks would be dead parameters.
    idx = torch.randint(0, 37, (4, 24))
    logits, aux = model(idx, collect_aux=True)
    (F.cross_entropy(logits.reshape(-1, 37), idx.reshape(-1))
     + aux[0]["router_aux_loss"]).backward()
    g = routed.expert.q_w.grad.reshape(L, M, -1).norm(dim=(1, 2))
    assert (g > 0).all(), g
    # ...and they must be genuinely distinct blocks, not aliases of one set
    assert not torch.allclose(routed.expert.q_w[:M], routed.expert.q_w[M:2 * M])
    print(f"RT-UNFOLD PASSED: prefill == decode ({err:.2e}); {L} independent "
          f"weight sets, per-depth grad norms {[round(float(x), 4) for x in g]}")


def test_heterogeneous_fifo():
    """K_list gives each expert its own FIFO length at unchanged traffic."""
    from dataclasses import replace
    from core.base_model import _RoutedExpertBlock
    base = routed_cfg()
    Ks = (1, 2, 3, 6)
    cfg = replace(base, cores=[replace(base.cores[0], K=6, K_list=Ks)] * 4)
    err, model = _prefill_decode_err(cfg, 14)
    assert err < 3e-6, err        # ring padded to K_max must agree step-wise
    ex = model.cores[0].expert
    assert ex.K == 6 and ex.Ks == list(Ks)
    for m, km in enumerate(Ks):   # gate open inside K_m, shut past it
        assert float(ex.k_gate[m, 0, :km].abs().max()) == 0.0
        assert km == ex.K or float(ex.k_gate[m, 0, km:].max()) < -1e3

    # Functional: expert 0 has K=1, so its output at the newest slot may not
    # depend on ANY earlier resident. Expert 3 (K=6) must depend on them.
    torch.manual_seed(15)
    blk = _RoutedExpertBlock(48, replace(base.cores[0], K=6, K_list=Ks), 4)
    x = torch.randn(4, 6, 48)
    valid = torch.ones(4, 6, dtype=torch.bool)
    row = torch.zeros(4, 6, dtype=torch.long)
    with torch.no_grad():
        a = blk.forward_packed(x, valid, row, 0)
        # Perturb the OLDEST resident. Must NOT be a constant offset: the
        # block layer-norms each token, which would erase it and make both
        # assertions below pass vacuously.
        x2 = x.clone(); x2[:, 0] = torch.randn_like(x2[:, 0]) * 3.0
        b = blk.forward_packed(x2, valid, row, 0)
    d_short = float((a[0, -1] - b[0, -1]).abs().max())
    d_long = float((a[3, -1] - b[3, -1]).abs().max())
    # Absolute size is small (residual scale is 0.1 at init and one slot of
    # six gets diluted); the separation is what matters.
    assert d_short < 1e-6, d_short   # K=1 expert: newest slot sees only itself
    assert d_long > 1e-4, d_long     # K=6 expert: reaches back to slot 0
    assert d_long > 100 * max(d_short, 1e-12), (d_short, d_long)
    print(f"RT-MULTIK PASSED: prefill == decode ({err:.2e}); K=1 expert "
          f"unmoved by a distant token ({d_short:.1e}), K=6 expert moves "
          f"{d_long:.1e} ({d_long / max(d_short, 1e-12):.0f}x)")


def test_multik_compute():
    from scripts.m5_arch import presets, flops_per_token
    T = 2048
    P = presets(T)
    out = {}
    for n in ("smoke_cores_top1_loopmix", "smoke_cores_top1_multik"):
        cfg = P[n]
        out[n] = flops_per_token(SWTransformer(cfg), cfg, T)
    rel = out["smoke_cores_top1_multik"] / out["smoke_cores_top1_loopmix"] - 1
    assert 0 < rel < 0.01, (out, rel)   # K_max band costs a little, not a lot
    print(f"RT-MULTIK-FLOPS PASSED: {out['smoke_cores_top1_multik']:,} vs "
          f"{out['smoke_cores_top1_loopmix']:,} baseline ({rel:+.3%})")


def test_bounded_learned_rates():
    """Range penalty is silent inside the band, bites outside; prior anneals."""
    from dataclasses import replace
    base = routed_cfg()
    cfg = replace(base, cores=[replace(base.cores[0], router_bias=True,
                                       hash_anneal_iters=10, rate_lo=0.10,
                                       rate_hi=0.40,
                                       router_hash_scale=0.5)] * 4)
    torch.manual_seed(16)
    model = SWTransformer(cfg)
    core = model.cores[0]
    assert core.r_bias.shape == (4,)

    # penalty is EXACTLY zero on a balanced share vector, and positive once an
    # expert goes dominant or dead
    load = torch.full((4,), 0.25)
    inside = core._balance_term(load, torch.tensor([0.25, 0.25, 0.25, 0.25]))
    dom = core._balance_term(load, torch.tensor([0.85, 0.05, 0.05, 0.05]))
    assert float(inside) == 0.0, float(inside)
    assert float(dom) > 0.1, float(dom)

    # the positional prior decays to exactly zero and stays there
    scales = []
    for s in (0, 5, 10, 25):
        core._step.fill_(s)
        scales.append(float(torch.as_tensor(core._hash_scale())))
    assert scales[0] > scales[1] > scales[2] == 0.0 and scales[3] == 0.0, scales

    # the bias must actually receive gradient — it is the only term that can
    # move traffic, since router rows are renormalised to unit length
    core._step.zero_()
    model.train()
    idx = torch.randint(0, 37, (4, 24))
    logits, aux = model(idx, collect_aux=True)
    (F.cross_entropy(logits.reshape(-1, 37), idx.reshape(-1))
     + aux[0]["router_aux_loss"]).backward()
    assert core.r_bias.grad is not None and float(core.r_bias.grad.abs().max()) > 0
    assert int(core._step) == 1, int(core._step)   # counter advanced once
    model.eval()
    before = int(core._step)
    model(idx)
    assert int(core._step) == before               # frozen outside training
    print(f"RT-FREERATE PASSED: penalty 0.0 inside band / {float(dom):.3f} on a "
          f"dominant expert; prior anneals {scales[0]:.3f}->0; bias grad live")


def test_param_matched_control():
    """The dense control matches unfold on PARAMS and costs ~2x the FLOPs."""
    from scripts.m5_arch import presets, flops_per_token
    T = 2048
    P = presets(T)
    got = {}
    for n in ("smoke_cores_top1_unfold", "smoke_dense_local_pm",
              "smoke_cores_top1_unfold_freerate"):
        m = SWTransformer(P[n])
        got[n] = (m.num_params(), flops_per_token(m, P[n], T))
    unf, pm, comb = (got[k] for k in
                     ("smoke_cores_top1_unfold", "smoke_dense_local_pm",
                      "smoke_cores_top1_unfold_freerate"))
    assert abs(pm[0] - unf[0]) / unf[0] < 0.03, (pm[0], unf[0])
    assert pm[1] / unf[1] > 1.9, pm[1] / unf[1]
    # combining the two winners must not change the compute budget: untying
    # touches expert weights, the router bias adds M scalars
    assert comb[1] == unf[1] and 0 < comb[0] - unf[0] <= 64, comb
    print(f"RT-PARAM-MATCH PASSED: dense_pm {pm[0]:,}p vs unfold {unf[0]:,}p "
          f"({(pm[0] - unf[0]) / unf[0]:+.2%}) at {pm[1] / unf[1]:.2f}x FLOPs; "
          f"unfold+freerate FLOPs unchanged")


def test_modern_recipe():
    """RMSNorm + SwiGLU + qk-norm + tied embeddings, on BOTH arms.

    The gate that matters is prefill == decode: the recipe touches four places
    where the packed training path and the single-token decode path are
    written separately (`_ln` / `_ln_one`, `_ffn` / the step_one FFN,
    SWAttention.forward / .step), and a norm applied in one and not the other
    is invisible in a loss curve — it only shows up as a model that trains
    fine and generates garbage.
    """
    from dataclasses import replace
    base = routed_cfg()
    cfg = replace(base, rmsnorm=True, swiglu=True, qk_norm=True,
                  tie_embeddings=True)
    err, model = _prefill_decode_err(cfg, 15)
    assert err < 3e-6, err
    exp = model.cores[0].expert
    assert exp.rmsnorm and exp.swiglu
    assert not hasattr(exp, "ln1_b"), "RMSNorm must not carry a bias"
    assert hasattr(exp, "f3_w"), "SwiGLU needs the gate projection"
    assert model.blocks[0].attn.q_norm.weight.shape == (48 // 4,)
    assert model.head.weight is model.tok_emb.weight, "embeddings not tied"

    # tying is a real saving, and parameters() must not double-count the
    # shared tensor
    plain = SWTransformer(base)
    tied_only = SWTransformer(replace(base, tie_embeddings=True))
    saved = plain.num_params() - tied_only.num_params()
    assert saved == 37 * 48, saved

    # flipping swiglu alone must not move the parameter budget much: the 2/3
    # width rule is what keeps it a nonlinearity change rather than a size one
    from core.base_model import ffn_hidden
    assert ffn_hidden(48, 4, swiglu=False) == 192
    assert ffn_hidden(48, 4, swiglu=True) == 128         # 2/3 of 192
    assert ffn_hidden(48, 4, swiglu=True, explicit=64) == 64
    # at a real width: 4*384 -> 1536, and 2/3 of that is exactly 1024, so the
    # multiple-of-8 rounding costs nothing. (At toy widths the rounding
    # dominates — 2/3 of 256 is 170.7, rounded to 176, a 3% jump — which is
    # why this is checked at a width the presets actually use.)
    dense = ModelConfig(vocab_size=37, d_model=384, n_layers=2, n_heads=6,
                        window=8, max_seq_len=32, core_layer=0, cores=[])
    a = SWTransformer(dense).num_params()
    b = SWTransformer(replace(dense, swiglu=True)).num_params()
    assert abs(b - a) / a < 0.005, (a, b)
    print(f"RT-RECIPE PASSED: prefill == decode ({err:.2e}); rmsnorm biasless, "
          f"swiglu gated, qk-norm per-head, tying saved {saved:,} params "
          f"({(b - a) / a:+.2%} FFN drift)")


def test_flops_accounting():
    """keys_per_token, the embedding lookup, and the LM head."""
    from core.base_model import keys_per_token
    from scripts.m5_arch import presets, flops_per_token
    T = 4096
    # causal full attention averages (T+1)/2 keys, not T
    assert keys_per_token(T, T) == (T + 1) / 2
    assert keys_per_token(10 ** 9, T) == (T + 1) / 2
    # a sliding window is the window itself, less the causal ramp over the
    # first `window` positions (248.03 at window 256, T 4096)
    assert 0.96 * 256 < keys_per_token(256, T) < 256
    assert keys_per_token(1, T) == 1.0

    P = presets(T)
    cfg = P["ref_dense_130m"]
    model = SWTransformer(cfg)
    f = flops_per_token(model, cfg, T)
    body = model.num_params() - model.tok_emb.weight.numel()
    head = 2 * cfg.d_model * cfg.vocab_size
    attn = cfg.n_layers * 4 * cfg.d_model * keys_per_token(cfg.window, T)
    assert abs(f - (2 * body + head + attn)) < 1.0, f

    # Tying moves V*d parameters and ZERO FLOPs — the head matmul runs either
    # way, and the lookup is free either way. If those two numbers do not come
    # out equal, the table is being billed as arithmetic somewhere.
    from dataclasses import replace
    ucfg = replace(cfg, tie_embeddings=False)
    untied = SWTransformer(ucfg)
    fu = flops_per_token(untied, ucfg, T)
    assert abs(fu - f) < 1.0, (f, fu)
    assert untied.num_params() - model.num_params() == cfg.vocab_size * cfg.d_model
    # and the naive "2 * every parameter" over-counts by exactly one head
    assert abs((2 * untied.num_params() + attn) - (fu + head)) < 1.0
    assert model.num_params() < 130e6 and body > 99e6, (model.num_params(), body)
    print(f"RT-FLOPS-ACCT PASSED: ref_dense_130m {model.num_params()/1e6:.1f}M "
          f"params ({body/1e6:.1f}M non-embedding), {f/1e6:.1f}M FLOPs/token "
          f"= body {2*body/1e6:.1f} + head {head/1e6:.1f} + attn {attn/1e6:.1f}")


if __name__ == "__main__":
    test_top1_exact_and_gradients()
    test_top1_prefill_decode()
    test_top1_compute_match()
    test_ablation_nomix()
    test_ablation_unfold()
    test_heterogeneous_fifo()
    test_multik_compute()
    test_bounded_learned_rates()
    test_param_matched_control()
    test_modern_recipe()
    test_flops_accounting()
    print("ALL ROUTED GATES PASSED")
