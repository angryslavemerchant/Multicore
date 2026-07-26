"""M0/M1/M2 gates. Runnable via pytest or `python tests/test_invariants.py`.

M0 — zero-init cores leave logits bit-identical to the core-free model.
M1 — gather->window->scatter agrees with the O(T^2) reference mask spec.
M2 — full prefill logits match token-by-token incremental decode (the cache
     test: if this fails, an invariant in CORE_ROUTING_PLAN.md section 6 is
     broken).
MB — MultiCore (M cores batched into one pass) == M independent Cores summed,
     and its own prefill/decode cache test.
PK — flat (varlen) packing on IMBALANCED rows (5 / 10 / 200 admitted of
     T=256) still equals the O(T^2) reference, and rows do not leak into each
     other across the packed buffer's row boundaries.
SW — banded base sliding-window attention == the dense (T,T) mask it replaced,
     in forward AND in gradients.
OG — gate directions are mutually orthogonal after reprojection, each row's
     NORM is preserved, and they stay orthogonal through a training loop.
     Guards the measured bug: 8 gate directions collapsed to |cos| 0.999 over
     11k joint steps, making eight cores one core replicated.
TQ — the sort+lerp tau quantile (which, unlike torch.quantile, traces under
     dynamo) returns bit-identical taus.
OG-D — the aux fields the section-10 diagnostics read are the RMS of the delta
     actually returned and the h actually passed in.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core import CoreConfig, ModelConfig, SWTransformer, Core, MultiCore
from core.base_model import SWAttention, _rope_cos_sin, _rope_apply
from core.core_module import _quantile_lerp
from core.resident import (resident_mask_reference, compact_indices,
                           pack_indices, window_mask)

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


def test_pk_packed_imbalanced_rows():
    """PK — the case flat packing exists for.

    Rows admitting 5 / 10 / 200 of T=256: the old rectangle padded all three
    out to 200 wide (600 slots for 215 tokens) and let the mask throw the
    padding away. The packed buffer is 215 long and carries a row id instead.
    Ground truth is the untouched O(T^2) `forward_reference`.
    """
    torch.manual_seed(6)
    B, T, d = 3, 256, 32
    counts = [5, 10, 200]
    cc = CoreConfig(K=8, d_core=32, n_heads=2, n_core_layers=2)

    core = Core(d, cc).eval()
    torch.nn.init.normal_(core.out_proj.weight, std=0.1)
    torch.nn.init.normal_(core.out_proj.bias, std=0.1)
    for lay in core.layers:
        torch.nn.init.normal_(lay.rel_bias, std=0.5)
    core.k_dir.data.zero_()
    core.k_dir.data[0] = 1.0                 # gate reads channel 0
    core.tau.data.fill_(0.0)                 # admitted iff h[..., 0] > 0

    h = torch.randn(B, T, d)
    h[..., 0] = -1.0
    for b, c in enumerate(counts):
        h[b, torch.randperm(T)[:c], 0] = 1.0

    with torch.no_grad():
        fast, aux = core(h)
        ref, _ = core.forward_reference(h)
    m = aux["m"]
    assert m.sum(1).tolist() == counts, \
        f"PK setup FAILED: admitted {m.sum(1).tolist()} != {counts}"
    assert ref.abs().max() > 1e-3, "PK vacuous: reference delta is ~zero"
    err = (fast - ref).abs().max().item()
    assert torch.allclose(fast, ref, atol=1e-5), \
        f"PK FAILED: packed vs O(T^2) reference, max err {err}"

    # the packing itself: one buffer of exactly sum(counts), rows contiguous
    # and in (b, then t) order, every admitted token present exactly once
    flat, row, valid = pack_indices(m[None])
    assert flat.shape == (1, sum(counts)) and bool(valid.all()), \
        f"PK FAILED: buffer is {tuple(flat.shape)}, want (1, {sum(counts)})"
    assert torch.equal(row[0], torch.repeat_interleave(
        torch.arange(B), torch.tensor(counts))), "PK FAILED: rows not in order"
    for b in range(B):
        tb = flat[0][row[0] == b] % T
        assert torch.equal(tb, m[b].nonzero().flatten()), \
            f"PK FAILED: row {b} slots are not its passers in order"

    # row separation. Row 1's first passer sits at buffer slot 5, so with K=8
    # it looks back over slots 0..4 -- all of row 0. Only the row-id mask
    # stops that, and a leak would show up as a gradient into row 0.
    hg = h.clone().requires_grad_(True)
    t1 = int(m[1].nonzero()[0])
    core(hg)[0][1, t1].sum().backward()
    reach = (hg.grad.abs().sum(-1) > 0).nonzero().tolist()
    assert reach == [[1, t1]], \
        f"PK FAILED: delta[1, {t1}] reaches {reach}, want only [[1, {t1}]]"

    # the batched path, with a DIFFERENT imbalance per core so the two packed
    # buffers have different lengths (215 vs 243 -> real cross-core padding)
    mc = MultiCore(d, cc, 2).eval()
    randomize_multicore(mc, scale=0.1)
    for lay in mc.layers:
        torch.nn.init.normal_(lay.rel_bias, std=0.5)
    torch.nn.init.normal_(mc.ln_w, mean=1.0, std=0.1)
    torch.nn.init.normal_(mc.ln_b, std=0.1)
    mc.k_dir.data.zero_()
    mc.k_dir.data[0, 0] = 1.0                # core 0 gates on channel 0
    mc.k_dir.data[1, 1] = 1.0                # core 1 gates on channel 1
    mc.tau.data.zero_()
    counts2 = [200, 3, 40]
    h2 = h.clone()
    h2[..., 1] = -1.0
    for b, c in enumerate(counts2):
        h2[b, torch.randperm(T)[:c], 1] = 1.0

    with torch.no_grad():
        got, auxes = mc(h2)
        want = torch.zeros_like(h2)
        for i in range(2):
            c = Core(d, cc).eval()
            c.load_state_dict(mc.core_state_dict(i))
            want = want + c.forward_reference(h2)[0]
    tot = [int(a["m"].sum()) for a in auxes]
    assert tot == [sum(counts), sum(counts2)], \
        f"PK setup FAILED: core totals {tot}"
    assert got.abs().max() > 1e-3, "PK vacuous: MultiCore delta is ~zero"
    err2 = (got - want).abs().max().item()
    assert torch.allclose(got, want, atol=1e-5), \
        f"PK FAILED: packed MultiCore vs O(T^2) reference, max err {err2}"

    # pack_indices on random masks, incl. empty rows and empty cores
    for G, Bg, Tg, p in ((1, 4, 37, 0.3), (3, 5, 64, 0.05), (2, 3, 16, 0.9)):
        mm = torch.rand(G, Bg, Tg) < p
        mm[0, 0] = False                                    # an empty row
        if G > 1:
            mm[-1] = False                                  # an empty core
        fl, rw, vd = pack_indices(mm)
        cnt = mm.reshape(G, -1).sum(1)
        assert fl.shape[1] == max(int(cnt.max()), 1), \
            f"PK FAILED: Npad {fl.shape[1]} != max core total {int(cnt.max())}"
        for gi in range(G):
            wnt = mm[gi].reshape(-1).nonzero().flatten()    # (b, t) order
            assert torch.equal(fl[gi][vd[gi]], wnt) and \
                torch.equal(rw[gi][vd[gi]], wnt // Tg) and \
                int(vd[gi].sum()) == int(cnt[gi]), \
                f"PK FAILED: pack_indices wrong for G={G} p={p} core {gi}"

    print(f"PK PASSED: packed rows {counts} == O(T^2) reference "
          f"(err {err:.2e}); MultiCore totals {tot} (err {err2:.2e}); "
          f"buffer {sum(counts)} slots vs rectangle {B * max(counts)}; "
          f"no cross-row leak")


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


def _abs_cos_max(k):
    """Max pairwise ABSOLUTE cosine between the rows of k (n, d)."""
    n = k.detach().float()
    n = n / n.norm(dim=1, keepdim=True)
    c = (n @ n.t()).abs()
    c.fill_diagonal_(0.0)
    return c.max().item()


def test_og_orthogonal_gates():
    """OG — the gate-direction orthogonality mechanism.

    The bug it exists for was measured, not hypothesised: after 11k steps of a
    real joint run, all eight of a MultiCore's k_dir rows had pairwise absolute
    cosine 0.999 (min 0.999, max 1.000) with norms 24-29 — eight cores admitting
    one token set, which is why extra cores bought no loss.

    Two halves, and both matter: DIRECTIONS become orthogonal, NORMS do not
    move. The gate is sigmoid((s - tau)/gate_temp), so ||k_dir[c]|| is that
    core's learned gate sharpness; normalising it away would fix collapse by
    destroying what collapsed.
    """
    torch.manual_seed(7)
    M, d = 8, 64
    cc = CoreConfig(K=8, d_core=32, n_heads=2, n_core_layers=2,
                    target_rate=0.25)
    mc = MultiCore(d, cc, M)

    # deliberately collapsed: near-identical directions, DIFFERENT norms (so a
    # mechanism that normalised the rows would be caught)
    base = torch.randn(d)
    with torch.no_grad():
        for i in range(M):
            mc.k_dir[i] = base * (1.0 + 0.3 * i) + 1e-3 * torch.randn(d)
    norms0 = mc.k_dir.detach().norm(dim=1).clone()
    before = _abs_cos_max(mc.k_dir)
    assert before > 0.99, f"OG setup FAILED: |cos| {before} is not collapsed"
    assert float(norms0.max() / norms0.min()) > 2.0, \
        "OG setup FAILED: row norms are too similar to test preservation"

    mc.reproject_gates()
    after = _abs_cos_max(mc.k_dir)
    dn = (mc.k_dir.detach().norm(dim=1) - norms0).abs().max().item()
    assert after < 1e-4, f"OG FAILED: |cos| {after:.3e} after reprojection"
    assert dn < 1e-5, f"OG FAILED: row norms moved by {dn:.3e}"
    # row 0's direction is the QR anchor and must not have rotated at all
    d0 = 1.0 - float(F.cosine_similarity(mc.k_dir[0].detach(), base, dim=0))
    assert d0 < 1e-6, f"OG FAILED: row 0 direction moved (1-cos = {d0:.3e})"

    # ---- and they STAY orthogonal under a real optimizer. Run the identical
    # 20 steps with and without the reprojection: the ON arm must hold, and the
    # OFF arm must drift, or the assertion is measuring nothing.
    def train_20(reproject):
        torch.manual_seed(7)
        m = MultiCore(d, cc, M)
        with torch.no_grad():
            for i in range(M):
                m.k_dir[i] = base * (1.0 + 0.3 * i) + 1e-3 * torch.randn(d)
        randomize_multicore(m, scale=0.1)    # nonzero out_w -> gate gets grad
        m.reproject_gates()                  # randomize_multicore reset k_dir
        k_start = m.k_dir.detach().clone()
        h = torch.randn(4, 48, d)
        target = torch.randn(4, 48, d) * 0.5
        # AdamW at m5's own weight decay: the parametrizations.orthogonal route
        # goes NaN here (the decay lands on the Householder generator);
        # reprojection does not care.
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.1)
        m.train()
        worst = 0.0
        for _ in range(20):
            delta, _ = m(h)
            opt.zero_grad(set_to_none=True)
            F.mse_loss(delta, target).backward()
            opt.step()
            if reproject:
                m.reproject_gates()
            worst = max(worst, _abs_cos_max(m.k_dir))
        return worst, (m.k_dir.detach() - k_start).abs().max().item()

    worst, moved = train_20(True)
    free, _ = train_20(False)
    assert moved > 1e-6, \
        f"OG vacuous: k_dir never moved over 20 steps (max {moved:.3e})"
    assert worst < 1e-3, f"OG FAILED: |cos| reached {worst:.3e} during training"
    assert free > 1e-3, (f"OG vacuous: unconstrained gates only reached "
                         f"|cos| {free:.3e}, so the 1e-3 bound proves nothing")

    # the heterogeneous path: N separate Cores, one (d_model,) k_dir each,
    # orthogonalised ACROSS modules by SWTransformer.reproject_gates
    cfg = ModelConfig(
        vocab_size=96, d_model=64, n_layers=3, n_heads=4, window=16,
        max_seq_len=256, core_layer=1,
        cores=[CoreConfig(K=4, d_core=32, n_heads=2, target_rate=0.3),
               CoreConfig(K=8, d_core=32, n_heads=2, target_rate=0.1),
               CoreConfig(K=16, d_core=16, n_heads=2, target_rate=0.05)])
    model = SWTransformer(cfg)
    assert not model.batched_cores and len(model.cores) == 3
    with torch.no_grad():
        for i, c in enumerate(model.cores):
            c.k_dir.copy_(base * (1.0 + 0.5 * i) + 1e-3 * torch.randn(d))
    het = torch.stack([c.k_dir for c in model.cores])
    hn0 = het.detach().norm(dim=1).clone()
    assert _abs_cos_max(het) > 0.99, "OG setup FAILED: het cores not collapsed"
    model.reproject_gates()
    het = torch.stack([c.k_dir for c in model.cores])
    hc = _abs_cos_max(het)
    hdn = (het.detach().norm(dim=1) - hn0).abs().max().item()
    assert hc < 1e-4, f"OG FAILED: heterogeneous |cos| {hc:.3e}"
    assert hdn < 1e-5, f"OG FAILED: heterogeneous norms moved by {hdn:.3e}"

    # M == 1 has nothing to orthogonalise and must not touch k_dir
    solo = MultiCore(d, cc, 1)
    k1 = solo.k_dir.detach().clone()
    solo.reproject_gates()
    assert torch.equal(solo.k_dir.detach(), k1), "OG FAILED: M=1 changed k_dir"

    print(f"OG PASSED: |cos| {before:.3f} -> {after:.2e} with norms fixed to "
          f"{dn:.1e}; 20 AdamW steps stayed under {worst:.2e} vs {free:.2e} "
          f"unconstrained; heterogeneous 3-core path {hc:.2e} "
          f"(norms {hdn:.1e})")


def test_tq_quantile_tau_unchanged():
    """TQ — the sort+lerp tau controller is BIT-identical to torch.quantile.

    `torch.quantile` calls numel() on its input, which throws under dynamo's
    symbolic shapes, so it killed `--compile` for every multi-core preset.
    Replacing it is only safe if tau is unchanged to the last bit: tau is a
    threshold, and a shifted threshold silently moves the admitted set.
    """
    torch.manual_seed(9)
    cases = 0
    for shape in ((1000,), (1024, 8), (7,), (4096, 3), (1, 5), (2048, 16)):
        for rate in (1 / 8, 1 / 64, 0.3, 0.5, 0.055, 0.9, 1 / 2, 0.001):
            for dt in (torch.float32, torch.float64):
                x = (torch.randn(*shape) * 2.0).to(dt)
                q = 1.0 - rate
                want = (torch.quantile(x, q, dim=0) if x.dim() > 1
                        else torch.quantile(x, q))
                got = _quantile_lerp(x, q, 0)
                assert torch.equal(want, got), (
                    f"TQ FAILED at shape={shape} rate={rate} {dt}: "
                    f"max diff {(want - got).abs().max():.3e}")
                cases += 1

    # and through the gates that actually set tau
    d, M = 32, 4
    cc = CoreConfig(K=8, d_core=16, n_heads=2, target_rate=1 / 8)
    h = torch.randn(4, 256, d) * 1.5
    core = Core(d, cc).train()
    torch.nn.init.normal_(core.k_dir, std=1.0)
    core.gate(h)
    s = (h @ core.k_dir).detach().float().flatten()
    assert torch.equal(core.tau, torch.quantile(s, 1 - cc.target_rate)), \
        "TQ FAILED: Core.gate tau != torch.quantile"

    mc = MultiCore(d, cc, M).train()
    torch.nn.init.normal_(mc.k_dir, std=1.0)
    mc.gate(h)
    s = torch.einsum('btd,md->btm', h, mc.k_dir).detach().float().reshape(-1, M)
    assert torch.equal(mc.tau, torch.quantile(s, 1 - cc.target_rate, dim=0)), \
        "TQ FAILED: MultiCore.gate tau != torch.quantile"
    print(f"TQ PASSED: sort+lerp quantile bit-identical to torch.quantile on "
          f"{cases} cases, and tau unchanged in both gates")


def test_og_delta_diagnostics():
    """OG-D — the aux fields the section-10 diagnostics read.

    delta_rms / h_rms / delta_group must be present on BOTH paths and must
    actually be the RMS of the delta returned and the h passed in — a
    diagnostic that silently reports the wrong tensor is worse than none.
    """
    torch.manual_seed(8)
    d = 48
    cc = CoreConfig(K=8, d_core=32, n_heads=2, n_core_layers=2,
                    target_rate=0.25)
    h = torch.randn(3, 64, d)

    def rms(t):
        return float(t.detach().float().pow(2).mean().sqrt())

    core = Core(d, cc).eval()
    torch.nn.init.normal_(core.out_proj.weight, std=0.1)
    torch.nn.init.normal_(core.k_dir, std=1.0)
    delta, aux = core(h)
    assert int(aux["delta_group"]) == 1
    assert abs(float(aux["h_rms"]) - rms(h)) < 1e-5, "OG-D FAILED: Core h_rms"
    assert abs(float(aux["delta_rms"]) - rms(delta)) < 1e-6, \
        "OG-D FAILED: Core delta_rms"
    assert float(aux["delta_rms"]) > 0, "OG-D vacuous: Core delta is zero"

    mc = MultiCore(d, cc, 4).eval()
    randomize_multicore(mc, scale=0.1)
    delta, auxes = mc(h)
    assert len(auxes) == 4
    for a in auxes:
        # one summed delta shared by all M dicts, so group == M
        assert int(a["delta_group"]) == 4, "OG-D FAILED: delta_group != M"
        assert abs(float(a["delta_rms"]) - rms(delta)) < 1e-6, \
            "OG-D FAILED: MultiCore delta_rms is not the SUMMED delta"
        assert abs(float(a["h_rms"]) - rms(h)) < 1e-5, \
            "OG-D FAILED: MultiCore h_rms"
    assert float(auxes[0]["delta_rms"]) > 0, "OG-D vacuous: delta is zero"

    # zero-init cores: the delta really is zero, and the ratio must say so
    # rather than divide by zero
    fresh = MultiCore(d, cc, 4).eval()
    _, auxes0 = fresh(h)
    assert float(auxes0[0]["delta_rms"]) == 0.0, \
        "OG-D FAILED: zero-init MultiCore reports nonzero delta_rms"
    print("OG-D PASSED: delta_rms/h_rms/delta_group correct on both paths")


if __name__ == "__main__":
    test_m0_bit_identical()
    test_m1_mask_equivalence()
    test_m2_cache_correctness()
    test_mb_multicore_equivalence()
    test_mb_multicore_cache()
    test_pk_packed_imbalanced_rows()
    test_sw_attention_equivalence()
    test_og_orthogonal_gates()
    test_tq_quantile_tau_unchanged()
    test_og_delta_diagnostics()
    print("ALL GATES PASSED")
