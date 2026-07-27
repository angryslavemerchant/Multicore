"""Tiny sliding-window transformer with optional cores at one layer.

The base is architecturally unable to attend past `window` tokens — cores are
the only long-range channel. Both a full-sequence forward and an incremental
(KV-cached) decode path are provided; M2 asserts they agree.

Prefill attention is BANDED (see `_sw_band_layout`): the sliding window makes
the (T, T) attention matrix a causal band of width `window`, so we cut it into
query blocks and hand SDPA (nb, C, W) blocks instead of one (T, T) problem —
O(T*window), never O(T*T), in both memory and FLOPs. Same trick the cores use
on the compacted passer stream (`core_module._banded_attend`); the layouts are
kept separate because the core band also carries a rank-relative bias and a
passer-validity mask that mean nothing here.

Passing the dense (T, T) bool mask to `F.scaled_dot_product_attention` was
costing both, and how badly depends on the build. It always disqualifies the
flash kernel; where nothing else takes a mask it falls through to the math
backend, which keeps a (B, H, T, T) probability tensor for backward (402 MB
per layer at B=4, H=6, T=2048 — measured 5.60 GB peak for the 8-layer base,
vs 2.69 GB banded). Where a fused masked kernel does exist the memory is fine
but the FLOPs are still quadratic: the kernel dutifully scores all T keys per
query and throws away the (1 - window/T) that the mask forbids — measured
764 ms -> 392 ms per fwd+bwd step at B=4, T=2048, window=256.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, CoreConfig
from .core_module import (Core, MultiCore, TokenAdapter, orthonormalize_rows_,
                          _banded_attend, _stacked, _bl)
from .losses import ce_sum
from .resident import pack_indices

# Query-block size floor for banded attention. C = `window` is the natural
# block (key window W = 2*window-1 — the least wasted work per query); the
# floor only stops tiny windows from degenerating into thousands of blocks of
# a couple of rows each.
SW_MIN_BLOCK = 64


def _rope_cos_sin(d_head, positions):
    """positions: (...,) long -> cos/sin (..., d_head/2)."""
    inv = 1.0 / (10000 ** (torch.arange(0, d_head, 2,
                                        device=positions.device) / d_head))
    ang = positions[..., None].float() * inv
    return ang.cos(), ang.sin()


def _rope_apply(x, cos, sin):
    """x: (B, H, T, dh); cos/sin broadcastable to (T, dh/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# ------------------------------------------------------------- recipe pieces
# RMSNorm / SwiGLU / qk-norm, the three places open-sci-ref-0.01 (and every
# modern small dense model) differs from the LayerNorm + GELU base this repo
# started from. Each is opt-in via ModelConfig so the byte-era runs stay
# reproducible; see core/config.py.
RMS_EPS = 1e-5


def _rms(x, eps=RMS_EPS):
    """RMS-normalise the last axis, in fp32.

    fp32 on purpose, matching what autocast already does to `F.layer_norm`:
    the reduction is over d_model terms of a bf16 residual stream, and doing
    it in bf16 costs ~3 decimal digits on the scale every downstream matmul
    is multiplied by. Returns fp32; the following matmul casts it back under
    autocast, exactly as with the LayerNorm path.
    """
    f = x.float()
    return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return (_rms(x, self.eps) * self.weight).to(x.dtype)


def _norm(cfg, d):
    return RMSNorm(d) if cfg.rmsnorm else nn.LayerNorm(d)


class SwiGLU(nn.Module):
    """down(silu(gate(x)) * up(x)) — three matrices where GELU-MLP has two."""

    def __init__(self, d, hidden, bias=True):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=bias)
        self.up = nn.Linear(d, hidden, bias=bias)
        self.down = nn.Linear(hidden, d, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def ffn_hidden(d, mult=4, swiglu=False, explicit=0, multiple_of=8):
    """Inner FFN width. `explicit` (ModelConfig/CoreConfig ffn_hidden) wins.

    Otherwise `mult * d`, scaled by 2/3 under SwiGLU: three matrices of width h
    cost 3*d*h against the GELU MLP's 2*d*h, so the unscaled width would make
    `swiglu=True` a 1.5x parameter and FLOPs change wearing a nonlinearity's
    name. With the scaling, flipping the flag changes the nonlinearity and
    nothing else — which is the only way the ablation means anything.

    An `explicit` width is taken at face value and NOT rescaled, because it
    cannot be known whether the author already sized it for SwiGLU (2256 in
    open-sci-ref-0.01: yes) or for GELU (704 in the routed presets: no). Turn
    swiglu on over an explicit width and you have asked for 1.5x the FFN;
    m5_arch prints the resulting hidden width and parameter count.
    """
    if explicit:
        return explicit
    h = mult * d
    if swiglu:
        h = 2 * h // 3
    return -(-h // multiple_of) * multiple_of


def _mlp(cfg, d):
    h = ffn_hidden(d, cfg.ffn_mult, cfg.swiglu, cfg.ffn_hidden)
    if cfg.swiglu:
        return SwiGLU(d, h)
    return nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))


def keys_per_token(window, T):
    """Mean number of keys a query attends, averaged over a length-T causal
    sequence with attention window `window`. The FLOPs multiplier for both
    QK^T and A@V, so attention costs 4 * d_model * this, per layer per token.

    Exact for both regimes rather than the `min(window, T)` it replaces:
    query i attends min(i+1, window) keys, so full attention (window >= T)
    averages (T+1)/2, not T. That factor of 2 is not a rounding detail — at
    T=4096, d=512, 22 layers it is 92M FLOPs/token against a ~99M body.
    """
    w = min(window, T)
    return (w * (w + 1) / 2 + max(T - w, 0) * w) / T


# ------------------------------------------------- banded sliding-window
def _sw_band_layout(T, window, C, device):
    """Block-banded layout for causal sliding-window attention.

    Query i attends key j iff `j <= i and i - j < window`, so the attention
    matrix is a causal band of width `window` and a dense (T, T) mask is pure
    waste. We walk the band in BLOCKS of C queries: block b owns queries
    [b*C, b*C+C) and those queries need exactly the contiguous key range
    [b*C-(window-1), b*C+C), of length W = C+window-1. Every tensor
    downstream is O(T*W); no (T, T) tensor is ever built.

    Returns (nb, W, Tp, jc, ok):
      nb   number of blocks, Tp = nb*C the block-padded position axis
      jc   (nb*W,) long   key index of each window slot, clamped into [0, Tp)
      ok   (nb, C, W) bool  slot is inside the band and is a real key.
           The diagonal (j == i) is always on, so padded query rows still
           have a non-empty softmax (their output is sliced off) — same
           guard `_banded_attend` uses for padded ranks.
    """
    W = C + window - 1
    nb = (T + C - 1) // C
    Tp = nb * C
    j = (torch.arange(nb, device=device)[:, None] * C - (window - 1)
         + torch.arange(W, device=device)[None, :])           # (nb, W)
    dlt = (torch.arange(C, device=device)[:, None]
           - torch.arange(W, device=device)[None, :] + (window - 1))  # i - j
    band = (dlt >= 0) & (dlt < window)                        # (C, W)
    ok = (band & (j >= 0)[:, None, :]) | (dlt == 0)[None]     # (nb, C, W)
    return nb, W, Tp, j.clamp(0, Tp - 1).reshape(-1), ok


def _sliding_window_attend(q, k, v, window, C):
    """Causal window-`window` attention, banded. q, k, v (B, H, T, dh).

    Semantics are exactly the dense mask
    `(i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < window)`
    fed to scaled_dot_product_attention — same attended set, same scaling.

    The blocks go back through SDPA rather than being attended by hand: the
    block axis rides in the head slot, so SDPA sees (B*H, nb, C, dh) against
    (B*H, nb, W, dh) under a (1, nb, C, W) mask that broadcasts over B*H, and
    a fused masked kernel then never materialises even the (nb, C, W) logits.
    Where no such kernel exists SDPA falls back to math and the peak is
    O(T*W) — still the smaller problem, by T/W.
    """
    B, H, T, dh = q.shape
    nb, W, Tp, jc, ok = _sw_band_layout(T, window, C, q.device)
    if Tp != T:                        # pad the position axis to whole blocks
        pad = (0, 0, 0, Tp - T)
        q, k, v = F.pad(q, pad), F.pad(k, pad), F.pad(v, pad)
    qb = q.reshape(B * H, nb, C, dh)          # blocks partition the queries
    kb = k.index_select(2, jc).view(B * H, nb, W, dh)   # ..and overlap by W-C
    vb = v.index_select(2, jc).view(B * H, nb, W, dh)
    out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=ok[None])
    return out.reshape(B, H, Tp, dh)[:, :, :T]


class SWAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, window=None):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.window = cfg.window if window is None else window
        self.rope = cfg.rope
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        # qk-norm: RMSNorm over each head's d_head, applied BEFORE RoPE (which
        # is a rotation, so it commutes with a per-head rescale only if the
        # norm comes first — and that is the order open-sci-ref uses).
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

    def _split(self, x, B, T):
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _qk(self, q, k):
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        return q, k

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q, B, T), self._split(k, B, T), self._split(v, B, T)
        q, k = self._qk(q, k)
        if self.rope:
            cos, sin = _rope_cos_sin(self.d_head, torch.arange(T, device=x.device))
            q, k = _rope_apply(q, cos, sin), _rope_apply(k, cos, sin)
        if self.window >= T:
            # the window covers the whole sequence (the *_dense_full presets
            # set window=T): the band IS the causal mask, and is_causal says
            # so in the one way that leaves every SDPA kernel eligible.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = _sliding_window_attend(
                q, k, v, self.window,
                min(max(self.window, SW_MIN_BLOCK), T))
        return self.proj(out.transpose(1, 2).reshape(B, T, C))

    def step(self, x, cache, t):
        """x: (B, 1, C) at absolute position t. cache holds up to `window`
        past ROTATED k (rope is position-absolute per entry, so cached k
        stay valid) and raw v."""
        B, _, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q, B, 1), self._split(k, B, 1), self._split(v, B, 1)
        q, k = self._qk(q, k)
        if self.rope:
            cos, sin = _rope_cos_sin(
                self.d_head, torch.tensor(t, device=x.device))
            q, k = _rope_apply(q, cos, sin), _rope_apply(k, cos, sin)
        if cache["k"] is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        k, v = k[:, :, -self.window:], v[:, :, -self.window:]
        cache["k"], cache["v"] = k, v
        out = F.scaled_dot_product_attention(q, k, v)  # full attn over window
        return self.proj(out.transpose(1, 2).reshape(B, 1, C))


class _RoutedExpertBlock(nn.Module):
    """Eight independent transformer blocks, one selected per token.

    With `tie_loops` (the default) one set of expensive weights is reused at
    every recurrent depth; LayerNorm affine parameters and residual scales are
    loop-specific, so the repeated block can learn different roles without
    duplicating its QKV/O or FFN weights. With `tie_loops=False` the recurrence
    is UNFOLDED: `n_loops` independent weight sets are stored back-to-back on
    the leading axis and depth `l` slices out its own, making the stack
    n_loops ordinary layers at identical FLOPs.
    """

    def __init__(self, d: int, cfg: CoreConfig, M: int, rmsnorm=False,
                 swiglu=False):
        super().__init__()
        # Heterogeneous FIFO lengths: every ring is padded to K_max so the
        # banded kernel keeps one width, and `k_gate` masks expert m down to
        # its own K_m. Gap 0 (the token itself) is never masked, so no softmax
        # row can be empty.
        Ks = list(cfg.K_list) if cfg.K_list else [cfg.K] * M
        assert len(Ks) == M and min(Ks) >= 1, Ks
        self.Ks = Ks
        self.d, self.M, self.K = d, M, max(Ks)
        self.n_heads = cfg.n_heads
        self.d_head = d // cfg.n_heads
        self.n_loops = cfg.n_loops
        self.rmsnorm, self.swiglu = rmsnorm, swiglu
        hidden = ffn_hidden(d, cfg.ffn_mult, swiglu, cfg.ffn_hidden)
        assert d % cfg.n_heads == 0
        self.hidden = hidden
        # One weight set when tied, n_loops of them when unfolded. `_slice`
        # below turns a depth index into the right block of the leading axis;
        # when tied it is always slice(0, M), i.e. the whole tensor, so the
        # tied path is bit-identical to the pre-unfold code.
        self.n_sets = 1 if cfg.tie_loops else cfg.n_loops
        S = self.n_sets * M
        self.q_w, self.q_b = _stacked(S, d, d), nn.Parameter(torch.zeros(S, d))
        self.k_w, self.k_b = _stacked(S, d, d), nn.Parameter(torch.zeros(S, d))
        self.v_w, self.v_b = _stacked(S, d, d), nn.Parameter(torch.zeros(S, d))
        self.o_w, self.o_b = _stacked(S, d, d), nn.Parameter(torch.zeros(S, d))
        self.f1_w = _stacked(S, d, hidden)
        self.f1_b = nn.Parameter(torch.zeros(S, hidden))
        self.f2_w = _stacked(S, hidden, d)
        self.f2_b = nn.Parameter(torch.zeros(S, d))
        if swiglu:                       # the gate branch; f1 becomes the "up"
            self.f3_w = _stacked(S, d, hidden)
            self.f3_b = nn.Parameter(torch.zeros(S, hidden))
        self.rel_bias = nn.Parameter(torch.zeros(S, cfg.n_heads, self.K))
        # Additive, non-learnable: 0 where the gap is inside that expert's
        # FIFO, a large negative otherwise. Finite rather than -inf so an
        # unattended run can never produce a NaN in the backward pass.
        gate = torch.zeros(self.n_sets * M, 1, self.K)
        for m, km in enumerate(Ks):
            if km < self.K:
                gate[m::M, :, km:] = -1e4
        self.register_buffer("k_gate", gate)
        shape = (cfg.n_loops, M, d)
        self.ln1_w = nn.Parameter(torch.ones(shape))
        self.ln2_w = nn.Parameter(torch.ones(shape))
        if not rmsnorm:                  # RMSNorm is scale-only, no bias
            self.ln1_b = nn.Parameter(torch.zeros(shape))
            self.ln2_b = nn.Parameter(torch.zeros(shape))
        scale = torch.full((cfg.n_loops, M), cfg.residual_scale_init)
        self.attn_scale = nn.Parameter(scale.clone())
        self.ffn_scale = nn.Parameter(scale.clone())

    def _ln(self, x, loop, which):
        """Per-(depth, expert) norm over the packed (M, P, d) stream."""
        w = (self.ln1_w if which == 1 else self.ln2_w)[loop]
        if self.rmsnorm:
            return _rms(x) * w[:, None, :]
        b = (self.ln1_b if which == 1 else self.ln2_b)[loop]
        return F.layer_norm(x, (x.shape[-1],)) * w[:, None, :] + b[:, None, :]

    def _ln_one(self, x, loop, expert, which):
        """Same norm for the decode path, where x is (S, d) or (S, K, d) and
        the affine is a single expert's (d,) row."""
        w = (self.ln1_w if which == 1 else self.ln2_w)[loop, expert]
        if self.rmsnorm:
            return _rms(x) * w
        b = (self.ln1_b if which == 1 else self.ln2_b)[loop, expert]
        return F.layer_norm(x, (self.d,)) * w + b

    def _ffn(self, xn, s):
        """Packed FFN for weight-block `s`: GELU MLP, or SwiGLU when enabled."""
        up = _bl(xn, self.f1_w[s], self.f1_b[s])
        h = (F.silu(_bl(xn, self.f3_w[s], self.f3_b[s])) * up if self.swiglu
             else F.gelu(up))
        return _bl(h, self.f2_w[s], self.f2_b[s])

    def _heads(self, x):
        M, P, _ = x.shape
        return x.view(M, P, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def _slice(self, loop):
        """Leading-axis block holding depth `loop`'s weights (all M experts)."""
        o = (loop % self.n_sets) * self.M
        return slice(o, o + self.M)

    def forward_packed(self, x, valid, row, loop):
        s = self._slice(loop)
        xn = self._ln(x, loop, 1)
        q = self._heads(_bl(xn, self.q_w[s], self.q_b[s]))
        k = self._heads(_bl(xn, self.k_w[s], self.k_b[s]))
        v = self._heads(_bl(xn, self.v_w[s], self.v_b[s]))
        a = _banded_attend(q, k, v, valid, row,
                           self.rel_bias[s] + self.k_gate[s],
                           self.K, self.M)
        a = a.permute(0, 2, 1, 3).reshape(self.M, x.shape[1], self.d)
        x = x + self.attn_scale[loop, :, None, None] * _bl(a, self.o_w[s],
                                                           self.o_b[s])
        f = self._ffn(self._ln(x, loop, 2), s)
        return x + self.ffn_scale[loop, :, None, None] * f

    def step_one(self, expert, loop, x, xkv, valid, query_rank, kv_rank):
        """One expert slice: x (S,d), xkv (S,K,d) -> (S,d).

        `expert` indexes the per-depth LayerNorm affine and residual scales,
        which are always (n_loops, M); `e` indexes the expensive weights,
        which carry the extra depth blocks when the recurrence is unfolded.
        """
        e = self._slice(loop).start + expert
        qn = self._ln_one(x, loop, expert, 1)
        kn = self._ln_one(xkv, loop, expert, 1)

        def heads(z):
            S, P, _ = z.shape
            return z.view(S, P, self.n_heads, self.d_head).transpose(1, 2)

        q = heads((qn @ self.q_w[e] + self.q_b[e])[:, None, :])
        k = heads(kn @ self.k_w[e] + self.k_b[e])
        v = heads(kn @ self.v_w[e] + self.v_b[e])
        gap = (query_rank[:, None] - kv_rank).clamp(0, self.K - 1)
        bias = (self.rel_bias[e] + self.k_gate[e])[:, gap] \
            .permute(1, 0, 2)[:, :, None, :]
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        a = a.transpose(1, 2).reshape(x.shape[0], self.d)
        x = x + self.attn_scale[loop, expert] * (
            a @ self.o_w[e] + self.o_b[e])
        xn = self._ln_one(x, loop, expert, 2)
        up = xn @ self.f1_w[e] + self.f1_b[e]
        h = (F.silu(xn @ self.f3_w[e] + self.f3_b[e]) * up if self.swiglu
             else F.gelu(up))
        return x + self.ffn_scale[loop, expert] * (h @ self.f2_w[e]
                                                   + self.f2_b[e])


class Top1LoopedMultiCore(nn.Module):
    """Top-1 routed recurrent middle stack with inter-expert communication.

    Every token executes exactly one of M expert blocks per loop. After each
    loop, tokens return to sequence order and pass through one shared causal
    sliding-window attention mixer, then route again from the updated state.
    """

    is_top1_routed = True

    def __init__(self, model_cfg: ModelConfig, cfg: CoreConfig, M: int):
        super().__init__()
        d = model_cfg.d_model
        assert cfg.d_core == d, "top1 recurrent cores require d_core == d_model"
        assert cfg.n_core_layers == 1, "top1 recurrent experts contain one tied block"
        assert M > 1 and cfg.n_loops > 0 and cfg.inter_core_window >= 0
        self.cfg, self.M, self.d = cfg, M, d
        # inter_core_window == 0 removes the shared mixer entirely: experts
        # then never exchange information and rerouting sees an unmixed state.
        # The module is not built at all, so its parameters and FLOPs are gone
        # rather than merely unused.
        self.use_mixer = cfg.inter_core_window > 0
        self.router_w = nn.Parameter(torch.randn(M, d) * 0.02)
        with torch.no_grad():
            orthonormalize_rows_(self.router_w)
        if cfg.router_bias:
            self.r_bias = nn.Parameter(torch.zeros(M))
        # Training-step counter for the hash anneal. Kept as a buffer and used
        # as a tensor throughout, so reading it never forces a device sync.
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))
        self.expert = _RoutedExpertBlock(d, cfg, M, model_cfg.rmsnorm,
                                         model_cfg.swiglu)
        if self.use_mixer:
            self.mixer_ln = nn.ModuleList(
                _norm(model_cfg, d) for _ in range(cfg.n_loops))
            self.mixer = SWAttention(model_cfg, window=cfg.inter_core_window)
            self.mix_scale = nn.Parameter(torch.full(
                (cfg.n_loops,), cfg.residual_scale_init))

    def zero_out_(self):
        # Deliberately not zeroed: zeroing the outer path starves every expert
        # parameter of gradient at initialization. Small residual scales provide
        # the stable near-identity start instead.
        pass

    def _hash_scale(self):
        """Positional-prior strength, linearly annealed to 0 if configured.

        Stays a tensor so no `.item()` sync creeps into the forward pass.
        """
        s = self.cfg.router_hash_scale
        if not self.cfg.hash_anneal_iters:
            return s
        frac = 1.0 - self._step.float() / self.cfg.hash_anneal_iters
        return s * frac.clamp_min(0.0)

    def _balance_term(self, load, importance):
        """Uniform Switch loss, or the bounded-band range penalty.

        The range penalty is EXACTLY zero while every expert's share is inside
        [rate_lo, rate_hi], so rates are free to differentiate; it only pushes
        back on experts going dead or dominant. Gradient flows through
        `importance` (the mean softmax mass), since the argmax load is not
        differentiable.
        """
        if not self.cfg.router_bias:
            return self.M * (load.detach() * importance).sum()
        lo, hi = self.cfg.rate_lo, self.cfg.rate_hi
        pen = (F.relu(lo - importance).pow(2)
               + F.relu(importance - hi).pow(2)).sum()
        return self.cfg.router_range_weight * pen

    def _route(self, x, positions=None, loop=0):
        xn = F.layer_norm(x.float(), (self.d,))
        rw = self.router_w.float()
        rw = rw / rw.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logits = torch.einsum("btd,md->btm", xn, rw)
        if self.cfg.router_bias:
            logits = logits + self.r_bias.float()
        if positions is not None and self.cfg.router_hash_scale:
            preferred = (positions + loop) % self.M
            prior = F.one_hot(preferred, self.M).to(logits.dtype)
            logits = logits + self._hash_scale() * prior
        choice = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1).to(x.dtype)
        hard = F.one_hot(choice, self.M).to(x.dtype)
        straight_through = (hard - probs).detach() + probs
        return probs, hard.bool(), straight_through

    def forward(self, h, m_override=None):
        if m_override is not None:
            raise ValueError("gate_override is not defined for top-1 routing")
        B, T, d = h.shape
        if self.training:
            self._step += 1
        entry, x = h, h
        loop_masks, loads, balances, pack_utils, pack_overheads = [], [], [], [], []
        drops = []
        entropies, margins = [], []
        for li in range(self.cfg.n_loops):
            pos = torch.arange(T, device=x.device)
            probs, hard, route_weight = self._route(x, pos, li)
            masks = hard.permute(2, 0, 1)               # (M,B,T)
            flat, row, valid = pack_indices(masks, self.cfg.capacity_factor)
            P = flat.shape[1]
            fl = flat.reshape(-1)
            packed = x.reshape(B * T, d).index_select(0, fl).view(self.M, P, d)
            updated = self.expert.forward_packed(packed, valid, row, li)
            delta = updated - packed
            selected_weight = route_weight.permute(2, 0, 1).reshape(
                self.M, B * T).gather(1, flat)
            delta = delta * (selected_weight * valid.to(x.dtype))[..., None]
            scattered = torch.zeros(B * T, d, device=x.device, dtype=x.dtype)
            scattered.index_add_(0, fl, delta.reshape(-1, d))
            x = x + scattered.view(B, T, d)
            if self.use_mixer:
                x = x + self.mix_scale[li] * self.mixer(self.mixer_ln[li](x))

            load = hard.float().mean(dim=(0, 1))
            importance = probs.float().mean(dim=(0, 1))
            balances.append(self._balance_term(load, importance))
            loop_masks.append(masks)
            # tokens routed here but over their expert's cap: they skip the
            # expert entirely and ride the residual. Nonzero means the cap is
            # binding, which is a fact about the router, not a free lunch.
            drops.append(1.0 - valid.sum() / masks.sum().clamp_min(1))
            pack_utils.append(valid.float().mean())
            pack_overheads.append(valid.float().mean().clamp_min(1e-9).reciprocal())
            p32 = probs.float().clamp_min(1e-9)
            loads.append(load.detach())
            entropies.append(-(p32 * p32.log()).sum(-1).mean() / math.log(self.M))
            top2 = probs.float().topk(2, dim=-1).values
            margins.append((top2[..., 0] - top2[..., 1]).mean())

        delta = x - entry
        masks = torch.stack(loop_masks)                  # (L,M,B,T)
        rates = masks.float().mean(dim=(0, 2, 3))
        hr = h.float().square().mean().sqrt().detach()
        dr = delta.float().square().mean().sqrt().detach()
        aux = []
        for mi in range(self.M):
            aux.append({"rate": rates[mi].detach(), "tau": rates.new_zeros(()),
                        "tau_z": rates.new_zeros(()), "m": masks[:, mi].detach(),
                        "h_rms": hr, "delta_rms": dr, "delta_group": self.M})
        aux[0].update({
            # the range penalty is already absolutely scaled; only the
            # uniform Switch objective takes router_aux_weight
            "router_aux_loss": (torch.stack(balances).mean() if
                                self.cfg.router_bias else
                                self.cfg.router_aux_weight
                                * torch.stack(balances).mean()),
            "router_entropy": torch.stack(entropies).mean().detach(),
            "router_margin": torch.stack(margins).mean().detach(),
            "pack_util": torch.stack(pack_utils).mean().detach(),
            "pack_overhead": torch.stack(pack_overheads).mean().detach(),
            "drop_rate": torch.stack(drops).mean().detach(),
            "loop_rates": masks.float().mean(dim=(2, 3)).detach(),
            "rate_min": rates.min().detach(), "rate_max": rates.max().detach(),
            "rate_cv": (rates.std() / rates.mean().clamp_min(1e-9)).detach(),
        })
        return delta, aux

    def init_ring(self, B, device, dtype=torch.float32):
        L, M, K, d = self.cfg.n_loops, self.M, self.expert.K, self.d
        return {"x": torch.zeros(L, M, B, K, d, device=device, dtype=dtype),
                "rank": torch.zeros(L, M, B, K, device=device, dtype=torch.long),
                "count": torch.zeros(L, M, B, device=device, dtype=torch.long),
                "mixer": [{"k": None, "v": None} for _ in range(L)]}

    def step(self, h, cache, t=0):
        entry, x = h, h
        ar = torch.arange(self.expert.K, device=h.device)
        for li in range(self.cfg.n_loops):
            pos = torch.tensor(t, device=x.device)
            _, hard, _ = self._route(x[:, None, :], pos, li)
            choice = hard[:, 0].long().argmax(-1)
            out = x.clone()
            for mi in range(self.M):
                bidx = (choice == mi).nonzero(as_tuple=True)[0]
                if bidx.numel() == 0:
                    continue
                pos = cache["count"][li, mi, bidx] % self.expert.K
                rank = cache["count"][li, mi, bidx].clone()
                cache["x"][li, mi, bidx, pos] = x[bidx]
                cache["rank"][li, mi, bidx, pos] = rank
                cache["count"][li, mi, bidx] += 1
                valid = ar[None, :] < cache["count"][li, mi, bidx, None].clamp(
                    max=self.expert.K)
                out[bidx] = self.expert.step_one(
                    mi, li, x[bidx], cache["x"][li, mi, bidx], valid, rank,
                    cache["rank"][li, mi, bidx])
            x = out
            if self.use_mixer:
                mixed = self.mixer.step(self.mixer_ln[li](x)[:, None, :],
                                        cache["mixer"][li], t)[:, 0, :]
                x = x + self.mix_scale[li] * mixed
        return x - entry

    def estimated_flops_parts(self, T=None):
        d, hidden, L = self.d, self.expert.hidden, self.cfg.n_loops
        n_ffn = 3 if self.expert.swiglu else 2
        expert_linear = 2 * (4 * d * d + n_ffn * d * hidden)
        # the FIFO is K entries deep and always full past the warm-up, so
        # unlike the base attention there is no causal ramp to average over
        expert_attention = 4 * self.expert.K * d
        if not self.use_mixer:
            return (L * (expert_linear + expert_attention), 0)
        mixer_linear = 2 * 4 * d * d
        w = self.cfg.inter_core_window
        mixer_attention = 4 * d * keys_per_token(w, T or w)
        return (L * (expert_linear + expert_attention),
                L * (mixer_linear + mixer_attention))

    def estimated_flops_per_token(self, T=None):
        return sum(self.estimated_flops_parts(T))

    @torch.no_grad()
    def reproject_router(self):
        orthonormalize_rows_(self.router_w)
class Block(nn.Module):


    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = _norm(cfg, cfg.d_model)
        self.attn = SWAttention(cfg)
        self.ln2 = _norm(cfg, cfg.d_model)
        self.mlp = _mlp(cfg, cfg.d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

    def step(self, x, cache, t):
        x = x + self.attn.step(self.ln1(x), cache, t)
        return x + self.mlp(self.ln2(x))


class SWTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = None if cfg.rope else nn.Embedding(cfg.max_seq_len,
                                                          cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = _norm(cfg, cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cls = TokenAdapter if cfg.adapter else Core
        self.top1_routed = (bool(cfg.cores) and
                            cfg.cores[0].routing == "top1_recurrent")
        if self.top1_routed:
            assert not cfg.adapter and len(cfg.cores) > 1
            assert all(c == cfg.cores[0] for c in cfg.cores)
        # Identical core configs (the common case: N copies of one CoreConfig)
        # are computed by a single batched MultiCore instead of N sequential
        # modules — same math, one set of kernel launches. Heterogeneous
        # configs keep the per-core path.
        self.batched_cores = (not self.top1_routed and not cfg.adapter and
                              len(cfg.cores) > 1 and
                              all(c == cfg.cores[0] for c in cfg.cores))
        # The threshold-gated Core/MultiCore path is still LayerNorm + GELU:
        # the recipe flags were plumbed through the base blocks and the routed
        # experts, which are the two arms under comparison. Refuse the
        # combination rather than ship a half-converted model whose loss would
        # be attributed to the recipe.
        assert not (cfg.cores and not self.top1_routed
                    and (cfg.rmsnorm or cfg.swiglu)), \
            "rmsnorm/swiglu are not implemented for threshold-gated cores"
        if self.top1_routed:
            self.cores = nn.ModuleList(
                [Top1LoopedMultiCore(cfg, cfg.cores[0], len(cfg.cores))])
        elif self.batched_cores:
            self.cores = nn.ModuleList(
                [MultiCore(cfg.d_model, cfg.cores[0], len(cfg.cores))])
        else:
            self.cores = nn.ModuleList(cls(cfg.d_model, c) for c in cfg.cores)
        self.cores_enabled = True
        self.apply(self._init)
        # AFTER _init, so the head does not get its own draw and then have it
        # discarded. Tying saves V*d parameters (25.75M at the open-sci-ref
        # shape, i.e. a fifth of that model) and is what every model we compare
        # against does; `parameters()` de-duplicates the shared tensor, so
        # num_params() and the optimizer both see it once.
        if cfg.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        # re-zero core outputs (self.apply above overwrote them)
        for c in self.cores:
            c.zero_out_()

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, collect_aux=False, gate_override=None,
                return_hidden=False, targets=None, ce_chunk=0):
        """idx: (B, T) -> logits (B, T, V), aux list (one dict per core).
        gate_override (B, T) bool: oracle admission for all cores.

        `return_hidden` returns the final normed hidden states (B, T, d)
        INSTEAD of logits, so the caller can fuse the head into a chunked
        cross-entropy. At vocab 50304 and T=4096 the fp32 logits are 824 MB
        per sequence, which bounds the batch long before anything else does.

        `targets` (B, T) computes that cross-entropy HERE and returns the mean
        loss instead. Use this one under DistributedDataParallel: DDP hooks the
        parameters touched inside the wrapped forward, so a loss computed
        outside it leaves `head.weight`'s gradient un-all-reduced — and with
        tied embeddings, racing rather than merely missing. Single-GPU callers
        may use either; see core/losses.py.
        """
        B, T = idx.shape
        h = self.tok_emb(idx)
        if self.pos_emb is not None:
            h = h + self.pos_emb(torch.arange(T, device=idx.device))[None]
        auxes = []
        for li, blk in enumerate(self.blocks):
            h = blk(h)
            if li == self.cfg.core_layer and self.cores and self.cores_enabled:
                for core in self.cores:
                    delta, aux = core(h, m_override=gate_override)
                    h = h + delta
                    # MultiCore returns one aux dict per core it batches, so
                    # the aux list keeps the same shape either way
                    if isinstance(aux, list):
                        auxes.extend(aux)
                    else:
                        auxes.append(aux)
        h = self.ln_f(h)
        if targets is not None:
            tgt = targets.reshape(-1)
            out = ce_sum(h.reshape(-1, h.shape[-1]), self.head.weight, tgt,
                         ce_chunk) / tgt.numel()
        elif return_hidden:
            out = h
        else:
            out = self.head(h)
        return (out, auxes) if collect_aux else out

    @torch.no_grad()
    def reproject_gates(self):
        """Make every gate direction at the core layer mutually orthogonal,
        preserving each one's norm. Call after `optimizer.step()`; no-op with
        fewer than two gate directions.

        The rows are collected ACROSS core modules, so this covers both paths:
        a single batched `MultiCore` (its k_dir is already (M, d_model)) and
        the heterogeneous path, where N separate `Core`s each hold a (d_model,)
        k_dir and would otherwise be free to converge on one direction exactly
        as the batched cores were measured doing. Rate diversity does not save
        them — cores at different rates sharing one direction give NESTED
        selections, one salience notion at several zooms (plan 5.9).

        Not applied at construction, so a run with the mechanism switched off
        (`m5_arch.py --no-ortho`) is bit-identical to the pre-mechanism code.
        """
        for core in self.cores:
            if hasattr(core, "reproject_router"):
                core.reproject_router()
        ks = [c.k_dir for c in self.cores if hasattr(c, "k_dir")]
        if sum(1 if k.dim() == 1 else k.shape[0] for k in ks) < 2:
            return
        w = torch.cat([k.data.reshape(-1, self.cfg.d_model) for k in ks], 0)
        orthonormalize_rows_(w)
        off = 0
        for k in ks:
            n = 1 if k.dim() == 1 else k.shape[0]
            k.data.copy_(w[off:off + n].reshape(k.shape))
            off += n

    def freeze_base(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for c in self.cores:
            for p in c.parameters():
                p.requires_grad_(True)

    # ---- incremental decode ----
    def init_caches(self, B, device):
        return {"blocks": [{"k": None, "v": None} for _ in self.blocks],
                "rings": [c.init_ring(B, device) for c in self.cores],
                "t": 0}

    def forward_step(self, idx_t, caches):
        """idx_t: (B,) one token -> logits (B, V). Mutates caches."""
        B = idx_t.shape[0]
        h = self.tok_emb(idx_t)[:, None, :]
        if self.pos_emb is not None:
            pos = torch.full((B,), caches["t"], device=idx_t.device,
                             dtype=torch.long)
            h = h + self.pos_emb(pos)[:, None, :]
        for li, blk in enumerate(self.blocks):
            h = blk.step(h, caches["blocks"][li], caches["t"])
            if li == self.cfg.core_layer and self.cores and self.cores_enabled:
                for core, ring in zip(self.cores, caches["rings"]):
                    h = h + core.step(h[:, 0, :], ring,
                                      caches["t"])[:, None, :]
        caches["t"] += 1
        return self.head(self.ln_f(h))[:, 0, :]

    def num_params(self, trainable_only=False):
        ps = [p for p in self.parameters() if p.requires_grad or not trainable_only]
        return sum(p.numel() for p in ps)
