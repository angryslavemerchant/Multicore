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

    def _split(self, x, B, T):
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q, B, T), self._split(k, B, T), self._split(v, B, T)
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

    def __init__(self, d: int, cfg: CoreConfig, M: int):
        super().__init__()
        self.d, self.M, self.K = d, M, cfg.K
        self.n_heads = cfg.n_heads
        self.d_head = d // cfg.n_heads
        self.n_loops = cfg.n_loops
        hidden = cfg.ffn_hidden or cfg.ffn_mult * d
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
        self.rel_bias = nn.Parameter(torch.zeros(S, cfg.n_heads, cfg.K))
        shape = (cfg.n_loops, M, d)
        self.ln1_w = nn.Parameter(torch.ones(shape))
        self.ln1_b = nn.Parameter(torch.zeros(shape))
        self.ln2_w = nn.Parameter(torch.ones(shape))
        self.ln2_b = nn.Parameter(torch.zeros(shape))
        scale = torch.full((cfg.n_loops, M), cfg.residual_scale_init)
        self.attn_scale = nn.Parameter(scale.clone())
        self.ffn_scale = nn.Parameter(scale.clone())

    @staticmethod
    def _ln(x, w, b):
        return F.layer_norm(x, (x.shape[-1],)) * w[:, None, :] + b[:, None, :]

    def _heads(self, x):
        M, P, _ = x.shape
        return x.view(M, P, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def _slice(self, loop):
        """Leading-axis block holding depth `loop`'s weights (all M experts)."""
        o = (loop % self.n_sets) * self.M
        return slice(o, o + self.M)

    def forward_packed(self, x, valid, row, loop):
        s = self._slice(loop)
        xn = self._ln(x, self.ln1_w[loop], self.ln1_b[loop])
        q = self._heads(_bl(xn, self.q_w[s], self.q_b[s]))
        k = self._heads(_bl(xn, self.k_w[s], self.k_b[s]))
        v = self._heads(_bl(xn, self.v_w[s], self.v_b[s]))
        a = _banded_attend(q, k, v, valid, row, self.rel_bias[s],
                           self.K, self.M)
        a = a.permute(0, 2, 1, 3).reshape(self.M, x.shape[1], self.d)
        x = x + self.attn_scale[loop, :, None, None] * _bl(a, self.o_w[s],
                                                           self.o_b[s])
        xn = self._ln(x, self.ln2_w[loop], self.ln2_b[loop])
        f = _bl(F.gelu(_bl(xn, self.f1_w[s], self.f1_b[s])),
                self.f2_w[s], self.f2_b[s])
        return x + self.ffn_scale[loop, :, None, None] * f

    def step_one(self, expert, loop, x, xkv, valid, query_rank, kv_rank):
        """One expert slice: x (S,d), xkv (S,K,d) -> (S,d).

        `expert` indexes the per-depth LayerNorm affine and residual scales,
        which are always (n_loops, M); `e` indexes the expensive weights,
        which carry the extra depth blocks when the recurrence is unfolded.
        """
        e = self._slice(loop).start + expert
        w1, b1 = self.ln1_w[loop, expert], self.ln1_b[loop, expert]
        qn = F.layer_norm(x, (self.d,)) * w1 + b1
        kn = F.layer_norm(xkv, (self.d,)) * w1 + b1

        def heads(z):
            S, P, _ = z.shape
            return z.view(S, P, self.n_heads, self.d_head).transpose(1, 2)

        q = heads((qn @ self.q_w[e] + self.q_b[e])[:, None, :])
        k = heads(kn @ self.k_w[e] + self.k_b[e])
        v = heads(kn @ self.v_w[e] + self.v_b[e])
        gap = (query_rank[:, None] - kv_rank).clamp(0, self.K - 1)
        bias = self.rel_bias[e][:, gap].permute(1, 0, 2)[:, :, None, :]
        bias = bias.masked_fill(~valid[:, None, None, :], float("-inf"))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        a = a.transpose(1, 2).reshape(x.shape[0], self.d)
        x = x + self.attn_scale[loop, expert] * (
            a @ self.o_w[e] + self.o_b[e])
        w2, b2 = self.ln2_w[loop, expert], self.ln2_b[loop, expert]
        xn = F.layer_norm(x, (self.d,)) * w2 + b2
        f = F.gelu(xn @ self.f1_w[e] + self.f1_b[e])
        f = f @ self.f2_w[e] + self.f2_b[e]
        return x + self.ffn_scale[loop, expert] * f


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
        self.expert = _RoutedExpertBlock(d, cfg, M)
        if self.use_mixer:
            self.mixer_ln = nn.ModuleList(
                nn.LayerNorm(d) for _ in range(cfg.n_loops))
            self.mixer = SWAttention(model_cfg, window=cfg.inter_core_window)
            self.mix_scale = nn.Parameter(torch.full(
                (cfg.n_loops,), cfg.residual_scale_init))

    def zero_out_(self):
        # Deliberately not zeroed: zeroing the outer path starves every expert
        # parameter of gradient at initialization. Small residual scales provide
        # the stable near-identity start instead.
        pass

    def _route(self, x, positions=None, loop=0):
        xn = F.layer_norm(x.float(), (self.d,))
        rw = self.router_w.float()
        rw = rw / rw.norm(dim=1, keepdim=True).clamp_min(1e-12)
        logits = torch.einsum("btd,md->btm", xn, rw)
        if positions is not None and self.cfg.router_hash_scale:
            preferred = (positions + loop) % self.M
            prior = F.one_hot(preferred, self.M).to(logits.dtype)
            logits = logits + self.cfg.router_hash_scale * prior
        choice = logits.argmax(dim=-1)
        probs = torch.softmax(logits, dim=-1).to(x.dtype)
        hard = F.one_hot(choice, self.M).to(x.dtype)
        straight_through = (hard - probs).detach() + probs
        return probs, hard.bool(), straight_through

    def forward(self, h, m_override=None):
        if m_override is not None:
            raise ValueError("gate_override is not defined for top-1 routing")
        B, T, d = h.shape
        entry, x = h, h
        loop_masks, loads, balances, pack_utils, pack_overheads = [], [], [], [], []
        entropies, margins = [], []
        for li in range(self.cfg.n_loops):
            pos = torch.arange(T, device=x.device)
            probs, hard, route_weight = self._route(x, pos, li)
            masks = hard.permute(2, 0, 1)               # (M,B,T)
            flat, row, valid = pack_indices(masks)
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
            balances.append(self.M * (load.detach() * importance).sum())
            loop_masks.append(masks)
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
            "router_aux_loss": self.cfg.router_aux_weight * torch.stack(balances).mean(),
            "router_entropy": torch.stack(entropies).mean().detach(),
            "router_margin": torch.stack(margins).mean().detach(),
            "pack_util": torch.stack(pack_utils).mean().detach(),
            "pack_overhead": torch.stack(pack_overheads).mean().detach(),
            "loop_rates": masks.float().mean(dim=(2, 3)).detach(),
            "rate_min": rates.min().detach(), "rate_max": rates.max().detach(),
            "rate_cv": (rates.std() / rates.mean().clamp_min(1e-9)).detach(),
        })
        return delta, aux

    def init_ring(self, B, device, dtype=torch.float32):
        L, M, K, d = self.cfg.n_loops, self.M, self.cfg.K, self.d
        return {"x": torch.zeros(L, M, B, K, d, device=device, dtype=dtype),
                "rank": torch.zeros(L, M, B, K, device=device, dtype=torch.long),
                "count": torch.zeros(L, M, B, device=device, dtype=torch.long),
                "mixer": [{"k": None, "v": None} for _ in range(L)]}

    def step(self, h, cache, t=0):
        entry, x = h, h
        ar = torch.arange(self.cfg.K, device=h.device)
        for li in range(self.cfg.n_loops):
            pos = torch.tensor(t, device=x.device)
            _, hard, _ = self._route(x[:, None, :], pos, li)
            choice = hard[:, 0].long().argmax(-1)
            out = x.clone()
            for mi in range(self.M):
                bidx = (choice == mi).nonzero(as_tuple=True)[0]
                if bidx.numel() == 0:
                    continue
                pos = cache["count"][li, mi, bidx] % self.cfg.K
                rank = cache["count"][li, mi, bidx].clone()
                cache["x"][li, mi, bidx, pos] = x[bidx]
                cache["rank"][li, mi, bidx, pos] = rank
                cache["count"][li, mi, bidx] += 1
                valid = ar[None, :] < cache["count"][li, mi, bidx, None].clamp(
                    max=self.cfg.K)
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
        expert_linear = 2 * (4 * d * d + 2 * d * hidden)
        expert_attention = 4 * self.cfg.K * d
        if not self.use_mixer:
            return (L * (expert_linear + expert_attention), 0)
        mixer_linear = 2 * 4 * d * d
        mixer_attention = 4 * min(self.cfg.inter_core_window, T or self.cfg.inter_core_window) * d
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
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = SWAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model))

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
        self.ln_f = nn.LayerNorm(cfg.d_model)
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

    def forward(self, idx, collect_aux=False, gate_override=None):
        """idx: (B, T) -> logits (B, T, V), aux list (one dict per core).
        gate_override (B, T) bool: oracle admission for all cores."""
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
        logits = self.head(self.ln_f(h))
        return (logits, auxes) if collect_aux else logits

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
