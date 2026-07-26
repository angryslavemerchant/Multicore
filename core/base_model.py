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
from .core_module import Core, MultiCore, TokenAdapter

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
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.window = cfg.window
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
        # Identical core configs (the common case: N copies of one CoreConfig)
        # are computed by a single batched MultiCore instead of N sequential
        # modules — same math, one set of kernel launches. Heterogeneous
        # configs keep the per-core path.
        self.batched_cores = (not cfg.adapter and len(cfg.cores) > 1
                              and all(c == cfg.cores[0] for c in cfg.cores))
        if self.batched_cores:
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
                    h = h + core.step(h[:, 0, :], ring)[:, None, :]
        caches["t"] += 1
        return self.head(self.ln_f(h))[:, 0, :]

    def num_params(self, trainable_only=False):
        ps = [p for p in self.parameters() if p.requires_grad or not trainable_only]
        return sum(p.numel() for p in ps)
