"""Tiny sliding-window transformer with optional cores at one layer.

The base is architecturally unable to attend past `window` tokens — cores are
the only long-range channel. Both a full-sequence forward and an incremental
(KV-cached) decode path are provided; M2 asserts they agree.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig, CoreConfig
from .core_module import Core, TokenAdapter


class SWAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.window = cfg.window
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def _split(self, x, B, T):
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q, B, T), self._split(k, B, T), self._split(v, B, T)
        i = torch.arange(T, device=x.device)
        mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < self.window)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))

    def step(self, x, cache):
        """x: (B, 1, C). cache holds up to `window` past k/v (including none)."""
        B, _, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._split(q, B, 1), self._split(k, B, 1), self._split(v, B, 1)
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

    def step(self, x, cache):
        x = x + self.attn.step(self.ln1(x), cache)
        return x + self.mlp(self.ln2(x))


class SWTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cls = TokenAdapter if cfg.adapter else Core
        self.cores = nn.ModuleList(cls(cfg.d_model, c) for c in cfg.cores)
        self.cores_enabled = True
        self.apply(self._init)
        # re-zero core outputs (self.apply above overwrote them)
        for c in self.cores:
            nn.init.zeros_(c.out_proj.weight)
            nn.init.zeros_(c.out_proj.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, collect_aux=False):
        """idx: (B, T) -> logits (B, T, V), aux list (one dict per core)."""
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(pos)[None]
        auxes = []
        for li, blk in enumerate(self.blocks):
            h = blk(h)
            if li == self.cfg.core_layer and self.cores and self.cores_enabled:
                for core in self.cores:
                    delta, aux = core(h)
                    h = h + delta
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
        pos = torch.full((B,), caches["t"], device=idx_t.device, dtype=torch.long)
        h = (self.tok_emb(idx_t) + self.pos_emb(pos))[:, None, :]
        for li, blk in enumerate(self.blocks):
            h = blk.step(h, caches["blocks"][li])
            if li == self.cfg.core_layer and self.cores and self.cores_enabled:
                for core, ring in zip(self.cores, caches["rings"]):
                    h = h + core.step(h[:, 0, :], ring)[:, None, :]
        caches["t"] += 1
        return self.head(self.ln_f(h))[:, 0, :]

    def num_params(self, trainable_only=False):
        ps = [p for p in self.parameters() if p.requires_grad or not trainable_only]
        return sum(p.numel() for p in ps)
