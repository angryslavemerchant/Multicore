"""A Core: threshold-gated private mini-transformer over a FIFO of admitted
tokens.

The core is an n_layer stack of (attention over residents + FFN) applied on
the COMPACTED passer sequence, with a learned rank-relative attention bias
(delta-rank in passer space, 0..K-1). Two layers + rank bias let the core
express the induction circuit internally: layer 1 binds each admitted token
to its rank-neighbours (e.g. a value to its adjacent key), layer 2 matches
those bindings and retrieves. A single-layer bag-of-residents core cannot do
this unless binding already exists in the base representations (measured:
it mostly doesn't — oracle-gated single-layer cores barely beat chance).

Commit semantics for multi-layer: resident j's layer-l state is computed AT
j's admission over the residents present then, and never recomputed. Layer
l+1 at a later token i attends over those committed layer-l states. In
prefill this is just l stacked causal sliding-window attentions on the
compacted sequence; in decode each admitted token computes its states from
the ring and stores them.

Gating: hard membership (quantile-controller tau), soft magnitude
g = sigmoid((s - tau)/temp) on members (gradient path to k_dir).
Output projection is zero-initialised: deltas are exactly 0 at init.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreConfig
from .resident import resident_mask_reference, compact_indices, window_mask


class _CoreLayer(nn.Module):
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        dc = cfg.d_core
        self.n_heads = cfg.n_heads
        self.d_head = dc // cfg.n_heads
        self.q = nn.Linear(dc, dc)
        self.k = nn.Linear(dc, dc)
        self.v = nn.Linear(dc, dc)
        self.ffn = nn.Sequential(
            nn.Linear(dc, cfg.ffn_mult * dc), nn.GELU(),
            nn.Linear(cfg.ffn_mult * dc, dc))
        # rank-relative attention bias: delta-rank 0..K-1 (query rank - key
        # rank in passer space), per head. This is what makes "the token
        # admitted right before me" addressable.
        self.rel_bias = nn.Parameter(torch.zeros(cfg.n_heads, cfg.K))

    def _heads(self, x):
        B, P, _ = x.shape
        return x.view(B, P, self.n_heads, self.d_head).transpose(1, 2)

    def attend(self, xq, xkv, mask, dq, dkv):
        """xq (B,P,dc) queries at ranks dq (B,P); xkv keys/values at ranks
        dkv (B,P); mask (B,P,P) bool."""
        qh = self._heads(self.q(xq))
        kh = self._heads(self.k(xkv))
        vh = self._heads(self.v(xkv))
        delta = (dq[:, :, None] - dkv[:, None, :]).clamp(0, self.rel_bias.shape[1] - 1)
        bias = self.rel_bias[:, delta]            # (H, B, P, P)
        bias = bias.permute(1, 0, 2, 3)           # (B, H, P, P)
        bias = bias.masked_fill(~mask[:, None, :, :], float("-inf"))
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        B, _, P, _ = out.shape
        return out.transpose(1, 2).reshape(B, P, -1)

    def forward(self, x, mask, ranks):
        """Full-sequence path on the compacted stream: queries and keys are
        the same set of positions."""
        a = self.attend(x, x, mask, ranks, ranks)
        return x + self.ffn(a)


class _CoreCompute(nn.Module):
    """Shared: gate params + in/out projections + layer stack."""

    def __init__(self, d_model: int, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        dc = cfg.d_core
        self.k_dir = nn.Parameter(torch.randn(d_model) * 0.02)
        # tau is a train-time quantile controller (EMA of the (1-r)-quantile
        # of scores -> hard admission rate ~= target_rate by construction),
        # frozen at inference (invariant I5). Not a learned parameter.
        self.register_buffer("tau", torch.zeros(()))
        self.register_buffer("tau_set", torch.zeros((), dtype=torch.bool))
        self.tau_momentum = 0.25
        self.ln = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, dc)
        self.layers = nn.ModuleList(_CoreLayer(cfg)
                                    for _ in range(cfg.n_core_layers))
        self.out_proj = nn.Linear(dc, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def gate(self, h):
        """h: (..., d) -> s, hard membership m, soft magnitude g."""
        s = h @ self.k_dir
        if self.training:
            with torch.no_grad():
                q = torch.quantile(s.detach().float().flatten(),
                                   1.0 - self.cfg.target_rate)
                if not self.tau_set:
                    # scores carry a large shared offset from the residual
                    # stream's mean direction; jump straight to the quantile
                    # instead of EMA-crawling from zero
                    self.tau.copy_(q)
                    self.tau_set.fill_(True)
                else:
                    self.tau.mul_(1 - self.tau_momentum).add_(
                        self.tau_momentum * q)
        m = s > self.tau
        g = torch.sigmoid((s - self.tau) / self.cfg.gate_temp)
        return s, m, g


class Core(_CoreCompute):
    """Cross-token core (the real thing)."""

    def forward(self, h, m_override=None):
        """h: (B, T, d) -> (delta (B, T, d), aux dict). Full-sequence path.
        m_override (B, T) bool: oracle admission for diagnosis runs —
        isolates transport/readback from gate discovery (g forced to 1)."""
        B, T, d = h.shape
        s, m, g = self.gate(h)
        if m_override is not None:
            m, g = m_override, torch.ones_like(g)
        aux = {"rate": m.float().mean().detach(),
               "tau": self.tau.detach().clone(), "m": m.detach()}
        if not m.any():
            return torch.zeros_like(h), aux
        idx, valid = compact_indices(m)
        P = idx.shape[1]
        hc = torch.gather(h, 1, idx[:, :, None].expand(-1, -1, d))
        gc = torch.gather(g, 1, idx)
        x = self.in_proj(self.ln(hc))
        mask = window_mask(P, self.cfg.K, valid)
        ranks = torch.arange(P, device=h.device)[None, :].expand(B, -1)
        for layer in self.layers:
            x = layer(x, mask, ranks)
        dc = self.out_proj(x) * (gc * valid.float())[:, :, None]
        delta = torch.zeros_like(h)
        delta.scatter_(1, idx[:, :, None].expand(-1, -1, d), dc)
        return delta, aux

    def forward_reference(self, h):
        """Same computation via the O(T^2) reference mask (tests only).
        Residents keep FULL-sequence positions; ranks come from cumsum."""
        B, T, d = h.shape
        s, m, g = self.gate(h)
        deltas = []
        for b in range(B):
            mask = resident_mask_reference(m[b], self.cfg.K)
            mask = (mask | torch.eye(T, dtype=torch.bool, device=h.device))
            ranks = (m[b].long().cumsum(0) - 1).clamp(min=0)[None, :]
            x = self.in_proj(self.ln(h[b:b + 1]))
            for layer in self.layers:
                x = layer(x, mask[None], ranks)
            d_b = self.out_proj(x)[0] * (g[b] * m[b].float())[:, None]
            deltas.append(d_b)
        return torch.stack(deltas), {"m": m.detach()}

    # ---- incremental decode ----
    def init_ring(self, B, device, dtype=torch.float32):
        dc, K, L = self.cfg.d_core, self.cfg.K, self.cfg.n_core_layers
        return {"x": torch.zeros(L, B, K, dc, device=device, dtype=dtype),
                "rank": torch.zeros(B, K, dtype=torch.long, device=device),
                "count": torch.zeros(B, dtype=torch.long, device=device)}

    def step(self, h, ring):
        """h: (B, d) one token per batch element -> delta (B, d).
        Mutates ring. Stores the committed layer-l input states of each
        admitted token; a new admit runs the layer stack against them."""
        B, d = h.shape
        s, m, g = self.gate(h)
        if not m.any():
            return torch.zeros_like(h)
        K = self.cfg.K
        pos = ring["count"] % K
        bidx = m.nonzero(as_tuple=True)[0]
        x = self.in_proj(self.ln(h))          # (B, dc) layer-0 input state
        my_rank = ring["count"].clone()       # rank of the incoming token
        # insert layer-0 state + rank for admitted tokens
        ring["x"][0][bidx, pos[bidx]] = x[bidx]
        ring["rank"][bidx, pos[bidx]] = my_rank[bidx]
        ring["count"] = ring["count"] + m.long()
        valid = torch.arange(K, device=h.device)[None, :] < \
            ring["count"].clamp(max=K)[:, None]
        mask = valid.clone()
        mask[:, 0] |= ~valid.any(dim=1)       # keep softmax rows non-empty
        for li, layer in enumerate(self.layers):
            a = layer.attend(x[:, None, :], ring["x"][li], mask[:, None, :],
                             my_rank[:, None], ring["rank"])
            x = x + layer.ffn(a)[:, 0, :]
            if li + 1 < len(self.layers):
                ring["x"][li + 1][bidx, pos[bidx]] = x[bidx]
        return self.out_proj(x) * (g * m.float())[:, None]


class TokenAdapter(_CoreCompute):
    """Per-token control: same gate, same layer stack sizes, NO cross-token
    attention (each layer's attention is replaced by a linear mix of the
    token's own value projection)."""

    def _solo(self, h):
        x = self.in_proj(self.ln(h))
        for layer in self.layers:
            x = x + layer.ffn(layer.v(x))
        return self.out_proj(x)

    def forward(self, h, m_override=None):
        s, m, g = self.gate(h)
        if m_override is not None:
            m, g = m_override, torch.ones_like(g)
        aux = {"rate": m.float().mean().detach(),
               "tau": self.tau.detach().clone(), "m": m.detach()}
        delta = self._solo(h) * (g * m.float())[..., None]
        return delta, aux

    def init_ring(self, B, device, dtype=torch.float32):
        return {}

    def step(self, h, ring):
        s, m, g = self.gate(h)
        return self._solo(h) * (g * m.float())[:, None]
