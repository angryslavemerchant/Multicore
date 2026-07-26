"""A Core: threshold-gated private attention over a FIFO of admitted tokens.

Gating: hard membership (s > 0 after tau fold-in), soft magnitude
g = sigmoid(s / temp) applied to members' deltas (gradient path to k_dir).
Output projection is zero-initialised: deltas are exactly 0 at init.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreConfig
from .resident import resident_mask_reference, compact_indices, window_mask


class _CoreCompute(nn.Module):
    """Shared machinery: gate params + projections + FFN + zero-init output."""

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
        self.q_proj = nn.Linear(d_model, dc)
        self.k_proj = nn.Linear(d_model, dc)
        self.v_proj = nn.Linear(d_model, dc)
        self.ffn = nn.Sequential(
            nn.Linear(dc, cfg.ffn_mult * dc), nn.GELU(),
            nn.Linear(cfg.ffn_mult * dc, dc))
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

    def _heads(self, x, B):
        H = self.cfg.n_heads
        return x.view(B, -1, H, self.cfg.d_core // H).transpose(1, 2)

    def attend(self, q, k, v, mask):
        """q,k,v: (B, P, dc); mask: (B, P, P) bool -> (B, P, dc)."""
        B = q.shape[0]
        qh, kh, vh = self._heads(q, B), self._heads(k, B), self._heads(v, B)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask[:, None, :, :])
        return out.transpose(1, 2).reshape(B, -1, self.cfg.d_core)

    def finish(self, a):
        """attention output -> delta direction (before gate scaling)."""
        f = a + self.ffn(a)
        return self.out_proj(f)


class Core(_CoreCompute):
    """Cross-token core (the real thing)."""

    def forward(self, h):
        """h: (B, T, d) -> (delta (B, T, d), aux dict). Full-sequence path."""
        B, T, d = h.shape
        s, m, g = self.gate(h)
        aux = {"rate": m.float().mean().detach(), "tau": self.tau.detach().clone(),
               "m": m.detach()}
        if not m.any():
            return torch.zeros_like(h), aux
        idx, valid = compact_indices(m)
        P = idx.shape[1]
        hc = torch.gather(h, 1, idx[:, :, None].expand(-1, -1, d))
        gc = torch.gather(g, 1, idx)
        x = self.ln(hc)
        mask = window_mask(P, self.cfg.K, valid)
        a = self.attend(self.q_proj(x), self.k_proj(x), self.v_proj(x), mask)
        dc = self.finish(a) * (gc * valid.float())[:, :, None]
        delta = torch.zeros_like(h)
        delta.scatter_(1, idx[:, :, None].expand(-1, -1, d), dc)
        return delta, aux

    def forward_reference(self, h):
        """Same computation via the O(T^2) reference mask (tests only)."""
        B, T, d = h.shape
        s, m, g = self.gate(h)
        x = self.ln(h)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        deltas = []
        for b in range(B):
            mask = resident_mask_reference(m[b], self.cfg.K)
            mask = mask | torch.eye(T, dtype=torch.bool, device=h.device)
            a = self.attend(q[b:b + 1], k[b:b + 1], v[b:b + 1], mask[None])
            d_b = self.finish(a)[0] * (g[b] * m[b].float())[:, None]
            deltas.append(d_b)
        return torch.stack(deltas), {"m": m.detach()}

    # ---- incremental decode ----
    def init_ring(self, B, device, dtype=torch.float32):
        dc = self.cfg.d_core
        return {"k": torch.zeros(B, self.cfg.K, dc, device=device, dtype=dtype),
                "v": torch.zeros(B, self.cfg.K, dc, device=device, dtype=dtype),
                "count": torch.zeros(B, dtype=torch.long, device=device)}

    def step(self, h, ring):
        """h: (B, d) one token per batch element -> delta (B, d). Mutates ring."""
        B, d = h.shape
        s, m, g = self.gate(h)
        if not m.any():
            return torch.zeros_like(h)
        x = self.ln(h)
        kn, vn, q = self.k_proj(x), self.v_proj(x), self.q_proj(x)
        pos = ring["count"] % self.cfg.K
        bidx = m.nonzero(as_tuple=True)[0]
        ring["k"][bidx, pos[bidx]] = kn[bidx]
        ring["v"][bidx, pos[bidx]] = vn[bidx]
        ring["count"] = ring["count"] + m.long()
        valid = torch.arange(self.cfg.K, device=h.device)[None, :] < \
            ring["count"].clamp(max=self.cfg.K)[:, None]
        # rows for non-admitted tokens are computed then zeroed; keep softmax sane
        mask = valid.clone()
        mask[:, 0] |= ~valid.any(dim=1)
        a = self.attend(q[:, None, :], ring["k"], ring["v"], mask[:, None, :])
        delta = self.finish(a)[:, 0, :] * (g * m.float())[:, None]
        return delta


class TokenAdapter(_CoreCompute):
    """Per-token control: same gate, same size, NO cross-token attention.
    An extra dc->dc linear stands in for the q/k attention params so total
    parameter count stays comparable."""

    def __init__(self, d_model: int, cfg: CoreConfig):
        super().__init__(d_model, cfg)
        self.mix = nn.Linear(cfg.d_core, cfg.d_core)

    def forward(self, h):
        s, m, g = self.gate(h)
        aux = {"rate": m.float().mean().detach(), "tau": self.tau.detach().clone(),
               "m": m.detach()}
        a = self.mix(self.v_proj(self.ln(h)))
        delta = self.finish(a) * (g * m.float())[..., None]
        return delta, aux

    def init_ring(self, B, device, dtype=torch.float32):
        return {}

    def step(self, h, ring):
        s, m, g = self.gate(h)
        a = self.mix(self.v_proj(self.ln(h)))
        return self.finish(a) * (g * m.float())[:, None]
