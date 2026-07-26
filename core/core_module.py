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

`MultiCore` computes M cores that share a CoreConfig (same K, d_core,
n_heads, n_core_layers, ffn_mult, target_rate) in one batched pass: every
weight carries a leading core dimension and the core index is folded into
the batch dim of the compaction / attention / scatter. The math is
identical to M independent `Core`s whose deltas SUM into h — only the
batching changes. Sequential cores are launch-bound (8 identical cores ran
at 128k tok/s vs 307k for 2), which is what this exists to fix.
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
        # tau is a train-time quantile controller (set to the exact per-batch
        # (1-r)-quantile of scores -> hard admission rate == target_rate by
        # construction), frozen at inference (invariant I5). Not a learned
        # parameter.
        self.register_buffer("tau", torch.zeros(()))
        self.ln = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, dc)
        self.layers = nn.ModuleList(_CoreLayer(cfg)
                                    for _ in range(cfg.n_core_layers))
        self.out_proj = nn.Linear(dc, d_model)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def zero_out_(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def gate(self, h):
        """h: (..., d) -> s, hard membership m, soft magnitude g."""
        s = h @ self.k_dir
        if self.training:
            with torch.no_grad():
                q = torch.quantile(s.detach().float().flatten(),
                                   1.0 - self.cfg.target_rate)
                # exact per-batch quantile, no EMA: under joint training the
                # score distribution drifts continuously and an EMA lags,
                # letting the admission rate creep above target.
                self.tau.copy_(q)
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


# ------------------------------------------------------------------ batched
def _stacked(*shape, std=0.02):
    p = nn.Parameter(torch.empty(*shape))
    nn.init.normal_(p, std=std)
    return p


def _bl(x, w, b):
    """Batched-over-cores linear: x (M,B,P,i) @ w (M,i,o) + b (M,o)."""
    return torch.einsum('mbpi,mio->mbpo', x, w) + b[:, None, None, :]


class _MultiCoreLayer(nn.Module):
    """One core layer for M cores at once. Weights are stored (M, in, out)
    (i.e. nn.Linear.weight transposed) so einsum/bmm can eat them directly."""

    def __init__(self, cfg: CoreConfig, M: int):
        super().__init__()
        dc, dh = cfg.d_core, cfg.ffn_mult * cfg.d_core
        self.M, self.K = M, cfg.K
        self.n_heads = cfg.n_heads
        self.d_head = dc // cfg.n_heads
        self.q_w, self.q_b = _stacked(M, dc, dc), nn.Parameter(torch.zeros(M, dc))
        self.k_w, self.k_b = _stacked(M, dc, dc), nn.Parameter(torch.zeros(M, dc))
        self.v_w, self.v_b = _stacked(M, dc, dc), nn.Parameter(torch.zeros(M, dc))
        self.f1_w, self.f1_b = _stacked(M, dc, dh), nn.Parameter(torch.zeros(M, dh))
        self.f2_w, self.f2_b = _stacked(M, dh, dc), nn.Parameter(torch.zeros(M, dc))
        self.rel_bias = nn.Parameter(torch.zeros(M, cfg.n_heads, cfg.K))

    # ---- full-sequence (batched over cores) ----
    def _heads(self, t, M, B, P):
        return t.view(M, B, P, self.n_heads, self.d_head) \
                .permute(0, 1, 3, 2, 4).reshape(M * B, self.n_heads, P, self.d_head)

    def attend(self, x, mask, drank):
        """x (M,B,P,dc); mask (M,B,P,P) bool; drank (P,P) long delta-ranks.
        Same math as _CoreLayer.attend with M folded into the SDPA batch."""
        M, B, P, dc = x.shape
        q = self._heads(_bl(x, self.q_w, self.q_b), M, B, P)
        k = self._heads(_bl(x, self.k_w, self.k_b), M, B, P)
        v = self._heads(_bl(x, self.v_w, self.v_b), M, B, P)
        bias = self.rel_bias[:, :, drank].to(q.dtype)      # (M,H,P,P)
        bias = bias[:, None].expand(M, B, self.n_heads, P, P)
        bias = bias.masked_fill(~mask[:, :, None, :, :], float("-inf"))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias.reshape(M * B, self.n_heads, P, P))
        return out.view(M, B, self.n_heads, P, self.d_head) \
                  .permute(0, 1, 3, 2, 4).reshape(M, B, P, dc)

    def ffn(self, a):
        return _bl(F.gelu(_bl(a, self.f1_w, self.f1_b)), self.f2_w, self.f2_b)

    def forward(self, x, mask, drank):
        return x + self.ffn(self.attend(x, mask, drank))

    # ---- single-core slice (decode path) ----
    def attend_one(self, mi, xq, xkv, mask, dq, dkv):
        """Exactly _CoreLayer.attend for core mi. xq (B,Pq,dc),
        xkv (B,Pk,dc), mask (B,Pq,Pk)."""
        def heads(t):
            B, P, _ = t.shape
            return t.view(B, P, self.n_heads, self.d_head).transpose(1, 2)
        qh = heads(xq @ self.q_w[mi] + self.q_b[mi])
        kh = heads(xkv @ self.k_w[mi] + self.k_b[mi])
        vh = heads(xkv @ self.v_w[mi] + self.v_b[mi])
        delta = (dq[:, :, None] - dkv[:, None, :]).clamp(0, self.K - 1)
        bias = self.rel_bias[mi][:, delta].permute(1, 0, 2, 3)   # (B,H,Pq,Pk)
        bias = bias.masked_fill(~mask[:, None, :, :], float("-inf"))
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        B, _, P, _ = out.shape
        return out.transpose(1, 2).reshape(B, P, -1)

    def ffn_one(self, mi, a):
        return F.gelu(a @ self.f1_w[mi] + self.f1_b[mi]) @ self.f2_w[mi] \
            + self.f2_b[mi]


class MultiCore(nn.Module):
    """M identically-shaped Cores computed in one batched pass.

    Parameters are independent per core (leading dim M); gates, taus and
    ring states are independent; the M deltas SUM into a single (B,T,d).
    forward returns (delta_sum, [aux_0 .. aux_{M-1}]) so callers that log
    per-core rates keep working.
    """

    def __init__(self, d_model: int, cfg: CoreConfig, M: int):
        super().__init__()
        self.cfg, self.d_model, self.M = cfg, d_model, M
        dc = cfg.d_core
        self.k_dir = nn.Parameter(torch.randn(M, d_model) * 0.02)
        self.register_buffer("tau", torch.zeros(M))
        self.ln_w = nn.Parameter(torch.ones(M, d_model))
        self.ln_b = nn.Parameter(torch.zeros(M, d_model))
        self.in_w = _stacked(M, d_model, dc)
        self.in_b = nn.Parameter(torch.zeros(M, dc))
        self.layers = nn.ModuleList(_MultiCoreLayer(cfg, M)
                                    for _ in range(cfg.n_core_layers))
        self.out_w = nn.Parameter(torch.zeros(M, dc, d_model))
        self.out_b = nn.Parameter(torch.zeros(M, d_model))

    def zero_out_(self):
        nn.init.zeros_(self.out_w)
        nn.init.zeros_(self.out_b)

    # ---- gate ----
    def gate(self, h):
        """h (...,d) -> s,m,g each (...,M). Per-core exact quantile tau."""
        s = torch.einsum('...d,md->...m', h, self.k_dir)
        if self.training:
            with torch.no_grad():
                q = torch.quantile(s.detach().float().reshape(-1, self.M),
                                   1.0 - self.cfg.target_rate, dim=0)
                self.tau.copy_(q)
        m = s > self.tau
        g = torch.sigmoid((s - self.tau) / self.cfg.gate_temp)
        return s, m, g

    def _aux(self, m):
        return [{"rate": m[..., i].float().mean().detach(),
                 "tau": self.tau[i].detach().clone(), "m": m[..., i].detach()}
                for i in range(self.M)]

    def _norm_in(self, hc):
        """hc (M,B,P,d) -> (M,B,P,dc) with per-core LN affine + in_proj."""
        x = F.layer_norm(hc, (self.d_model,))
        x = x * self.ln_w[:, None, None, :] + self.ln_b[:, None, None, :]
        return _bl(x, self.in_w, self.in_b)

    def forward(self, h, m_override=None):
        """h (B,T,d) -> (summed delta (B,T,d), list of M aux dicts)."""
        B, T, d = h.shape
        M, K = self.M, self.cfg.K
        s, m, g = self.gate(h)                      # (B,T,M)
        if m_override is not None:
            m = m_override[..., None].expand(B, T, M)
            g = torch.ones_like(g)
        aux = self._aux(m)
        if not m.any():
            return torch.zeros_like(h), aux
        # (M,B,T) -> rows r = m*B + b, so every (core, batch-row) is a row
        mp = m.permute(2, 0, 1).reshape(M * B, T)
        idx, valid = compact_indices(mp)            # (M*B,P)
        P = idx.shape[1]
        row_b = torch.arange(M * B, device=h.device) % B
        flat = row_b[:, None] * T + idx             # into h.view(B*T, d)
        hc = h.reshape(B * T, d).index_select(0, flat.reshape(-1)).view(M, B, P, d)
        gc = g.permute(2, 0, 1).reshape(M * B, T).gather(1, idx).view(M, B, P)
        x = self._norm_in(hc)
        mask = window_mask(P, K, valid).view(M, B, P, P)
        r = torch.arange(P, device=h.device)
        drank = (r[:, None] - r[None, :]).clamp(0, K - 1)
        for layer in self.layers:
            x = layer(x, mask, drank)
        out = _bl(x, self.out_w, self.out_b)
        out = out * (gc * valid.view(M, B, P).to(gc.dtype))[..., None]
        delta = torch.zeros(B * T, d, device=h.device, dtype=out.dtype)
        delta.index_add_(0, flat.reshape(-1), out.reshape(-1, d))
        return delta.view(B, T, d).to(h.dtype), aux

    # ---- incremental decode (loop over cores; correctness over speed) ----
    def init_ring(self, B, device, dtype=torch.float32):
        dc, K, L = self.cfg.d_core, self.cfg.K, self.cfg.n_core_layers
        M = self.M
        return {"x": torch.zeros(M, L, B, K, dc, device=device, dtype=dtype),
                "rank": torch.zeros(M, B, K, dtype=torch.long, device=device),
                "count": torch.zeros(M, B, dtype=torch.long, device=device)}

    def step(self, h, ring):
        """h (B,d) -> summed delta (B,d). Mutates ring. All cores gate on the
        SAME h (deltas sum), matching the forward path."""
        B, d = h.shape
        K = self.cfg.K
        s, m, g = self.gate(h)                      # (B,M)
        delta = torch.zeros_like(h)
        if not m.any():
            return delta
        ar = torch.arange(K, device=h.device)
        for mi in range(self.M):
            m_i, g_i = m[:, mi], g[:, mi]
            if not m_i.any():
                continue
            pos = ring["count"][mi] % K
            bidx = m_i.nonzero(as_tuple=True)[0]
            xn = F.layer_norm(h, (self.d_model,)) * self.ln_w[mi] + self.ln_b[mi]
            x = xn @ self.in_w[mi] + self.in_b[mi]          # (B, dc)
            my_rank = ring["count"][mi].clone()
            ring["x"][mi][0][bidx, pos[bidx]] = x[bidx]
            ring["rank"][mi][bidx, pos[bidx]] = my_rank[bidx]
            ring["count"][mi] = ring["count"][mi] + m_i.long()
            valid = ar[None, :] < ring["count"][mi].clamp(max=K)[:, None]
            mask = valid.clone()
            mask[:, 0] |= ~valid.any(dim=1)
            for li, layer in enumerate(self.layers):
                a = layer.attend_one(mi, x[:, None, :], ring["x"][mi][li],
                                     mask[:, None, :], my_rank[:, None],
                                     ring["rank"][mi])
                x = x + layer.ffn_one(mi, a)[:, 0, :]
                if li + 1 < len(self.layers):
                    ring["x"][mi][li + 1][bidx, pos[bidx]] = x[bidx]
            out = x @ self.out_w[mi] + self.out_b[mi]
            delta = delta + out * (g_i * m_i.float())[:, None]
        return delta

    # ---- interop with single-Core modules ----
    def core_state_dict(self, mi):
        """state_dict for core `mi` loadable by `Core(d_model, cfg)`."""
        sd = {"k_dir": self.k_dir[mi].detach().clone(),
              "tau": self.tau[mi].detach().clone(),
              "ln.weight": self.ln_w[mi].detach().clone(),
              "ln.bias": self.ln_b[mi].detach().clone(),
              "in_proj.weight": self.in_w[mi].detach().t().contiguous(),
              "in_proj.bias": self.in_b[mi].detach().clone(),
              "out_proj.weight": self.out_w[mi].detach().t().contiguous(),
              "out_proj.bias": self.out_b[mi].detach().clone()}
        for li, lay in enumerate(self.layers):
            p = f"layers.{li}."
            for name, w, b in (("q", lay.q_w, lay.q_b), ("k", lay.k_w, lay.k_b),
                               ("v", lay.v_w, lay.v_b),
                               ("ffn.0", lay.f1_w, lay.f1_b),
                               ("ffn.2", lay.f2_w, lay.f2_b)):
                sd[p + name + ".weight"] = w[mi].detach().t().contiguous()
                sd[p + name + ".bias"] = b[mi].detach().clone()
            sd[p + "rel_bias"] = lay.rel_bias[mi].detach().clone()
        return sd
