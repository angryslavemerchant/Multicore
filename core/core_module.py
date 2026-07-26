"""A Core: threshold-gated private mini-transformer over a FIFO of admitted
tokens.

The core is an n_layer stack of (attention over residents + FFN) applied on
the PACKED passer sequence, with a learned rank-relative attention bias
(delta-rank in passer space, 0..K-1). On that packed stream the resident
set of rank p is exactly ranks p-K+1..p, so prefill attention is BANDED and
runs block-wise (see `_band_layout`): O(P*K), never O(P*P). The dense P x P
mask/bias path survives only where the band does not apply -- the O(T^2)
reference and decode's 1-query-vs-K-ring step.

PACKED, not padded: the batch's admitted tokens go into ONE flat buffer per
core (`resident.pack_indices`), rows laid end to end, rather than a (B, P)
rectangle whose width is the widest row's passer count. A rectangle costs
max-over-rows for EVERY row, and gate scores are not evenly spread over a
batch -- measured on an imbalanced batch, P = 785 against a mean row of 128,
i.e. 6.1x of the core's arithmetic (and memory) spent on padding that the
mask then throws away. Packing costs sum-over-rows, which is the work
actually needed; what is left over is the spread between the M cores'
TOTALS, measured at 1%. The separation the rectangle got from its shape
becomes one extra mask term: a slot may only attend slots with the same row
id (`_banded_attend`). Rows are contiguous in the buffer, so within a row
the packed-slot difference is still exactly the passer-rank difference the
rel_bias indexes. Two layers + rank bias let the core
express the induction circuit internally: layer 1 binds each admitted token
to its rank-neighbours (e.g. a value to its adjacent key), layer 2 matches
those bindings and retrieves. A single-layer bag-of-residents core cannot do
this unless binding already exists in the base representations (measured:
it mostly doesn't — oracle-gated single-layer cores barely beat chance).

Commit semantics for multi-layer: resident j's layer-l state is computed AT
j's admission over the residents present then, and never recomputed. Layer
l+1 at a later token i attends over those committed layer-l states. In
prefill this is just l stacked causal sliding-window attentions on the
packed sequence; in decode each admitted token computes its states from
the ring and stores them.

Gating: hard membership (quantile-controller tau), soft magnitude
g = sigmoid((s - tau)/temp) on members (gradient path to k_dir).
Output projection is zero-initialised: deltas are exactly 0 at init.

`MultiCore` computes M cores that share a CoreConfig (same K, d_core,
n_heads, n_core_layers, ffn_mult, target_rate) in one batched pass: every
weight carries a leading core dimension and the core index IS the batch dim
of the packing / attention / scatter (one packed buffer per core, so the
packed tensor is (M, Npad, dc)). The math is identical to M independent
`Core`s whose deltas SUM into h — only the batching changes. Sequential
cores are launch-bound (8 identical cores ran at 128k tok/s vs 307k for 2),
which is what this exists to fix.

Gate-direction collapse (`orthonormalize_rows_`, `MultiCore.reproject_gates`):
M independently-initialised `k_dir` rows do NOT stay distinct. Measured on a
real 11k-step joint run, all eight collapsed onto one direction (pairwise
|cos| mean 0.999, min 0.999) — eight cores admitting one token set, i.e. one
core replicated, which is why adding cores or core params bought no loss.
CORE_ROUTING_PLAN.md 5.9 prescribes making collapse structurally impossible
rather than penalised; `reproject_gates()` re-orthogonalises the DIRECTIONS
after each optimizer step and leaves each row's NORM alone (see the function
docstring for why both halves matter).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CoreConfig
from .resident import resident_mask_reference, pack_indices


# --------------------------------------------------- gate-direction geometry
@torch.no_grad()
def orthonormalize_rows_(w):
    """In place: make the rows of `w` (n, d), n <= d, mutually ORTHOGONAL,
    preserving every row's original norm. Returns `w`.

    Direction only, on purpose. The gate is hard membership `s > tau` plus a
    soft magnitude `sigmoid((s - tau)/gate_temp)`, so ||k_dir[c]|| sets how
    sharply core c's magnitude saturates — normalise it away and every core
    loses its learned gate sharpness. Collapse is a statement about direction,
    so that is all this constrains.

    QR of `w.T` gives Q (d, n) with orthonormal columns, and column i is the
    Gram-Schmidt residual of row i against rows 0..i-1: row 0 does not move at
    all and row i moves as little as the constraint allows, which matters when
    this runs every optimizer step against live Adam moments. LAPACK picks Q's
    column signs by its own convention and a flipped sign would inverse the
    gate (admitting exactly the tokens it used to reject); diag(R) carries that
    sign, so undo it. A rank-deficient row gives diag(R) == 0 — no sign to
    undo, and LAPACK's arbitrary completion of the basis is exactly the escape
    from collapse we want, so 0 maps to +1.

    Chosen over `torch.nn.utils.parametrizations.orthogonal`; see
    `MultiCore.reproject_gates`.
    """
    n, d = w.shape
    assert n <= d, f"need n <= d for orthogonal gate directions, got {n} > {d}"
    nrm = w.norm(dim=1, keepdim=True)
    q, r = torch.linalg.qr(w.detach().float().t())          # (d,n), (n,n)
    sgn = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
    return w.copy_((q * sgn).t().to(w.dtype) * nrm)


def _quantile_lerp(x, q, dim=0):
    """`torch.quantile(x, q, dim)` (default 'linear' interpolation), via sort.

    BIT-identical, not merely close: quantile is *defined* as the lerp between
    the two order statistics either side of index q*(n-1), and the arithmetic
    here is ATen's own — rank computed in x's dtype (not double, which shifts
    the weight by ~1e-7 and so the returned tau), then `torch.lerp`. The TQ
    gate in tests/test_invariants.py holds it to `torch.equal`.

    It exists because `torch.quantile`'s kernel calls `numel()` on its input,
    which raises "Cannot call numel() on tensor with symbolic sizes" under
    dynamo — so the tau controller silently killed `--compile` for every
    multi-core preset (m5 compiles with dynamic=True on purpose: the packed
    passer buffer's length is data-dependent, so static shapes would recompile
    every step).

    `int(x.shape[dim])` guards on the scored axis (B*T, fixed within a run) and
    leaves the packed axis — the one that actually varies — dynamic.
    """
    n = int(x.shape[dim])
    srt = x.sort(dim=dim).values
    rank = float(torch.tensor(q, dtype=x.dtype) * (n - 1))
    lo = math.floor(rank)
    return torch.lerp(srt.select(dim, lo), srt.select(dim, min(lo + 1, n - 1)),
                      rank - lo)


def _rms(t):
    return t.detach().float().pow(2).mean().sqrt()


def _delta_diag(aux, h, delta, group=1):
    """Section-10 "delta norm / residual norm" into an aux dict.

    RMS of the core delta against the RMS of the residual stream it is added
    to: near zero says the core is dead weight, comparable to the residual
    says it is destabilising the base. Zero-init out_proj and the soft gate are
    two factors escaping zero together, so this needs watching early.

    Detached 0-dim tensors (never `.item()` — that would sync and break the
    compiled graph); two reductions over (B,T,d), cheap enough for every step.
    `group` is how many aux dicts share one delta (M for MultiCore, whose M
    cores sum into a single delta) so a consumer adding up per-core
    contributions does not count the same delta M times.
    """
    aux["h_rms"] = _rms(h)
    aux["delta_rms"] = (torch.zeros((), device=h.device) if delta is None
                        else _rms(delta))
    aux["delta_group"] = group
    return aux


# ------------------------------------------------------- banded attention
def _band_layout(P: int, K: int, device):
    """Block-banded layout for the compacted passer stream.

    On the compacted stream the resident set of query rank p is exactly the
    K ranks p-K+1 .. p, so the attention matrix is a band of width K and a
    dense P x P mask/bias is pure waste. We walk the band in BLOCKS of
    C = K queries: block i owns query ranks [i*C, i*C+C) and those ranks
    need exactly the contiguous key window [i*C-(K-1), i*C+C), of length
    W = C+K-1. Every tensor downstream is O(P*K); no P x P tensor exists.

    (Per-QUERY gathering of a (P, K, d_head) band would also be "O(P*K)" in
    slots but costs K*d_head floats per query -- 4096 for the K=64 /
    d_head=64 presets, i.e. worse than the P=1500 dense row it replaces.
    Blocking amortises the key gather over C queries instead.)

    Returns (nb, C, W, Pp, jc, okj, band, dlt, slf):
      nb   number of blocks, Pp = nb*C the block-padded rank axis
      jc   (nb*W,) long  key rank of each window slot, clamped into [0, Pp)
      okj  (nb, W) bool  window slot maps to a real rank (j >= 0)
      band (C, W) bool   delta-rank = qi - w + K-1 lies in [0, K-1]
      dlt  (C, W) long   that delta-rank, for the rel_bias lookup
      slf  (C, W) bool   delta-rank == 0, i.e. the diagonal j == p
    """
    C = K
    W = C + K - 1
    nb = (P + C - 1) // C
    Pp = nb * C
    j = (torch.arange(nb, device=device)[:, None] * C - (K - 1)
         + torch.arange(W, device=device)[None, :])          # (nb, W)
    qi = torch.arange(C, device=device)[:, None]
    dlt = qi - torch.arange(W, device=device)[None, :] + (K - 1)
    return (nb, C, W, Pp, j.clamp(0, Pp - 1).reshape(-1), j >= 0,
            (dlt >= 0) & (dlt < K), dlt.clamp(0, K - 1), dlt == 0)


def _banded_attend(q, k, v, valid, row, rel_bias, K, G):
    """Causal window-K attention on the PACKED passer stream, banded.

    q, k, v  (N, H, P, dh) heads-first; valid (N, P) bool marks real
    passers; row (N, P) long is the batch row each packed slot came from;
    rel_bias (G, H, K) indexed by delta-rank, with N a whole multiple of G
    and rows laid out g-major (N = G = 1 for a single Core, N = G = M for
    MultiCore, one packed buffer per core). Returns (N, H, P, dh).

    Key slot j is attended from query slot p iff 0 <= p-j < K, j is a real
    passer, and row[j] == row[p]; the diagonal p == j is always allowed so no
    softmax row is empty (an all -inf row would come back NaN). Since the
    packing lays each batch row down contiguously, p-j inside a row IS the
    passer-rank difference, so this is exactly the per-row
    `window_mask(P, K, valid)` + rel_bias the rectangle gave — the row
    separation just moved from the tensor shape into the mask.
    """
    N, H, P, dh = q.shape
    nb, C, W, Pp, jc, okj, band, dlt, slf = _band_layout(P, K, q.device)
    q = q * (dh ** -0.5)
    if Pp != P:                       # pad the rank axis to whole blocks
        pad = (0, 0, 0, Pp - P)
        q, k, v = F.pad(q, pad), F.pad(k, pad), F.pad(v, pad)
        valid = F.pad(valid, (0, Pp - P))
        row = F.pad(row, (0, Pp - P), value=-1)   # matches no real row
    qb = q.reshape(N, H, nb, C, dh)
    kb = k.index_select(2, jc).view(N, H, nb, W, dh)
    vb = v.index_select(2, jc).view(N, H, nb, W, dh)
    logits = torch.einsum('nhicd,nhiwd->nhicw', qb, kb)
    bias = rel_bias[..., dlt].to(logits.dtype)               # (G, H, C, W)
    logits = (logits.view(G, N // G, H, nb, C, W) + bias[:, None, :, None]) \
        .view(N, H, nb, C, W)
    vk = valid.index_select(1, jc).view(N, nb, W)            # key is a passer
    rk = row.index_select(1, jc).view(N, nb, W)
    same = row.reshape(N, nb, C)[:, :, :, None] == rk[:, :, None, :]
    ok = (band & okj[:, None, :] & vk[:, :, None, :] & same) | slf
    logits.masked_fill_(~ok[:, None], float("-inf"))         # (N, nb, C, W)
    a = torch.softmax(logits.float(), dim=-1).to(vb.dtype)
    out = torch.einsum('nhicw,nhiwd->nhicd', a, vb)
    return out.reshape(N, H, Pp, dh)[:, :, :P]


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

    def attend_banded(self, x, valid, row):
        """Prefill on the PACKED passer stream: queries and keys are the same
        P slots, so the mask is exactly the K-wide causal band (intersected
        with same-row) and we never build (P, P). x (1,P,dc), valid (1,P)
        bool, row (1,P) long."""
        B, P, _ = x.shape
        out = _banded_attend(self._heads(self.q(x)), self._heads(self.k(x)),
                             self._heads(self.v(x)), valid, row,
                             self.rel_bias[None], self.rel_bias.shape[1], 1)
        return out.transpose(1, 2).reshape(B, P, -1)

    def forward_banded(self, x, valid, row):
        return x + self.ffn(self.attend_banded(x, valid, row))

    def attend(self, xq, xkv, mask, dq, dkv):
        """Dense-mask attention. Only for the two paths where the band does
        not apply: the O(T^2) reference (arbitrary resident mask over FULL
        positions) and decode (Pq=1 query against the K-entry ring). The
        prefill path uses attend_banded.

        xq (B,Pq,dc) queries at ranks dq (B,Pq); xkv keys/values at ranks
        dkv (B,Pk); mask (B,Pq,Pk) bool."""
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
        """Dense full-sequence path (O(T^2)); used by forward_reference."""
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
                q = _quantile_lerp(s.detach().float().flatten(),
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
            return torch.zeros_like(h), _delta_diag(aux, h, None)
        # ONE flat buffer of every admitted token in the batch (rows laid end
        # to end), not a (B, P) rectangle padded to the widest row.
        flat, row, valid = pack_indices(m[None])         # (1, Npad)
        hf = h.reshape(B * T, d)
        hc = hf.index_select(0, flat[0])[None]           # (1, Npad, d)
        gc = g.reshape(B * T).index_select(0, flat[0])[None]
        x = self.in_proj(self.ln(hc))
        for layer in self.layers:
            x = layer.forward_banded(x, valid, row)
        out = self.out_proj(x) * (gc * valid.to(gc.dtype))[:, :, None]
        delta = torch.zeros(B * T, d, device=h.device, dtype=out.dtype)
        delta.index_add_(0, flat[0], out[0])
        delta = delta.view(B, T, d).to(h.dtype)
        return delta, _delta_diag(aux, h, delta)

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
        return delta, _delta_diag(aux, h, delta)

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
    """Batched-over-cores linear: x (M,P,i) @ w (M,i,o) + b (M,o). P is the
    PACKED slot axis — one buffer per core, the whole batch inside it."""
    return torch.einsum('mpi,mio->mpo', x, w) + b[:, None, :]


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
    def _heads(self, t, M, P):
        return t.view(M, P, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def attend(self, x, valid, row):
        """x (M,P,dc); valid (M,P) bool; row (M,P) long. Same math as
        _CoreLayer.attend_banded with the core index in the batch slot: the
        band is walked in blocks, so nothing here is ever (P, P)."""
        M, P, dc = x.shape
        q = self._heads(_bl(x, self.q_w, self.q_b), M, P)
        k = self._heads(_bl(x, self.k_w, self.k_b), M, P)
        v = self._heads(_bl(x, self.v_w, self.v_b), M, P)
        out = _banded_attend(q, k, v, valid, row, self.rel_bias, self.K, M)
        return out.permute(0, 2, 1, 3).reshape(M, P, dc)

    def ffn(self, a):
        return _bl(F.gelu(_bl(a, self.f1_w, self.f1_b)), self.f2_w, self.f2_b)

    def forward(self, x, valid, row):
        return x + self.ffn(self.attend(x, valid, row))

    # ---- single-core slice (decode path) ----
    def attend_one(self, mi, xq, xkv, mask, dq, dkv):
        """Exactly _CoreLayer.attend for core mi -- dense, but decode only
        ever runs it as 1 query against the K-entry ring. xq (B,Pq,dc),
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

    @torch.no_grad()
    def reproject_gates(self):
        """Re-orthogonalise the M gate directions in place (norms preserved).

        Call after `optimizer.step()`. The step's update to `k_dir` is small
        relative to `k_dir` itself, so the projection back onto the orthogonal
        set is also small and Adam's moments stay approximately valid — the
        same bargain projected-gradient methods make. No-op for M == 1.

        Why this and not `torch.nn.utils.parametrizations.orthogonal`, which
        was tried first (torch 2.10, RTX 2060, the exact autocast + compile
        setup m5 trains under):
          1. it produces an orthoNORMAL factor — every row norm becomes 1.0
             (measured: 19.5..33.2 -> all 1.0), destroying the per-core gate
             sharpness this deliberately keeps;
          2. AdamW's decoupled weight decay lands on the Householder
             *generator* `parametrizations.k_dir.original`, not on anything
             with a meaning, and shrinking reflectors toward zero makes
             `householder_product` degenerate: at m5's own wd=0.1, lr=3e-4
             the materialised matrix was NaN within 200 steps (wd=0 was fine);
          3. under bf16 autocast the factorisation itself runs in bf16 and
             orthogonality degrades from |cos| 4.7e-8 to 2.6e-4;
          4. it renames the parameter to `parametrizations.k_dir.original`,
             which breaks `core_state_dict()` interop with `Core` and every
             checkpoint already on disk.
        torch.compile handled it fine — it was the other four that decided it.
        """
        if self.M > 1:
            orthonormalize_rows_(self.k_dir.data)

    # ---- gate ----
    def gate(self, h):
        """h (...,d) -> s,m,g each (...,M). Per-core exact quantile tau."""
        s = torch.einsum('...d,md->...m', h, self.k_dir)
        if self.training:
            with torch.no_grad():
                q = _quantile_lerp(s.detach().float().reshape(-1, self.M),
                                   1.0 - self.cfg.target_rate, dim=0)
                self.tau.copy_(q)
        m = s > self.tau
        g = torch.sigmoid((s - self.tau) / self.cfg.gate_temp)
        return s, m, g

    def _aux(self, m):
        return [{"rate": m[..., i].float().mean().detach(),
                 "tau": self.tau[i].detach().clone(), "m": m[..., i].detach()}
                for i in range(self.M)]

    def _aux_delta(self, aux, h, delta):
        """`_delta_diag` for all M dicts off ONE pair of reductions. The M
        cores sum into a single delta, so the value is shared (group = M)."""
        hr = _rms(h)
        dr = (torch.zeros((), device=h.device) if delta is None
              else _rms(delta))
        for a in aux:
            a["h_rms"], a["delta_rms"], a["delta_group"] = hr, dr, self.M
        return aux

    def _norm_in(self, hc):
        """hc (M,P,d) -> (M,P,dc) with per-core LN affine + in_proj."""
        x = F.layer_norm(hc, (self.d_model,))
        x = x * self.ln_w[:, None, :] + self.ln_b[:, None, :]
        return _bl(x, self.in_w, self.in_b)

    def forward(self, h, m_override=None):
        """h (B,T,d) -> (summed delta (B,T,d), list of M aux dicts)."""
        B, T, d = h.shape
        M = self.M
        s, m, g = self.gate(h)                      # (B,T,M)
        if m_override is not None:
            m = m_override[..., None].expand(B, T, M)
            g = torch.ones_like(g)
        aux = self._aux(m)
        if not m.any():
            return torch.zeros_like(h), self._aux_delta(aux, h, None)
        # one flat buffer per core: (M, Npad) into the (B*T) token axis. Npad
        # is a per-core TOTAL over the batch, not a per-row max, so the M
        # buffers are near-equal in length and cross-core padding is ~1%.
        mp = m.permute(2, 0, 1)                     # (M,B,T)
        flat, row, valid = pack_indices(mp)         # (M,Npad)
        fl = flat.reshape(-1)
        P = flat.shape[1]
        hc = h.reshape(B * T, d).index_select(0, fl).view(M, P, d)
        gc = g.permute(2, 0, 1).reshape(M, B * T).gather(1, flat)
        x = self._norm_in(hc)
        for layer in self.layers:
            x = layer(x, valid, row)
        out = _bl(x, self.out_w, self.out_b)
        out = out * (gc * valid.to(gc.dtype))[..., None]
        delta = torch.zeros(B * T, d, device=h.device, dtype=out.dtype)
        delta.index_add_(0, fl, out.reshape(-1, d))
        delta = delta.view(B, T, d).to(h.dtype)
        return delta, self._aux_delta(aux, h, delta)

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
