"""Resident-set logic.

`resident_mask_reference` is the SPEC (O(T^2), tests only).
`pack_indices` supports the fast gather -> sliding-window -> scatter path: on
the packed sequence of passers, the resident condition is exactly causal
sliding-window attention with window K, restricted to slots from the same
batch row.

`compact_indices` is the older RECTANGULAR compaction (B, P) with
P = max passer count over rows. It is kept for the reference/mask tests; the
fast path uses `pack_indices` because the rectangle pays max-over-rows for
every row (measured P = 785 against a 128-token mean row: 6.1x of the core's
arithmetic spent on padding).
"""
import torch


def resident_mask_reference(m: torch.Tensor, K: int) -> torch.Tensor:
    """m: (T,) bool -> (T, T) bool where [i, j] = j is resident at time i."""
    T = m.shape[0]
    cnt = m.long().cumsum(0)
    j_leq_i = torch.tril(torch.ones(T, T, dtype=torch.bool, device=m.device))
    within_k = (cnt[:, None] - cnt[None, :]) < K
    both = m[:, None] & m[None, :]
    return both & j_leq_i & within_k


def compact_indices(m: torch.Tensor):
    """m: (B, T) bool -> (idx, valid) where idx (B, P) gathers passers to the
    front in original order and valid (B, P) marks real entries. P = max
    passer count in the batch (>= 1 so downstream shapes never go empty)."""
    B, T = m.shape
    order = torch.argsort((~m).to(torch.int8), dim=1, stable=True)
    counts = m.sum(dim=1)
    P = max(int(counts.max().item()), 1)
    idx = order[:, :P]
    valid = torch.arange(P, device=m.device)[None, :] < counts[:, None]
    return idx, valid


def pack_indices(m: torch.Tensor):
    """Flat (varlen) packing. m: (G, B, T) bool -> (flat, row, valid).

    G is the number of cores packed together (1 for a single `Core`, M for a
    `MultiCore`). Core g's admitted tokens across the WHOLE batch go into ONE
    contiguous buffer, ordered by (b, then t) — batch rows laid end to end,
    in order within a row — instead of each row being padded out to a
    rectangle of the widest row's width.

      flat  (G, Npad) long  index into the FLATTENED (B*T) token axis (b*T+t)
      row   (G, Npad) long  which batch row b the slot came from
      valid (G, Npad) bool  slot holds a real admitted token

    Npad = max over g of core g's TOTAL admitted count (>= 1 so downstream
    shapes never go empty). That total is a SUM over B rows, and sums have far
    lower relative spread than maxima do, so the padding here is the gap
    between cores' totals — measured at 1% — where the rectangle's was the
    gap between the widest row and the mean row (measured 170-610%).

    Rows land contiguously, so within a row the packed-slot difference IS the
    passer-rank difference (what the rank-relative bias indexes). The row
    separation the rectangle got from its shape is now bookkeeping: attention
    must additionally require row[query] == row[key] (see
    `core_module._banded_attend`).

    Padding slots carry flat=0, row=0, valid=False; they are masked out of
    attention and multiplied by 0 before the scatter, so they contribute
    nothing.
    """
    G, B, T = m.shape
    N = B * T
    mf = m.reshape(G, N)
    cum = mf.long().cumsum(1)                    # admitted count through slot i
    counts = cum[:, -1]                          # (G,) total per core
    Npad = max(int(counts.max()), 1)
    pos = torch.arange(N, device=m.device)
    # A two-bucket counting sort: admitted tokens keep their flat order at the
    # front (rank cum-1), everything else lands past Npad and is sliced off.
    # Every destination is distinct, so the scatter is a permutation.
    dest = torch.where(mf, cum - 1, Npad + pos[None, :] - cum)
    buf = torch.zeros(G, Npad + N, dtype=torch.long, device=m.device)
    buf.scatter_(1, dest, pos[None, :].expand(G, N))
    # slots [counts[g], Npad) were never written, so they are already 0
    flat = buf[:, :Npad]
    valid = pos[:Npad][None, :] < counts[:, None]
    return flat, torch.div(flat, T, rounding_mode="floor"), valid


def window_mask(P: int, K: int, valid: torch.Tensor) -> torch.Tensor:
    """(B, P, P) bool attention mask on the compacted sequence: causal window
    K over valid keys; the diagonal is always allowed so no softmax row is
    empty (invalid rows are zeroed after scatter anyway)."""
    device = valid.device
    i = torch.arange(P, device=device)
    causal_win = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] < K)
    mask = causal_win[None, :, :] & valid[:, None, :]
    eye = torch.eye(P, dtype=torch.bool, device=device)
    return mask | eye[None, :, :]
