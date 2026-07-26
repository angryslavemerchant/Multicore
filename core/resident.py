"""Resident-set logic.

`resident_mask_reference` is the SPEC (O(T^2), tests only).
`compact_indices` supports the fast gather -> sliding-window -> scatter path:
on the compacted sequence of passers, the resident condition is exactly causal
sliding-window attention with window K.
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
