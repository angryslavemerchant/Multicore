"""Chunked next-token cross-entropy, computed from hidden states + head weight.

Lives in `core` rather than in the training script for one reason: under
DistributedDataParallel, DDP registers its gradient hooks on the parameters
used inside the WRAPPED module's forward. A loss computed outside that forward
touches `head.weight` where DDP cannot see it, so that gradient is never
all-reduced (and with tied embeddings it is worse than that -- the hook for the
shared tensor can fire before the outside contribution lands, which is a race,
not a missing term). Putting the loss inside `SWTransformer.forward` is what
makes multi-GPU correct, so the loss has to be importable from `core`.

The chunking itself exists for memory: at vocab 50304 and T=4096 the fp32
logits are 824 MB per sequence and bound the batch long before activations do.
"""
import torch
import torch.nn.functional as F


def ce_chunk_default(vocab_size):
    """Rows per chunk. Chunking is pure overhead at vocab 256 (4 MB of logits
    per sequence) and is not optional at vocab 50304 (824 MB), so the default
    follows the vocabulary rather than waiting to be remembered on a CLI."""
    return 0 if vocab_size <= 1024 else 1024


def _ce_chunk(h, w, t):
    return F.cross_entropy(F.linear(h, w).float(), t, reduction="sum")


def ce_sum(h, w, targets, chunk=0):
    """Summed next-token cross-entropy for logits = h @ w.T, without ever
    materialising the full (N, V) fp32 logit tensor.

    h (N, d), w (V, d), targets (N,). Chunking the rows and recomputing each
    chunk's logits in backward (torch.utils.checkpoint) caps the peak at one
    chunk -- 206 MB at chunk=1024 -- for one extra head matmul in backward,
    2*d*V per token. Memory bought with a modest amount of arithmetic.

    fp32 for the softmax is not the expensive part and is not negotiable: bf16
    has 8 mantissa bits, and a logsumexp over 50304 terms in bf16 loses the
    small differences the loss is made of. Training stays bf16 autocast; only
    this reduction is widened.

    chunk <= 0 is the plain path, identical to
    F.cross_entropy(logits.float(), targets, reduction="sum") -- and the right
    choice when there is VRAM to spare, since it skips the recompute.
    """
    N = h.shape[0]
    if chunk <= 0 or chunk >= N:
        return _ce_chunk(h, w, targets)
    from torch.utils.checkpoint import checkpoint
    total = None
    for i in range(0, N, chunk):
        part = checkpoint(_ce_chunk, h[i:i + chunk], w, targets[i:i + chunk],
                          use_reentrant=False)
        total = part if total is None else total + part
    return total


@torch.no_grad()
def ce_per_token(h, w, targets, chunk=0):
    """Per-position cross-entropy (N,). Eval only: the induction slice needs
    the per-position values, and with no graph to keep there is nothing to
    checkpoint -- chunking alone bounds the peak."""
    N = h.shape[0]
    if chunk <= 0 or chunk >= N:
        return F.cross_entropy(F.linear(h, w).float(), targets,
                               reduction="none")
    return torch.cat([
        F.cross_entropy(F.linear(h[i:i + chunk], w).float(),
                        targets[i:i + chunk], reduction="none")
        for i in range(0, N, chunk)])
