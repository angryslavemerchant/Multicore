"""Multi-query associative recall (MQAR-style) generator.

Sequence layout (length T): key->value pairs (adjacent tokens) scattered in
the context region [0, ctx_end); queries in [ctx_end, T). A query is the key
token; the label at that position is the paired value (which then appears in
the input at the next position, teacher-forced). Loss/accuracy only at query
positions. Gap = query position - value position of its pair: gaps <= window
are solvable by the sliding-window base; longer gaps require the core.

Vocab layout: [0, n_filler) filler, [n_filler, n_filler+n_keys) keys,
[n_filler+n_keys, n_filler+n_keys+n_vals) values.
"""
import torch


class MQAR:
    def __init__(self, T=512, n_pairs=8, n_queries=4, n_filler=64, n_keys=64,
                 n_vals=64, ctx_frac=0.75):
        self.T, self.n_pairs, self.n_queries = T, n_pairs, n_queries
        self.n_filler, self.n_keys, self.n_vals = n_filler, n_keys, n_vals
        self.ctx_end = int(T * ctx_frac)
        self.vocab_size = n_filler + n_keys + n_vals
        assert self.ctx_end >= 2 * n_pairs and T - self.ctx_end >= 2 * n_queries

    def key_token(self, k):
        return self.n_filler + k

    def val_token(self, v):
        return self.n_filler + self.n_keys + v

    def gen_batch(self, B, device="cpu", generator=None):
        """Returns idx (B,T), labels (B,T) (-100 off-query), gap_map (B,T)
        (pair->query gap at query positions, -1 elsewhere)."""
        T, P, Q = self.T, self.n_pairs, self.n_queries
        idx = torch.randint(0, self.n_filler, (B, T), generator=generator)
        labels = torch.full((B, T), -100, dtype=torch.long)
        gap_map = torch.full((B, T), -1, dtype=torch.long)
        for b in range(B):
            keys = torch.randperm(self.n_keys, generator=generator)[:P]
            vals = torch.randint(0, self.n_vals, (P,), generator=generator)
            slots = torch.randperm((self.ctx_end - 1) // 2,
                                   generator=generator)[:P] * 2
            for p in range(P):
                s = int(slots[p])
                idx[b, s] = self.key_token(keys[p])
                idx[b, s + 1] = self.val_token(vals[p])
            qs = torch.randperm((T - self.ctx_end - 1) // 2,
                                generator=generator)[:Q] * 2 + self.ctx_end
            which = torch.randint(0, P, (Q,), generator=generator)
            for qi in range(Q):
                qpos, p = int(qs[qi]), int(which[qi])
                idx[b, qpos] = self.key_token(keys[p])
                idx[b, qpos + 1] = self.val_token(vals[p])  # teacher forcing
                labels[b, qpos] = self.val_token(vals[p])
                gap_map[b, qpos] = qpos - (int(slots[p]) + 1)
        return idx.to(device), labels.to(device), gap_map.to(device)


DEFAULT_BUCKETS = ((0, 64), (64, 128), (128, 256), (256, 10 ** 9))


def eval_recall(model, task: MQAR, buckets=DEFAULT_BUCKETS, n_batches=8, B=64,
                device="cpu"):
    """Accuracy at query positions, bucketed by pair->query gap.
    Returns {"(lo,hi)": acc, ..., "overall": acc}."""
    stats = {bk: [0, 0] for bk in buckets}
    tot = [0, 0]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            idx, labels, gap_map = task.gen_batch(B, device=device)
            pred = model(idx).argmax(-1)
            mask = labels != -100
            correct = (pred == labels) & mask
            for (lo, hi) in buckets:
                sel = mask & (gap_map >= lo) & (gap_map < hi)
                stats[(lo, hi)][0] += int((correct & sel).sum())
                stats[(lo, hi)][1] += int(sel.sum())
            tot[0] += int(correct.sum())
            tot[1] += int(mask.sum())
    if was_training:
        model.train()
    out = {f"acc_gap_{lo}_{hi}": (c / n if n else float("nan"))
           for (lo, hi), (c, n) in stats.items()}
    out["acc_overall"] = tot[0] / max(tot[1], 1)
    return out
