"""M4: inspect what the gate learned to admit.

Loads the trained core run, generates MQAR batches, and reports admission
rate by token type (filler / pair-key / pair-value / query-key). Writes a
selection-map figure to figures/selection.png.

Gate for M4: admissions track key/value tokens, not position or noise.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.m3_mechanism import make_task, make_cfg
from core import SWTransformer


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/m3_core_s0/latest.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = make_task()
    model = SWTransformer(make_cfg(task, "core")).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    torch.manual_seed(123)
    idx, labels, gap_map = task.gen_batch(64, device=device)
    with torch.no_grad():
        _, auxes = model(idx, collect_aux=True)
    m = auxes[0]["m"]  # (B, T) admissions

    is_key = (idx >= task.n_filler) & (idx < task.n_filler + task.n_keys)
    is_val = idx >= task.n_filler + task.n_keys
    is_query = labels != -100
    types = {"filler": ~(is_key | is_val),
             "pair_key": is_key & ~is_query,
             "value": is_val,
             "query_key": is_query}
    rates = {name: float(m[sel].float().mean()) for name, sel in types.items()}
    print(json.dumps({"admission_rate_by_type": rates,
                      "overall_rate": float(m.float().mean())}, indent=2))

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    tt = torch.zeros_like(idx)
    tt[types["pair_key"]] = 1
    tt[types["value"]] = 2
    tt[types["query_key"]] = 3
    ax[0].imshow(tt[:16].cpu(), aspect="auto", interpolation="nearest",
                 cmap="viridis")
    ax[0].set_ylabel("token type\n(0=filler 1=key 2=val 3=query)")
    ax[1].imshow(m[:16].cpu(), aspect="auto", interpolation="nearest",
                 cmap="gray")
    ax[1].set_ylabel("admitted")
    ax[1].set_xlabel("position")
    fig.suptitle("Core admissions vs token types (16 sequences)")
    fig.tight_layout()
    fig.savefig("figures/selection.png", dpi=120)
    print("wrote figures/selection.png")


if __name__ == "__main__":
    main()
