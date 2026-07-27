"""Score a published HuggingFace causal LM on OUR eval set.

This is what makes an external baseline usable. Published numbers are on the
publisher's held-out split, at their sequence length, with their tokenizer and
their loss reduction -- a table of theirs and a table of ours are two
measurements, not one comparison, and the difference between them is not
small. Running their weights over the identical windows of the identical file
that our runs are evaluated on removes every one of those degrees of freedom
at once. What is left is the models.

Default target: open-sci-ref-v0.01-0.13b-fineweb-edu-1.4t-300B-4096 --
22 layers, d 512, 8 heads, FFN 2256 SwiGLU, RMSNorm, per-head qk-norm, RoPE
10k, tied embeddings, GPT-NeoX vocab 50304, sequence 4096, trained on
FineWeb-Edu with a WSD schedule (lr 4e-3, 25k warmup) at 1008x4096 = 4.13M
tokens per step. 37 revisions are published: `main` plus iter_0002000 ..
iter_0072000 every 2000 steps, i.e. 8.3B tokens per checkpoint step -- so
`--revision iter_0002000` is the one to compare a 8.3B-token run against, not
`main` (300B).

The model ships its own modelling code (`model_type: opensci`, an `auto_map`
in config.json), which is why `trust_remote_code=True` is required: from_
pretrained downloads and imports modeling_opensci.py from the repo. Read it
if that matters to you; it is a LLaMA-shaped decoder with qk-norm.

TRANSFORMERS VERSION. That downloaded module was written against transformers
4.49 and is frozen at whatever the library looked like on publication day. It
does not import under 5.x -- `ImportError: cannot import name 'LossKwargs'`,
observed 2026-07-27 against transformers 5.14.1. Give it its own environment
rather than pinning the whole project back:

  /venv/main/bin/python -m venv --system-site-packages /workspace/tfv4
  /workspace/tfv4/bin/pip install "transformers==4.49.0"
  /workspace/tfv4/bin/python scripts/score_ref.py --revision iter_0002000

`--system-site-packages` so it reuses the image's torch instead of pulling a
second multi-GB copy. m5_suite takes --score-ref-python for exactly this, and
treats a failure here as non-fatal: this is a supplementary anchor and must
never take down a training ladder.

  python scripts/score_ref.py --revision iter_0002000 --data-shards 11
  python scripts/score_ref.py --model EleutherAI/pythia-160m --batch 4
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from m5_arch import IND_REF_WINDOW, ce_chunk_default, stream_batches
from m5_data import DEFAULT_SHARDS, open_data

DEFAULT_MODEL = "open-sci/open-sci-ref-v0.01-0.13b-fineweb-edu-1.4t-300B-4096"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=None,
                    help="hub branch: iter_0002000 (8.3B tokens) .. "
                         "iter_0072000, or main (300B). Compare against the "
                         "checkpoint at YOUR token budget, not the final one.")
    # The eval windows are drawn by ONE seeded generator that emits B starts
    # at a time, so the set of positions scored is a function of
    # (seed, batch, eval_batches, seq_len, data_shards) TOGETHER -- change any
    # of them and this scores a different sample of the corpus. All five must
    # equal the m5_arch run being compared against; --batch is not a free
    # performance knob here the way it is during training.
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ind-window", type=int, default=IND_REF_WINDOW)
    ap.add_argument("--data-shards", type=int, default=DEFAULT_SHARDS)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None, help="write metrics json here")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.to(device).eval()
    V = int(model.config.vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight
    n_body = n_params - emb.numel()
    if model.get_output_embeddings() is not None and \
            model.get_output_embeddings().weight is not emb:
        n_body -= model.get_output_embeddings().weight.numel()
    print(f"{args.model}@{args.revision or 'main'}: {n_params/1e6:.1f}M params "
          f"({n_body/1e6:.1f}M non-embedding), vocab {V}, loaded in "
          f"{time.time() - t0:.0f}s on {device}", flush=True)
    assert V > 256, "this scores TOKEN models; the byte corpus is not comparable"

    data = open_data(True, args.data_shards, args.data_dir)
    stream = stream_batches(args.batch, args.seq_len, args.ind_window, device,
                            seed=args.seed + 999, with_masks=True,
                            split="eval", data=data, vocab_size=V)
    chunk = ce_chunk_default(V)
    print(f"[ref] scoring the eval windows of seed={args.seed}, "
          f"batch={args.batch}, eval_batches={args.eval_batches}, "
          f"seq_len={args.seq_len}, data_shards={args.data_shards} -- all five "
          f"must match the run this is compared against", flush=True)

    tot, tot_n, ind, ind_n = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            idx, mask = next(stream)
            assert int(idx.max()) < V, (
                f"token id {int(idx.max())} exceeds this model's vocab {V}: "
                f"the cache was built with a different tokenizer")
            logits = model(idx[:, :-1]).logits
            tgt = idx[:, 1:].reshape(-1)
            flat = logits.reshape(-1, logits.shape[-1])
            # chunked like m5_arch's eval, for the same reason: 4096x50304
            # fp32 logits are 824 MB per sequence
            c = chunk if chunk > 0 else flat.shape[0]
            ce = torch.cat([
                F.cross_entropy(flat[i:i + c].float(), tgt[i:i + c],
                                reduction="none")
                for i in range(0, flat.shape[0], c)])
            tot += float(ce.sum()); tot_n += ce.numel()
            sel = mask.reshape(-1)
            ind += float(ce[sel].sum()); ind_n += int(sel.sum())

    m = {"model": args.model, "revision": args.revision or "main",
         "params": n_params, "params_non_embedding": n_body, "vocab_size": V,
         "seq_len": args.seq_len, "seed": args.seed,
         "eval_batches": args.eval_batches, "batch": args.batch,
         "ind_window": args.ind_window, "ind_units": "tokens",
         "eval_loss": tot / max(tot_n, 1),
         "eval_loss_induction": ind / max(ind_n, 1),
         "eval_induction_frac": ind_n / max(tot_n, 1),
         "eval_ppl": float(torch.tensor(tot / max(tot_n, 1)).exp())}
    print(json.dumps(m, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)


if __name__ == "__main__":
    main()
