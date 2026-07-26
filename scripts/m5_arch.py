"""M5 architecture test: from-scratch joint training on byte-level text.

Three matched configs (see CORE_ROUTING_PLAN.md section 4, phase 2):
  cores       — sliding-window base + hefty core(s). The proposal.
  dense_full  — full-attention transformer, PARAM-matched to `cores` total.
  dense_local — sliding-window transformer, FLOPs-matched to `cores`.

Primary metrics: loss vs tokens (logged with FLOPs/token so loss-vs-compute
can be plotted), and STRATIFIED loss — positions whose 8-gram context
reoccurs from > window bytes back ("induction positions", the
recall-dependent slice where H2 predicts the cores' gain concentrates).

Data: FineWeb-Edu as UTF-8 bytes (no tokenizer), downloaded once as parquet
shards and cached as a flat local uint8 file, then sampled deterministically
-- every config sees the identical byte stream for a given seed, and there is
no live hub connection to drop mid-run. See scripts/m5_data.py; build the
cache up front with `python scripts/m5_data.py --shards 4`. Use --synthetic
for a no-network smoke test.

Usage:
  python scripts/m5_arch.py --preset smoke_cores --iters 200 --synthetic
  python scripts/m5_arch.py --preset base_cores --tokens 2e9 --wandb
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from core import CoreConfig, ModelConfig, SWTransformer
from m5_data import (DEFAULT_SHARDS, ByteData, build_byte_cache,
                     induction_mask)

# ---------------------------------------------------------------- presets
# window=T -> full attention. Cores: hefty at moderate rate (+ small memory
# core at low rate in the *_mem variants).


def presets(T):
    # north star: MOST params in the cores, activated for a fraction of
    # tokens. Two hefty cores at rate 1/8 with independent gates.
    hefty = CoreConfig(K=64, d_core=1024, n_heads=8, ffn_mult=4,
                       n_core_layers=2, target_rate=1 / 8)
    memory = CoreConfig(K=64, d_core=128, n_heads=4, ffn_mult=2,
                        n_core_layers=2, target_rate=1 / 64)
    # eight narrower/deeper cores instead of two wide ones: same base, same
    # per-core rate, more independent gates (SWTransformer builds one Core
    # module per list entry, so repeating the read-only config is fine)
    octo = CoreConfig(K=64, d_core=512, n_heads=8, ffn_mult=4,
                      n_core_layers=3, target_rate=1 / 8)
    return {
        # ---- smoke scale (minutes per run) ----
        # base ~14M + cores ~2x21M -> ~57M params, ~62% in cores;
        # FLOPs/token ~= a ~19M dense model + attention terms
        "smoke_cores": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[hefty, hefty]),
        "smoke_cores_8x": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[octo] * 8),
        "smoke_cores_mem": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[hefty, hefty, memory]),
        "smoke_dense_full": ModelConfig(   # param-matched to smoke_cores
            vocab_size=256, d_model=640, n_layers=12, n_heads=10, window=T,
            max_seq_len=T, core_layer=4, cores=[]),
        "smoke_dense_local": ModelConfig(  # FLOPs-matched to smoke_cores
            vocab_size=256, d_model=384, n_layers=11, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[]),
        # ---- base scale (~180M total for cores config) ----
        "base_cores": ModelConfig(
            vocab_size=256, d_model=768, n_layers=10, n_heads=12, window=512,
            max_seq_len=T, core_layer=5,
            cores=[CoreConfig(K=64, d_core=2048, n_heads=16, ffn_mult=4,
                              target_rate=1 / 8),
                   CoreConfig(K=64, d_core=256, n_heads=4, ffn_mult=2,
                              target_rate=1 / 64)]),
        "base_dense_full": ModelConfig(
            vocab_size=256, d_model=1024, n_layers=14, n_heads=16, window=T,
            max_seq_len=T, core_layer=5, cores=[]),
        "base_dense_local": ModelConfig(
            vocab_size=256, d_model=768, n_layers=11, n_heads=12, window=512,
            max_seq_len=T, core_layer=5, cores=[]),
    }


def flops_per_token(model, cfg, T):
    """Estimated forward FLOPs/token: 2*active_params + attention terms.
    Core params count at their admission rate (that's the whole point)."""
    core_params = sum(sum(p.numel() for p in c.parameters())
                      for c in model.cores)
    base_params = model.num_params() - core_params
    f = 2 * base_params
    f += cfg.n_layers * 2 * 2 * min(cfg.window, T) * cfg.d_model  # base attn
    for c, cc in zip(model.cores, cfg.cores):
        cp = sum(p.numel() for p in c.parameters())
        f += cc.target_rate * (2 * cp + 2 * 2 * cc.K * cc.d_core)
    return f


# ---------------------------------------------------------------- data
def stream_batches(B, T, window, device, synthetic=False, seed=0,
                   with_masks=False, split="train", data=None,
                   data_shards=DEFAULT_SHARDS, data_dir=None):
    """Yields (idx (B,T+1) long, ind_mask (B,T) bool or None) forever.

    Non-synthetic: random contiguous windows over the cached local byte file
    (scripts/m5_data.py), drawn from a seeded numpy Generator -- so a given
    (seed, B, T) gives every model config the identical sequence of batches.
    split="eval" draws from the held-out tail, which training never touches.
    """
    if synthetic:
        g = torch.Generator().manual_seed(seed)
        while True:
            idx = torch.randint(0, 256, (B, T + 1), generator=g)
            masks = (torch.stack([induction_mask(idx[b, :-1], window)
                                  for b in range(B)])
                     if with_masks else None)
            yield idx.to(device), masks.to(device) if masks is not None else None
        return
    if data is None:
        data = ByteData(build_byte_cache(data_shards, data_dir))
    yield from data.batches(B, T, window, device, seed=seed, split=split,
                            with_masks=with_masks)


# ---------------------------------------------------------------- train
def evaluate(model, eval_data, device):
    model.eval()
    tot, tot_n, ind, ind_n = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for idx, mask in eval_data:
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                logits = model(idx[:, :-1])
            ce = F.cross_entropy(logits.reshape(-1, 256).float(),
                                 idx[:, 1:].reshape(-1), reduction="none")
            tot += float(ce.sum()); tot_n += ce.numel()
            sel = mask.reshape(-1)
            ind += float(ce[sel].sum()); ind_n += int(sel.sum())
    model.train()
    return {"eval_loss": tot / max(tot_n, 1),
            "eval_loss_induction": ind / max(ind_n, 1),
            "eval_induction_frac": ind_n / max(tot_n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--tokens", type=float, default=None,
                    help="overrides --iters: train until this many tokens")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-shards", type=int, default=DEFAULT_SHARDS,
                    help="FineWeb-Edu parquet shards to cache locally; each "
                         "yields 3.47 GB of bytes (a run consumes ~1.5e9)")
    ap.add_argument("--data-dir", default=None,
                    help="where the byte cache lives (default runs/data)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = args.seq_len
    # data first: a missing/failed cache should cost nothing (and the build is
    # a one-off download, so it must not race the model onto the GPU)
    data = None if args.synthetic else ByteData(
        build_byte_cache(args.data_shards, args.data_dir))
    cfg = presets(T)[args.preset]
    model = SWTransformer(cfg).to(device)
    if args.compile:
        try:
            model = torch.compile(model, dynamic=True)
        except Exception as e:
            print(f"torch.compile unavailable ({e}); running eager", flush=True)
    fpt = flops_per_token(model, cfg, T)
    n_params = model.num_params()
    if args.tokens:
        args.iters = int(args.tokens / (args.batch * T))
    print(f"{args.preset}: {n_params/1e6:.1f}M params, "
          f"~{fpt/1e6:.1f}M FLOPs/token, {args.iters} iters "
          f"({args.iters*args.batch*T/1e9:.2f}B tokens) on {device}",
          flush=True)

    run_name = args.run_name or f"m5_{args.preset}_s{args.seed}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join("runs", "LATEST"), "w") as f:
        f.write(run_name)
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project="multicore", name=run_name,
                        config={**vars(args), "params": n_params,
                                "flops_per_token": fpt})

    def log(m):
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in m.items()}), flush=True)
        if wb:
            wb.log(m)

    # held-out eval set: fixed windows from the tail of the byte file, which
    # the train split never reaches. Same bytes for every config and run.
    eval_stream = stream_batches(args.batch, T, cfg.window, device,
                                 synthetic=args.synthetic, seed=args.seed + 999,
                                 with_masks=True, split="eval", data=data)
    eval_data = [next(eval_stream) for _ in range(args.eval_batches)]
    train_stream = stream_batches(args.batch, T, cfg.window, device,
                                  synthetic=args.synthetic, seed=args.seed,
                                  split="train", data=data)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    def lr_at(it):
        if it < args.warmup:
            return args.lr * it / args.warmup
        import math
        p = (it - args.warmup) / max(args.iters - args.warmup, 1)
        return args.lr * 0.5 * (1 + math.cos(math.pi * p))

    model.train()
    t0 = time.time()
    final = {}
    for it in range(1, args.iters + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(it)
        idx, _ = next(train_stream)
        with torch.autocast(device, dtype=torch.bfloat16,
                            enabled=(device == "cuda")):
            logits, auxes = model(idx[:, :-1], collect_aux=True)
            loss = F.cross_entropy(logits.reshape(-1, 256).float(),
                                   idx[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if it % args.eval_every == 0 or it == args.iters:
            m = evaluate(model, eval_data, device)
            m.update({"iter": it, "loss": float(loss),
                      "tokens": it * args.batch * T,
                      "tok_per_s": it * args.batch * T / (time.time() - t0)})
            for ci, a in enumerate(auxes):
                m[f"core{ci}_rate"] = float(a["rate"])
            final = m
            log(m)
            torch.save({"model": model.state_dict(), "iter": it,
                        "config": args.preset},
                       os.path.join(run_dir, "latest.pt"))
    final["preset"] = args.preset
    final["params"] = n_params
    final["flops_per_token"] = fpt
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=2)
    if wb:
        wb.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
