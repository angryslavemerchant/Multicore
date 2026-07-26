"""M3 mechanism test.

Stage `pretrain`: train the sliding-window base (no cores) on MQAR. It masters
gaps < window and is architecturally unable to do longer ones.
Stage `core`: freeze the base, attach one core (or the per-token adapter
control, or nothing), train, report accuracy by gap bucket.

Gate for M3: variant=core lifts long-gap accuracy far above variant=none and
variant=adapter, which fail by construction.

Usage:
  python scripts/m3_mechanism.py --stage all --variant core --wandb
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from core import CoreConfig, ModelConfig, SWTransformer, MQAR, eval_recall

BUCKETS = ((0, 64), (64, 128), (128, 256), (256, 10 ** 9))


def make_task():
    return MQAR(T=512, n_pairs=8, n_queries=4, n_filler=64, n_keys=64, n_vals=64)


def make_cfg(task, variant):
    cores = [] if variant == "none" else [CoreConfig(
        K=32, d_core=128, n_heads=4, target_rate=0.055)]
    return ModelConfig(
        vocab_size=task.vocab_size, d_model=256, n_layers=4, n_heads=4,
        window=64, max_seq_len=task.T, core_layer=2, cores=cores,
        adapter=(variant == "adapter"))


def make_oracle_fn(task):
    """Force-admit every key/value token (pairs and queries): isolates
    transport/readback from gate discovery."""
    return lambda idx: idx >= task.n_filler


def run_stage(model, task, iters, B, lr, device, log, eval_every=500,
              run_dir=None, tag="", oracle_fn=None):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    model.train()
    t0 = time.time()
    for it in range(1, iters + 1):
        idx, labels, _ = task.gen_batch(B, device=device)
        ov = oracle_fn(idx) if oracle_fn else None
        logits, auxes = model(idx, collect_aux=True, gate_override=ov)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                               labels.view(-1), ignore_index=-100)
        # rate is held by the tau quantile controller, not a loss.
        # with a frozen base and zero admissions there is no grad path at all;
        # skip the step (tau still updated inside gate()).
        if loss.requires_grad:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()
        if it % eval_every == 0 or it == iters:
            metrics = eval_recall(model, task, BUCKETS, n_batches=8, B=64,
                                  device=device, oracle_fn=oracle_fn)
            metrics.update({f"{tag}loss": loss.item(),
                            f"{tag}iter": it,
                            f"{tag}it_per_s": it / (time.time() - t0)})
            if auxes:
                metrics[f"{tag}gate_rate"] = float(sum(
                    a["rate"] for a in auxes) / len(auxes))
            log(metrics)
            if run_dir:
                torch.save(model.state_dict(),
                           os.path.join(run_dir, "latest.pt"))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain", "core", "all"], default="all")
    ap.add_argument("--variant", choices=["core", "adapter", "none", "oracle"],
                    default="core")
    ap.add_argument("--pretrain-iters", type=int, default=3000)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--core-lr", type=float, default=1e-3)
    ap.add_argument("--base-ckpt", default="runs/m3_base/base.pt")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joint", action="store_true",
                    help="do NOT freeze the base in the core stage: tests the "
                         "mechanism under joint training (the stage-2 regime) "
                         "instead of the frozen-retrofit regime")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = make_task()
    run_name = args.run_name or f"m3_{args.variant}_s{args.seed}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join("runs", "LATEST"), "w") as f:
        f.write(run_name)

    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project="multicore", name=run_name, config=vars(args))

    def log(metrics):
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in metrics.items()}), flush=True)
        if wb:
            wb.log(metrics)

    # ---- stage: pretrain ----
    if args.stage in ("pretrain", "all"):
        cfg = make_cfg(task, "none")
        model = SWTransformer(cfg).to(device)
        print(f"pretrain: {model.num_params()/1e6:.2f}M params on {device}",
              flush=True)
        run_stage(model, task, args.pretrain_iters, args.batch, args.lr,
                  device, log, run_dir=run_dir, tag="pre_")
        os.makedirs(os.path.dirname(args.base_ckpt), exist_ok=True)
        torch.save(model.state_dict(), args.base_ckpt)
        print(f"base saved to {args.base_ckpt}", flush=True)

    # ---- stage: core ----
    final = {}
    if args.stage in ("core", "all"):
        cfg = make_cfg(task, args.variant)
        model = SWTransformer(cfg).to(device)
        base_sd = torch.load(args.base_ckpt, map_location=device)
        missing, unexpected = model.load_state_dict(base_sd, strict=False)
        assert not unexpected, f"ckpt keys not in model: {unexpected[:5]}"
        if not args.joint:
            model.freeze_base()
        n_train = model.num_params(trainable_only=True)
        print(f"{args.variant}: {n_train/1e6:.3f}M trainable "
              f"({model.num_params()/1e6:.2f}M total)", flush=True)
        oracle_fn = make_oracle_fn(task) if args.variant == "oracle" else None
        if args.variant == "none":
            final = eval_recall(model, task, BUCKETS, n_batches=16, B=64,
                                device=device)
            log(final)
        else:
            final = run_stage(model, task, args.iters, args.batch,
                              args.core_lr, device, log, run_dir=run_dir,
                              oracle_fn=oracle_fn)
        final["variant"] = args.variant
        final["trainable_params"] = n_train
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=2)
    if wb:
        wb.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
