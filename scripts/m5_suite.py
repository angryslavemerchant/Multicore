"""M5 suite: run matched configs sequentially on one instance. Each run is a
separate process (own wandb run). Fails fast if any run crashes.

  python scripts/m5_suite.py --tokens 1.5e9 --batch 32

TOKEN LADDER. `--token-ladder 100e6,300e6,1e9` runs every preset once per
budget, each with a COMPLETE schedule of its own (with --warmup-frac, warmup
and cooldown both scale to the run). That is deliberately not the same thing
as taking three checkpoints out of one 1B-token run: under any decaying
schedule an intermediate checkpoint sits at a high learning rate and its loss
is inflated, so it does not answer "what does a run of this length reach".
Three real endpoints cost 1.4B tokens against 1.0B — 40% more for numbers that
mean what they say.

--score-ref runs scripts/score_ref.py BEFORE the ladder. Two reasons, and the
second is the practical one: it anchors our eval set to a published model, and
it builds and exercises the token cache on real data in ~15 minutes, so a
broken data pipeline costs that instead of the whole ladder.
"""
import argparse, subprocess, sys, os

PRESETS = ["smoke_cores", "smoke_dense_local", "smoke_dense_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1.5e9")
    ap.add_argument("--token-ladder", default=None,
                    help="comma-separated budgets, e.g. 100e6,300e6,1e9; each "
                         "preset runs once per budget with its own schedule")
    ap.add_argument("--batch", default="32",
                    help="passed to m5_arch as the MICRO batch")
    ap.add_argument("--grad-accum", default=None,
                    help="passed to m5_arch: micro-batches per optimizer step, "
                         "so every preset in the suite shares one step size")
    ap.add_argument("--seq-len", default="2048")
    ap.add_argument("--schedule", default=None, choices=("cosine", "wsd"))
    ap.add_argument("--lr", default=None)
    ap.add_argument("--warmup-frac", default=None,
                    help="fraction of steps, so warmup scales with the budget")
    ap.add_argument("--eval-every", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--free-rate", action="store_true",
                    help="passed to m5_arch: learned tau, so each core picks "
                         "its own admission rate (no-op for the dense presets)")
    ap.add_argument("--no-ortho", action="store_true",
                    help="passed to m5_arch: disable orthogonal gate "
                         "directions (the without arm of the ablation)")
    ap.add_argument("--data-shards", default=None,
                    help="passed to m5_arch (a sample/100BT shard is 2.0 GB of "
                         "parquet, ~0.70B NeoX tokens)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--presets", nargs="*", default=PRESETS)
    ap.add_argument("--score-ref", default=None,
                    help="hub revision of the external reference to score on "
                         "our eval set first, e.g. iter_0002000")
    ap.add_argument("--score-ref-model", default=None)
    ap.add_argument("--eval-batches", default="8")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    def run(cmd, label):
        print(f"SUITE_RUN {label}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"SUITE_FAILED {label} rc={rc}", flush=True)
            sys.exit(rc)

    if args.score_ref:
        cmd = [sys.executable, os.path.join(here, "score_ref.py"),
               "--revision", args.score_ref, "--seq-len", args.seq_len,
               "--batch", args.batch, "--eval-batches", args.eval_batches,
               "--out", os.path.join("runs", f"ref_{args.score_ref}.json")]
        for flag, val in (("--model", args.score_ref_model),
                          ("--data-shards", args.data_shards),
                          ("--data-dir", args.data_dir), ("--seed", args.seed)):
            if val:
                cmd += [flag, str(val)]
        run(cmd, f"score_ref@{args.score_ref}")

    budgets = ([b.strip() for b in args.token_ladder.split(",")]
               if args.token_ladder else [args.tokens])
    for preset in args.presets:
        for tokens in budgets:
            cmd = [sys.executable, os.path.join(here, "m5_arch.py"),
                   "--preset", preset, "--tokens", tokens,
                   "--batch", args.batch, "--seq-len", args.seq_len]
            if len(budgets) > 1:
                # distinct wandb names, or the ladder overwrites itself
                tag = f"{float(tokens):.0e}".replace("+0", "").replace("+", "")
                cmd += ["--run-name", f"m5_{preset}_{tag}"]
            for flag in ("--wandb", "--synthetic", "--compile", "--free-rate",
                         "--no-ortho"):
                if getattr(args, flag[2:].replace("-", "_")):
                    cmd.append(flag)
            for flag, val in (("--data-shards", args.data_shards),
                              ("--data-dir", args.data_dir),
                              ("--grad-accum", args.grad_accum),
                              ("--schedule", args.schedule), ("--lr", args.lr),
                              ("--warmup-frac", args.warmup_frac),
                              ("--eval-every", args.eval_every),
                              ("--eval-batches", args.eval_batches),
                              ("--seed", args.seed)):
                if val:
                    cmd += [flag, str(val)]
            run(cmd, f"{preset}@{tokens}")
    print("SUITE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
