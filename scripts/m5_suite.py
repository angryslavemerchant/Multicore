"""M5 suite: run matched configs sequentially on one instance. Each run is a
separate process (own wandb run). Fails fast if any run crashes.

  python scripts/m5_suite.py --tokens 1.5e9 --batch 32

WSD LADDER (`--wsd-ladder 100e6,300e6,1e9`). ONE trunk plus one short cooldown
per budget, which is the reason open-sci-ref picked WSD in the first place:
constant lr means the trunk is at no particular point in any schedule, so you
can branch off it anywhere and anneal to an honest endpoint.

  trunk     warmup, then constant lr, out to (1 - decay_frac) x the largest
            budget, writing a full checkpoint at each branch point
  branches  resume each checkpoint (weights AND Adam moments AND the stream
            position) and decay linearly to zero over decay_frac x that budget

At 100M/300M/1B with a 20% cooldown that is a 800M trunk plus 20M + 60M + 200M
of cooldown = 1.08B tokens, against 1.4B for three independent runs and 1.0B
for the thing you must not do -- reading loss straight off a mid-run cosine
checkpoint, which is sitting at high lr and reports worse than a run of that
length actually reaches.

Each finished branch has seen 0..B of the stream exactly once: the trunk gave
it 0..0.8B and the cooldown continues at 0.8B rather than replaying.

TOKEN LADDER (`--token-ladder`) is the independent-runs alternative: every
preset run once per budget with a complete schedule of its own. More expensive
and only needed if you want the budgets to be genuinely independent samples.

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
    ap.add_argument("--wsd-ladder", default=None,
                    help="comma-separated budgets: one constant-lr trunk with "
                         "a checkpoint per branch point, then one cooldown per "
                         "budget. Cheaper than --token-ladder and the reason "
                         "WSD exists. Implies --schedule wsd.")
    ap.add_argument("--decay-frac", default="0.2")
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
    ap.add_argument("--warmup", default=None,
                    help="absolute steps. Prefer this for --wsd-ladder: one "
                         "trunk means ONE warmup serving every rung, and a "
                         "fraction of the trunk is a fraction of the LARGEST "
                         "budget, which would be most of the smallest one.")
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
    ap.add_argument("--score-ref-python", default=None,
                    help="interpreter to run score_ref.py with. Published "
                         "trust_remote_code models are pinned to the "
                         "transformers that existed when they shipped, so the "
                         "anchor often needs its own venv (see score_ref.py).")
    ap.add_argument("--eval-batches", default="8")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))

    def run(cmd, label, fatal=True):
        print(f"SUITE_RUN {label}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"SUITE_FAILED {label} rc={rc}", flush=True)
            if fatal:
                sys.exit(rc)
            print(f"SUITE_CONTINUE {label} was not fatal", flush=True)
        return rc

    if args.score_ref:
        cmd = [args.score_ref_python or sys.executable,
               os.path.join(here, "score_ref.py"),
               "--revision", args.score_ref, "--seq-len", args.seq_len,
               "--batch", args.batch, "--eval-batches", args.eval_batches,
               "--out", os.path.join("runs", f"ref_{args.score_ref}.json")]
        for flag, val in (("--model", args.score_ref_model),
                          ("--data-shards", args.data_shards),
                          ("--data-dir", args.data_dir), ("--seed", args.seed)):
            if val:
                cmd += [flag, str(val)]
        # NOT fatal. This is a supplementary anchor that runs somebody else's
        # `trust_remote_code` module, pinned to whatever transformers existed
        # when they published it -- open-sci-ref's needs 4.x and dies on 5.x.
        # Letting that abort a multi-hour ladder on a rented GPU is the wrong
        # trade by a very wide margin.
        run(cmd, f"score_ref@{args.score_ref}", fatal=False)

    def base(preset, tokens, name):
        cmd = [sys.executable, os.path.join(here, "m5_arch.py"),
               "--preset", preset, "--tokens", f"{tokens:g}",
               "--batch", args.batch, "--seq-len", args.seq_len,
               "--run-name", name]
        for flag in ("--wandb", "--synthetic", "--compile", "--free-rate",
                     "--no-ortho"):
            if getattr(args, flag[2:].replace("-", "_")):
                cmd.append(flag)
        for flag, val in (("--data-shards", args.data_shards),
                          ("--data-dir", args.data_dir),
                          ("--grad-accum", args.grad_accum),
                          ("--lr", args.lr),
                          ("--eval-every", args.eval_every),
                          ("--eval-batches", args.eval_batches),
                          ("--seed", args.seed)):
            if val:
                cmd += [flag, str(val)]
        return cmd

    sys.path.insert(0, here)
    from m5_arch import token_tag as tag       # one namer, or paths diverge

    if args.wsd_ladder:
        budgets = sorted(float(b) for b in args.wsd_ladder.split(","))
        df = float(args.decay_frac)
        assert 0 < df < 1, df
        trunk_end = budgets[-1] * (1 - df)
        for preset in args.presets:
            name = f"m5_{preset}_trunk"
            cmd = base(preset, trunk_end, name)
            cmd += ["--schedule", "wsd", "--decay-frac", "0",
                    "--save-at", ",".join(f"{b * (1 - df):g}" for b in budgets)]
            for flag, val in (("--warmup", args.warmup),
                              ("--warmup-frac", args.warmup_frac)):
                if val:
                    cmd += [flag, str(val)]
            run(cmd, f"{preset}@trunk->{trunk_end:g}")
            for b in budgets:
                ck = os.path.join("runs", name, f"ckpt_{tag(b * (1 - df))}.pt")
                cmd = base(preset, b * df, f"m5_{preset}_{tag(b)}")
                cmd += ["--schedule", "cooldown", "--resume", ck]
                run(cmd, f"{preset}@cooldown->{b:g}")
        print("SUITE_COMPLETE", flush=True)
        return

    budgets = ([float(b) for b in args.token_ladder.split(",")]
               if args.token_ladder else [float(args.tokens)])
    for preset in args.presets:
        for tokens in budgets:
            name = (f"m5_{preset}_{tag(tokens)}" if len(budgets) > 1
                    else f"m5_{preset}_s{args.seed or 0}")
            cmd = base(preset, tokens, name)
            for flag, val in (("--schedule", args.schedule),
                              ("--warmup", args.warmup),
                              ("--warmup-frac", args.warmup_frac)):
                if val:
                    cmd += [flag, str(val)]
            run(cmd, f"{preset}@{tokens:g}")
    print("SUITE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
