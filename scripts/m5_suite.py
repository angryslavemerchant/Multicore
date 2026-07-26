"""M5 smoke suite: run the three matched configs sequentially on one
instance. Each config is a separate process (own wandb run). Fails fast if
any run crashes.

  python scripts/m5_suite.py --tokens 1.5e9 --batch 32
"""
import argparse, subprocess, sys, os

PRESETS = ["smoke_cores", "smoke_dense_local", "smoke_dense_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1.5e9")
    ap.add_argument("--batch", default="32")
    ap.add_argument("--seq-len", default="2048")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--presets", nargs="*", default=PRESETS)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for preset in args.presets:
        cmd = [sys.executable, os.path.join(here, "m5_arch.py"),
               "--preset", preset, "--tokens", args.tokens,
               "--batch", args.batch, "--seq-len", args.seq_len]
        if args.wandb:
            cmd.append("--wandb")
        if args.synthetic:
            cmd.append("--synthetic")
        if args.compile:
            cmd.append("--compile")
        print(f"SUITE_RUN {preset}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"SUITE_FAILED {preset} rc={rc}", flush=True)
            sys.exit(rc)
    print("SUITE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
