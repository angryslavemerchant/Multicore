"""
vast/upload_results.py — attach eval visualisations and final checkpoints to
the training run on wandb.

Resumes the run identified by WANDB_RUN_ID (exported by run_training.sh so
training and this step share one run), logs every PNG in --viz_dir as an
image panel, and uploads best.pt / latest.pt (+ any --extra files, e.g. the
benchmark JSON) as a model artifact. Download later with:

    wandb artifact get <entity>/<project>/asfnet-ae-<run_id>:latest
"""

import argparse
import glob
import os
import sys
import time

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_dir",  type=str, default="viz_ae")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_asfnet_ae")
    parser.add_argument("--extra",    type=str, nargs="*", default=[])
    args = parser.parse_args()

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "asfnetAE"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume="allow",
    )

    pngs = sorted(glob.glob(os.path.join(args.viz_dir, "*.png")))
    if pngs:
        run.log({f"eval/{os.path.basename(p)}": wandb.Image(p) for p in pngs})
        print(f"Logged {len(pngs)} eval image(s)")

    def build_artifact():
        # Neocore runs live in the `neocore` project — brand their final
        # artifact to match train_neocore.py's periodic ones.
        prefix = "neocore" if run.project == "neocore" else "asfnet-ae"
        artifact = wandb.Artifact(f"{prefix}-{run.id}", type="model")
        added = False
        for name in ("best.pt", "latest.pt"):
            path = os.path.join(args.ckpt_dir, name)
            if os.path.exists(path):
                artifact.add_file(path)
                added = True
        for path in args.extra:
            if os.path.exists(path):
                artifact.add_file(path)
        return artifact if added else None

    # The instance SELF-DESTROYS on this script's success, so the upload
    # must be VERIFIED before we exit 0: log_artifact is async and failed
    # silently during the 2026-07-15 storage outage — .wait() blocks until
    # the artifact is actually committed and raises otherwise. On final
    # failure exit non-zero so run_training.sh keeps the instance alive
    # (AWAITING_PULL fallback) instead of destroying the only copy.
    if build_artifact() is not None:
        for attempt in range(6):
            try:
                logged = run.log_artifact(build_artifact(), aliases=["final"])
                logged.wait()
                print("Checkpoint artifact uploaded and verified")
                break
            except Exception as e:
                if attempt == 5:
                    run.finish()
                    sys.exit(f"artifact upload failed after retries: {e!r}")
                wait = min(300, 60 * (attempt + 1))
                print(f"artifact upload failed ({e!r}) — "
                      f"retry {attempt + 1}/5 in {wait}s")
                time.sleep(wait)

    run.finish()


if __name__ == "__main__":
    main()
