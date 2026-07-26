# Vast.ai training automation — SmallCore

Rent a GPU, health-check it, train, push results to wandb, destroy the
instance — driven from `vast/launch.py` on the local machine.

Ported 2026-07-25 from the NeocoreEpisodic setup. `BLUEPRINT.md` is the
portable spec and `OFFER_JUDGEMENT.md` the machine-selection ledger; both carry
operational history worth reading before renting anything.

## What is different about this project

- **There is no dataset.** SmallCore generates its own walks in-process from a
  20 KB committed JSON. The download, Drive-bank and jpeg-decode gate tests
  measure a path this project never takes and are skipped
  (`--skip download,bank,cpu`). `thresholds_smallcore.json` keeps only the
  broken-hardware floors.
- **SHOP FOR THE CPU, NOT THE GPU.** The recurrence is kernel-launch bound and
  launch issue rate is *CPU-side* work, so the host CPU is the bottleneck. On
  the identical 21x21 / 20-dim job, measured 2026-07-25:

  | host | $/hr | s/iter | cost per 3000-iter run |
  |---|---|---|---|
  | **RTX A4000 + i7-13700** | **0.088** | **0.268** | **$0.02** |
  | RTX 5090 + Core Ultra 9 285K | 0.268 | 0.25 | $0.06 |
  | RTX 5090 + EPYC 7B12 | 0.308 | 1.46 | $0.38 |
  | local workstation | — | 2.46 | — |

  A $0.088 A4000 with a good desktop CPU is within 7% of a $0.268 RTX 5090 and
  **19x better value than a 5090 on an EPYC**. GPU class is nearly irrelevant;
  single-thread CPU is the entire story. **Buy the cheapest card attached to a
  Ryzen 5000+/12th-gen-Core-or-newer chip**, and use
  `--thresholds vast/thresholds_cheap.json` so the boot gate does not reject
  small GPUs for being small (the default floor of 50 bf16 TFLOPS was
  calibrated on a 5090 at 236 and rejects a perfectly usable 3060).

  Note this inverts the previous project's rule, which preferred many-core
  server CPUs because that workload was GPU-bound. This one is not.
- **`destroy --all` is scoped to this repo.** The account runs instances for
  other projects concurrently; `--all` destroys only what
  `.vast/instances.json` records. `--all-remote` is the unscoped version and
  must be asked for by name.

## One-time setup

`vast/secrets.env` (gitignored — this repo is PUBLIC) with:

```
VAST_API_KEY=...
WANDB_API_KEY=...
HF_TOKEN=...
```

## Commands

```bash
python vast/launch.py search                       # candidate offers
python vast/launch.py scan --n 3                   # bench 3 machines, suggest thresholds
python vast/launch.py launch --smoke               # tiny pipeline test (keep-alive)
python vast/launch.py launch                       # default: m2_train.py, 8000 iters
python vast/launch.py launch --train-script scripts/m1_position.py \
                             --train-args "--iters 3000"
python vast/launch.py status                       # live instances (all projects)
python vast/launch.py logs [--id ID]
python vast/launch.py pull [--id ID]               # copy runs/ back before destroying
python vast/launch.py destroy [--id ID | --all]
```

## Lifecycle

1. `launch.py` picks an offer (median price, never the cheapest — the bottom of
   the range over-samples lemons) and creates the instance.
2. Onstart clones this repo and runs `vast/onstart.sh`: installs
   `requirements.txt`, runs the health gate (a sick machine **destroys
   itself**), then starts `vast/run_training.sh` in tmux.
3. `run_training.sh` runs the training script, uploads checkpoints and figures
   to the wandb run, and **destroys the instance** on success. On failure it
   stays alive for inspection. `--keep-alive` disables auto-destroy.

## Monitoring

- wandb project `smallcore` — losses and eval metrics live; checkpoints under
  Artifacts.
- `python vast/launch.py logs` — markers: `ONSTART_BEGIN`,
  `BENCHMARK_JSON {...}`, `GATE_PASSED`/`GATE_FAILED`, `TRAIN_LAUNCHED`,
  `TRAIN_EXIT`, `RUN_COMPLETE`, `SELF_DESTROY`.
- **`vastai logs` can be silent on a perfectly healthy instance.** On image tag
  `pytorch_cuda-13.2.1-auto` (2026-07-25) provisioning output never reached
  `vastai logs` at all: the instance had cloned, installed, passed its health
  gate and started training, while the log showed only ssh port-forward noise.
  The old rule "empty logs past ~8 minutes means a zombie, destroy it" would
  have killed it. **SSH in and read `/workspace/onstart.log` and
  `/workspace/train.log` before destroying anything.**

      python vast/launch.py ssh --id <ID>      # prints ssh://user@host:port
      ssh -p <port> root@<host> 'tail -30 /workspace/onstart.log'
      ssh -p <port> root@<host> 'tail -20 /workspace/train.log'

  Only treat an instance as dead if SSH also shows nothing running.

## Training scripts

Any script under `scripts/` works as `--train-script`. They log to wandb with
`--wandb` and write `runs/<name>/{best,latest}.pt` plus `metrics.json`, with
`runs/LATEST` naming the current run — which is what the upload step reads.
