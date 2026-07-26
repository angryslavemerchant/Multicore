# Blueprint: Vast.ai training automation for any project

This document is a portable spec. Give it (plus this `vast/` folder as a
reference implementation) to a Claude Code instance in a NEW project and say
"set this up here" — it contains everything needed to recreate the
functionality, including every pitfall we hit building it the first time.

## What the system does

One local command rents a cloud GPU, health-checks it, trains, evaluates,
uploads all results to wandb, and destroys the instance. Zero idle billing on
success. Sick machines self-destruct at boot before wasting money. The user
monitors from wandb and can kill everything at any time with one command.

```
launch.py (local, thin wrapper over `vastai` CLI)
   └─ rents cheapest offer passing filters
        └─ onstart-cmd: clones repo from GitHub, runs vast/onstart.sh
             ├─ pip install deps
             ├─ benchmark.py --gate thresholds.json   → FAIL: self-destroy
             └─ tmux: run_training.sh
                  ├─ python <TRAIN SCRIPT> $TRAIN_ARGS   (logs to wandb)
                  ├─ python <EVAL SCRIPT>                (writes PNGs)
                  ├─ upload_results.py                   (PNGs + ckpts → wandb)
                  └─ success: self-destroy | failure: stay alive for inspection
```

## Files (all in `vast/`, plus repo-root dotfiles)

| File | Role | Project-specific? |
|---|---|---|
| `launch.py` | Local CLI: search / launch / scan / status / logs / ssh / destroy | constants at top: `REPO_URL`, `IMAGE`, `DISK_GB`; wandb project name in `create_instance`; default `--train-args`; GPU/price filters in arg defaults |
| `onstart.sh` | Boot: deps → gate → tmux | only if deps install differs from `pip install -r requirements.txt` |
| `run_training.sh` | train → eval → upload → destroy | the train/eval command lines and checkpoint/viz paths |
| `benchmark.py` | Health metrics + `--gate` | HF dataset URL in `bench_download` (points at YOUR dataset so the test measures the real path) |
| `thresholds.json` | Boot-gate minimums | re-derive per project/GPU via `scan` |
| `upload_results.py` | Attach PNGs + checkpoints to the wandb run | checkpoint filenames if not `best.pt`/`latest.pt` |
| `secrets.env` | `VAST_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN` | NEVER commit; add to .gitignore FIRST |
| `README.md` | Ops runbook for humans/Claude | project names |

Porting = copy the folder, edit the table's right column, push, smoke test.

## Design decisions that matter (keep these)

- **The repo is the deployment mechanism.** The instance clones from GitHub;
  cloud-side behavior only changes after a push. Repo must be public (or add
  a deploy key). Never put secrets in the repo — they travel as container
  env vars via `vastai create instance --env "-e KEY=VAL ..."`.
- **Train inside tmux**, output tee'd to a file. Onstart is not a supervisor;
  tmux survives it.
- **`--ssh --direct` launch mode** keeps the container alive after onstart
  exits, and `vastai logs` still captures onstart/benchmark output.
- **Self-destroy from inside the instance**: instance id is
  `${VAST_CONTAINERLABEL#C.}`; `pip install vastai` on-instance, then
  `vastai destroy instance $ID --api-key $VAST_API_KEY -y`. This requires
  passing the Vast key into the container (accepted tradeoff for autonomy).
- **Fail-open on failure, fail-closed on cost**: successful runs and failed
  health gates destroy the instance; failed TRAINING keeps it alive so logs
  can be inspected (destroying deletes all evidence). `KEEP_ALIVE=1` env
  disables auto-destroy for smoke tests.
- **One wandb run shared across processes**: `run_training.sh` generates
  `WANDB_RUN_ID` once and exports it; training and `upload_results.py`
  (resume="allow") both attach to it, so eval panels land in the training run.
- **Checkpoint insurance**: training uploads `latest.pt` as a wandb artifact
  every N epochs (`--artifact_every`), so an instance dying mid-run loses at
  most N epochs. Requires a small patch to the train script if it doesn't
  have it.
- **Grep-able log markers** (`ONSTART_BEGIN`, `BENCHMARK_JSON {...}`,
  `GATE_PASSED/FAILED`, `TRAIN_LAUNCHED`, `TRAIN_EXIT status=`,
  `RUN_COMPLETE`, `SELF_DESTROY`): the local orchestrator scrapes
  `vastai logs` for these — they ARE the API between cloud and local.
- **Health gate at every boot** (`benchmark.py --gate thresholds.json`):
  same-GPU machines vary wildly (measured: 48% spread in bf16 TFLOPS, 80% in
  PCIe bandwidth, 4x in disk; one of three machines had a host driver that
  couldn't run CUDA at all). Metrics: download Mbps (from YOUR dataset's
  CDN), sequential disk write, multi-core JPEG-decode/s, pinned H2D PCIe
  GB/s, bf16 matmul TFLOPS. A missing metric (test crashed) fails the gate —
  that is how broken-driver machines get caught.
- **`scan` subcommand** rents N cheap machines in bench-only mode
  (`BENCH_ONLY=1`: benchmark → report → self-destroy, ~2 cents each),
  collects `BENCHMARK_JSON` from logs, suggests thresholds at 70% of median.
  Re-run when switching GPU class or dataset host.

## Setup checklist for a new project

1. `pip install vastai` locally; `vastai set api-key <key>` (key persists in
   `~/.config/vastai/vast_api_key`).
2. Copy `vast/` folder; add `vast/secrets.env` + `.vast/` to `.gitignore`
   BEFORE anything else; create `secrets.env`.
3. Add `.gitattributes` with `*.sh text eol=lf` (see gotchas).
4. Edit the project-specific spots (table above). Pick the Docker image to
   match the project's CUDA needs (`pytorch/pytorch:<ver>-cudaXX.X-cudnn9-devel`).
5. Ensure the train script: reads creds from env (wandb/HF auto-read their
   env vars), has resumable checkpoints, and ideally `--artifact_every`.
6. Push. Run `launch.py scan --n 3` → update `thresholds.json` → push.
7. `launch.py launch --smoke` (a `--smoke` flag = tiny run + `KEEP_ALIVE`):
   verify wandb run appears, eval images upload, then `destroy`.
8. Real run: `launch.py launch --train-args "..."`.

## Gotchas (each cost real debugging time — do not rediscover)

1. **`vastai destroy instance` prompts "[y/N]" since CLI ~1.4** — always pass
   `-y`, everywhere (on-instance scripts AND local subprocess calls). Without
   it, "self-destroyed" instances silently keep billing.
2. **Vast query-language units ≠ response units**: in `search offers`,
   `cpu_ram` is GB (`cpu_ram>=32`), but the response field is MB. Filters
   that silently match zero offers are the symptom.
3. **Cloudflare speed-test 403s python-urllib's default User-Agent** — set a
   browser-ish UA; better, benchmark against your dataset's actual host
   (HF CDN: `https://huggingface.co/api/datasets/<id>/parquet/<config>/<split>`
   lists shard URLs; stream one, read N MB, stop).
4. **CRLF kills onstart**: shell scripts written on Windows and committed
   without `.gitattributes` (`*.sh text eol=lf`) arrive on Linux with CRLF
   and bash dies on line 1. Set it before the first commit.
5. **Windows console is cp1252**: printing instance logs crashes Python —
   `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at CLI start;
   also pass `encoding="utf-8", errors="replace"` to every `subprocess.run`.
6. **Buffered stdout hides progress** of long-running local commands run in
   the background — launch pollers with `python -u`.
7. **Image-pull time varies minutes to 15+** — scan/poll timeouts must be
   generous (30 min+), and "no logs yet" usually means "still pulling", not
   "broken".
8. **Error 804 "forward compatibility"** in CUDA calls = host driver too old
   for the image's CUDA. Nothing you can fix; the gate should destroy it.
9. **Some "identical" offers have 6 CPU cores** — filter
   `cpu_cores_effective>=8 cpu_ram>=32` or dataloading starves the GPU.
10. **Don't trust local state** (`.vast/instances.json`): the ground truth is
    `vastai show instances`. Always offer a `destroy --all` that queries the
    API, for orphans created by crashed orchestrator runs.
11. **Vast env-var values can't contain spaces** via `--env` — anything with
    spaces (like TRAIN_ARGS) goes into the onstart command string as
    `export VAR='...'` instead.
12. **`wandb artifact` names**: use `<project>-<run_id>` so multiple runs
    never collide; alias `final` for the end-of-run version.
13. **tmux output is invisible to `vastai logs`** (docker captures only
    PID 1's stdout). Pipe the tmux payload through
    `tee /workspace/train.log >> /proc/1/fd/1` so remote log polling can see
    training markers too.
14. **Use Vast's own template images** (`vastai/pytorch:<tag>` — find current
    tags via `vastai search templates --raw`), not vanilla DockerHub images.
    Hosts pre-cache the official template images, so instances boot in ~1
    min; a cold `pytorch/pytorch` devel pull adds 10-30 min per fresh host
    and is why manual template-based rentals feel much faster.
15. **`vastai/pytorch` images have no bare `python` on PATH** — the ML stack
    lives in `/venv/main`. Resolve the interpreter at the top of onstart
    (`[ -x /venv/main/bin/python ] && PY=/venv/main/bin/python || PY=python3`)
    and use `$PY` / `$PY -m pip` everywhere. Symptom otherwise:
    `python: command not found` and a spurious gate failure.
16. **Image-bundled `vastai` CLIs are old** (no `-y` support). After
    `$PY -m pip install vastai`, call `$(dirname $PY)/vastai` explicitly,
    with a `|| echo y | vastai destroy ...` fallback.
17. **Don't assume `/workspace` exists** — newer vastai image tags
    (e.g. `cuda-13.2.1-auto`) don't create it, and `cd /workspace && ...`
    silently kills the whole onstart chain. Always `mkdir -p` first.
    Symptom: instance `running`, sshd up, zero markers in logs.
18. **Instances can wedge in `created` state and never boot** — e.g.
    `status_msg: "Error response from daemon: ... OCI runtime create
    failed"` (host kernel/docker incompatibility). No onstart, no logs, no
    self-destroy possible; only `vastai show instance --raw` reveals it.
    Watchers should check `actual_status`/`status_msg`, and after ~15-20 min
    stuck in `created`, destroy and relaunch on a DIFFERENT `machine_id`.

## Operating cost intuition (2026 figures)

RTX 4090 ≈ $0.31–0.40/hr. Bench-only scan instance ≈ 2–3 cents. Failed gate
≈ 2 cents. A 300-epoch MAE-style pretrain on ImageNet-100 (~13 GB dataset,
small model) ≈ overnight ≈ $3–6. The gate + scan pay for themselves the
first time they reject one slow machine from an overnight run.
