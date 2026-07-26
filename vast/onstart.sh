#!/usr/bin/env bash
# vast/onstart.sh — runs once at instance boot, invoked by the onstart
# command that vast/launch.py builds (repo is already cloned, cwd is set,
# and TRAIN_ARGS / BENCH_ONLY / KEEP_ALIVE are exported there).
#
# Secrets (WANDB_API_KEY, HF_TOKEN, VAST_API_KEY) arrive as container env
# vars via `vastai create instance --env`.
#
# Log markers scraped by vast/launch.py:
#   ONSTART_BEGIN / BENCH_ONLY_DONE / GATE_FAILED / GATE_PASSED /
#   TRAIN_LAUNCHED / SELF_DESTROY  (+ BENCHMARK_JSON from benchmark.py)

set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export DEBIAN_FRONTEND=noninteractive

# runs/ holds checkpoints and metrics. Kept outside the repo because the
# provision command `rm -rf`s the repo on every (re)start, so a stop/start on
# the same machine would otherwise discard a run's only local copy.
mkdir -p /workspace/runs
[ -L runs ] || { rm -rf runs; ln -s /workspace/runs runs; }

INSTANCE_ID="${VAST_CONTAINERLABEL#C.}"
echo "ONSTART_BEGIN instance=${INSTANCE_ID} $(date -u +%FT%TZ)"

# Resolve the real interpreter: vastai/pytorch images keep the ML stack in
# /venv/main with no bare `python` on PATH; pytorch/pytorch has `python`.
if [ -x /venv/main/bin/python ]; then
    PY=/venv/main/bin/python
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    PY=python3
fi
export PY
echo "PYTHON=$PY ($($PY --version 2>&1))"

"$PY" -m pip install -q vastai
# Prefer the CLI we just installed (supports -y); image-bundled ones are old.
VAST_CLI="$(dirname "$PY")/vastai"
[ -x "$VAST_CLI" ] || VAST_CLI=vastai
export VAST_CLI

self_destroy() {
    echo "SELF_DESTROY instance=${INSTANCE_ID}"
    sleep 20   # let the log collector catch the final lines
    "$VAST_CLI" destroy instance "${INSTANCE_ID}" --api-key "${VAST_API_KEY}" -y \
        || echo y | "$VAST_CLI" destroy instance "${INSTANCE_ID}" --api-key "${VAST_API_KEY}"
}

command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; }

# --- Benchmark-only mode: measure, report, self-destroy -------------------
if [ -n "${BENCH_ONLY:-}" ]; then
    "$PY" -m pip install -q numpy
    timeout -k 60 900 "$PY" vast/benchmark.py --skip download,bank,cpu --out /workspace/benchmark.json
    echo "BENCH_ONLY_DONE"
    self_destroy
    exit 0
fi

# --- Full provisioning -----------------------------------------------------
echo "INSTALLING_DEPS"
"$PY" -m pip install -q -r requirements.txt

# --- Health gate: refuse to train on a sick machine -------------------------
# KEEP_ALIVE (smoke tests) preserves the instance so the failing metrics
# can be inspected — otherwise the destroy takes the evidence with it.
# timeout = outer belt; benchmark.py also enforces per-test hard limits.
if ! timeout -k 60 900 "$PY" vast/benchmark.py --skip download,bank,cpu --gate "${THRESHOLDS_FILE:-vast/thresholds_cheap.json}" --out /workspace/benchmark.json; then
    if [ -n "${KEEP_ALIVE:-}" ]; then
        echo "GATE_FAILED — KEEP_ALIVE set, instance left up for inspection"
        exit 1
    fi
    echo "GATE_FAILED — instance below thresholds, destroying"
    self_destroy
    exit 1
fi
echo "GATE_PASSED"

# --- Launch training detached so it survives everything ---------------------
# tee to the container's PID-1 stdout so `vastai logs` sees training output
# (tmux output is otherwise invisible to docker logs).
tmux new-session -d -s train "bash vast/run_training.sh 2>&1 | tee /workspace/train.log >> /proc/1/fd/1"
echo "TRAIN_LAUNCHED tmux=train log=/workspace/train.log"
