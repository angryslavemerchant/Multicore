# Top-1 Recurrent Core Handoff

This note records why the top-1 recurrent-core change altered the M5
architecture and what the next experiment is intended to test. Read this
before modifying or launching the routed preset.

## What failed before

The original core layer was not a conventional transformer block. Its update
was effectively `x = x + FFN(Attention(x))`: no attention residual, no
per-layer pre-norm, and no attention output projection. In addition, the old
8-core preset put roughly 72.48M parameters in low-rate cores (86.87M total),
so each core parameter saw far fewer tokens than a compute-matched dense-local
parameter. Saved M3 checkpoints showed almost no behavioral change when cores
were ablated, and the experimental absolute learned threshold could drift to
zero admissions, an absorbing state.

## New experiment: `smoke_cores_top1_loopmix`

The new path is opt-in; legacy threshold Core/MultiCore behavior is unchanged.

- Eight independent experts, each `d_core == d_model == 384`.
- Every token executes exactly one expert per recurrent loop.
- Three loops. Expensive expert Q/K/V/O and FFN weights are tied across loops.
- Each loop has its own LayerNorm affine parameters and residual scales.
- Expert block is conventional pre-norm:
  `x += scale * O(Attention(LN1(x)))`, then
  `x += scale * FFN(LN2(x))`.
- After every expert loop, tokens scatter to original sequence order and pass
  through one shared causal sliding-window attention-only mixer (window 256).
- Tokens reroute after each mixer, so one token may visit different experts at
  different loop depths.
- Router is content-based top-1 with a deterministic causal prior favoring
  expert `(position + loop) % 8`. The prior keeps experts alive early while
  learned logits can override it. Router rows are re-orthogonalized after each
  optimizer step. A small Switch-style load-balancing loss remains enabled.
- Incremental decode owns separate expert rings and mixer KV caches for every
  loop depth even though weights are tied.

## Fairness numbers at sequence length 2048

| preset | parameters | estimated FLOPs/token |
|---|---:|---:|
| `smoke_base_only` | 14,393,088 | 31,931,904 |
| `smoke_dense_local` | 19,716,480 | 43,758,336 |
| `smoke_cores_top1_loopmix` | 24,094,771 | 43,728,384 |
| old `smoke_cores_8x` | 86,871,808 | 50,067,968 |

The routed preset is 0.068% below dense-local in the static estimator while
holding about 22.2% more parameters. `ffn_hidden=704` is chosen for this match.
The estimator also logs `flops_per_token_padded`: the current `(M, max_count)`
packed kernel performs padding work when routing is imbalanced. Do not claim a
wall-clock win from theoretical FLOPs; read `pack_util`, `pack_overhead`, and
measured throughput.

A local synthetic smoke initially routed unevenly but reached approximately
balanced final-batch rates by step 120 with the causal prior. This is not proof
that the 1.5B-token text run stays balanced. Watch per-loop/core rates,
`router_cos_max`, `rate_cv`, and padding overhead.

## Validation

```powershell
C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe tests/test_invariants.py
C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe tests/test_routed.py
C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe tests/test_data.py
```

All passed before the commit. Routed-specific coverage includes exactly-one
assignment, live router/expert gradients, router reprojection, separate loop
rings/mixer caches, prefill versus incremental decode (`3.28e-7` max error),
and the 2048-token FLOP match.

Local Windows `--compile` reaches TorchInductor but cannot finish because
ToastEnv has no working Triton installation. The Vast PyTorch image is expected
to provide Triton; verify the cloud smoke rather than treating the local error
as an architecture failure.

## Intended cloud run

```bash
python scripts/m5_arch.py \
  --preset smoke_cores_top1_loopmix \
  --tokens 1.5e9 --batch 32 --data-shards 1 --compile --wandb \
  --run-name m5_smoke_cores_top1_loopmix
```

Compare against the existing `smoke_dense_local` run at matched token budget.
Primary metrics are held-out loss, wall-clock throughput, theoretical and
padding-adjusted FLOPs, expert balance, and the induction slice. The first
question is simply whether conditional recurrent depth beats three additional
dense-local blocks at almost identical estimated compute.

## Context-length scaling

Both this routed preset and `smoke_dense_local` use fixed sliding windows, so
both have approximately constant per-token attention cost after sequence length
exceeds their windows. The routed model does **not** become increasingly cheaper
relative to dense-local solely from longer context. It does become increasingly
cheaper relative to a full-attention model, whose total attention cost is
quadratic in sequence length. The expert FIFO also provides a passer-space
horizon of roughly `K / rate` (about 512 positions at K=64 and balanced 1/8
routing), but that is a receptive-field difference rather than an asymptotic
compute advantage over dense-local.
