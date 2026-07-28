# Routed-core performance: final report

Answers the 15 items in `ROUTED_PERF_OPTIMIZATION_INSTRUCTIONS.md`.

**Status: PARTIAL, and deliberately so.** Stage 1 (baseline + profile) is
complete. Stage 2 (compile matrix) is in flight with a separate agent and is
marked PENDING below rather than guessed at. Stages 3–5 (fused mixer, fused
expert attention, ragged grouped GEMM) were not started — they are the actual
project and the brief gates them on Stage 1. Per the brief's own stopping
rules, nothing here is claimed without profiler or timing evidence, and no
kernel is described as fused without a backend confirmation.

One finding *outside* the brief's scope dominates everything in it, and is
reported in §16: the multi-GPU configuration was giving back the entire
`torch.compile` win.

---

## 1. Commits and exact commands

Branch `tokens-and-ref-baseline`.

| commit | what |
|---|---|
| `9c0a737` | DDP gradient buckets sized for fusion (§16) |
| `8ef865d` | `no_sync()` during accumulation in the trainer |
| `fa17ada` | torch 2.12 renamed the profiler's `cuda_*` accessors to `device_*` |
| `5693cee` | exact per-batch quantile tau; `smoke_cores_8x` preset |

```bash
# routed throughput sweep (synthetic data, no dataset needed)
python scripts/bench_batch.py --preset cores_620m --seq-len 4096 \
    --micro 1,2 --step-tokens 262144 --compile --capacity 1.25

# shape-aware profile
python scripts/bench_batch.py --preset cores_620m --seq-len 4096 --micro 1 \
    --step-tokens 262144 --compile --capacity 1.25 \
    --profile --profile-accum 8 --trace-out trace.json

# the 8x ladder actually being trained
torchrun --standalone --nproc_per_node=8 scripts/m5_suite.py \
    --presets cores_620m --wsd-ladder 100e6,300e6,1e9 --decay-frac 0.2 \
    --batch 1 --grad-accum 8 --seq-len 4096 --lr 6e-4 --warmup 100 \
    --eval-every 100 --eval-batches 32 --data-shards 4 --compile --wandb
```

## 2. Environment

Two boxes, and the difference matters when comparing numbers:

| | profile box | ladder box |
|---|---|---|
| GPU | 1× RTX 5090, 32 GB | 8× RTX 5090, 32 GB |
| driver | 610.43.02 | 575.51.03 |
| torch | 2.12.0+cu130 | 2.11.0+cu128 |
| triton | 3.7.0 | — |
| host | — | AMD EPYC 9654, 192 cores, 773 GB |
| measured bf16 gate | 237.3 TFLOPS | ~232 TFLOPS |

bf16 autocast, fused AdamW, chunked cross-entropy throughout. GeForce P2P is
disabled, so GPU↔GPU traffic routes through host RAM.

**Host variance is real and must be controlled for.** A same-box control found
a 23% throughput difference between two nominally identical 5090 hosts. Any
cross-box comparison below 25% is noise.

## 3. Reproduced baselines

Optimizer batch pinned at 262,144 tokens throughout.

| config | s/step | tok/s | peak VRAM |
|---|---:|---:|---:|
| `ref_dense_130m`, micro 4 | 2.53 | 104,224 | 20.6 GB |
| `cores_620m` micro 1, capacity 2.5 | 6.402 | 40,949 | 18.3 GB |
| `cores_620m` micro 2, capacity 2.5 | 6.406 | 40,919 | 27.6 GB |
| `cores_620m` micro 4 | — | OOM | >32 GB |
| `cores_620m` micro 1, capacity 1.5 | 5.123 | 51,171 | 16.3 GB |
| **`cores_620m` micro 1, capacity 1.25** | **5.148** | **50,921** | **15.7 GB** |
| `cores_620m` micro 1, capacity 1.1 | 4.972 | 52,722 | 15.5 GB |

The dense row was measured on a 224-TFLOPS 5090; scaled to 234.7 that is
~109k tok/s equivalent. **The routed gap at capacity 1.25 is ~2.14× at
essentially equal semantic FLOPs.** On the 237.3-TFLOPS profile box the same
routed config reproduced at **54,410 tok/s compiled** and **18,938 eager**
(2.87×), 15.9 GB peak — consistent with the table after scaling.

Parameter counts: `ref_dense_130m` 125,240,512 (99,484,864 non-embedding);
`cores_620m` 621.5M (595.7M non-embedding).

## 4. Useful versus executed FLOPs — corrected

An earlier draft claimed "342.6M defined vs 342.8M dense, so any gap is pure
overhead." That was wrong; it compared the *semantic* count to the dense
count and ignored what the kernels actually run.

| | FLOPs/token |
|---|---:|
| `cores_620m` semantic (defined) | 342.6M |
| `cores_620m` **executed** | **392.3M (1.145×)** |
| `ref_dense_130m` | 342.8M |

**This corrects `ROUTED_PERF.md`, which reports 389.8M (1.14×).** That value
corresponds to a 64-wide band block; the committed kernel tiles at `C = K`
(`core_module.py`), i.e. 128, which the profile independently confirms as
`(experts, heads, blocks, C=128, W=255)`. The document contradicted itself.
392.3M is what the preset actually constructs:

```
semantic 342.6M  executed 392.3M  ratio 1.145x
```

Two implementation facts inflate the executed count:

- **Capacity padding.** The packed buffer is `capacity × N/M` slots per expert
  and `einsum('mpi,mio->mpo')` multiplies every slot — `valid` is not applied
  until scatter time. +25% on the expert linear term at capacity 1.25, every
  step, bound or not. **Capacity is a direct multiplier on expert GEMM FLOPs.**
- **Band overscan.** `_banded_attend` gives each block of C=128 queries the
  union of their windows, W=255, so keys are duplicated ~2× (`W/C ≈ 2`).

Live training confirms this: `flops_per_token_padded` runs **382–391M** against
`flops_per_token_measured` 342.6M, tracking the packing efficiency step to step.

So the honest headline is **~2.14× wall-clock for 1.145× executed FLOPs** —
roughly 1.9× unexplained by arithmetic, which is the thing to attack.

**Note for whoever owns the band-fusion work:** `band_block_size` is a live,
uncommitted tunable in the working tree (`core/config.py`,
`core/base_model.py`, `core/core_module.py`, plus `bench_banded_blocks.py`
and `bench_mixer_blocks.py`). It must land as one complete commit with its
own gate. A partial push of just its *use* in the `cores_620m` preset took
all 8 ranks down with
`TypeError: CoreConfig.__init__() got an unexpected keyword argument`.
`flops_per_token_executed` now reads it via `getattr(..., 0) or K`, so the
accounting is correct with or without the feature — but any block size other
than 128 changes the executed FLOP count and therefore every FLOP-normalised
comparison in this report. Re-derive §4 when it lands.

## 5. Region profile and dominant operator shapes

`record_function` regions are **silently dropped under `torch.compile`** ("will
be ignored" in the dynamo log), so the region breakdown required a separate
eager run. Regions instrumented: `route`, `pack_indices`, `gather`,
`expert_block`, `scatter`, `mixer`, and inside `forward_packed`
`expert_norm_qkv`, `expert_banded_attn`, `expert_out_proj`, `expert_ffn`.

Removing the optimizer (genuinely amortised 64:1) leaves **70.3 ms per
micro-step**; the reconstruction checks out at `64 × 70.3 ms + 33 ms = 4.5 s`
against 5.148 s measured, so the shares are usable.

| | share of fwd+bwd |
|---|---:|
| `aten::bmm` — batched expert GEMMs | **39%** |
| `aten::mm` — dense layers + LM head | 14% |
| **everything else** | **47%** |

Split by shape, `aten::bmm` is **2,688 calls / 196.9 ms** (expert projections)
versus **768 calls / 21.8 ms** (the band) — **90/10**.

**The expert GEMMs are already fast and are not the problem.** At capacity
1.25 micro 1, the FFN bmm is `(8, 640, 512) @ (8, 512, 2152)` = 11.28 GFLOP in
64.0 µs = **~176 TFLOPS, 76% of this card's measured 232**. Do not start here.

Named offenders in the ~36% remainder:

- `aten::add_` — 33.8 ms.
- The band's K-gather triton kernel — **18.7 ms / 128 calls / 145.8 µs each**,
  moving ~21 MB per call. That is **~10× off achievable bandwidth** and is the
  single most concrete target in the profile.

*Caveat carried from the source profile:* the profiled region was one
micro-step plus one optimizer step, so `AdamW.step` at 32.07% is inflated ~64×
(really ~0.6%) and ran without `fused=True`. Re-profile with ≥8 accumulation
steps before trusting any optimizer number.

## 6. Attention backends — confirmed

All three confirmed from the profiler, not inferred:

| path | backend |
|---|---|
| dense prefix / suffix layers | **FlashAttention** |
| W=256 inter-core mixer | **memory-efficient (cutlass) — NOT flash** |
| expert K=128 band | **no fused backend at all** |

The expert band is raw `bmm` + `masked_fill` + fp32 softmax. It cannot use SDPA
as written because it needs a rank-relative bias indexed by
`query_rank − key_rank` plus a same-row mask. It materialises
`(experts, heads, blocks, C=128, W=255)` logits across ~5 passes.

The mixer landing on cutlass rather than flash is an unforced loss and is the
cheapest of the remaining fusion targets.

## 7. Static compilation and compile modes — **PENDING**

Owned by a separate agent, not yet returned. What is already settled and
should frame that table:

- `compile_dynamic()` **already auto-selects `dynamic=False`** for capacity-capped
  routed presets, `True` uncapped, `None` dense. So `--dynamic true` is now the
  *legacy* row, not the baseline.
- The historical reason for dynamic shapes is gone: `pack_indices` used to size
  its buffer by `int(counts.max())`, a host sync on a data-dependent value.
  Capacity capping made the buffer static.
- Expected spread on a healthy 5090: **~54,410 compiled vs 18,938 eager**. A row
  within ~5% of 18.9k has silently fallen back to eager — check the tag per row.

## 8–10. Fused mixer / fused expert attention / ragged grouped execution — **NOT STARTED**

Deliberately not begun; the brief gates them on Stage 1, and Stage 2 is still
out. Priority order implied by §5 and §6: **mixer fusion first** (it is a
backend selection problem, not a new kernel), then the **band K-gather**
(10× off bandwidth, a concrete number), then **ragged execution** (which
subsumes the capacity-padding tax in §4).

## 11. Correctness tests and tolerances

Four gate suites, all green at `9c0a737` (36 gates):

| gate | tolerance / result |
|---|---|
| VOC | init loss == ln(vocab) at 37 (3.611) and 50304 (10.826) |
| WRP | unwrap/FLOPs/evaluate survive DDP, compile(DDP), DDP(compile) |
| SYN | trainer *and* benchmark suppress all-reduce on all but the last micro-step |
| BKT | DDP buckets → ~4 graph splits, not ~95 (§16) |
| CHK | chunked CE == unchunked in loss *and* both gradients, chunk 1..4096 |
| ACC | 4 micro-batches of 2 == one batch of 8; worst relative grad diff **5.94e-07** |
| RSM | resumed == uninterrupted to **0.0e+00**; dropping Adam moments diverges 3.6e-03 |
| CAP | capacity collapses buffer 128→40 slots, **0 drops** on balanced traffic; prefill == decode 2.2e-07 |
| RT-RECIPE | prefill == decode **2.38e-07** |
| RT-FLOPS-ACCT | `ref_dense_130m` 342.8M/token = body 199.0 + head 51.5 + attn 92.3 |

Numerical comparison tolerance for "unchanged architecture" claims: **prefill ==
decode at ~2e-07**, and gradient agreement at **~6e-07**. Any optimization that
moves either by more than an order of magnitude should be treated as an
ablation, not an implementation change.

## 12. Per-depth expert loads and capacity drops

From live training at 157M tokens (8 experts × 16 unfolded loops, `rate_mean`
pinned at 0.125 = 1/8 by construction).

**Read these as rank 0's shard, not a world aggregate.** There is no
`all_reduce` in `m5_arch.py` — DDP averages gradients, not reported losses —
so every logged metric, `eval_loss` included, covers `eval_batches` windows on
one rank rather than `world × eval_batches`. Unbiased (ranks draw disjointly
from one stream) but ~√8 ≈ 2.8× noisier than the run's batch size suggests.
Consistent across arms, so comparisons hold; do not quote a single eval to
more decimal places than that supports.

| metric | value |
|---|---:|
| `pack_util` | 0.76–0.78 |
| `pack_overhead` | 1.28–1.34 |
| `rate_min` / `rate_max` | 0.085 / 0.187 |
| `rate_cv` | 0.17–0.34, falling |
| `router_entropy` | 0.70–0.83 |
| drops at capacity 1.25 | **0 on balanced traffic** (CAP gate) |

**Load is not uniform across depth, and the imbalance is systematic rather than
noisy.** Individual experts drift monotonically across the 16 loops — e.g. at
157M tokens core 2 climbs 0.065 → 0.196 from loop 0 to loop 15 while core 3
falls 0.165 → 0.077. The routing is depth-dependent, which is the intended
behaviour, but it means **a single global capacity factor is provisioned for
the worst loop**. Per-depth capacity is a real, unexploited saving and belongs
on the Stage 5 list.

`rate_cv` falling from 0.34 to 0.17 over the first 600 steps says the balance
is improving with training, so capacity headroom chosen at init is
over-provisioned later.

## 13. End-to-end tok/s and VRAM after each change

See the attribution table below. Multi-GPU rows are per-GPU so they compare
directly against the single-card baseline.

## 14. Exact-architecture changes versus ablations

**Exact** (bit-comparable forward, gates hold at the §11 tolerances): static
compilation, compile mode selection, mixer fusion, expert-attention fusion,
ragged grouped execution, DDP bucket sizing, `no_sync` accumulation.

**Ablations** (change what the model computes — must be reported separately and
never used as a headline speedup): capacity factor below `M × rate_hi`, window
or K changes, `tie_loops`, reduced `n_loops`.

Capacity is the trap here. It looks like a tuning knob and reads as a 1.24×
speedup from 2.5 → 1.25, but below `M × rate_hi` it starts dropping tokens and
becomes an ablation. **At `rate_hi = 0.15` and M = 8, the floor is 1.20.** The
current 1.25 sits just above it, and the curve is flat below 1.5 — **this lever
is spent.**

## 15. Remaining bottlenecks, highest value first

1. **The ~47% "everything else."** Neither bmm nor mm. This is the whole game:
   the routed model runs 2.14× slower for 1.14× the executed FLOPs.
2. **Band K-gather at ~10× off bandwidth** (18.7 ms / 128 calls). The most
   concrete single number in the profile.
3. **Mixer on cutlass instead of flash.** A backend-selection loss.
4. **Expert band has no fused backend at all** — the largest structural win, and
   the hardest, because of the rank-relative bias.
5. **Per-depth capacity** (§12): one global factor provisions for the worst loop.

**Next highest-value experiment:** get the Stage 2 compile matrix in, then fuse
the mixer. It is the only item that is a configuration problem rather than a
kernel-authoring problem, and §6 already proves the current backend is
suboptimal.

## 16. Outside the brief: the multi-GPU config was cancelling the compile win

Found while verifying the 8×5090 ladder, and it is larger than anything in §5.

`torch.compile(DDP(model))` enables dynamo's DDPOptimizer, which splits the
compiled graph once per DDP gradient bucket so each bucket's all-reduce can
overlap the next bucket's backward. Every split is also a **hard fusion
barrier**, and `bucket_cap_mb` defaults to **25 MB**. At 621.5M params — 2.49 GB
of fp32 gradients — that is **~99 buckets, so ~99 subgraphs**. The inductor
cache held 12,968 generated kernel files.

The routed model's entire compile win is fusing many small gather and
elementwise ops (§5: 47% is neither bmm nor mm), so chopping the graph 99 ways
gives all of it back:

| | tok/s per GPU |
|---|---:|
| 8×5090, DDP + compile, 25 MB buckets | **18,992** |
| 8×5090, DDP + compile, **593 MB buckets** | **20,200** |
| single 5090, compiled, `grad_accum=64` | 50,921 |
| single 5090, **eager**, `grad_accum=64` | 18,938 |

### The fix measured 1.064×, not the ~2.3× predicted. Read this part.

Sizing the buckets was correct and is worth keeping, but graph fragmentation
was a **6.4%** effect, not a 65% one. The hypothesis was mostly wrong, and the
reason it looked so convincing is a trap worth naming:

**18,992 and 50,921 are not commensurable.** The single-GPU reference ran
`grad_accum=64`; the ladder runs `grad_accum=8` per GPU at the same 262,144
global batch. `clip_grad_norm_` and the AdamW step are **O(params) and
independent of batch size**, so at 8 GPUs they amortize **8× worse**. The
coincidence that per-GPU throughput landed within 0.3% of the eager number was
exactly that — a coincidence, and it anchored the whole diagnosis.

Backing the step out (per GPU, 1,622 ms measured):

| | ms | share |
|---|---:|---:|
| 8 micro-steps × 70.3 ms | 562 | 35% |
| per-step fixed: `clip_grad_norm_` + AdamW over 621.5M params | ~616 | 38% |
| all-reduce, 4.35 GB at ~9.8 GB/s | ~444 | 27% |

The 616 ms is not fitted: the single-GPU bench shows the same residual
(`5,148 − 64 × 70.3 − 33 = 616 ms`) per optimizer step, where it is hidden by
64:1 amortization.

Power draw (380–408 W of 575) did correctly rule out an idle-waiting-on-PCIe
failure — but "compute-bound" was read as "the *model* is computing", when a
third of it is a 621.5M-param gradient-norm reduction and optimizer update,
which draw power too.

**Consequence for planning: 50,921/GPU was never reachable at this scaling
point.** 8×5090 at a fixed 262k global batch is near its strong-scaling limit
for this model, and the limit is O(params) per-step work, not compile quality.
Levers, in order: raise the global batch (amortizes both fixed cost and
comms), use fewer GPUs with deeper accumulation, or attack clip+optimizer
(fused/foreach `clip_grad_norm_`, or fold clipping into the optimizer).

**Comms was ruled out by power draw, not by arithmetic.** The box ran at
**380–408 W of 575 W** at 84–98% util and 2.8 GHz — computing, not waiting on
PCIe. The genuinely comms-bound failure on this same box (a missing `no_sync`,
`8ef865d`) looked completely different: **285 W**.

The trade is worse under accumulation. `no_sync()` means only the last of
`grad_accum` micro-steps all-reduces at all, so the fusion penalty is paid on
**every** micro-step to buy overlap on **one in eight**.

Fix (`9c0a737`): `ddp_bucket_mb()` sizes buckets for ~4 splits — 593 MB at this
model — floored at DDP's own 25 MB so it can only ever widen them.
`--ddp-bucket-mb` overrides. Gate **BKT** covers it and, like SYN, reads the
*trainer's* source: a helper nothing calls is a no-op.

**Lesson for whoever picks this up:** three separate bugs this week were the
same shape — correct on one GPU, wrong the moment a wrapper appeared, invisible
to any single-process test (`no_sync`, the double-wrapper `unwrap`, bucket
sizing). **Any throughput number from a multi-GPU box must be differentiated
from two eval points, not read off a cumulative counter**, and should be
sanity-checked against per-GPU single-card numbers before being believed.

## Attribution table

Per-GPU tok/s so multi-GPU rows compare against the single-card baseline.
Capacity 1.25, micro 1, optimizer batch 262,144, seq 4096.

| revision | exact architecture? | change | tok/s | speedup vs routed baseline | peak VRAM | tests |
|---|---:|---|---:|---:|---:|---|
| baseline (1 GPU, eager) | yes | current implementation | 18,938 | 0.37× | 15.9 GB | all gates |
| **baseline (1 GPU, compiled)** | yes | `torch.compile`, `dynamic=False` auto | **50,921–54,410** | **1.00×** | 15.7–15.9 GB | all gates |
| 8×5090, pre-`8ef865d` | yes | DDP, no `no_sync` | 11,245 | 0.22× | — | SYN fails |
| 8×5090, `8ef865d` | yes | `no_sync` during accumulation | 18,992 | 0.37× | 22.7 GB | SYN, BKT fails |
| 8×5090, `9c0a737` | yes | DDP buckets 593 MB (~4 splits) | 20,200 | 0.40× | — | all gates |
| static | yes | static compile matrix | **PENDING** | | | |
| mixer | yes | fused W=256 mixer | not started | | | |
| expert | yes | fused K=128 expert attention | not started | | | |
| ragged | yes | actual-load grouped execution | not started | | | |
| capacity 2.5 → 1.25 | **no** (ablation) | capacity factor, floor is 1.20 | 40,949 → 50,921 | 1.24× | 18.3 → 15.7 GB | CAP: 0 drops balanced |

Aggregate for planning: the 8×5090 ladder ran at **151,938 tok/s** total
before `9c0a737` and **161,602 tok/s** after — both differentiated across two
eval points, never read off the cumulative counter.

**Caveat on the per-GPU column.** Multi-GPU rows run `grad_accum=8`; the
single-GPU rows run `grad_accum=64` at the same 262,144 global batch. They are
therefore *not* a clean speedup comparison — see §16. The `0.40×` is a real
statement about this 8-GPU configuration, not a claim that 60% of the
single-card performance is being lost to a bug.

**The ladder was stopped before completing.** No cores-vs-dense result was
produced; the dense baseline (§3, complete at 100M/300M/1B) still stands
alone. Anything below is throughput work only.
