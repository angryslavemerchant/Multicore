# Routed-core performance: where the ~2x goes

Handoff note for optimisation work on the top-1 routed stack. Everything here
is measured on rented RTX 5090s unless marked as an estimate. Read the
"already ruled out" section first — three plausible explanations have already
been tested and killed, and repeating them costs an hour each.

## The model

`cores_620m` in `scripts/m5_arch.py`. 621,454,488 params (595.7M
non-embedding, 94% of those in the cores), **342.6M FLOPs/token forward**.

```
 4 dense prefix layers      full attention, T=4096, d=512, SwiGLU FFN 2256
16 routed layers            top-1 over 8 experts, weights UNFOLDED (each depth
                            has its own expert weights), SwiGLU FFN 2152,
                            K=128 same-expert FIFO, capacity 1.25 PAIRED
                            WITH rate_hi=0.15 (8 x 0.15 = 1.2 < 1.25, so the
                            cap provably never binds: zero drops)
                            + a shared causal W=256 mixer between depths
 4 dense suffix layers
     RMSNorm, per-head qk-norm before RoPE(10k), tied 50304 NeoX embed/head
```

It is SEMANTICALLY FLOPs-matched to `ref_dense_130m` (our reimplementation of
`open-sci/open-sci-ref-v0.01-0.13b-fineweb-edu-1.4t-300B-4096`): 342.6M vs
342.8M FLOPs/token, 0.06% apart, at 6.0x the parameters.

**CORRECTION (this doc said otherwise on first writing).** 342.6M is what the
architecture is *defined* to compute. It is not what the GPU *runs*. Two
implementation facts inflate the executed count:

  capacity padding   the packed buffer is `capacity * N/M` slots per expert and
                     the batched matmul multiplies every slot -- `valid` is not
                     applied until scatter time. +25% on the expert linear term
                     at capacity 1.25, every step, bound or not.
  band overscan      `_banded_attend` gives each block of C=K queries the
                     W = C+K-1 contiguous keys it needs, so every query scores
                     ~2x the keys in its own FIFO and masks the rest. Same for
                     the mixer's sliding window.

`flops_per_token_executed()` in `scripts/m5_arch.py` returns both:

| | semantic | executed | |
|---|---:|---:|---|
| `ref_dense_130m` | 342.8M | 342.8M | 1.00x |
| `cores_620m` | 342.6M | **389.8M** | **1.14x** |

So the measured **2.14x** wall-clock gap decomposes as **1.14x more arithmetic
executed x 1.88x lower efficiency**. Only the second factor is what fusion work
can recover; the first needs ragged execution or a narrower band.

## The measurement

RTX 5090, 32 GB, measured 232-234 TFLOPS bf16 on a dense-matmul gate.
`torch.compile` on, bf16 autocast, fused AdamW, chunked cross-entropy.
Optimizer batch pinned at 262,144 tokens throughout.

| config | s/step | tok/s | peak VRAM |
|---|---:|---:|---:|
| `ref_dense_130m`, micro 4 | 2.53 | **104,224** | 20.6 GB |
| `cores_620m` micro 1, capacity 2.5 | 6.402 | 40,949 | 18.3 GB |
| `cores_620m` micro 2, capacity 2.5 | 6.406 | 40,919 | 27.6 GB |
| `cores_620m` micro 4 | — | OOM | >32 GB |
| `cores_620m` micro 1, capacity 1.5 | 5.123 | 51,171 | 16.3 GB |
| **`cores_620m` micro 1, capacity 1.25** | **5.148** | **50,921** | **15.7 GB** |
| `cores_620m` micro 1, capacity 1.1 | 4.972 | 52,722 | 15.5 GB |

The dense row was measured on a 224-TFLOPS 5090; scaled to this box's 234.7
that is ~109k tok/s equivalent. **So the gap at capacity 1.25 is ~2.14x at
identical FLOPs/token.** Closing it is the task.

## The profile

`python scripts/bench_batch.py --preset cores_620m --seq-len 4096 --micro 1
--step-tokens 262144 --compile --capacity 1.25 --profile`

**Caveat, and it matters:** the profiled region is ONE micro-step plus one
optimizer step. The real loop runs 64 micro-steps per optimizer step, so
`AdamW.step` at 32.07% is inflated ~64x (really ~0.6%), and it also ran without
`fused=True` which the training loop uses. Re-profile with >= 8 accumulation
steps inside the profiled region before trusting any optimizer number.

Raw, sorted by self CUDA time:

```
Optimizer.step#AdamW.step        33.198ms   32.07%      1 call     <- artifact, see above
aten::bmm                        27.646ms   26.70%    432 calls   64.0us avg
aten::mm                          9.851ms    9.52%    208 calls   47.4us avg
cutlass_80_wmma_tensorop_bf16     7.831ms    7.56%     64 calls  122.4us avg
multi_tensor_apply (foreach_mul)  6.721ms    6.49%     60 calls    <- optimizer
foreach_addcdiv                   6.347ms    6.13%     31 calls    <- optimizer
cutlass_80_wmma_tensorop_bf16     6.052ms    5.85%     96 calls   63.0us avg
foreach_addcmul                   4.857ms    4.69%     31 calls    <- optimizer
foreach_lerp                      4.855ms    4.69%     30 calls    <- optimizer
cutlass_80_tensorop_bf16_s1681    4.228ms    4.08%     40 calls  105.7us avg
cutlass_80_wmma_tensorop_bf16     3.391ms    3.28%     80 calls   42.4us avg
foreach_div / add / sqrt          ~10ms      ~9.7%                 <- optimizer
```

Removing the optimizer (which is genuinely amortised 64:1) leaves **70.3 ms per
micro-step**, and the reconstruction checks out: `64 x 70.3ms + 33ms = 4.5 s`
against the 5.148 s measured, so the remaining shares are usable.

| | share of fwd+bwd |
|---|---:|
| `aten::bmm` — the batched expert GEMMs | **39%** |
| `aten::mm` — dense layers + LM head | 14% |
| **everything else** | **47%** |

## The finding

**47% of forward+backward is not matrix multiplication.** For a dense
transformer that share is normally 20-30%. That is the routed tax, and it is
*diffuse* — spread across many small elementwise/gather/softmax ops rather
than sitting in one kernel.

**The expert GEMMs are already fast and are not the problem.** At capacity
1.25, micro 1, the FFN bmm is `(8, 640, 512) @ (8, 512, 2152)` = 11.28 GFLOP
in 64.0 us = **~176 TFLOPS, 76% of this card's measured 232**. Do not start
here.

## Already ruled out (do not re-test)

1. **Launch overhead / small micro-batches.** micro 1 = 40,949 tok/s, micro 2
   = 40,919. *Identical.* Halving the number of micro-steps per optimizer step
   changed nothing, which rules out per-launch cost as the dominant term.
2. **Padded expert GEMMs.** Real but partial. The packed buffer is
   `capacity * N/M` slots per expert and `einsum('mpi,mio->mpo')` runs over
   every slot including padding (`valid` is only applied after the expert
   block, at scatter time). Going 2.5 -> 1.1 bought **1.29x**, where the FLOPs
   arithmetic predicted 1.55x. Capacity is now 1.25 and the curve is flat
   below 1.5 — this lever is spent.
3. **Gather/scatter bandwidth.** ~480 MB per forward at 1.8 TB/s is ~0.8 ms
   against a 78 ms micro-step. Under 1%. The data movement is free; it is the
   surrounding work that is not.

## Prime suspects (unverified — this is where to look)

- **`core/core_module.py::_banded_attend`** — hand-rolled attention on the
  packed stream. It materialises the full `(experts, heads, blocks, C=128,
  W=255)` logits tensor and passes over it ~5 times (einsum, +bias, masked_fill,
  fp32 softmax, second einsum). A fused kernel would tile it in SRAM and never
  write it. It also gathers K/V with `index_select` at `W/C ~= 2`, so keys are
  duplicated ~2x and each query scores 255 keys when its FIFO is only 128 deep.
  It cannot call SDPA as-is because it needs a rank-relative bias indexed by
  `query_rank - key_rank` plus a same-row mask — expressing those as a dense
  `attn_mask` disqualifies the flash backend (see the module docstring in
  `core/base_model.py`, which documents exactly this trap for the base
  attention). A FlexAttention or Triton rewrite is the obvious candidate.
- **`core/resident.py::pack_indices`** — cumsum + scatter into a
  `(G, Npad+N)` buffer, allocated fresh, 16x per forward. With a fixed
  capacity the width is now a compile-time constant, so the buffer could be
  preallocated and reused, and the whole thing is a candidate for fusion.
- **`core/base_model.py::_RoutedExpertBlock.forward_packed`** — per-(depth,
  expert) RMSNorm and SwiGLU implemented as separate elementwise ops over the
  packed tensor.
- **The scatter-back `index_add_`** in `Top1LoopedMultiCore.forward`.

## Highest-value directions

1. **Re-profile properly first** (>= 8 accumulation steps inside the region,
   `group_by_input_shape=True`, or export a trace). The 47% needs to be broken
   down before anyone optimises inside it. Everything below is a guess until
   that exists.
2. **Fuse the routed block.** The diffuse shape of the overhead says the win
   is in fusion, not in any single kernel — a Triton/CUDA implementation of
   gather -> norm -> attn -> FFN -> scatter, or adopting a ScatterMoE /
   MegaBlocks-style grouped GEMM that indexes inside the kernel so the
   permuted tensor is never materialised.
3. **CUDA graphs.** Shapes are now static (the capacity factor removed the
   `int(counts.max())` host sync that used to force `dynamic=True`), so
   `torch.compile(mode="reduce-overhead")` is newly available and untested on
   this model. Cheap to try.
4. **`_banded_attend` via FlexAttention**, which can express a score-mod
   (the rank-relative bias) and a mask-mod (same-row) while keeping a fused
   backend.

## Reproducing

```bash
# throughput sweep, synthetic data, no dataset needed
python scripts/bench_batch.py --preset cores_620m --seq-len 4096 \
    --micro 1,2 --step-tokens 262144 --compile --capacity 1.25

# with the profiler
python scripts/bench_batch.py --preset cores_620m --seq-len 4096 --micro 1 \
    --step-tokens 262144 --compile --capacity 1.25 --profile

# the dense yardstick at matched FLOPs/token
python scripts/bench_batch.py --preset ref_dense_130m --seq-len 4096 \
    --micro 4 --step-tokens 262144 --compile
```

Correctness gates that must keep passing (`python tests/test_routed.py`, and
the other three suites in `tests/`): prefill == decode to < 3e-6 for every
routed variant, expert capacity bounds the buffer under total router collapse,
and the kept tokens are a *prefix* of the uncapped packing — the cap may
truncate but must never reorder, or the rank-relative bias is meaningless.
