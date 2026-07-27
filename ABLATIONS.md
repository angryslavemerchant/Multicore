# M5 routed-core ablations

Every run below: **1.5B bytes** of FineWeb-Edu (shard 0), byte-level (vocab 256,
no tokenizer), sequence length **2048**, sliding window **256**, batch 32, seed 0,
lr 3e-4, 22,888 iterations, `--compile`, one RTX PRO 6000. Identical data order
across runs, so losses are directly comparable.

`overall` is held-out cross-entropy in nats/byte. `induction` is the same loss
restricted to positions whose 8-gram context last occurred **more than 256 bytes
back** — the slice a 256-token window cannot reach directly. That reference
distance is fixed at 256 for *every* config (`IND_REF_WINDOW`), including the
full-attention ones, so the compared positions are the same everywhere.

BPB = bits per byte = nats / ln 2. Tokenizer-independent, and the unit published
byte-level work reports.

## Results

| run | params | in cores | FLOPs/tok | overall | BPB | induction | ind BPB | tok/s | pack_util |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_local_pm` | 41,244,160 | — | 89,304,064 | **0.86936** | **1.2542** | 0.88107 | 1.2711 | 429,441 | — |
| `unfold_freerate` | 42,230,843 | 65.9% | 43,728,384 | 0.89451 | 1.2905 | 0.88246 | 1.2731 | 437,643 | 0.613 |
| `unfold` | 42,230,835 | 65.9% | 43,728,384 | 0.89557 | 1.2920 | **0.87894** | **1.2680** | 500,801 | 0.959 |
| `freerate` | 24,094,779 | 40.3% | 43,728,384 | 0.90702 | 1.3086 | 0.88577 | 1.2779 | 433,025 | 0.556 |
| `loopmix` (baseline) | 24,094,771 | 40.3% | 43,728,384 | 0.90915 | 1.3116 | 0.88629 | 1.2786 | 511,731 | 0.953 |
| `dense_local` | 19,716,480 | — | 43,758,336 | 0.91037 | 1.3134 | 0.92035 | 1.3278 | 668,649 | — |
| `multik` | 24,097,907 | 40.3% | 44,023,296 | 0.91116 | 1.3145 | 0.89759 | 1.2949 | 495,396 | 0.959 |
| `nomix` | 23,501,104 | 38.8% | 39,009,792 | 0.91154 | 1.3151 | 0.89501 | 1.2912 | 614,977 | 0.964 |

`tok/s` is the wandb cumulative average, which includes `torch.compile` warm-up
and so understates steady state by ~3%. Same-GPU back-to-back steady-state
measurements: `dense_local` 691,439 vs `loopmix` 527,081 — the routed path costs
**31.2% more wall-clock at 0.99932x the theoretical FLOPs**. Never quote a
compute win from the FLOPs column alone.

## The two baselines, and why both are needed

- **`dense_local`** — 11-layer sliding-window transformer, **FLOPs-matched** to the
  routed presets (43.76M vs 43.73M, 0.07% apart). Answers "at equal compute."
- **`dense_local_pm`** — 13 layers at d=512, **parameter-matched** to `unfold`
  (41.24M vs 42.23M, -2.3%) and therefore burning **2.04x the FLOPs**. Answers
  "at equal parameters," which is the north-star framing. Shape chosen as
  d=512/L=13 rather than the closer-on-params d=384/L=24 because aspect ratio 39
  is an ordinary transformer and 16 is not; a badly proportioned opponent would
  flatter the cores for a reason unrelated to conditional compute.

They disagree, and both are true:

- **At equal FLOPs the cores win clearly.** `unfold` beats `dense_local` by
  **0.0148 overall / 0.0414 induction**.
- **At equal parameters dense wins on aggregate.** `dense_local_pm` beats
  `unfold` by **0.0262 overall**, while *losing* the induction slice by 0.0021.

## The baseline architecture (`loopmix`)

Eight expert transformer blocks at `d_core == d_model == 384`, sitting after base
layer 4 of an 8-layer sliding-window base. Every token executes **exactly one**
expert per loop, chosen by content-based top-1 argmax. Three loops, expert
weights **tied** across them (only LayerNorm affines and residual scales are
per-depth). After each loop tokens scatter back into sequence order and pass
through one shared causal sliding-window attention mixer (window 256), then
re-route — so a token may visit different experts at different depths. Each
expert keeps a K=64 FIFO of the tokens routed to it; at 1/8 traffic that FIFO
spans ~512 original-sequence positions.

Routing is kept balanced by a Switch-style load-balancing loss plus a
deterministic positional prior favouring expert `(position + loop) % 8`, and the
router rows are re-orthonormalised after every optimizer step.

---

## Ablation 1 — `nomix`: remove the inter-core mixer

**Change.** `inter_core_window=0`. The shared mixer is not built at all, so its
parameters and its 4.72M FLOPs/token are gone rather than merely unused. Experts
never exchange information, and re-routing sees an unmixed state.

**Why.** The mixer is 10.8% of the compute budget. Is cross-core communication
load-bearing, or is it paying for a channel the base model already provides?

**Result.** 0.91154 / 0.89501 — worse than baseline by +0.0024 overall and
+0.0087 induction, while running **20% faster**. `delta_rms_ratio` falls
0.942 -> 0.689.

**Verdict.** Wall-clock-adjusted this is roughly break-even on aggregate loss
(arguably favourable — 20% more tokens is worth ~0.006) but clearly worse on the
induction slice. **The mixer buys long-range quality specifically, not general
quality.** Not compute-matched to anything; compare only to `loopmix`.

## Ablation 2 — `unfold`: untie the recurrence

**Change.** `tie_loops=False`. Three *independent* expert weight sets instead of
one applied three times, sliced per depth off a single leading axis. The mixer
stays exactly where it was. **FLOPs identical** to baseline; expert parameters
tripled, so 65.9% of the model now lives in the cores (up from 40.3%).

**Why.** Is the "recurrence" doing real work, or is weight sharing just a
constraint? Three loops of one block versus three ordinary layers.

**Result.** **0.89557 / 0.87894 — the best routed run.** Beats the baseline by
0.0136 overall and 0.0074 induction, at only 2.1% lower throughput.

**Verdict.** **The biggest single win, and the only configuration that beats
`dense_local` outright.** Weight tying was costing 0.0136 nats. The stack wants
three real layers, not one block applied three times. It also survives the
wall-clock correction against `dense_local`: dense is 33.5% faster, worth ~0.010
nats off its own tail slope, and `unfold` still wins both slices by ~0.005 and
~0.032.

## Ablation 3 — `multik`: heterogeneous FIFO lengths

**Change.** `K_list = [16,16,32,32,64,64,128,128]`. Horizon is `K/p`, so with
traffic `p` pinned at 1/8 for every expert, varying `K` buys temporal diversity
without touching the traffic balance: horizons of 128/128/256/256/512/512/
1024/1024 positions. Implemented by padding every ring to `K_max` and masking
each expert to its own `K_m` with a non-learnable additive gate, keeping the
banded kernel single-width. Costs +0.674% FLOPs (everyone pays the K_max
attention term).

**Why.** ROUTED_CORE_NOTES.md's designated "important next test", and the safe
route to timescale diversity — varying `p` collapses, varying `K` cannot.

**Result.** 0.91116 / 0.89759 — **worse than baseline on both slices**
(+0.0020 / +0.0113). The mechanism worked as designed (`rate_cv` stayed pinned at
0.016, traffic untouched); the idea simply did not pay.

**Verdict.** **Negative result.** Best available explanation: the two K=16
experts have a 128-position horizon, which is *shorter than the shared mixer's
256 window* — they are redundant with a channel that already exists. A retry
should start every K above 256.

## Ablation 4 — `freerate`: bounded learned rates

**Change.** Three pieces. A learnable scalar routing bias per expert (the only
term that can actually move traffic, since router rows are renormalised to unit
length and x is layer-normed, capping learned logit spread near +-1 sigma). The
positional prior annealed linearly to zero over 2000 steps. And the
exact-uniform Switch objective replaced by a range penalty that is **exactly
zero** while every expert sits inside [3%, 30%], punishing only dead or dominant
experts.

**Why.** The long-standing goal: let the model choose what each core computes
over. Previous unconstrained attempts collapsed onto one or two experts.

**Result.** 0.90702 / 0.88577 — marginally better than baseline (-0.0021 /
-0.0005). Final traffic shares
`[0.056, 0.078, 0.107, 0.123, 0.129, 0.149, 0.158, 0.200]`: a **3.54x spread**,
`rate_cv` 0.354 against the baseline's 0.0093, and the range penalty read
**exactly 0.00000** at the end — no expert ever touched a guardrail, so this is
the router choosing, not a constraint holding experts apart. The spread persisted
to the end of training rather than decaying.

**Cost.** `pack_util` collapsed 0.953 -> 0.556. The `(M, max_count)` kernel sizes
every row by the busiest expert, so unequal rates waste ~44% of the expert work:
**15% slower**, which eats the quality gain.

**Verdict.** **The mechanism works and is stable — it just does not buy quality.**
The packing cost is a fixable varlen-kernel problem, not a flaw in the idea.

**Known deviation:** the penalty acts on mean softmax mass, not measured argmax
traffic, because the argmax load has no gradient. Measured rates therefore drift
*outside* the nominal band — `unfold_freerate` ran with `rate_min` at 0.025
against a 0.03 floor without the penalty ever firing. Nothing died, but the
guardrail is looser than it reads.

## Ablation 5 — `unfold_freerate`: combine the two that won

**Change.** Both of the above. Independent mechanisms — untying touches expert
weights, the bias touches the router. FLOPs unchanged; the eight bias scalars are
the only new parameters.

**Result.** 0.89451 / 0.88246 — a hair better than `unfold` overall (-0.0011),
**worse** on induction (+0.0035), and 12.6% slower. Rates differentiated *more*
than standalone `freerate` (**5.03x** spread, 0.041 -> 0.207) and again never hit
a guardrail.

**Verdict.** **A wash.** The gains do not stack. Untying is doing the work;
learned rates add nothing on top of it and cost throughput through packing.

---

## Cross-cutting findings

**1. The routed family has a real, reproducible long-range advantage.** For every
dense model the induction slice is *harder* than its own average (+0.010 to
+0.028). For every routed model it is *easier* (-0.005 to -0.017). `unfold`'s
absolute induction loss (0.87894) beats the parameter-matched dense model's
(0.88107) *despite trailing it by 0.026 overall*. Six routed runs, three dense.

**2. Timescale diversity is not supported — two independent nulls.** Explicit
horizons (`multik`) and learned horizons (`freerate`) both came back neutral or
slightly negative, by different mechanisms.

**3. The theoretical FLOPs advantage does not convert to wall-clock.** Effective
model-FLOP throughput: `dense_local_pm` 39.0, `dense_local` 30.3, `unfold` ~22.5
(TFLOP/s-equivalent; x3 for fwd+bwd). The 1.73x efficiency gap factors cleanly
into **1.35x routing overhead** (gather/scatter, packing, band duplication — at
matched d=384) times **1.29x width** (d=512 GEMMs simply hit tensor cores better
than d=384; a config choice, not a tax). Only the first factor is a routing
problem. It is *not* host syncs — three `.item()` calls per forward against a
145ms step is ~0.2%, and the GPU sat at 99% utilization throughout.

**4. Router collapse at init is real but transient.** At step 0, 96.8% of tokens
route to a single expert (after four base layers the residual stream has a
dominant shared direction that layer-norm does not remove, so argmax over
unit-norm router rows picks the same row for nearly every token). The load
balancing loss digs it out by ~step 250. It matters only because the
`(M, max_count)` buffer sizes to the collapsed expert: **it OOMs a 32GB card at
iteration 0.** These runs need a 96GB card for a transient, not for steady state.

## Not tested

- Longer context. Everything here is T=2048 with a 256 window, where 11 stacked
  windows already reach ~2.8k via multi-hop, so the window costs `dense_local`
  very little and the cores' O(K)-forever property has nothing to bite on.
- Wider cores (`d_core=512`), which factor 3 above suggests is free efficiency.
- More than one seed. **Every number here is a single run.** The `unfold` gap is
  6x the baseline-vs-dense gap and comparable to the `dense_local`-vs-`base_only`
  positive control (0.0217), so it is probably real, but a seed repeat is the
  cheap way to be sure before it becomes load-bearing.
- Any external benchmark. All comparisons here are internal controls; see the
  BPB column for the unit that would make these numbers comparable to published
  byte-level work (enwik8/text8 being the natural target at this parameter scale).
