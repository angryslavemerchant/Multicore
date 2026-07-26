# Multi-Core Sparse Side-Channel — Project Plan

Working name: **Core**. A cheap local-attention transformer plus a set of fixed-capacity attention modules ("cores"). Each core admits a sparse, threshold-selected subset of tokens, lets those tokens attend to each other privately, and returns the result as a residual delta to those tokens only.

The one non-negotiable idea: **everything is causal and the KV cache is never invalidated.** Every mechanism below was chosen to satisfy that constraint. Section 6 states the invariants; treat them as hard.

---

## 1. The idea in plain language

The base model is a small transformer with **sliding-window attention** — cheap, fast, and architecturally unable to see past its window. At one layer, each token's hidden state is scored against a learned direction, per core. Tokens above threshold are "admitted": they join that core's club (K slots), attend over the other current members, get processed by the core's machinery, and receive an additive correction to their hidden state. Everyone else is untouched. When a club is full, the oldest member is ejected but keeps the correction it already received; ejection only changes what *future* entrants see.

Two consequences, which give cores two distinct roles:

- **Memory.** A core's horizon is set by its admission rate, not by position: rate 1/1000 with 64 slots reaches ~64,000 tokens back. Low-rate cores are the model's *only* long-range channel — a learned, content-addressed transport mechanism at fixed cost.
- **Conditional compute.** A core's parameters are touched only by admitted tokens, so a core can be *hefty* — hundreds of millions of params — while adding compute only at its rate. High-rate hefty cores are where most of the model's parameters live, spent only on the tokens that need them.

Same mechanism, two knobs (rate, size), two roles.

## 2. The bet, and the compute math

**The bet:** most tokens are locally predictable and need only a cheap local model; a sparse subset needs heavy computation and long-range access. Spend parameters where they're rarely needed, and compute only when they are.

**The math.** Dense transformer: every param touched every token, FLOPs/token ≈ 2·N, plus O(T²) attention. Here:

```
FLOPs/token ≈ 2·N_base + Σ_c 2·r_c·N_c        (+ O(T·W) base attention, O(P_c·K) core attention)
```

Params and compute decouple through the admission rate. Worked example: base 12L/d768 (~85M params, window 256) + two hefty cores (~150M params each, rate 1/8) + two small memory cores (rate 1/64 and 1/1000). Total ≈ 390M params; FLOPs/token ≈ 2·85M + 2·(1/8)·2·150M ≈ that of a ~130M dense model. Param-matched to a dense 390M model at a third of its per-token compute — and the attention savings on top grow with context length.

**Precedents this rhymes with** (the pattern is proven; this combination is not):
- **MoE** — conditional params, but per-token (every token gets k experts). Here sparsity is over the *sequence*: most tokens get nothing extra.
- **CoLT5** (Ainslie et al. 2023) — light branch for all tokens, heavy branch for routed ones. The closest precedent for hefty cores; encoder-side, so it never had to solve causality. This is the causal, cache-safe, decoder version.
- **Mixture-of-Depths** — tokens skipping blocks; hit exactly the top-k causality problem this design avoids by construction.

**Two design rules that fall out:**

1. **Size cores by their token budget: params ∝ rate.** A core at rate r gets gradient from r of the tokens. MoE experience says 1/8–1/32 of tokens feeds an expert fine; 1/1000 cannot feed 150M params. So: hefty cores at moderate rates (compute role), small cores at low rates (memory role). A hefty 1/1000 core is a gradient-starvation bug, not a feature.
2. **FLOPs ≠ wall-clock until proven.** The gather→process→scatter path is ragged; kernel overhead can eat the theoretical win at small scale. The heavy compute does run *dense* over the compacted passer sequence (GPU-friendly), but report FLOPs and wall-clock separately and claim speed only from measurements.

## 3. Hypotheses

**H1 — Capacity confound (null).** Added params help however you attach them. Design comparisons so this can't masquerade as a result (params-matched and FLOPs-matched baselines, section 4).

**H2 — Long-range channel.** The gate learns what to carry; the FIFO carries it; tokens read information from far outside their window. Prediction: gains concentrate on **recall-dependent tokens** (repeated identifiers, re-mentioned entities, queried keys), not spread evenly.

**H3 — Conditional compute.** At matched params, local-base-plus-hefty-cores approaches dense-full-attention quality at a fraction of the FLOPs; at matched FLOPs, it beats dense. This is the MoE-style win with sequence-sparse routing.

**H4 — Multi-timescale.** Cores at different rates capture different structure and their contributions stack. Only meaningful after H2/H3.

Even a negative result yields the selection maps — what a learned salience gate admits, and whether different rates select different *content* — which are cheap interpretability data regardless.

## 4. Experiments

Ordered so each phase is cheap and gates the next.

### Phase 1 — Mechanism test (frozen tiny model, synthetic recall)

Pretrain a tiny sliding-window transformer (~4L, d≈256, W≈64) on **multi-query associative recall** (MQAR-style, per Arora et al. "Zoology"): scattered `key→value` pairs, filler, then queries; exact-match accuracy on value tokens, stratified by pair→query gap. The model masters short gaps and is architecturally unable to do gaps > W. Freeze it. Bolt on one small core. Train core + gate only.

Why first: the base fails long gaps *by construction*, and a matched-parameter per-token control (MLP on the query token alone, no cross-token attention) fails *by construction* — it cannot know the value. Long-gap success is attributable to the mechanism, full stop, and it exercises the crux: **can the gate learn from task gradient alone to admit the right tokens?** If gate learning fails here it fails everywhere, and hefty cores would just waste params on misrouted tokens. Minutes per run on one GPU; sweep gaps, rates, K.

### Phase 2 — Architecture test (from-scratch, param- and FLOPs-matched)

Train jointly from scratch at small scale (~300–400M total params, T = 4–8k, ordinary text + some long-context data):

- **(A)** Local base + hefty core(s) + small memory core(s) — the proposal.
- **(B)** Dense full-attention transformer, **param-matched** to (A)'s total. The quality target, at much higher compute.
- **(C)** Dense sliding-window transformer, **FLOPs-matched** to (A). Does sparse-heavy beat dense-cheap at equal compute? (The MoE-style question.)

Report: loss vs FLOPs, loss vs wall-clock, and **stratified loss** — loss on long-range-dependent tokens (repeated rare tokens/identifiers beyond W, second entity mentions, long-range repeated n-grams) separately from local tokens. H2 predicts the (A)−(C) gap concentrates in the stratified slice; aggregate perplexity alone would smear it.

Joint training notes: zero-init core outputs retained as an init/stability choice (the frozen-phase "bit-identical baseline" property no longer applies — it's a from-scratch model; the cache invariants are unaffected). Expect gate and core to co-adapt; watch the phase-1 diagnostics to confirm gates don't collapse to trivial selectors early.

### Phase 3 — Rate ladder / multi-core

The full configuration: hefty compute cores (r = 1/4–1/32) + small memory cores (r = 1/50–1/1000), orthogonal keys, one non-learned control core. Gate: beats the best phase-2 single-core config, with gains attributable to diversity (section 10), not param count. Note the training-length caveat in 5.7 — a 1/1000 core's turnover is untrained until sequences approach its horizon.

---

## 5. Architecture

### 5.1 Placement and output form

At a chosen layer `L` (or a few layers — see deferred), for each core `c`:

```
delta_c(i) = g_c(i) * Core_c(h_i, residents_c(i))   if token i admitted to core c
delta_c(i) = 0                                       otherwise

h_out(i) = h_in(i) + sum_c delta_c(i)
```

- Core output projections **zero-initialised**; deltas start at exactly zero.
- `g_c(i)` is the soft admission strength (5.3).
- **Private side-channel**: non-admitted tokens read *nothing* from core `c`. No broadcast, no shared summary. A routing error stays local to one token.

### 5.2 Admission: per-token threshold

Each core has a scoring vector `k_c`:

```
s_c(i) = <k_c, h_i>
member:  m_c(i) = 1  iff  s_c(i) > tau_c
```

Admission depends **only on token i's own representation** — never a top-k, never a ranking. Tokens may join multiple cores; gates are independent, deltas sum, no competition.

**Hard membership is load-bearing twice over.** For memory: soft membership would put every token in every core, degenerating the K-slot FIFO into "the last K tokens" and killing the K/rate horizon. For compute: hard membership is what makes core FLOPs conditional at all. Soft *magnitude* on members is fine (and needed — next section).

### 5.3 Gate training: hard membership, soft magnitude

A hard threshold has no gradient; without more, `k_c` never learns *what* to select. Decision:

- **Membership** (who enters the ring) is the hard decision `s_c(i) > tau_c`.
- **Magnitude**: members' deltas scale by `g_c(i) = sigmoid((s_c(i) - tau_c) / T)`. Task gradient reaches `k_c` through members' delta magnitudes. Per-token, causal, cache-safe; use the same scaling at inference.
- Limitation: gradient flows only through admitted tokens; a wrongly-rejected token generates no signal. Mitigations in order: (a) the rate loss keeps gates open enough to explore; (b) optional train-time exploration — admit a small random fraction of below-threshold tokens (causal; disabled at inference); (c) straight-through estimator as fallback if phase-1 selection analysis shows the gate failing to learn.
- With hefty cores the stakes rise: misrouting wastes the model's main parameter budget. Phase 1's selection analysis (M4) is the checkpoint; don't scale past it on hope.

`tau_c` is driven by a low-weighted **rate-deviation loss**: `(mean(sigmoid((s-tau)/T)) - r_c)^2` toward target rate `r_c`. Calibrate to rates, not raw tau — score distributions drift; rate is the interpretable quantity.

### 5.4 Turnover: FIFO with commit semantics

Each core holds `K` slots. New admit takes a slot; if full, the **oldest resident is ejected** and **keeps the delta it already computed**. Ejection only changes what future entrants see; nothing is ever recomputed. This is the entire reason the cache survives.

### 5.5 Resident set, formally

Let `cnt_c(i)` = number of tokens ≤ i passing core c's gate. Token `j` is resident at time `i` iff:

```
m_c(j) = 1  AND  m_c(i) = 1  AND  j <= i  AND  cnt_c(i) - cnt_c(j) < K
```

(fewer than K passers since `j`; includes `j = i`, so admitted tokens attend to themselves). All prefix-determined — one cumsum per core, fully vectorised, no sequential scan.

### 5.6 What "hefty" means

Heft goes where params scale cheaply and legally:

- **Wide internal width**: project `h_i` up to `d_core >> d` inside the core; attention and FFN at `d_core`.
- **Fat FFN**: the bulk of a hefty core's params. Applied per admitted token — the compute runs dense over the compacted passer sequence.
- **Multiple internal layers**: attention-over-ring → FFN → attention-over-ring → FFN, all at time `i` over the ring's *committed* entries. Causal and commit-safe: reprocessing stored ring states with more layers computes token i's delta more richly without touching any past delta. Cost O(K·d_core²)-ish per admitted token — fine at small K.

What heft must **not** do: change what's stored in the ring after commit, or read anything outside `(h_i, residents)`.

### 5.7 Memory horizon

Resident lifetime ≈ K/rate positions: rate 1/2 → ~128 tokens (K=64); 1/50 → ~3,200; 1/1000 → ~64,000. **Caveat:** a core's ejection path is only trained when sequences are long enough to cycle it. On 8k training sequences a 1/1000 core never completes one turnover — its advertised horizon is untested until trained near that scale.

### 5.8 Position information inside the core

Residents arrive from wildly different positions. Default: **relative position by passer rank**, which the sliding-window kernel over the compacted sequence provides natively (free). Optionally inject original-position gaps (`i − j`) if phase 1 shows order sensitivity matters; decide empirically.

### 5.9 Multi-core diversity

Separate weights per core. Uniform K *within a size class* so same-class cores batch into one kernel (hefty and memory cores needn't match each other). Two independent diversity axes: **rate** → timescale, **direction** → content — if all cores share one direction, selections are nested (one salience notion at several zooms).

Mechanisms, in build order: (1) **orthogonal parameterisation of `{k_c}`** (Householder/Cayley/QR-reproject) — key collapse becomes structurally impossible; cheap; from day one of multi-core. (2) **One non-learned control core** (recency or `||h_i||`) — free reference for whether learned cores beat a dumb heuristic. (3) **Disjoint input slices** — each core scores its own chunk of the residual stream; zero params, guaranteed difference.

Collapse-risk calibration: independent gates + summed deltas are multi-head-attention dynamics (differentiates fine, no auxiliary loss), not MoE dynamics (collapse driven by winner-take-all exclusivity, absent here). Remaining risk is mild convergence to one easy direction, which orthogonal keys address. Don't over-engineer.

---

## 6. Invariants — do not break these

Any change violating one silently breaks incremental decoding.

1. **I1 — Admission is per-token.** A function of `h_i` alone. Never top-k, never a ranking.
2. **I2 — Resident sets are prefix-determined.** `residents_c(i)` depends only on tokens ≤ i.
3. **I3 — Commit semantics.** Once computed, `delta_c(i)` is final. Ejection and later arrivals never trigger recomputation.
4. **I4 — No broadcast.** Non-members read nothing from any core. If they ever do, I1–I3 stop being sufficient and the cache dies.
5. **I5 — `tau_c` frozen at inference.** Inference-time adaptive thresholds depend on the prefix retroactively — exactly the failure this design avoids.
6. **I6 — A gradient path to `k_c` exists** (5.3). A trainability invariant: without it the gate is decorative and hefty cores burn params on noise.

**Rejected alternatives** (recorded so they don't creep back):

| Alternative | Why rejected |
|---|---|
| Global top-k admission | Non-causal: a late high scorer evicts an incumbent, changing its delta, invalidating upper-layer KV for the whole suffix. Expert-choice routing — known-incompatible with autoregressive decoding. |
| Fully soft membership | Causal, but every token occupies every core: FIFO degenerates to a K-token sliding window (kills the horizon) and core FLOPs become unconditional (kills the compute win). Soft *magnitude* on members is fine and used. |
| Refuse-when-full | Core seals shut after K passers. |
| Time-to-live expiry | Doesn't bound occupancy; needs a cap plus an overflow rule. FIFO does both with one mechanism. |
| Pinned/permanent slots | Subsumed: low rates give ~64k horizons without permanence. |
| Broadcast core output | Any turnover changes what every token reads, including the prefix. Full invalidation. |
| Writable state slots | A recurrence → sequential training. Deferred (section 9). |

---

## 7. Implementation notes

### 7.1 Gather → sliding window → scatter

**Never materialise the T×T resident mask.** On the compacted passer sequence, the resident condition is exactly causal sliding-window attention with window K:

1. Gather indices where `m_c = 1` → compact sequence of length `P_c`.
2. Causal SW attention (window K) over it — any flash-attention SW kernel, O(P_c·K). Hefty-core FFN/layers also run here, dense over `P_c` tokens.
3. Scatter outputs back; non-passers get zero.

`P_c` is ragged per (batch element, core): varlen/nested-tensor API, or pad to max with a validity mask. With hefty cores this path carries most of the model's params — its efficiency *is* the wall-clock claim; profile it early.

The cumsum formula (5.5) is the *spec*; gather-window-scatter is the *implementation*. **Write the equivalence test first:**

```python
# Reference implementation of the spec, for testing only. O(T^2) — small T only.
def resident_mask_reference(m, K):
    # m: (T,) bool  ->  (T, T) bool, [i, j] = j is resident at time i
    cnt = m.cumsum(0)
    j_leq_i = torch.tril(torch.ones(T, T, dtype=torch.bool))
    within_k = (cnt[:, None] - cnt[None, :]) < K
    both = m[:, None] & m[None, :]
    return both & j_leq_i & within_k
```

### 7.2 Batching across cores

Same-size cores stack into one batched matmul / block-diagonal attention. Without this, M separate small attentions are kernel-launch-bound and dominate step time despite negligible arithmetic. Expect this to be the first performance surprise for the small memory cores; the hefty cores are compute-dense enough to stand alone.

### 7.3 Inference

Per core: a K-entry ring buffer of keys/values (at `d_core` for hefty cores). Per decode step: score `h_i`; below threshold → delta 0, done. Else insert into ring, run the core over valid entries, scale by `g_c(i)`. Cost is O(core) *only on admitted steps* — most decode steps touch only the cheap base. The base cache is untouched.

### 7.4 Cold start

Early tokens see a partly-filled core — mask unfilled slots. Skipping this presents as "trains but slightly worse," not as an error.

---

## 8. Milestones

**M0 — Scaffolding.** Zero-init cores inserted (frozen-base setting). *Gate: logits bit-identical to base.*

**M1 — Mask correctness.** Gather-window-scatter vs `resident_mask_reference`, small random inputs, several K and rates. *Gate: exact agreement.*

**M2 — Cache correctness.** Full prefill vs token-by-token incremental decode, several lengths, cores actively turning over. *Gate: logits match to tolerance. Failure = a section 6 invariant is broken; find which before proceeding.*

**M3 — Mechanism test** (phase 1). Frozen tiny SW model + one small core on MQAR, vs no-core and per-token-adapter controls. *Gate: long-gap recall well above both controls — which fail by construction if the mechanism is what works.*

**M4 — Selection analysis.** Inspect admissions. *Gate: the gate visibly admits key/value tokens, not position or noise. If M3 passed but this looks wrong, understand it before spending params on the gate's judgment.*

**M5 — Architecture test** (phase 2). From-scratch (A) local+cores vs (B) param-matched dense full-attention vs (C) FLOPs-matched dense local. *Gate: (A) beats (C) overall, with the gap concentrated in stratified long-range loss; report distance to (B) honestly, plus FLOPs and wall-clock.*

**M6 — Rate ladder** (phase 3). Hefty compute cores + small memory cores, orthogonal keys, control core. *Gate: beats best M5 single-core config; gains attributable to diversity (M7).*

**M7 — Diagnostics and ablations.** Section 10.

---

## 9. Deferred (v2+)

None before M3 clears — all add cost and risk to an untested mechanism.

- **Novelty gating** — admit only if dissimilar to current residents. Different *in kind* (depends on core state); cache-safe but **destroys the cumsum trick** (admission depends on prior admissions → recurrence → sequential training). Compromise: novelty vs residents as of the previous 256-token block.
- **MLP scorers** — non-monotone selection criteria ("entity-like AND not recently mentioned"); ~d² params/core.
- **Multiple insertion depths** — surface features early, semantics late; composes with timescale diversity. Costs cross-depth batching; do in groups.
- **Alternative scoring bases** — change detection (`h_i − h_{i−1}`), predictive entropy.
- **Writable slots** — admitted tokens write into learned slot vectors; strictly more expressive, but a recurrence. If pursued: segment-boundary updates (RMT-style) or associative writes admitting a parallel scan (linear-attention/SSM route). Natural v2 if M3+M6 clear.

---

## 10. Diagnostics

**For H2: stratified loss is primary** — loss on long-range-dependent tokens (repeated rare tokens beyond W, second entity mentions, long-range repeated n-grams) vs local tokens. Gains should concentrate in the former.

**For H3: loss-vs-FLOPs and loss-vs-wall-clock curves** against baselines (B) and (C). Also per-core **FLOPs utilisation**: actual admitted-token throughput per core vs its budget r_c·T.

**For multi-core: pairwise selection overlap** (Jaccard over admitted sets). Near 1.0 → redundant regardless of loss; near 0.0 → check nobody's dead.

Also track:
- **Per-core passing rate** vs target (drift = rate loss / calibration off).
- **Occupancy over time** — a never-filling core is over-strict, or the training length is too short (5.7).
- **Delta norm / residual norm** per core. Near zero = dead weight; comparable to residual = destabilising. Watch early: zero-init + soft gate = two factors escaping zero together.
- **Effective horizon** (mean resident age at ejection) vs predicted K/r.
- **Per-core ablation at eval** — zero one core's delta, measure loss increase. With hefty cores this doubles as a params-actually-used check.
- **Overlap with the control core** — a learned core tracking pure recency/norm hasn't learned anything.
- **Selection maps on real text** — dump admitted tokens over documents and *look*. Cheapest interpretability available; a deliverable in its own right.

---

## 11. Background reading

Verify against the papers — summaries approximate.

**Closest precedents**: **CoLT5** (Ainslie et al. 2023) — light/heavy conditional branches, heavy for routed tokens; the nearest thing to hefty cores, but encoder-side (no causality problem to solve). Raposo et al., *Mixture-of-Depths* (2024) — token-skipping with exactly the top-k causality problem this design avoids; read the routing section. Goyal et al., *Shared Global Workspace* (2021) — nearest in spirit to the private-membership side-channel. Perceiver — fixed latent bottleneck, learned latents rather than selected tokens.

**Conditional params**: MoE literature — expert-choice vs token-choice routing (precisely why the rejected top-k design fails); Switch Transformer for load-balance losses and for expert-utilisation folklore relevant to the params-∝-rate rule.

**The phase-1 task**: Arora et al., *Zoology* / MQAR — associative recall as the discriminating synthetic; Olsson et al., induction heads — basis for the repeated-n-gram loss stratification.

**Sparse attention topology**: BigBird, Longformer — local windows + global tokens; cores are the learned-selection version of global tokens.

**Commit-semantics argument**: StreamingLLM, H2O, SnapKV — evicting cache entries is safe because it only affects future reads. Same argument here.

**If pursuing writable slots**: Recurrent Memory Transformer; Compressive Transformer; Mamba/Jamba/Samba; Coconut.

---

## 12. What a result means

The load-bearing question is still M3: **can the gate learn, from task gradient alone, to select and transport long-range information that the rest of the model then uses?** Everything — especially the hefty-core bet, which hands the gate the model's parameter budget — is conditional on yes. If M3 fails, M4 says which link broke: selection never learned (trainability — try exploration/STE) vs. selected correctly but the write-back went unused (legibility — more fundamental). Both are findings; both arrive cheap.

If M3 clears, M5 is the headline experiment: **param-matched to dense at a fraction of the compute** — does the local-plus-cores model close most of the gap to dense full attention while beating the FLOPs-matched dense baseline? A yes, with the gains concentrated where H2 predicts (recall-dependent tokens), is a real architecture result. A no still localises *why* (utilisation diagnostics, stratified loss, selection maps), which is what makes the next iteration cheap.

Hold the semantic-specialisation story loosely: MoE chased "one expert per concept" for years and mostly got surface-feature specialisation (token identity, position, frequency). Design so the thing works even if the learned split is boring; interpretable specialisation is a bonus, not the load-bearing assumption.
