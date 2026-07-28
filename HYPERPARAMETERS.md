# Hyperparameters — dense baseline vs routed-core models

Reference for the three models trained on 2026-07-27/28. Every number here is
read from the wandb run configs or the preset definitions in
`scripts/m5_arch.py`, not from memory.

The claim under test: **most parameters in sparsely-activated cores, at the
same compute per token.**

---

## 1. Training hyperparameters — IDENTICAL across `ref_dense_130m` and `cores_620m`

| | value |
|---|---|
| optimizer batch | **262,144 tokens/step** |
| micro-batching | dense `--batch 4 --grad-accum 16`; cores `--batch 1 --grad-accum 64` |
| seq_len | 4096 |
| peak LR | **6e-4** |
| schedule | WSD — constant trunk, **linear** cooldown over the final 20% |
| warmup | 100 steps (trunk), 0 on cooldown branches |
| final_lr_frac | 0 |
| optimizer | AdamW (fused), betas (0.9, 0.95), weight_decay 0.1, grad-clip 1.0 |
| seed | 0 |
| corpus | FineWeb-Edu `sample/100BT`, 4 shards, GPT-NeoX-20B tokenizer, vocab 50304 |
| eval | every 100 steps, 32 batches, `ind_window` 256 |
| precision | bf16 autocast, `torch.compile` (`dynamic=False`), chunked CE (`ce_chunk` 1024) |

The two micro-batch shapes are **mathematically identical** — `batch x
grad_accum x seq_len` is 262,144 either way. Micro-batch is a pure throughput
knob; only the product changes the optimization.

## 2. WSD ladder structure

One trunk yields several honest endpoints, which is why WSD was chosen over
cosine: a trunk at constant LR can be branched anywhere and annealed to a real
endpoint, whereas a mid-run cosine checkpoint sits at high LR and reads too
high.

```
trunk       800M tokens, schedule=wsd, decay_frac=0, warmup=100
            --save-at 80e6,240e6,800e6          (full model+optimizer)
branches    schedule=cooldown, decay_frac=0.2, warmup=0
            100M arm  <- resume ckpt_80M.pt   + 20M annealed
            300M arm  <- resume ckpt_240M.pt  + 60M annealed
            1B   arm  <- resume ckpt_800M.pt  + 200M annealed
```

## 3. Architecture

| | `ref_dense_130m` | `cores_620m` | `cores_700m_fixed2bank` |
|---|---|---|---|
| params | 125.2M | **621.5M** | **699.1M** |
| non-embedding | 99.5M | 595.7M (94% in cores) | — |
| d_model | 512 | 512 | 512 |
| n_layers | 22 (all full-attention) | 4 prefix + **16 unfolded routed** + 4 suffix | 10, cores inserted at layers (3, 5) |
| n_heads | 8 | 8 | 8 |
| ffn_hidden | 2256 | 2256 backbone / **2152 per expert** | 2256 backbone / **5696 per expert** |
| experts | — | 8, top-1 routed | 8, `top1_fixed` |
| n_loops | — | **16**, `tie_loops=False` | **4**, `tie_loops=False` |
| expert memory | — | K=128 FIFO | K=128 FIFO |
| inter-core mixer | — | **W=256** chronological | **0 (disabled)** |
| capacity_factor | — | 1.25 | 1.25 |
| rate_lo / rate_hi | — | 0.03 / 0.15 | 0.03 / 0.15 |
| router | — | learned bias, hash anneal 2000 steps, hash scale 0.5 | learned bias, hash anneal 16000, hash scale 3.0 |
| band_block_size | — | default (C = K = 128) | 64 |
| norm / act | RMSNorm, SwiGLU, qk-norm, tied embeddings | same | same |
| **semantic FLOPs/token** | **342.8M** | **342.6M** | **342.8M** |
| **executed FLOPs/token** | 342.8M (1.00x) | **394.7M (1.15x)** | **382.5M (1.12x)** |
| peak LR | 6e-4 | 6e-4 | **3e-4** |

**The expert FFN width of 2152 is not a round number by accident.** It is the
value that lands `cores_620m` at 342.6M FLOPs/token against the dense model's
342.8M — 0.06% apart — so the comparison is "~5x the parameters at identical
compute" and nothing else.

## 4. Results

### Annealed endpoints (eval loss / induction loss)

Only these are directly comparable across arms — each is a real WSD endpoint at
lr = 0. `cores_700m` has no annealed endpoint yet.

| tokens | `ref_dense_130m` | `cores_620m` | delta eval | delta induction |
|---|---|---|---|---|
| 100M | 5.2328 / 5.3918 | 5.1053 / 5.5460 | **-0.128** | +0.154 |
| 300M | 3.8326 / 3.1131 | 3.7466 / 3.0589 | **-0.086** | **-0.054** |
| 1B | 3.2773 / 2.2482 | 3.2192 / 2.1643 | **-0.058** | **-0.084** |

`cores_620m` wins on eval at every budget. Induction starts *worse* at 100M
and crosses over, widening to -0.084 by 1B — routing looks like a capability
that costs something to learn before it pays.

### Trunk, matched tokens (constant LR, NOT annealed)

Trunk numbers read systematically high — dense trunk at 800M is 3.4171 against
3.2773 annealed at 1B. Use these only trunk-to-trunk.

| tokens | dense | `cores_620m` | `cores_700m` |
|---|---|---|---|
| 200M | 4.1636 / 3.6531 | **4.0637** / 3.6612 | 4.4869 / 4.0475 |
| 400M | 3.6962 / 2.8477 | **3.6298** / 2.7785 | 3.7465 / 2.8399 |
| 600M | 3.5120 / 2.5684 | **3.4538** / 2.4983 | 3.5265 / 2.5669 |
| 800M | 3.4171 / 2.4071 | **3.3604** / 2.3622 | 3.3905 / 2.4180 |
| 1.2B | — | — | 3.2700 / 2.3091 |
| 1.8B | — | — | 3.1641 / 2.2074 |

## 5. Caveats that change how these read

1. **`cores_700m` runs at half the learning rate** (3e-4 vs 6e-4). Its weak
   start (+0.323 vs dense at 200M) and slow convergence are most likely the LR,
   not the architecture. It is **not** a controlled comparison against the other
   two, and should not be quoted as one.
2. **"FLOP-matched" holds on defined arithmetic, not on kernels.** The cores
   models execute 12-15% more real FLOPs (capacity padding + band overscan).
3. **Trunk != annealed.** Never mix rows from the two tables above.
4. **The wandb config for `cores_620m` is misleading if read raw.** All four
   ladder arms wrote into a single wandb run, so the stored config is whichever
   arm ran *last* — it shows `schedule=cooldown`, `warmup=0`, `tokens=2e8`,
   `decay_frac=0.2`, which is the **1B cooldown arm**. The trunk used
   `schedule=wsd`, `decay_frac=0`, `warmup=100`, `tokens=8e8`, matching dense.
5. `ddp_bucket_mb`, `dynamic`, `world_size` appear only in the cores config.
   They are flags added during this session, not deliberate differences; both
   runs were effectively single-process.
6. **Induction loss is a stratified slice**, not a separate eval set: positions
   whose 8-gram context recurs from more than `ind_window`=256 symbols back. It
   is keyed to a FIXED reference window across configs, never to each model's
   own window.

## 6. Provenance

| run | wandb id | state |
|---|---|---|
| `m5_ref_dense_130m_trunk` | `n2o7uy5x` | finished |
| `m5_ref_dense_130m_100M/300M/1B` | `fsvwabgx` / `pq1d56ae` / `rcvon85z` | finished |
| `m5_cores_620m` (all 4 arms, one run) | `99de0g3l` | finished |
| `m5_cores_700m_fixed2bank_trunk` | `6tmbhsv8` -> `luwuslul` -> `mw1a0qzp` | 2 crashes, resumed twice |

Entity/project: `luckymushy-individual/multicore`. Code: branch
`ddp-perf-fixes`. Throughput and profiling notes live in `ROUTED_PERF.md` and
`ROUTED_PERF_REPORT.md`.
