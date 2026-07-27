"""M5 architecture test: from-scratch joint training on byte-level text.

Three matched configs (see CORE_ROUTING_PLAN.md section 4, phase 2):
  cores       — sliding-window base + hefty core(s). The proposal.
  dense_full  — full-attention transformer, PARAM-matched to `cores` total.
  dense_local — sliding-window transformer, FLOPs-matched to `cores`.

Primary metrics: loss vs tokens (logged with FLOPs/token so loss-vs-compute
can be plotted), and STRATIFIED loss — positions whose 8-gram context
reoccurs from > IND_REF_WINDOW bytes back ("induction positions", the
recall-dependent slice where H2 predicts the cores' gain concentrates).
That reference distance is FIXED across configs (`--ind-window`), not each
model's own `cfg.window`: the point of the metric is to hold the set of
positions constant while the architecture varies. Keyed to cfg.window it was
not a comparison at all — the *_dense_full presets set window == seq_len, so
no position could reoccur from further back than the window and the slice was
EMPTY, reporting eval_loss_induction 0.0 as if it were a result.

Data: FineWeb-Edu, either as UTF-8 bytes (vocab 256) or as GPT-NeoX subword
tokens (vocab 50304) -- the preset's `vocab_size` picks the corpus, so a token
model can never be handed byte data and quietly report a byte-scale loss.
Downloaded once as parquet shards and cached as a flat local file, then sampled
deterministically: every config sees the identical symbol stream for a given
seed, and there is no live hub connection to drop mid-run. See
scripts/m5_data.py; build the cache up front with
`python scripts/m5_data.py --mode tokens --shards 11`. Use --synthetic for a
no-network smoke test.

Usage:
  python scripts/m5_arch.py --preset smoke_cores --iters 200 --synthetic
  python scripts/m5_arch.py --preset base_cores --tokens 2e9 --wandb
"""
import argparse, json, math, os, sys, time
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from core import CoreConfig, ModelConfig, SWTransformer
from core.base_model import keys_per_token
from core.losses import ce_chunk_default, ce_per_token, ce_sum
from m5_data import (DEFAULT_SHARDS, IND_NGRAM_BYTES, IND_NGRAM_TOKENS,
                     induction_mask, open_data)

# Reference distance for the induction slice, in symbols. 256 is the window the
# cores / dense_local presets run, i.e. "beyond what the sliding-window arms can
# reach" — but it is applied to EVERY config, including the full-attention ones,
# so the stratified loss compares the same positions everywhere.
#
# The UNIT changes with the corpus: 256 bytes of an 8-byte context, or 256
# tokens of a 4-token context. Both are recorded in metrics.json, because a
# byte-era eval_loss_induction and a token-era one are different measurements
# over different position sets and must never be put in the same table.
IND_REF_WINDOW = 256

# ---------------------------------------------------------------- presets
# window=T -> full attention. Cores: hefty at moderate rate (+ small memory
# core at low rate in the *_mem variants).


def presets(T):
    # north star: MOST params in the cores, activated for a fraction of
    # tokens. Two hefty cores at rate 1/8 with independent gates.
    hefty = CoreConfig(K=64, d_core=1024, n_heads=8, ffn_mult=4,
                       n_core_layers=2, target_rate=1 / 8)
    memory = CoreConfig(K=64, d_core=128, n_heads=4, ffn_mult=2,
                        n_core_layers=2, target_rate=1 / 64)
    # eight narrower/deeper cores instead of two wide ones: same base, same
    # per-core rate, more independent gates (SWTransformer builds one Core
    # module per list entry, so repeating the read-only config is fine)
    octo = CoreConfig(K=64, d_core=512, n_heads=8, ffn_mult=4,
                      n_core_layers=3, target_rate=1 / 8)
    routed = CoreConfig(
        K=64, d_core=384, n_heads=6, n_core_layers=1, ffn_mult=2,
        routing="top1_recurrent", n_loops=3, ffn_hidden=704,
        inter_core_window=256, residual_scale_init=0.1,
        router_aux_weight=0.01, router_hash_scale=0.5)

    return {
        # ---- smoke scale (minutes per run) ----
        # base ~14M + cores ~2x21M -> ~57M params, ~62% in cores;
        # FLOPs/token ~= a ~19M dense model + attention terms
        "smoke_cores": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[hefty, hefty]),
        "smoke_cores_8x": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[octo] * 8),
        "smoke_cores_top1_loopmix": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[routed] * 8),
        # ---- ablations of smoke_cores_top1_loopmix (compare to IT, not to
        # dense_local: only the unfold variant stays FLOPs-matched) ----
        # (1) no cross-core mixing: experts never exchange information and
        # rerouting sees an unmixed state. Drops 4.72M FLOPs/token (-10.8%).
        "smoke_cores_top1_nomix": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            cores=[replace(routed, inter_core_window=0)] * 8),
        # (2) unfolded recurrence: 3 independent expert weight sets instead of
        # one applied 3x. IDENTICAL FLOPs, 3x the expert parameters — tests
        # whether weight tying was costing anything.
        "smoke_cores_top1_unfold": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            cores=[replace(routed, tie_loops=False)] * 8),
        # (3) HETEROGENEOUS FIFO (ROUTED_CORE_NOTES.md "Important next test").
        # Horizon = K/p. Traffic p stays pinned at 1/8 for every expert — the
        # thing that collapses when you free it — and TEMPORAL diversity comes
        # from K instead: horizons 128/128/256/256/512/512/1024/1024 positions.
        # Every expert still sees the same number of training tokens, so no
        # expert is starved and packing stays balanced.
        "smoke_cores_top1_multik": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            cores=[replace(routed, K=128,
                           K_list=(16, 16, 32, 32, 64, 64, 128, 128))] * 8),
        # (4) BOUNDED LEARNED RATES (ROUTED_CORE_NOTES.md, the item after
        # heterogeneous FIFO). Traffic shares become learnable via a per-expert
        # routing bias; the positional prior anneals away over 2000 steps; the
        # exact-uniform Switch loss is replaced by a penalty that is EXACTLY
        # zero while every expert sits in [3%, 30%]. Inside that band rates are
        # free to differentiate — a 10x spread, i.e. horizons 213..2133 at
        # K=64 — and only dead or dominant experts are pushed back.
        "smoke_cores_top1_freerate": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            cores=[replace(routed, router_bias=True, hash_anneal_iters=2000,
                           rate_lo=0.03, rate_hi=0.30,
                           router_range_weight=1.0)] * 8),
        # (5) the two ablations that WON, combined: three independent expert
        # weight sets AND learnable traffic shares. Independent mechanisms —
        # untying touches the expert weights, the bias touches the router.
        "smoke_cores_top1_unfold_freerate": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            cores=[replace(routed, tie_loops=False, router_bias=True,
                           hash_anneal_iters=2000, rate_lo=0.03, rate_hi=0.30,
                           router_range_weight=1.0)] * 8),
        # PARAM-matched control for the unfold preset (42.23M params): the same
        # parameter budget spent DENSELY. 41.24M params (-2.3%) but 89.3M
        # FLOPs/token = 2.04x unfold's — which is the whole claim under test,
        # i.e. whether conditional routing reaches dense quality at half the
        # compute. d=512/L=13 rather than the closer-on-params d=384/L=24:
        # aspect ratio 39 is a normal transformer shape, 16 is not, and a
        # badly-shaped opponent would flatter the cores for the wrong reason.
        # Head dim stays 64, as in every other preset here.
        "smoke_dense_local_pm": ModelConfig(
            vocab_size=256, d_model=512, n_layers=13, n_heads=8, window=256,
            max_seq_len=T, core_layer=6, cores=[]),

        # ---- subword tokens (vocab 50304, GPT-NeoX). vocab_size > 256 is what
        # selects the token corpus, so these cannot be run on byte data.
        # THE EXTERNAL REFERENCE: open-sci-ref-v0.01-0.13b-fineweb-edu-1.4t-
        # 300B-4096, reproduced shape-for-shape. 22 layers, d=512, 8 heads
        # (head_dim 64), FFN 2256 with SwiGLU, RMSNorm, per-head qk-norm, RoPE
        # 10k, tied embeddings, full attention at seq 4096. ~99M non-embedding
        # + 25.75M tied embedding params. Their trained checkpoints (37 of
        # them, iter_2000..iter_72000, 4.13M tokens each) are on the hub, so
        # this preset exists to train OUR arm at their shape and to check the
        # parameter arithmetic — scripts/score_ref.py scores THEIR weights on
        # OUR eval set, which is what makes the two numbers comparable.
        "ref_dense_130m": ModelConfig(
            vocab_size=50304, d_model=512, n_layers=22, n_heads=8, window=T,
            max_seq_len=T, core_layer=11, cores=[], ffn_hidden=2256,
            rmsnorm=True, swiglu=True, qk_norm=True, tie_embeddings=True),
        # THE CORE ARCHITECTURE, at the reference's compute. 4 full-attention
        # prefix layers, 16 UNFOLDED top-1 routed layers of 8 experts each, 4
        # suffix layers; K=128 expert FIFO, W=256 chronological mixer between
        # depths, bounded learned traffic shares, tied NeoX head.
        #
        # core FFN 2152 is not a round number, it is the one that makes this
        # 342.6M FLOPs/token against ref_dense_130m's 342.8M -- 0.06% apart. So
        # the claim under test is exactly "6.0x the parameters at identical
        # compute": 621.5M params, 595.7M non-embedding, 94% of them in cores.
        #
        # capacity 2.5 >= M * rate_hi = 8 * 0.30, so the cap never binds before
        # the range penalty does (core/config.py). Without a cap this preset
        # OOMs at initialisation, not in steady state -- the router collapse is
        # transient but the memory spike is not survivable.
        #
        # Chinchilla for 595.7M non-embedding is ~11.9B tokens; at 1B it sits
        # at 1.7 tokens/param and every expert is starved. Read a short run as
        # a point on the loss-vs-compute curve against ref_dense_130m at the
        # SAME budget (both are FLOPs-matched per token, so equal tokens is
        # equal compute), never as a verdict on the architecture.
        "cores_620m": ModelConfig(
            vocab_size=50304, d_model=512, n_layers=8, n_heads=8, window=T,
            max_seq_len=T, core_layer=3, ffn_hidden=2256,
            rmsnorm=True, swiglu=True, qk_norm=True, tie_embeddings=True,
            cores=[CoreConfig(
                K=128, d_core=512, n_heads=8, n_core_layers=1,
                ffn_hidden=2152, routing="top1_recurrent",
                n_loops=16, tie_loops=False, inter_core_window=256,
                residual_scale_init=0.1, router_bias=True,
                hash_anneal_iters=2000, rate_lo=0.03, rate_hi=0.30,
                router_range_weight=1.0, router_hash_scale=0.5,
                capacity_factor=2.5)] * 8),

        # Cheap token-corpus smoke tests: same shapes as the byte smoke presets
        # so the pipeline (uint16 cache, chunked CE, 4-token induction) can be
        # exercised in minutes before anything is rented. The architecture to
        # actually run against the reference is not decided yet.
        "smoke_tok_dense": ModelConfig(
            vocab_size=50304, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[],
            rmsnorm=True, swiglu=True, qk_norm=True, tie_embeddings=True),
        "smoke_tok_top1": ModelConfig(
            vocab_size=50304, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            # 704 -> 472: `routed` sets ffn_hidden explicitly, and an explicit
            # width is not rescaled for SwiGLU's third matrix (see
            # base_model.ffn_hidden), so leaving it would be a silent 1.5x FFN.
            cores=[replace(routed, tie_loops=False, router_bias=True,
                           hash_anneal_iters=2000, ffn_hidden=472)] * 8,
            rmsnorm=True, swiglu=True, qk_norm=True, tie_embeddings=True),

        # ---- rate sweep: SAME conditional FLOPs (rate x core params fixed),
        # trading sparsity against how much data each core param sees.
        # rate 1/8 => a core param gets gradient from 12.5% of tokens; at a
        # 1.5B-token budget that is ~4 tokens/param (starved). Higher rate,
        # smaller cores => same compute, far more signal per param.
        "smoke_rate_half": ModelConfig(     # rate 1/2, small cores
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            cores=[CoreConfig(K=64, d_core=512, n_heads=8, ffn_mult=4,
                              n_core_layers=2, target_rate=1 / 2)] * 2),
        "smoke_rate_quarter": ModelConfig(  # rate 1/4, mid cores
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4,
            cores=[CoreConfig(K=64, d_core=736, n_heads=8, ffn_mult=4,
                              n_core_layers=2, target_rate=1 / 4)] * 2),
        "smoke_cores_mem": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[hefty, hefty, memory]),
        # THE MISSING CONTROL: the cores presets' own base, cores removed.
        # Every cores-vs-dense comparison so far used smoke_dense_local, which
        # has 11 layers against the cores base's 8 -- so a cores loss could not
        # be told apart from "three fewer layers". This isolates what the cores
        # actually add.
        "smoke_base_only": ModelConfig(
            vocab_size=256, d_model=384, n_layers=8, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[]),
        "smoke_dense_full": ModelConfig(   # param-matched to smoke_cores
            vocab_size=256, d_model=640, n_layers=12, n_heads=10, window=T,
            max_seq_len=T, core_layer=4, cores=[]),
        "smoke_dense_local": ModelConfig(  # FLOPs-matched to smoke_cores
            vocab_size=256, d_model=384, n_layers=11, n_heads=6, window=256,
            max_seq_len=T, core_layer=4, cores=[]),
        # ---- base scale (~180M total for cores config) ----
        "base_cores": ModelConfig(
            vocab_size=256, d_model=768, n_layers=10, n_heads=12, window=512,
            max_seq_len=T, core_layer=5,
            cores=[CoreConfig(K=64, d_core=2048, n_heads=16, ffn_mult=4,
                              target_rate=1 / 8),
                   CoreConfig(K=64, d_core=256, n_heads=4, ffn_mult=2,
                              target_rate=1 / 64)]),
        "base_dense_full": ModelConfig(
            vocab_size=256, d_model=1024, n_layers=14, n_heads=16, window=T,
            max_seq_len=T, core_layer=5, cores=[]),
        "base_dense_local": ModelConfig(
            vocab_size=256, d_model=768, n_layers=11, n_heads=12, window=512,
            max_seq_len=T, core_layer=5, cores=[]),
    }


def flops_per_token(model, cfg, T, rates=None):
    """Estimated forward FLOPs/token: 2*active_params + attention + LM head.
    Core params count at their admission rate (that's the whole point).

    Two things this used to get wrong. At the open-sci-ref shape (d=512,
    V=50304, T=4096, full attention) they together overstated the true 342.8M
    by 143.7M, i.e. 42%:

      EMBEDDINGS ARE A LOOKUP, not a matmul. Counting the (V, d) table at
      2*params added 2*V*d = 51.5M FLOPs/token that nothing computes. The
      table is subtracted here, and the LM head — which IS a matmul — is added
      back explicitly, so a tied-embedding model (where the two are one
      tensor) and an untied one are both counted once and correctly.

      CAUSAL ATTENTION AVERAGES (T+1)/2 KEYS, not T. The old min(window, T)
      was right for a sliding window and 2x too high for the full-attention
      presets; `keys_per_token` is exact for both.

    `rates` (one per core, in aux-dict order) replaces each core's ADVERTISED
    `cc.target_rate` with its MEASURED admission rate. With the quantile
    controller on, the two agree by construction and the default None is the
    honest static estimate; under `--free-rate` the advertised rate is only
    where the run started, and a loss can only be read against the compute the
    model actually chose to spend. See `flops_per_token_measured`.

    A batched `MultiCore` is ONE module holding M cores (its `parameters()`
    cover all M, and it emits M aux dicts), so its M rates collapse into their
    mean: every core in it is the same size, so mean-rate x all-M-params is
    exactly the per-core sum. That also keeps this arithmetic identical to the
    pre-`rates` version whenever rates == target_rate, measured or not.
    """
    raw = getattr(model, "_orig_mod", model)
    core_params = sum(sum(p.numel() for p in c.parameters())
                      for c in model.cores)
    emb = raw.tok_emb.weight.numel()
    if raw.pos_emb is not None:
        emb += raw.pos_emb.weight.numel()
    base_params = model.num_params() - core_params - emb
    f = 2 * base_params
    if cfg.tie_embeddings:      # the head IS the table we just subtracted
        f += 2 * cfg.d_model * cfg.vocab_size
    f += cfg.n_layers * 4 * cfg.d_model * keys_per_token(cfg.window, T)
    routed = [c for c in model.cores if getattr(c, "is_top1_routed", False)]
    if routed:
        assert len(routed) == 1 and len(model.cores) == 1
        return f + routed[0].estimated_flops_per_token(T)

    i = 0
    for c, cc in zip(model.cores, cfg.cores):
        n = getattr(c, "M", 1)                     # cores inside this module
        r = cc.target_rate if rates is None else sum(rates[i:i + n]) / n
        i += n
        cp = sum(p.numel() for p in c.parameters())
        f += r * (2 * cp + 2 * 2 * cc.K * cc.d_core)
    return f


def token_tag(tokens):
    """Filename/run-name tag for a token count: 8e8 -> '800M', 1e9 -> '1B'.

    Shared by the checkpoint writer and the suite that has to reconstruct the
    path. `%.0e` was the obvious choice and is wrong: it rounds 4.8e8 and
    5.0e8 to the same '5e8', so two branch points of one ladder would collide
    on one file and the second cooldown would silently resume the first's.
    """
    t = float(tokens)
    return f"{t / 1e9:g}B" if t >= 1e9 else f"{t / 1e6:g}M"


def lr_schedule(it, lr, iters, warmup, schedule, decay_frac=0.2,
                final_frac=0.0):
    """Learning rate at optimizer step `it`. Module-level so it is testable.

    cosine    linear warmup, then cosine from the peak to 0. Pythia's.
    wsd       linear warmup, then the peak held until the last `decay_frac` of
              steps, then LINEAR to `final_frac` of the peak. open-sci-ref's
              (verified: "constant learning rate with linear cooldown", 20%).
              decay_frac 0 makes it a pure trunk with no cooldown at all.
    cooldown  the branch half: no warmup, linear from the peak to `final_frac`
              over the whole run. Meant for `--resume`ing a trunk checkpoint.

    The trunk/cooldown split is what makes one run yield several honest
    endpoints: on a constant lr the trunk is at no particular point in any
    schedule, so branching anywhere and annealing gives the number a run of
    that length would have reached. Reading loss off a mid-run COSINE
    checkpoint does not — it is sitting at a high lr and reports too high.
    """
    if schedule == "cooldown":
        return lr * (1 - (it / max(iters, 1)) * (1 - final_frac))
    if it < warmup:
        return lr * it / warmup
    p = (it - warmup) / max(iters - warmup, 1)
    if schedule == "cosine":
        return lr * 0.5 * (1 + math.cos(math.pi * p))
    start = 1.0 - decay_frac
    if p <= start:
        return lr
    return lr * (1 - (p - start) / max(decay_frac, 1e-9) * (1 - final_frac))


def train_loss(model, head_w, idx, chunk=0):
    """The training objective on one micro-batch: mean next-token CE, plus any
    router aux losses. -> (ce, router_aux, auxes).

    Factored out of the loop so tests exercise the exact code that trains --
    a vocabulary bug here (the old hard-coded `reshape(-1, 256)`) returns a
    plausible finite number rather than crashing, so the only real gate is
    running this path and checking the value against ln(vocab).

    The loss is computed INSIDE the model's forward (`targets=`), not from
    returned hidden states. Under DDP that is the difference between correct
    and silently wrong: DDP hooks the parameters touched inside the wrapped
    forward, and a head matmul done outside it leaves `head.weight` out of the
    all-reduce. `head_w` is kept in the signature for callers that still hold
    one, and is unused.
    """
    ce, auxes = model(idx[:, :-1], collect_aux=True, targets=idx[:, 1:],
                      ce_chunk=chunk)
    aux = sum((a.get("router_aux_loss", 0.0) for a in auxes), start=0.0)
    return ce, aux, auxes


def flops_per_token_measured(model, cfg, T, auxes):
    """FLOPs/token at the admission rates the cores were MEASURED running.

    The static estimate is a property of the config; this is a property of the
    batch, and under a free rate they are different numbers. Logged every eval
    as `flops_per_token_measured` next to the static `flops_per_token`, so
    loss-vs-compute is plotted against compute actually spent.
    """
    return flops_per_token(model, cfg, T,
                           rates=[float(a["rate"]) for a in auxes])


# ---------------------------------------------------------------- data
def stream_batches(B, T, ind_window, device, synthetic=False, seed=0,
                   with_masks=False, split="train", data=None,
                   data_shards=DEFAULT_SHARDS, data_dir=None, vocab_size=256,
                   skip=0):
    """Yields (idx (B,T+1) long, ind_mask (B,T) bool or None) forever.

    Non-synthetic: random contiguous windows over the cached local symbol file
    (scripts/m5_data.py), drawn from a seeded numpy Generator -- so a given
    (seed, B, T) gives every model config the identical sequence of batches.
    split="eval" draws from the held-out tail, which training never touches.

    `ind_window` is the induction slice's reference distance (IND_REF_WINDOW),
    NOT the model's attention window, and it only matters when with_masks --
    which is the eval stream only. The n-gram length comes from the corpus
    (8 bytes / 4 tokens); see m5_data.
    """
    if synthetic:
        g = torch.Generator().manual_seed(seed)
        n = IND_NGRAM_BYTES if vocab_size <= 256 else IND_NGRAM_TOKENS
        bits = 8 if vocab_size <= 256 else 16
        for _ in range(skip):
            torch.randint(0, vocab_size, (B, T + 1), generator=g)
        while True:
            idx = torch.randint(0, vocab_size, (B, T + 1), generator=g)
            masks = (torch.stack([induction_mask(idx[b, :-1], ind_window, n,
                                                 bits)
                                  for b in range(B)])
                     if with_masks else None)
            yield idx.to(device), masks.to(device) if masks is not None else None
        return
    if data is None:
        data = open_data(vocab_size > 256, data_shards, data_dir)
    yield from data.batches(B, T, ind_window, device, seed=seed, split=split,
                            with_masks=with_masks, skip=skip)


# ------------------------------------------------------------- diagnostics
def core_diagnostics(model, auxes):
    """CORE_ROUTING_PLAN.md section 10, the multi-core half. Once per eval.

    These are the metrics whose absence let gate collapse survive a full day of
    experiments: eight cores had silently become one replicated core, and loss
    curves alone cannot say that.

      gate_cos_mean / gate_cos_max — mean and max pairwise ABSOLUTE cosine
        between the gate directions. The collapse alarm. Absolute because a
        core reading -k admits the complement of k's tokens but is the same
        one-dimensional notion of salience. Random directions in d_model=384
        sit near sqrt(2/(pi*d)) ~= 0.04; the measured collapsed run was 0.999.
      sel_jaccard_mean — mean pairwise Jaccard overlap of the ADMITTED token
        sets on this batch. The primary multi-core diagnostic: near 1.0 is
        redundancy regardless of loss, near 0.0 means check nobody is dead.
        Orthogonal directions bound this loosely, not tightly — two orthogonal
        gates can still admit overlapping sets — so it is measured, not
        assumed. Independent gates at rate r would overlap at ~r/(2-r)
        (0.067 at r = 1/8).
      delta_rms_ratio — RMS of the summed core delta over the RMS of the
        residual stream the cores read. Near zero = the cores are dead weight;
        comparable to the residual = they are destabilising the base.
      rate_mean — mean MEASURED admission rate over the cores. With the
        quantile controller on this is target_rate by construction and is
        logged only as a check; under `--free-rate` it is the experiment's
        answer (what rate the model chooses), and the number
        `flops_per_token_measured` is computed from.
      tau_mean / tau_z_mean — the admission threshold, absolute and in units of
        the score distribution it is a threshold ON: tau_z = (tau - mean s)/
        std s. Under `--free-rate` tau is the learned parameter and rate_mean
        is its consequence, so a rate that hits 0 or 1 has to be read against
        tau_z: |tau_z| of a few means the model chose it, |tau_z| of 10 means
        the score distribution drifted away from a threshold that could not
        follow (measured — see core_module._tau_maintain_).

    Cosines come off the parameters, so they are exact; Jaccard and the delta
    ratio come off one eval batch's masks/deltas.
    """
    raw = getattr(model, "_orig_mod", model)
    out = {}
    if not auxes:
        return out
    n = len(auxes)
    for key, dst in (("rate", "rate_mean"), ("tau", "tau_mean"),
                     ("tau_z", "tau_z_mean")):
        out[dst] = sum(float(a[key]) for a in auxes) / n

    # ---- delta vs residual. `delta_group` says how many aux dicts share one
    # delta (MultiCore's M cores sum into one), so step over each group once;
    # the reference residual is the h the FIRST core read, i.e. pre-core.
    tot_sq, h_ref, i = 0.0, float(auxes[0]["h_rms"]), 0
    while i < len(auxes):
        a = auxes[i]
        tot_sq += float(a["delta_rms"]) ** 2
        i += int(a.get("delta_group", 1))
    out["delta_rms_ratio"] = tot_sq ** 0.5 / max(h_ref, 1e-9)

    # ---- gate direction collapse
    ks = [c.k_dir.detach().float() for c in raw.cores if hasattr(c, "k_dir")]
    if ks:
        k = torch.cat([x.reshape(-1, raw.cfg.d_model) for x in ks], 0)
        if k.shape[0] >= 2:
            n = k / k.norm(dim=1, keepdim=True).clamp_min(1e-12)
            c = (n @ n.t()).abs()
            off = ~torch.eye(k.shape[0], dtype=torch.bool, device=k.device)
            out["gate_cos_mean"] = float(c[off].mean())
            out["gate_cos_max"] = float(c[off].max())
    routers = [c.router_w.detach().float() for c in raw.cores
               if hasattr(c, "router_w")]
    if routers:
        r = torch.cat(routers, 0)
        rn = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
        rc = (rn @ rn.t()).abs()
        off = ~torch.eye(r.shape[0], dtype=torch.bool, device=r.device)
        out["router_cos_mean"] = float(rc[off].mean())
        out["router_cos_max"] = float(rc[off].max())

    # ---- selection overlap: |m_i & m_j| / |m_i | m_j| over all pairs, as one
    # (n_cores, B*T) float matmul rather than a python loop over pairs.
    if len(auxes) >= 2:
        ms = torch.stack([a["m"].reshape(-1) for a in auxes]).float()
        inter = ms @ ms.t()
        sz = ms.sum(1)
        j = inter / (sz[:, None] + sz[None, :] - inter).clamp_min(1.0)
        off = ~torch.eye(ms.shape[0], dtype=torch.bool, device=ms.device)
        out["sel_jaccard_mean"] = float(j[off].mean())
    for key in ("router_entropy", "router_margin", "pack_util", "pack_overhead",
                "rate_min", "rate_max", "rate_cv"):
        if key in auxes[0]:
            out[key] = float(auxes[0][key])
    # bounded-learned-rate mode: how far the per-expert routing bias has
    # actually moved. A spread pinned near 0 means the rates never
    # differentiated and the experiment is a null result, not a win.
    # read straight off the module rather than through the aux dict: the
    # forward runs under torch.compile and must not build host tensors
    for c in raw.cores:
        if getattr(c, "is_top1_routed", False) and c.cfg.hash_anneal_iters:
            frac = max(0.0, 1.0 - int(c._step) / c.cfg.hash_anneal_iters)
            out["hash_scale"] = c.cfg.router_hash_scale * frac
    biases = [c.r_bias.detach().float() for c in raw.cores
              if hasattr(c, "r_bias")]
    if biases:
        b = torch.cat(biases)
        out["router_bias_min"] = float(b.min())
        out["router_bias_max"] = float(b.max())
        out["router_bias_spread"] = float(b.max() - b.min())

    if "loop_rates" in auxes[0]:
        lr = auxes[0]["loop_rates"]
        for li in range(lr.shape[0]):
            for mi in range(lr.shape[1]):
                out[f"loop{li}_core{mi}_rate"] = float(lr[li, mi])
    return out


# ---------------------------------------------------------------- train
def evaluate(model, eval_data, device, cfg=None, T=None, chunk=0):
    model.eval()
    head_w = getattr(model, "_orig_mod", model).head.weight
    tot, tot_n, ind, ind_n = 0.0, 0, 0.0, 0
    diag = {}
    with torch.no_grad():
        for bi, (idx, mask) in enumerate(eval_data):
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                # collect_aux on every batch (one compiled graph, not two);
                # the diagnostics are read off the first batch only.
                hidden, auxes = model(idx[:, :-1], collect_aux=True,
                                      return_hidden=True)
            if bi == 0:
                diag = core_diagnostics(model, auxes)
                # tau is frozen in eval (I5), so these ARE the rates the
                # eval loss was produced at
                if auxes and cfg is not None:
                    diag["flops_per_token_measured"] = \
                        flops_per_token_measured(model, cfg, T, auxes)
                    raw = getattr(model, "_orig_mod", model)
                    routed = [c for c in raw.cores if
                              getattr(c, "is_top1_routed", False)]
                    if routed and "pack_util" in diag:
                        expert_f, mixer_f = routed[0].estimated_flops_parts(T)
                        base_f = diag["flops_per_token_measured"] - expert_f - mixer_f
                        diag["flops_per_token_padded"] = (
                            base_f + mixer_f + expert_f * diag["pack_overhead"])
            ce = ce_per_token(hidden.reshape(-1, hidden.shape[-1]), head_w,
                              idx[:, 1:].reshape(-1), chunk)
            tot += float(ce.sum()); tot_n += ce.numel()
            sel = mask.reshape(-1)
            ind += float(ce[sel].sum()); ind_n += int(sel.sum())
    model.train()
    return {"eval_loss": tot / max(tot_n, 1),
            "eval_loss_induction": ind / max(ind_n, 1),
            "eval_induction_frac": ind_n / max(tot_n, 1), **diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--tokens", type=float, default=None,
                    help="overrides --iters: train until this many tokens")
    ap.add_argument("--batch", type=int, default=16,
                    help="MICRO batch: sequences per forward. The optimizer "
                         "sees --batch * --grad-accum of them.")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches accumulated per optimizer step. What "
                         "makes the batch size a free variable: the reference "
                         "runs at 1008 sequences x 4096 = 4.13M tokens/step, "
                         "and a 16-sequence step is a different optimisation "
                         "problem at the same token count, not the same one "
                         "run cheaply. Set batch to what fits and grad_accum "
                         "to whatever reaches the target.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--warmup-frac", type=float, default=0.0,
                    help="if > 0, warmup = this fraction of --iters, "
                         "overriding --warmup. A fixed step count cannot serve "
                         "a token ladder: 300 steps is 8% of a 1B-token run "
                         "and 79% of a 100M-token one.")
    ap.add_argument("--schedule", choices=("cosine", "wsd", "cooldown"),
                    default="cosine",
                    help="cosine (to 0), warmup-stable-decay, or a bare "
                         "cooldown. open-sci-ref uses WSD -- constant lr, then "
                         "a LINEAR decay over the last --decay-frac -- and "
                         "Pythia uses cosine. `cooldown` is the branch half of "
                         "WSD: no warmup, decay from the peak over the whole "
                         "run, for resuming a --decay-frac 0 trunk at a "
                         "checkpoint. That is why WSD is worth the trouble: "
                         "one trunk yields an honest endpoint at every branch "
                         "point, where a cosine checkpoint pulled mid-run is "
                         "sitting at a high lr and reads worse than a run of "
                         "that length actually gets.")
    ap.add_argument("--decay-frac", type=float, default=0.2,
                    help="0 makes --schedule wsd a pure warmup+constant trunk")
    ap.add_argument("--final-lr-frac", type=float, default=0.0)
    ap.add_argument("--save-at", default=None,
                    help="comma-separated token counts to write a full "
                         "(model + optimizer) checkpoint at, e.g. "
                         "80e6,240e6,800e6. These are the WSD branch points.")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume from: restores weights, "
                         "OPTIMIZER STATE, and the training stream position. "
                         "Adam moments matter -- restarting them at zero puts "
                         "a transient right where the cooldown is supposed to "
                         "be measuring.")
    ap.add_argument("--ce-chunk", type=int, default=None,
                    help="rows per cross-entropy chunk (see ce_sum). Default "
                         "follows the vocabulary: 0 (off) at vocab <= 1024, "
                         "1024 above it.")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--ind-window", type=int, default=IND_REF_WINDOW,
                    help="reference distance (SYMBOLS -- bytes or tokens, "
                         "whichever the preset's vocab selects) for the "
                         "induction slice. FIXED across configs on purpose -- "
                         "using each model's own cfg.window made the metric "
                         "incomparable, and empty for window==seq_len presets.")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--data-shards", type=int, default=DEFAULT_SHARDS,
                    help="FineWeb-Edu parquet shards to cache locally; each "
                         "yields 3.47 GB of bytes, or ~0.75B NeoX tokens")
    ap.add_argument("--data-dir", default=None,
                    help="where the byte cache lives (default runs/data)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--free-rate", action="store_true",
                    help="learned_tau for every core in the preset: the "
                         "quantile controller runs once (so the run starts at "
                         "the preset's target_rate) and after that tau is a "
                         "parameter the task loss moves, i.e. the admission "
                         "rate is the model's to choose. Read rate_mean and "
                         "flops_per_token_measured, not the static estimate -- "
                         "and read tau_z_mean before believing either: an "
                         "absolute tau at the base lr is measurably too slow "
                         "to follow the early drift of the score distribution "
                         "(core_module._tau_maintain_).")
    ap.add_argument("--no-ortho", action="store_true",
                    help="disable the orthogonal gate-direction constraint "
                         "(plan 5.9). The controlled with/without arm: without "
                         "it, all 8 gate directions of smoke_cores_8x were "
                         "measured collapsing to |cos| 0.999 by 11k steps.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = args.seq_len
    # cfg first (it is pure and a typo'd preset should not cost a download),
    # then data -- the build is a one-off download and must not race the model
    # onto the GPU. The preset's vocab picks the corpus: <= 256 is the byte
    # cache, above it the uint16 NeoX-token cache. Nothing else is consulted,
    # so a token model can never be fed byte data and report a byte-scale loss
    # that looks like a triumph.
    cfg = presets(T)[args.preset]
    tokenized = cfg.vocab_size > 256
    data = None if args.synthetic else open_data(
        tokenized, args.data_shards, args.data_dir)
    ce_chunk = (ce_chunk_default(cfg.vocab_size) if args.ce_chunk is None
                else args.ce_chunk)
    if args.free_rate:
        # replace, not mutate: the presets hand out SHARED CoreConfig objects
        # ([octo] * 8), and dataclass equality is by value, so the batched
        # MultiCore path still sees eight identical configs.
        cfg.cores = [replace(c, learned_tau=True) for c in cfg.cores]
    model = SWTransformer(cfg).to(device)
    if args.compile:
        # dynamic=True ONLY when something in the graph has a data-dependent
        # shape. The cores do: `pack_indices` sizes its buffer by the busiest
        # expert on the batch, so static shapes would recompile every step.
        # A dense preset has no such axis, and forcing dynamic there is a
        # straight loss — symbolic shapes block the constant folding and
        # tiling choices inductor makes when it knows T and B. `None` lets
        # dynamo compile static first and fall back if a shape ever moves.
        dyn = True if cfg.cores else None
        try:
            compiled = torch.compile(model, dynamic=dyn)
            # torch.compile is LAZY: it returns a wrapper and does the work on
            # the first forward, so wrapping only the call above catches
            # nothing and a broken inductor (no C++ toolchain, a bad triton, a
            # kernel it cannot lower) takes the run down minutes into a rented
            # instance. Force the compile here, at the real shape, where it can
            # still fall back to eager and finish the run 30% slower instead of
            # not at all.
            model.eval()
            with torch.no_grad(), torch.autocast(
                    device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                compiled(torch.zeros(args.batch, T, dtype=torch.long,
                                     device=device))
            model = compiled
            print(f"[compile] on, dynamic={dyn}", flush=True)
        except Exception as e:
            print(f"[compile] unavailable ({type(e).__name__}: {e}); "
                  f"running EAGER", flush=True)
        model.train()
    raw_model = getattr(model, "_orig_mod", model)
    head_w = raw_model.head.weight
    fpt = flops_per_token(model, cfg, T)
    n_params = model.num_params()
    # Non-embedding params: the number that is comparable across vocabularies.
    # Embeddings scale V*d and the body scales L*d^2, so at vocab 50304 the
    # (V, d) table is 25.75M of a 125M "0.13B" model -- quoting totals would
    # make "% of params in the cores" partly a statement about a lookup table.
    emb_params = raw_model.tok_emb.weight.numel() + (
        0 if raw_model.pos_emb is None else raw_model.pos_emb.weight.numel())
    if not cfg.tie_embeddings:
        emb_params += head_w.numel()
    n_body = n_params - emb_params
    per_step = args.batch * args.grad_accum * T
    if args.tokens:
        args.iters = int(args.tokens / per_step)
    if args.warmup_frac > 0:
        args.warmup = max(1, int(args.warmup_frac * args.iters))
    if args.schedule == "cooldown":
        args.warmup = 0             # the trunk already warmed this model up

    # ---- resume. Loaded before the streams and the optimizer are built: the
    # stream has to fast-forward past what the trunk consumed, and the Adam
    # moments have to land on an optimizer that already exists.
    ck, skip, tok0 = None, 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        # every one of these changes which tokens the stream yields or what
        # the weights mean, so a mismatch is a silently wrong experiment
        for k, mine in (("preset", args.preset), ("batch", args.batch),
                        ("grad_accum", args.grad_accum), ("seq_len", T),
                        ("seed", args.seed)):
            assert ck[k] == mine, f"resume mismatch on {k}: {ck[k]} vs {mine}"
        raw_model.load_state_dict(ck["model"])
        skip, tok0 = ck["iter"] * ck["grad_accum"], ck["tokens"]
        print(f"[resume] {args.resume}: step {ck['iter']}, "
              f"{tok0/1e6:.4g}M tokens consumed; continuing the stream at "
              f"batch {skip}", flush=True)
    save_at = sorted(int(float(x)) for x in args.save_at.split(",")) \
        if args.save_at else []
    print(f"{args.preset}: {n_params/1e6:.1f}M params "
          f"({n_body/1e6:.1f}M non-embedding), ~{fpt/1e6:.1f}M FLOPs/token, "
          f"{args.iters} steps x {per_step/1e6:.3f}M tokens "
          f"({args.iters*per_step/1e9:.2f}B tokens) on {device}, "
          f"{'tokens' if tokenized else 'bytes'} corpus, ce_chunk={ce_chunk}, "
          f"lr {args.lr:g} {args.schedule} warmup {args.warmup}"
          f"{' [FREE RATE: tau learned]' if args.free_rate else ''}",
          flush=True)

    run_name = args.run_name or f"m5_{args.preset}_s{args.seed}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join("runs", "LATEST"), "w") as f:
        f.write(run_name)
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(project="multicore", name=run_name,
                        config={**vars(args), "params": n_params,
                                "flops_per_token": fpt})

    def fmt(v):
        """Console rounding that does not erase the diagnostics. round(v, 5)
        prints gate_cos_mean 1e-8 (orthogonal) and 3e-5 (drifting) both as
        0.0 -- i.e. the collapse alarm reads identically whatever it says.
        wandb always gets the raw float."""
        if not isinstance(v, float):
            return v
        return round(v, 5) if v == 0.0 or abs(v) >= 1e-4 else float(f"{v:.2e}")

    def log(m):
        print(json.dumps({k: fmt(v) for k, v in m.items()}), flush=True)
        if wb:
            wb.log(m)

    # held-out eval set: fixed windows from the tail of the byte file, which
    # the train split never reaches. Same bytes for every config and run.
    eval_stream = stream_batches(args.batch, T, args.ind_window, device,
                                 synthetic=args.synthetic, seed=args.seed + 999,
                                 with_masks=True, split="eval", data=data,
                                 vocab_size=cfg.vocab_size)
    eval_data = [next(eval_stream) for _ in range(args.eval_batches)]
    train_stream = stream_batches(args.batch, T, args.ind_window, device,
                                  synthetic=args.synthetic, seed=args.seed,
                                  split="train", data=data,
                                  vocab_size=cfg.vocab_size, skip=skip)

    # tau (only a parameter under --free-rate) is exempt from weight decay:
    # decay pulls it toward 0, and tau == 0 means "admit every token whose
    # score is positive", i.e. ~50% -- the optimizer's regulariser would be
    # choosing the admission rate that this run exists to measure. Without
    # --free-rate there are no tau parameters and `groups` IS model.parameters().
    taus = [p for n, p in model.named_parameters() if n.split(".")[-1] == "tau"]
    if taus:
        tau_ids = {id(p) for p in taus}
        groups = [{"params": [p for p in model.parameters()
                              if id(p) not in tau_ids]},
                  {"params": taus, "weight_decay": 0.0}]
    else:
        groups = model.parameters()
    try:
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.1,
                                betas=(0.9, 0.95), fused=(device == "cuda"))
    except (RuntimeError, ValueError) as e:
        # fused rejects some param layouts; foreach is the next best thing
        print(f"[opt] fused AdamW unavailable ({e}); using foreach", flush=True)
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.1,
                                betas=(0.9, 0.95))
    if ck is not None:
        opt.load_state_dict(ck["opt"])
        del ck                      # ~1 GB of Adam moments, already copied in

    def lr_at(it):
        return lr_schedule(it, args.lr, args.iters, args.warmup, args.schedule,
                           args.decay_frac, args.final_lr_frac)

    # gate directions are orthogonalised once before step 1 and re-projected
    # after every step, so collapse is structurally unreachable rather than
    # merely penalised (plan 5.9). Skipped entirely under --no-ortho, which
    # leaves the pre-mechanism behaviour bit-identical.
    if not args.no_ortho:
        raw_model.reproject_gates()

    model.train()
    t0 = time.time()
    final = {}
    for it in range(1, args.iters + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(it)
        opt.zero_grad(set_to_none=True)
        # Accumulated losses stay TENSORS across the inner loop: a float()
        # here would sync the device once per micro-batch, and at the
        # accumulation depths this exists for (63 micro-batches to reach the
        # reference's 1008 sequences) that is 63 stalls a step to log a number
        # read once every --eval-every.
        ce_acc = aux_acc = None
        for _ in range(args.grad_accum):
            idx, _ = next(train_stream)
            with torch.autocast(device, dtype=torch.bfloat16,
                                enabled=(device == "cuda")):
                ce_loss, router_aux_loss, auxes = train_loss(
                    model, head_w, idx, ce_chunk)
                loss = (ce_loss + router_aux_loss) / args.grad_accum
            loss.backward()
            ce_d = ce_loss.detach()
            aux_d = (router_aux_loss.detach()
                     if torch.is_tensor(router_aux_loss)
                     else torch.as_tensor(float(router_aux_loss),
                                          device=ce_d.device))
            ce_acc = ce_d if ce_acc is None else ce_acc + ce_d
            aux_acc = aux_d if aux_acc is None else aux_acc + aux_d
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not args.no_ortho:
            raw_model.reproject_gates()
        tokens_done = tok0 + it * per_step
        # WSD branch points. `tok0` carries the trunk's count across a resume,
        # so a checkpoint is named for the tokens the MODEL has seen, not for
        # how far this particular process got.
        #
        # The `it == args.iters` clause matters: iters = int(tokens/per_step)
        # truncates, so a run whose own budget is also a branch point (the
        # usual case — the trunk ends exactly at the largest one) finishes up
        # to per_step-1 tokens SHORT of it and would never write the last
        # checkpoint. Anything within one step of the target at the final step
        # is that target.
        reach = tokens_done + (per_step if it == args.iters else 0)
        while save_at and reach >= save_at[0]:
            path = os.path.join(run_dir,
                                f"ckpt_{token_tag(save_at.pop(0))}.pt")
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "tokens": tokens_done,
                        "preset": args.preset, "batch": args.batch,
                        "grad_accum": args.grad_accum, "seq_len": T,
                        "seed": args.seed, "lr": args.lr}, path)
            print(f"CHECKPOINT {path} at {tokens_done/1e6:.4g}M tokens "
                  f"(step {it})", flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            m = evaluate(model, eval_data, device, cfg, T, ce_chunk)
            ce_mean = float(ce_acc) / args.grad_accum
            aux_mean = float(aux_acc) / args.grad_accum
            m.update({"iter": it, "loss": ce_mean + aux_mean,
                      "ce_loss": ce_mean,
                      "router_aux_loss": aux_mean,
                      "lr": lr_at(it),
                      "tokens": tokens_done,
                      "tok_per_s": it * per_step / (time.time() - t0)})
            for ci, a in enumerate(auxes):
                m[f"core{ci}_rate"] = float(a["rate"])
            final = m
            log(m)
            # raw_model, not model: under --compile the wrapper prefixes every
            # key with _orig_mod., which no plain SWTransformer can load back
            torch.save({"model": raw_model.state_dict(), "iter": it,
                        "tokens": tokens_done, "config": args.preset},
                       os.path.join(run_dir, "latest.pt"))
    final["preset"] = args.preset
    final["params"] = n_params
    final["params_non_embedding"] = n_body
    final["flops_per_token"] = fpt
    final["tokens_per_step"] = per_step
    final["schedule"] = args.schedule
    final["lr_peak"] = args.lr
    # a cooldown branch's loss is only interpretable next to where it branched
    final["resumed_from"] = args.resume
    final["tokens_at_resume"] = tok0
    # recorded so a stratified loss can never be read against runs that used a
    # different (or, pre-fix, per-config) induction reference distance -- or,
    # now, a different UNIT: 256 bytes of 8-byte context and 256 tokens of
    # 4-token context are not the same slice of the same corpus.
    final["ind_window"] = args.ind_window
    final["ind_units"] = "tokens" if tokenized else "bytes"
    final["ind_ngram"] = IND_NGRAM_TOKENS if tokenized else IND_NGRAM_BYTES
    final["vocab_size"] = cfg.vocab_size
    # so a rate_mean can never be mistaken for the controller's target_rate
    final["free_rate"] = args.free_rate
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(final, f, indent=2)
    if wb:
        wb.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
