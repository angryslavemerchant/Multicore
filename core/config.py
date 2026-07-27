from dataclasses import dataclass, field


@dataclass
class CoreConfig:
    K: int = 16                 # slots (FIFO depth in passer space)
    d_core: int = 128           # internal width
    n_heads: int = 4
    n_core_layers: int = 2      # depth of the mini-transformer over the FIFO
    ffn_mult: int = 4
    target_rate: float = 0.06   # target admission rate
    gate_temp: float = 1.0      # temperature of the soft magnitude gate
    # FREE RATE. False (default): tau is a buffer pinned every training step to
    # the exact per-batch (1 - target_rate) score quantile, so the admission
    # rate IS target_rate by construction and the model has no say in it.
    # True: tau becomes an nn.Parameter moved only by the task loss (the
    # quantile controller is off after a one-time init at target_rate), so the
    # measured rate is what the model WANTS. See core_module._tau_maintain_.
    learned_tau: bool = False
    routing: str = "threshold"  # "threshold" or "top1_recurrent"
    n_loops: int = 1             # recurrent applications for routed cores
    # True: one set of expert Q/K/V/O + FFN weights reused at every loop depth
    # (LayerNorm affine and residual scales stay per-depth either way).
    # False: UNFOLD the recurrence — n_loops independent expert weight sets, so
    # the stack is n_loops ordinary layers rather than one block applied
    # n_loops times. Same FLOPs, n_loops x the expert parameters.
    tie_loops: bool = True
    # Per-expert FIFO lengths for routed cores; () means "all use K".
    # Horizon in original-sequence positions is K/p, so with traffic held at
    # p = 1/M, varying K per expert buys TEMPORAL diversity without touching
    # the traffic balance — unlike varying p, which collapses (see
    # ROUTED_CORE_NOTES.md). Implemented by padding every ring to max(K_list)
    # and masking each expert down to its own length, so the band kernel stays
    # single-width; costs the K_max attention term for everyone.
    K_list: tuple = ()
    # BOUNDED LEARNED RATES (ROUTED_CORE_NOTES.md). Rates are free to differ,
    # but only inside a safe band — the failure mode of unconstrained top-1 is
    # collapse onto one or two experts, which starves the rest and destroys the
    # max-count-padded kernel. Three pieces, all off by default:
    #   router_bias        a learnable scalar per expert added to the routing
    #                      logits. Router rows are re-orthonormalised to unit
    #                      norm and x is layer-normed, which caps the learned
    #                      logit spread near +-1 sigma; this bias is what
    #                      actually lets traffic shares move.
    #   hash_anneal_iters  linearly decay router_hash_scale to 0 over N steps,
    #                      so the positional prior keeps experts alive early
    #                      and then gets out of the way. 0 disables.
    #   rate_lo/rate_hi    range penalty bounds. Exactly zero penalty while
    #                      every expert is inside the band.
    router_bias: bool = False
    hash_anneal_iters: int = 0
    rate_lo: float = 0.03
    rate_hi: float = 0.30
    router_range_weight: float = 1.0
    ffn_hidden: int = 0          # 0 -> ffn_mult * d_core
    inter_core_window: int = 0   # shared causal mixer window; 0 disables
    residual_scale_init: float = 0.1
    router_hash_scale: float = 0.0  # deterministic (position+loop)%M prior
    router_aux_weight: float = 0.01


@dataclass
class ModelConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    window: int = 64            # sliding-window attention width (includes self)
    max_seq_len: int = 1024
    core_layer: int = 2         # cores inserted after this block (0-based)
    cores: list = field(default_factory=list)   # list[CoreConfig]
    adapter: bool = False       # per-token control variant instead of cores
    rope: bool = True           # rotary positions (False: learned absolute)
