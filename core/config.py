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
