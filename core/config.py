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
