from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "silu"


@dataclass
class ObjectiveConfig:
    eps: float = 0.1
    tau: float = 1.0          # только для SU-VarEOT
    num_noise_samples: int = 128  # K
    exp_clip: float | None = None


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0


@dataclass
class InferenceConfig:
    num_steps: int = 200
    step_size: float = 0.01
    score_clip: float | None = None
    ema_momentum: float = 0.999


@dataclass
class TrainingConfig:
    num_steps: int = 10_000
    batch_size: int = 256
    device: str = "cuda"
    seed: int = 42
    log_every: int = 100
    validate_every: int = 1_000
    checkpoint_every: int = 1_000


@dataclass
class VaSUOTConfig:
    model: ModelConfig
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
