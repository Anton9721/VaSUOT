"""Typed configuration shared by all VaSUOT experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class ModelConfig:
    """Architecture of one scalar network: potential or log-normalizer."""

    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "silu"

@dataclass
class ObjectiveConfig:
    eps: float = 0.1
    tau: float = 1.0
    num_noise_samples: int = 128
    exp_clip: float | None = None

@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0

@dataclass
class InferenceConfig:
    num_steps: int = 200
    step_size: float = 0.01
    score_clip: float | None = None
    ema_momentum: float = 0.999

@dataclass
class EvaluationConfig:
    """Shared held-out evaluation and ablation settings."""

    num_eval_samples: int = 2_048
    num_conditional_samples: int = 16
    num_seeds: int = 3
    wasserstein_max_samples: int | None = 2_000
    energy_max_samples: int | None = 5_000
    ula_step_sizes: tuple[float, ...] = (0.001, 0.003, 0.01)
    ula_num_steps: tuple[int, ...] = (50, 100, 200)

@dataclass
class TrainingConfig:
    num_steps: int = 10_000
    batch_size: int = 256
    device: str = "cuda"
    seed: int = 42
    log_every: int = 100
    validate_every: int = 1_000
    checkpoint_every: int = 1_000
    track_peak_memory: bool = True

@dataclass
class VaSUOTConfig:
    """Top-level configuration with independently ablatable network capacities.

    model remains the shared fallback for compatibility. If potential_model or
    normalizer_model is omitted, a copy of model is used for that network.
    """

    model: ModelConfig
    potential_model: ModelConfig | None = None
    normalizer_model: ModelConfig | None = None
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def resolved_potential_model(self) -> ModelConfig:
        return replace(self.model) if self.potential_model is None else self.potential_model

    def resolved_normalizer_model(self) -> ModelConfig:
        return replace(self.model) if self.normalizer_model is None else self.normalizer_model
