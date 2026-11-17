"""Typed configuration shared by all VaSUOT experiments."""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class ModelConfig:
    """Architecture of one scalar network: potential or log-normalizer."""

    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "silu"

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths.")


@dataclass
class ObjectiveConfig:
    eps: float = 0.1
    tau: float = 1.0
    num_noise_samples: int = 128
    exp_clip: float | None = None

    def __post_init__(self) -> None:
        if self.eps <= 0 or self.tau <= 0:
            raise ValueError("eps and tau must be positive.")
        if self.num_noise_samples <= 0:
            raise ValueError("num_noise_samples must be positive.")
        if self.exp_clip is not None and self.exp_clip <= 0:
            raise ValueError("exp_clip must be positive or None.")


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if len(self.betas) != 2 or not all(0 <= beta < 1 for beta in self.betas):
            raise ValueError("betas must contain two values in [0, 1).")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0:
            raise ValueError("grad_clip_norm must be positive or None.")


@dataclass
class InferenceConfig:
    num_steps: int = 200
    step_size: float = 0.01
    score_clip: float | None = None
    ema_momentum: float = 0.999

    def __post_init__(self) -> None:
        if self.num_steps < 0 or self.step_size <= 0:
            raise ValueError("num_steps must be non-negative and step_size positive.")
        if self.score_clip is not None and self.score_clip <= 0:
            raise ValueError("score_clip must be positive or None.")
        if not 0 <= self.ema_momentum < 1:
            raise ValueError("ema_momentum must lie in [0, 1).")


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

    def __post_init__(self) -> None:
        if self.num_eval_samples <= 0 or self.num_conditional_samples <= 0:
            raise ValueError("Evaluation sample counts must be positive.")
        if self.num_seeds <= 0:
            raise ValueError("num_seeds must be positive.")
        for name, value in (
            ("wasserstein_max_samples", self.wasserstein_max_samples),
            ("energy_max_samples", self.energy_max_samples),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None.")
        if not self.ula_step_sizes or any(step <= 0 for step in self.ula_step_sizes):
            raise ValueError("ula_step_sizes must contain positive values.")
        if not self.ula_num_steps or any(steps < 0 for steps in self.ula_num_steps):
            raise ValueError("ula_num_steps must contain non-negative values.")


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

    def __post_init__(self) -> None:
        if self.num_steps <= 0 or self.batch_size <= 0:
            raise ValueError("num_steps and batch_size must be positive.")
        if min(self.log_every, self.validate_every, self.checkpoint_every) <= 0:
            raise ValueError("Logging, validation, and checkpoint intervals must be positive.")


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
