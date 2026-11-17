"""Training loop for semi-unbalanced variational entropic optimal transport."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional
import time

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from config import OptimizerConfig, TrainingConfig
    from objectives import ObjectiveOutput, SemiUnbalancedVarEOTObjective
    from samplers import Sampler


@dataclass(frozen=True)
class TrainStepMetrics:
    """Scalar diagnostics emitted after one successful optimizer update."""

    step: int
    loss: float
    dual_objective: float
    additive_constant: float
    target_term: float
    normalizer_term: float
    partition_term: float
    target_clipped_fraction: float
    partition_clipped_fraction: float
    target_logit_max: float
    partition_logit_max: float
    gradient_norm: float
    step_seconds: float
    peak_memory_bytes: int | None


@dataclass(frozen=True)
class ValidationMetrics:
    """Held-out metrics emitted by an experiment-provided validation callback."""

    step: int
    values: Mapping[str, float]


StepCallback = Callable[["VaSUOTTrainer", TrainStepMetrics], None]
ValidationCallback = Callable[["VaSUOTTrainer", TrainStepMetrics], Mapping[str, float]]
CheckpointCallback = Callable[["VaSUOTTrainer", ValidationMetrics], None]


def build_optimizer(
    parameters: Iterable[nn.Parameter], config: OptimizerConfig
) -> torch.optim.Optimizer:
    """Construct the optimizer currently supported by the experiment config."""
    parameter_list = list(parameters)
    if not parameter_list:
        raise ValueError("Cannot build an optimizer with no trainable parameters.")
    if config.name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer {config.name!r}; only 'adamw' is supported.")
    return torch.optim.AdamW(
        parameter_list,
        lr=config.learning_rate,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )


class VaSUOTTrainer:
    """Optimize ``g_theta`` and ``xi_phi`` using independent source/target batches.

    The objective returns a loss for gradient descent.  The EMA copy contains
    only the potential, because ``xi_phi`` is auxiliary and is not used by ULA
    inference after training.
    """

    def __init__(
        self,
        *,
        potential: nn.Module,
        normalizer: nn.Module,
        objective: SemiUnbalancedVarEOTObjective,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        device: torch.device | str,
        grad_clip_norm: Optional[float],
        ema_momentum: float = 0.999,
        track_peak_memory: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if grad_clip_norm is not None and grad_clip_norm <= 0:
            raise ValueError(
                f"grad_clip_norm must be positive or None, got {grad_clip_norm}."
            )
        if not 0.0 <= ema_momentum < 1.0:
            raise ValueError(
                f"ema_momentum must lie in [0, 1), got {ema_momentum}."
            )

        self.device = torch.device(device)
        self.potential = potential.to(self.device)
        self.normalizer = normalizer.to(self.device)
        self.objective = objective
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.ema_momentum = ema_momentum
        self.track_peak_memory = track_peak_memory
        self.step = 0
        self.history: list[TrainStepMetrics] = []
        self.validation_history: list[ValidationMetrics] = []
        self.best_validation: ValidationMetrics | None = None

        self._parameters = list(self.potential.parameters()) + list(self.normalizer.parameters())
        if not self._parameters:
            raise ValueError("potential and normalizer must contain trainable parameters.")
        potential_ids = {id(parameter) for parameter in self.potential.parameters()}
        normalizer_ids = {id(parameter) for parameter in self.normalizer.parameters()}
        if potential_ids & normalizer_ids:
            raise ValueError("potential and normalizer must not share trainable parameters.")

        self.ema_potential = copy.deepcopy(self.potential).eval()
        for parameter in self.ema_potential.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_config(
        cls,
        *,
        potential: nn.Module,
        normalizer: nn.Module,
        objective: SemiUnbalancedVarEOTObjective,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        ema_momentum: float,
    ) -> "VaSUOTTrainer":
        """Construct a trainer and AdamW optimizer from repository configs."""
        optimizer = build_optimizer(
            list(potential.parameters()) + list(normalizer.parameters()), optimizer_config
        )
        return cls(
            potential=potential,
            normalizer=normalizer,
            objective=objective,
            optimizer=optimizer,
            batch_size=training_config.batch_size,
            device=training_config.device,
            grad_clip_norm=optimizer_config.grad_clip_norm,
            ema_momentum=ema_momentum,
            track_peak_memory=training_config.track_peak_memory,
        )

    @property
    def inference_potential(self) -> nn.Module:
        """EMA-smoothed potential to pass into ``ULASampler`` after fitting."""
        return self.ema_potential

    def train_step(self, source_sampler: Sampler, target_sampler: Sampler) -> TrainStepMetrics:
        """Perform one joint update using fresh, independent draws from ``p0`` and ``p1``."""
        self.potential.train()
        self.normalizer.train()
        self._reset_peak_memory()
        self._synchronize_if_cuda()
        started_at = time.perf_counter()
        source = source_sampler.sample(self.batch_size, device=self.device)
        target = target_sampler.sample(self.batch_size, device=self.device)

        self.optimizer.zero_grad()
        objective_output = self.objective(source, target, self.potential, self.normalizer)
        self._ensure_finite_loss(objective_output)
        objective_output.loss.backward()

        gradient_norm = self._gradient_norm_and_clip()
        if not torch.isfinite(gradient_norm):
            self.optimizer.zero_grad()
            raise FloatingPointError("Encountered a non-finite gradient norm; optimizer step skipped.")

        self.optimizer.step()
        self._update_ema()
        self._synchronize_if_cuda()
        step_seconds = time.perf_counter() - started_at
        self.step += 1
        metrics = self._to_metrics(
            objective_output,
            gradient_norm,
            step_seconds=step_seconds,
            peak_memory_bytes=self._peak_memory_bytes(),
        )
        self.history.append(metrics)
        return metrics

    def fit(
        self,
        source_sampler: Sampler,
        target_sampler: Sampler,
        *,
        num_steps: int,
        callback: Optional[StepCallback] = None,
        validation_callback: Optional[ValidationCallback] = None,
        validation_every: Optional[int] = None,
        monitor: Optional[str] = None,
        monitor_mode: str = "min",
        checkpoint_callback: Optional[CheckpointCallback] = None,
    ) -> list[TrainStepMetrics]:
        """Run ``num_steps`` updates and return the metrics produced in this call."""
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}.")
        if validation_callback is None and (
            validation_every is not None or monitor is not None or checkpoint_callback is not None
        ):
            raise ValueError("Validation settings require validation_callback.")
        if validation_callback is not None:
            if validation_every is None or validation_every <= 0:
                raise ValueError("validation_every must be positive when validation_callback is set.")
            if monitor is None:
                raise ValueError("monitor must name a validation metric when validation_callback is set.")
            if monitor_mode not in {"min", "max"}:
                raise ValueError("monitor_mode must be either 'min' or 'max'.")

        metrics_for_run: list[TrainStepMetrics] = []
        for _ in range(num_steps):
            metrics = self.train_step(source_sampler, target_sampler)
            metrics_for_run.append(metrics)
            if callback is not None:
                callback(self, metrics)
            if validation_callback is not None and self.step % validation_every == 0:
                validation = self._run_validation(validation_callback, metrics)
                self.validation_history.append(validation)
                if self._is_best(validation, monitor, monitor_mode):
                    self.best_validation = validation
                    if checkpoint_callback is not None:
                        checkpoint_callback(self, validation)
        return metrics_for_run

    def _gradient_norm_and_clip(self) -> Tensor:
        if self.grad_clip_norm is None:
            norms = [parameter.grad.detach().norm(2) for parameter in self._parameters if parameter.grad is not None]
            if not norms:
                return torch.zeros((), device=self.device)
            return torch.stack(norms).pow(2).sum().sqrt()
        return torch.nn.utils.clip_grad_norm_(self._parameters, self.grad_clip_norm)

    def _update_ema(self) -> None:
        with torch.no_grad():
            for current, averaged in zip(self.potential.parameters(), self.ema_potential.parameters()):
                averaged.mul_(self.ema_momentum).add_(current, alpha=1.0 - self.ema_momentum)
            for current, averaged in zip(self.potential.buffers(), self.ema_potential.buffers()):
                if torch.is_floating_point(averaged):
                    averaged.mul_(self.ema_momentum).add_(current, alpha=1.0 - self.ema_momentum)
                else:
                    averaged.copy_(current)

    def _reset_peak_memory(self) -> None:
        if self.track_peak_memory and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory_bytes(self) -> int | None:
        if not self.track_peak_memory or self.device.type != "cuda":
            return None
        return int(torch.cuda.max_memory_allocated(self.device))

    def _synchronize_if_cuda(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _run_validation(
        self, callback: ValidationCallback, train_metrics: TrainStepMetrics
    ) -> ValidationMetrics:
        values = dict(callback(self, train_metrics))
        if not values:
            raise ValueError("validation_callback must return at least one metric.")
        for name, value in values.items():
            if not isinstance(value, (float, int)) or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"Validation metric {name!r} must be finite.")
        return ValidationMetrics(
            step=self.step,
            values={name: float(value) for name, value in values.items()},
        )

    def _is_best(
        self, validation: ValidationMetrics, monitor: str, monitor_mode: str
    ) -> bool:
        if monitor not in validation.values:
            raise KeyError(f"Validation callback did not return monitored metric {monitor!r}.")
        if self.best_validation is None or monitor not in self.best_validation.values:
            return True
        current = validation.values[monitor]
        previous = self.best_validation.values[monitor]
        return current < previous if monitor_mode == "min" else current > previous

    @staticmethod
    def _ensure_finite_loss(output: ObjectiveOutput) -> None:
        if output.loss.ndim != 0:
            raise ValueError(f"objective loss must be scalar, got shape {tuple(output.loss.shape)}.")
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Encountered a non-finite objective loss; optimizer step skipped.")

    def _to_metrics(
        self,
        output: ObjectiveOutput,
        gradient_norm: Tensor,
        *,
        step_seconds: float,
        peak_memory_bytes: int | None,
    ) -> TrainStepMetrics:
        return TrainStepMetrics(
            step=self.step,
            loss=float(output.loss.detach().cpu()),
            dual_objective=float(output.dual_objective.cpu()),
            additive_constant=float(output.additive_constant.cpu()),
            target_term=float(output.target_term.cpu()),
            normalizer_term=float(output.normalizer_term.cpu()),
            partition_term=float(output.partition_term.cpu()),
            target_clipped_fraction=float(output.target_clipped_fraction.cpu()),
            partition_clipped_fraction=float(output.partition_clipped_fraction.cpu()),
            target_logit_max=float(output.target_logit_max.cpu()),
            partition_logit_max=float(output.partition_logit_max.cpu()),
            gradient_norm=float(gradient_norm.detach().cpu()),
            step_seconds=step_seconds,
            peak_memory_bytes=peak_memory_bytes,
        )


__all__ = [
    "CheckpointCallback",
    "TrainStepMetrics",
    "ValidationCallback",
    "ValidationMetrics",
    "VaSUOTTrainer",
    "build_optimizer",
]
