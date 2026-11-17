"""Conditional transport sampling with unadjusted Langevin dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional
import warnings

import torch
from torch import Tensor

if TYPE_CHECKING:
    from config import InferenceConfig
    from samplers import Sampler


ScalarNetwork = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ULADiagnostics:
    """Finite-run diagnostics; they are not a proof of Markov-chain mixing."""

    mean_score_norm: float
    max_score_norm: float
    quadratic_reference_variance_ratio: float


class ULASampler:
    """Sample the SU-VarEOT conditional plan using Equation (34).

    For a learned potential ``g`` the target conditional density is

    ``q(y | x) ∝ exp(-(0.5 * ||x - y||² + tau * g(y)) / eps)``.

    This class applies unadjusted Langevin dynamics (ULA), not a deterministic
    transport map.  Hence multiple runs for the same source point produce
    different valid conditional samples.
    """

    def __init__(
        self,
        *,
        eps: float,
        tau: float,
        step_size: float,
        num_steps: int,
        score_clip: Optional[float] = None,
    ) -> None:
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}.")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        if not isinstance(num_steps, int) or num_steps < 0:
            raise ValueError(f"num_steps must be a non-negative integer, got {num_steps!r}.")
        if score_clip is not None and score_clip <= 0:
            raise ValueError(f"score_clip must be positive or None, got {score_clip}.")

        self.eps = float(eps)
        self.tau = float(tau)
        self.step_size = float(step_size)
        self.num_steps = num_steps
        self.score_clip = score_clip
        if score_clip is not None:
            warnings.warn(
                "score_clip modifies the ULA transition kernel and therefore its target "
                "distribution. Use it only as an explicitly evaluated numerical safeguard.",
                stacklevel=2,
            )
        if step_size >= 2.0 * eps:
            warnings.warn(
                "step_size >= 2 * eps is unstable even for the quadratic reference "
                "case g(y)=0. Decrease step_size before running ULA.",
                stacklevel=2,
            )

    @classmethod
    def from_config(
        cls, config: InferenceConfig, *, eps: float, tau: float
    ) -> "ULASampler":
        """Build an ULA sampler from inference and objective hyperparameters."""
        return cls(
            eps=eps,
            tau=tau,
            step_size=config.step_size,
            num_steps=config.num_steps,
            score_clip=config.score_clip,
        )

    def score(self, current: Tensor, source: Tensor, potential: ScalarNetwork) -> Tensor:
        """Evaluate ``∇_y log q(y | x)`` for current states of shape ``(B, K, D)``."""
        self._validate_source(source)
        self._validate_current(current, source)

        # ULA needs ∇g even when callers wrap evaluation in torch.no_grad().
        with torch.enable_grad():
            state = current.detach().requires_grad_(True)
            potential_value = potential(state)
            expected_shape = (*state.shape[:-1], 1)
            if not isinstance(potential_value, Tensor) or tuple(potential_value.shape) != expected_shape:
                observed = tuple(potential_value.shape) if isinstance(potential_value, Tensor) else type(potential_value).__name__
                raise ValueError(
                    f"potential must return shape {expected_shape}, got {observed}."
                )
            gradient = torch.autograd.grad(
                potential_value.sum(), state, create_graph=False, retain_graph=False
            )[0]
        score = (source[:, None, :] - state - self.tau * gradient) / self.eps
        if self.score_clip is not None:
            score = score.clamp(min=-self.score_clip, max=self.score_clip)
        return score.detach()

    @property
    def quadratic_reference_variance_ratio(self) -> float:
        """ULA variance / exact variance for the reference case ``g(y)=0``.

        The value is ``1 / (1 - step_size / (2 * eps))`` when stable. It
        quantifies discretization bias, not finite-chain mixing error.
        """
        if self.step_size >= 2.0 * self.eps:
            return float("inf")
        return 1.0 / (1.0 - self.step_size / (2.0 * self.eps))

    def diagnose(self, current: Tensor, source: Tensor, potential: ScalarNetwork) -> ULADiagnostics:
        """Measure final score magnitudes and the analytically known ULA bias."""
        score = self.score(current, source, potential)
        norms = score.norm(dim=-1)
        return ULADiagnostics(
            mean_score_norm=float(norms.mean().cpu()),
            max_score_norm=float(norms.max().cpu()),
            quadratic_reference_variance_ratio=self.quadratic_reference_variance_ratio,
        )

    def sample_conditional(
        self,
        source: Tensor,
        potential: ScalarNetwork,
        *,
        num_samples: int = 1,
        initial: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generate ``num_samples`` conditional target samples for every source point.

        Returns a tensor of shape ``(batch_size, num_samples, event_dim)``.
        """
        self._validate_source(source)
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer, got {num_samples!r}.")
        if generator is not None and generator.device != source.device:
            raise ValueError("generator and source must be on the same device.")

        expected_shape = (source.shape[0], num_samples, source.shape[1])
        if initial is None:
            current = source[:, None, :].expand(expected_shape).clone()
        else:
            self._validate_current(initial, source, num_samples=num_samples)
            current = initial.detach().clone()

        noise_scale = (2.0 * self.step_size) ** 0.5
        for _ in range(self.num_steps):
            drift = self.score(current, source, potential)
            noise = torch.randn(
                current.shape,
                device=current.device,
                dtype=current.dtype,
                generator=generator,
            )
            current = current + self.step_size * drift + noise_scale * noise
        return current.detach()

    def sample_marginal(
        self,
        source_sampler: Sampler,
        potential: ScalarNetwork,
        *,
        num_source_samples: int,
        num_conditional_samples: int = 1,
        device: Optional[torch.device | str] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Draw source points and transport each one through the conditional plan."""
        source = source_sampler.sample(num_source_samples, device=device)
        return self.sample_conditional(
            source,
            potential,
            num_samples=num_conditional_samples,
            generator=generator,
        )

    @staticmethod
    def _validate_source(source: Tensor) -> None:
        if not isinstance(source, Tensor):
            raise TypeError("source must be a torch.Tensor.")
        if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
            raise ValueError(
                "source must have shape (positive_batch_size, positive_event_dim), "
                f"got {tuple(source.shape)}."
            )
        if not torch.is_floating_point(source):
            raise TypeError("source must have a floating-point dtype.")

    @staticmethod
    def _validate_current(
        current: Tensor, source: Tensor, *, num_samples: Optional[int] = None
    ) -> None:
        if not isinstance(current, Tensor):
            raise TypeError("current must be a torch.Tensor.")
        if current.ndim != 3:
            raise ValueError("current must have shape (batch_size, num_samples, event_dim).")
        expected_num_samples = current.shape[1] if num_samples is None else num_samples
        expected_shape = (source.shape[0], expected_num_samples, source.shape[1])
        if tuple(current.shape) != expected_shape:
            raise ValueError(f"current must have shape {expected_shape}, got {tuple(current.shape)}.")
        if current.device != source.device or current.dtype != source.dtype:
            raise ValueError("current and source must have the same device and dtype.")


__all__ = ["ULADiagnostics", "ULASampler"]
