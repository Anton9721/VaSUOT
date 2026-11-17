"""Conditional transport sampling with unadjusted Langevin dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence
import time

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


@dataclass(frozen=True)
class ULARunResult:
    """Samples and diagnostics from one ULA hyperparameter setting."""

    step_size: float
    num_steps: int
    runtime_seconds: float
    diagnostics: ULADiagnostics
    samples: Tensor


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
        self.eps = float(eps)
        self.tau = float(tau)
        self.step_size = float(step_size)
        self.num_steps = num_steps
        self.score_clip = score_clip

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
        # ULA needs ∇g even when callers wrap evaluation in torch.no_grad().
        with torch.enable_grad():
            state = current.detach().requires_grad_(True)
            potential_value = potential(state)
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
        expected_shape = (source.shape[0], num_samples, source.shape[1])
        if initial is None:
            current = source[:, None, :].expand(expected_shape).clone()
        else:
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

    @classmethod
    def sweep(
        cls,
        source: Tensor,
        potential: ScalarNetwork,
        *,
        eps: float,
        tau: float,
        step_sizes: Sequence[float],
        num_steps_values: Sequence[int],
        num_conditional_samples: int,
        score_clip: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> list[ULARunResult]:
        """Run the Cartesian ULA sensitivity sweep from the evaluation protocol.

        Each setting receives a generator reset to the same seed, so output
        differences are attributable to step size and iteration count rather
        than random-number stream ordering.
        """
        results: list[ULARunResult] = []
        for step_size in step_sizes:
            for num_steps in num_steps_values:
                sampler = cls(
                    eps=eps,
                    tau=tau,
                    step_size=step_size,
                    num_steps=num_steps,
                    score_clip=score_clip,
                )
                generator = None
                if seed is not None:
                    generator = torch.Generator(device=source.device)
                    generator.manual_seed(seed)
                _synchronize_if_cuda(source.device)
                started_at = time.perf_counter()
                samples = sampler.sample_conditional(
                    source,
                    potential,
                    num_samples=num_conditional_samples,
                    generator=generator,
                )
                _synchronize_if_cuda(source.device)
                elapsed = time.perf_counter() - started_at
                results.append(
                    ULARunResult(
                        step_size=step_size,
                        num_steps=num_steps,
                        runtime_seconds=elapsed,
                        diagnostics=sampler.diagnose(samples, source, potential),
                        samples=samples,
                    )
                )
        return results

def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = ["ULADiagnostics", "ULARunResult", "ULASampler"]
