"""Variational objectives for semi-unbalanced entropic optimal transport."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from config import ObjectiveConfig


ScalarNetwork = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class ObjectiveOutput:
    """Differentiable loss and detached terms used for experiment logging."""

    loss: Tensor
    dual_objective: Tensor
    additive_constant: Tensor
    target_term: Tensor
    normalizer_term: Tensor
    partition_term: Tensor
    target_clipped_fraction: Tensor
    partition_clipped_fraction: Tensor
    target_logit_max: Tensor
    partition_logit_max: Tensor


class SemiUnbalancedVarEOTObjective:
    """Monte Carlo SU-VarEOT objective from Equation (32) of the tech report.

    The report maximizes the variational dual

    ``L = -tau E_p1[exp(g(y))] - eps E_p0[xi(x)]
         - eps E_p0,z[exp(-tau * g(x + sqrt(eps) z) / eps - xi(x))]``.

    This class returns ``-L`` (without constants) as ``loss`` so a conventional
    optimizer can use gradient descent. ``dual_objective`` restores the
    dimension-dependent constant from Equation (30), allowing absolute dual
    values to be compared across ``eps`` and ``tau``.

    ``exp_clip=None`` is the exact objective. Supplying a finite threshold is
    an explicit numerical approximation, reported in the diagnostics.
    """

    def __init__(
        self,
        *,
        eps: float,
        tau: float,
        num_noise_samples: int,
        exp_clip: Optional[float] = None,
    ) -> None:
        self.eps = float(eps)
        self.tau = float(tau)
        self.num_noise_samples = num_noise_samples
        self.exp_clip = exp_clip

    @classmethod
    def from_config(cls, config: ObjectiveConfig) -> "SemiUnbalancedVarEOTObjective":
        """Construct the objective from the repository's configuration type."""
        return cls(
            eps=config.eps,
            tau=config.tau,
            num_noise_samples=config.num_noise_samples,
            exp_clip=config.exp_clip,
        )

    def __call__(
        self,
        source: Tensor,
        target: Tensor,
        potential: ScalarNetwork,
        normalizer: ScalarNetwork,
        *,
        noise: Optional[Tensor] = None,
    ) -> ObjectiveOutput:
        """Estimate the negative variational dual from independent batches.

        Args:
            source: Draws from ``p0`` with shape ``(N0, D)``.
            target: Independent draws from ``p1`` with shape ``(N1, D)``.
            potential: Scalar network ``g_theta`` from the report.
            normalizer: Scalar auxiliary network ``xi_phi`` from the report.
            noise: Optional standard-normal samples of shape ``(N0, K, D)``.
                Providing it is useful for deterministic tests.
        """
        if noise is None:
            noise = torch.randn(
                source.shape[0],
                self.num_noise_samples,
                source.shape[1],
                device=source.device,
                dtype=source.dtype,
            )
        perturbed_source = source[:, None, :] + (self.eps**0.5) * noise
        g_target = potential(target).squeeze(-1)
        xi_source = normalizer(source).squeeze(-1)
        g_perturbed = potential(perturbed_source).squeeze(-1)

        # Evaluate exponentials in float64. This preserves the exact gradient
        # of the objective while substantially postponing overflow relative to
        # float32; non-finite values are then caught by the trainer.
        target_logits = g_target.to(torch.float64)
        partition_logits = (
            -(self.tau / self.eps) * g_perturbed - xi_source[:, None]
        ).to(torch.float64)
        target_exponent = self._clip_exponent(target_logits)
        partition_exponent = self._clip_exponent(partition_logits)

        target_term = self.tau * torch.exp(target_exponent).mean()
        normalizer_term = self.eps * xi_source.to(torch.float64).mean()
        partition_term = self.eps * torch.exp(partition_exponent).mean()
        loss = target_term + normalizer_term + partition_term

        additive_constant = self._additive_constant(source.shape[1], device=source.device)
        return ObjectiveOutput(
            loss=loss,
            dual_objective=(-loss + additive_constant).detach(),
            additive_constant=additive_constant.detach(),
            target_term=target_term.detach(),
            normalizer_term=normalizer_term.detach(),
            partition_term=partition_term.detach(),
            target_clipped_fraction=self._clipped_fraction(target_logits).detach(),
            partition_clipped_fraction=self._clipped_fraction(partition_logits).detach(),
            target_logit_max=target_logits.detach().max(),
            partition_logit_max=partition_logits.detach().max(),
        )

    def _clip_exponent(self, exponent: Tensor) -> Tensor:
        return exponent if self.exp_clip is None else exponent.clamp(max=self.exp_clip)

    def _clipped_fraction(self, logits: Tensor) -> Tensor:
        if self.exp_clip is None:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)
        return (logits > self.exp_clip).to(logits.dtype).mean()

    def _additive_constant(self, event_dim: int, *, device: torch.device) -> Tensor:
        return torch.tensor(
            self.tau + self.eps - 0.5 * self.eps * event_dim * math.log(2.0 * math.pi * self.eps),
            device=device,
            dtype=torch.float64,
        )

__all__ = ["ObjectiveOutput", "SemiUnbalancedVarEOTObjective"]
